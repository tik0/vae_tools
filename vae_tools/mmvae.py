from enum import Enum
import sys, os
import itertools
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from vae_tools import sampling, setfun, custom_variational_layer
from tensorflow.keras import metrics
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import concatenate as concat

from vae_tools.vae import *

class MmVae(GenericVae):

    def __init__(self,
                 z_dim,
                 encoder,
                 decoder,
                 encoder_inputs_dim,
                 beta,
                 beta_is_normalized = False,
                 beta_mutual = 1.0,
                 reconstruction_loss_metrics = [ReconstructionLoss.MSE],
                 latent_encoder = None,
                 sampling_obj = None,
                 name='MmVae'):
        super().__init__(z_dim=z_dim, encoder=encoder, decoder=decoder,name=name,
                         reconstruction_loss_metrics = reconstruction_loss_metrics,
                         beta = beta, beta_is_normalized = beta_is_normalized,
                         encoder_inputs_dim = encoder_inputs_dim)
        if sampling_obj is None: # Use standard sampling with linear mean and logvar layer
            self.sampling_obj = sampling.RandN(self.z_dim)
        else:
            self.sampling_obj = sampling_obj
        self.beta_mutual = beta_mutual
        self.latent_encoder = latent_encoder
        self.y = self._configure()

    def _configure(self):
        
        # build the encoder outputs for the whole powerset
        encoder_outputs_powerset = [] # Holds M=|encoder_powerset|-1 encoder networks
        for encoder_set in self.encoder_powerset:
            element_output = []
            for encoder_element in encoder_set:
                element_output.append(encoder_element[0]) # The first element is always an input layer
                # Append the next layer to the currently appended element in the set
                for encoder_layer in encoder_element[1:]:
                    element_output[-1] = encoder_layer(element_output[-1])
            # Concat all elements of the current set and append a latent encoder if desired
            if len(element_output) > 1:
                encoder_outputs_powerset.append(concat(element_output))
                if self.latent_encoder is not None: # define the hidden encoder network
                    for r, d, a in  zip(self.latent_encoder.is_relative, self.latent_encoder.layer_dimensions, self.latent_encoder.activations):
                        # Get the output shape of the former layer and calulate the size of the current layer
                        shape = d
                        if r:
                            shape *= encoder_outputs_powerset[-1].shape[-1]
                        encoder_outputs_powerset[-1] = Dense(int(shape), activation=a)(encoder_outputs_powerset[-1])
            else:
                encoder_outputs_powerset.append(element_output[0])

        # Add sampling for every permutation
        # TODO allow external definition of sampling layers (e.g. with pretrained weights)
        #self.Z = []
        #self.Z_mean = []
        #self.Z_logvar = []
        #for encoder_output, idx_set in zip(encoder_outputs_powerset, range(len(self.encoder_inputs_powerset))):
        #    self.Z_mean.append(Dense(self.z_dim, name="mean_" + str(idx_set))(encoder_output))
        #    self.Z_logvar.append(Dense(self.z_dim, name="logvar_" + str(idx_set))(encoder_output))
        #    self.Z.append(self.sampling_layer([self.Z_mean[-1], self.Z_logvar[-1]]))
        self.Z, self.Z_layers_powerset = self.sampling_obj.get_sampling(encoder_outputs_powerset)
        self.Z_mean = [layer["mean"] for layer in self.Z_layers_powerset]
        self.Z_logvar = [layer["logvar"] for layer in self.Z_layers_powerset]

        # Add a decoder output for every permutation
        self.decoder_outputs_powerset = []
        for z, decoder_set in zip(self.Z, self.decoder_powerset):
            element_output = []
            for decoder_element in decoder_set:
                element_output.append(decoder_element[0](z)) # Input the sample z into the first decoder layer
                # Append the next layer to the currently appended element in the set
                for decoder_layer in decoder_element[1:]:
                    element_output[-1] = decoder_layer(element_output[-1])
            self.decoder_outputs_powerset.append(element_output)
        
        # collection of loss layers
        self.loss_layers = []
        
        # Calculate entropy losses for all sets in the powerset
        reconstruction_loss = self._losses_reconstruction()

        # Calculate the prior losses for all sets in the powerset
        kl_prior_loss = self._losses_prior()
            
        # Calculate the mutual KL divergences for the to sets A and B of the powerset,
        # where |A|=|B|-1 and A is a proper subset of B (which is always valid for only one pair of sets)
        kl_mutual_loss = self._losses_mutual()

        loss_list = reconstruction_loss + kl_prior_loss + kl_mutual_loss
        #print("\n +++++++++++++++++++++++ \n")
        #print("loss_list: ", loss_list)
        #print("reconstruction_loss: ", reconstruction_loss)
        #print("kl_prior_loss: ", kl_prior_loss)
        #print("kl_mutual_loss: ", kl_mutual_loss)
        #print("\n +++++++++++++++++++++++ \n")
        return loss_list

    def _losses_reconstruction(self):
        ''' Returns the reconstruction losses for all sets'''
        reconstruction_loss = []
        # Traverse the sets of the powerset
        for x_set, x_decoded_mean_set, encoder_inputs_dim, reconstruction_loss_metrics, idx_set in \
                                                             zip(self.encoder_inputs_powerset, \
                                                                 self.decoder_outputs_powerset, \
                                                                 self.encoder_inputs_dim_powerset, \
                                                                 self.reconstruction_loss_metrics_powerset, \
                                                                 range(len(self.encoder_inputs_powerset))):
            reconstruction_loss_set = [] # Holds the loss for the whole powerset
            # Traverse the elements per set
            for x, x_decoded_mean, encoder_input_dim, reconstruction_loss_metric, idx_input in zip(x_set, \
                                                                                        x_decoded_mean_set, \
                                                                                        encoder_inputs_dim, \
                                                                                        reconstruction_loss_metrics, \
                                                                                        range(len(x_set))):

                # Choose the proper reconstruction loss metric
                loss_layer = self.get_reconstruction_loss(reconstruction_loss_metric, weight = encoder_input_dim,
                                                           name="loss_reconstruction_" + str(idx_set) + "_" + str(idx_input))
                self.loss_layers.append(loss_layer) # Backup the layer for callbacks, etc.
                loss = loss_layer([x, x_decoded_mean])
                reconstruction_loss_set.append(loss)
            reconstruction_loss.extend(reconstruction_loss_set)
        return reconstruction_loss

    def _losses_mutual(self):
        ''' Returns the mutual losses for all sets'''
        kl_mutual_loss = []
        subset_idx, superset_idx = setfun.find_proper_subsets(self.encoder_inputs_powerset, cardinality_difference = 1, debug = False)
        for A_idx, B_idx, idx_mutual in zip(subset_idx, superset_idx, range(len(superset_idx))):
            loss_layer = LosslayerDistributionGaussianMutual(weight=self.beta_mutual, name = "loss_mutual_" + str(idx_mutual))
            self.loss_layers.append(loss_layer) # Backup the layer for callbacks, etc.
            loss = loss_layer([self.Z_mean[B_idx], self.Z_mean[A_idx], self.Z_logvar[B_idx], self.Z_logvar[A_idx]])
            kl_mutual_loss.append(loss)
        return kl_mutual_loss

    def _losses_prior(self):
        ''' Returns the prior losses for all sets'''
        kl_prior_loss = []
        for z_mean, z_logvar, idx_set, encoder_inputs_dim in zip(self.Z_mean, self.Z_logvar, range(len(self.encoder_powerset)),
                                                                self.encoder_inputs_dim_powerset):
            loss_layer = LosslayerDistributionGaussianPrior(weight=self.get_beta(x_dim = sum(encoder_inputs_dim)), name="loss_prior_" + str(idx_set))
            self.loss_layers.append(loss_layer) # Backup the layer for callbacks, etc.
            loss = loss_layer([z_mean, z_logvar])
            kl_prior_loss.append(loss)
        return kl_prior_loss
    
    def get_callback_functions(self):
        '''Returns the callback functions for warmup for now'''
        return [cb for cb in self.loss_layers.get_cb_warmup()]
    
    def get_model(self, get_new_model = False, extra_inputs = []):
        #print("\n ------------------------ \n")
        #print("self.encoder_inputs", self.encoder_inputs)
        #print("self.y", self.y)
        if 'model' not in dir(self) or get_new_model:
            self.model = Model(inputs = self.encoder_inputs + extra_inputs, outputs = self.y)
        # add losses (i.e. output layers which name starts with 'loss_')
        # we assume that they already emit a scalar value for each batch, thus, the aggregation does alter the value
        for output in self.model.outputs:
            if output.name.split(sep='/')[0][:5] == "loss_":
                self.model.add_metric(output, name=output.name.split(sep='/')[0], aggregation="mean")
        return self.model
    
    def get_encoder_mean(self, encoder_input_list, extra_inputs = []):
        ''' Returns the decoder model with the mean layer
        encoder_input_list   (list): List of encoder input layers for which the decoder should be build
        extra_inputs         (list): List of additional inputs that are concatenated with the common input (see cvae.ipynb for an example)

        returns an encoder model with the desired configuration
        '''
        encoder_input_list_ref = [t.experimental_ref() for t in encoder_input_list]
        set_idx = setfun.get_set_idx_in_powerset(set(encoder_input_list_ref),
                                                 setfun.powerset(self.encoder_inputs_ref, sets_as_set = True))
        return Model(self.encoder_inputs_powerset[set_idx] + extra_inputs, self.Z_mean[set_idx])
    
    def get_encoder_logvar(self, encoder_input_list, extra_inputs = []):
        ''' Returns the decoder model with the logvar layer
        encoder_input_list   (list): List of encoder input layers for which the decoder should be build
        extra_inputs         (list): List of additional inputs that are concatenated with the common input (see cvae.ipynb for an example)

        returns an encoder model with the desired configuration
        '''
        encoder_input_list_ref = [t.experimental_ref() for t in encoder_input_list]
        set_idx = setfun.get_set_idx_in_powerset(set(encoder_input_list_ref),
                                                 setfun.powerset(self.encoder_inputs_ref, sets_as_set = True))
        return Model(self.encoder_inputs_powerset[set_idx] + extra_inputs, self.Z_logvar[set_idx])
    
    def get_decoder(self, latent_input = None, decoder_output_list = None, extra_inputs = []):
        ''' Returns the decoder model
        latent_input               : Some own keras layers which should be used as input
        decoder_output_list  (list): List of decoder output layers for which the decoder should be build
        extra_inputs         (list): List of additional inputs that are concatenated with the common input (see cvae.ipynb for an example)

        returns a decoder model with the desired decoder outputs
        '''
        if latent_input is None:
            latent_input = Input(shape=(self.z_dim,))
        decoder_outputs = []
        if decoder_output_list is None:
            decoder = self.decoder
        else: # Compare all decoder outputs with the desired outputs and store the net
            decoder = []
            for decoder_output in decoder_output_list:
                for decoder_net in self.decoder:
                    if decoder_net[-1] == decoder_output:
                        decoder.append(decoder_net)
                        break
        # Build the final decoder model
        for current_decoder in decoder:
            decoder_outputs.append(current_decoder[0](latent_input))
            for decoder_element in current_decoder[1:]:
                decoder_outputs[-1] = decoder_element(decoder_outputs[-1])
        return Model([latent_input] + extra_inputs, decoder_outputs)
    
    def get_encoder_decoder(self, encoder_input_list, extra_inputs = []):
        ''' Returns the decoder model
        ATTENTION: This model applies sampling in the hidden layer

        encoder_input_list   (list): List of encoder input layers for which the model should be build
        extra_inputs         (list): List of additional inputs that are concatenated with the common input (see cvae.ipynb for an example)

        returns an end-to-end encoder-decoder model with sampling in the hidden layer
        '''
        model_encoder_mean = self.get_encoder_mean(encoder_input_list)
        model_encoder_logvar = self.get_encoder_logvar(encoder_input_list)
        input_encoder_mean = Input(shape=(self.z_dim,))
        input_encoder_logvar = Input(shape=(self.z_dim,))
        z = self.sampling_layer([input_encoder_mean, input_encoder_logvar])
        model_z = Model([input_encoder_mean, input_encoder_logvar], z)
        model_decoder = self.get_decoder()
        output = model_decoder(model_z([model_encoder_mean(encoder_input_list), model_encoder_logvar(encoder_input_list)]))
        return Model(encoder_input_list + extra_inputs, output)
        #set_idx = setfun.get_set_idx_in_powerset(set(encoder_input_list), setfun.powerset(self.encoder_inputs, sets_as_set = True))
        #print("self.encoder_inputs_powerset[set_idx]", self.encoder_inputs_powerset[set_idx])
        #print("self.decoder_outputs", self.decoder_outputs)
        #return Model(self.encoder_inputs_powerset[set_idx], self.decoder_outputs_powerset[-1])
