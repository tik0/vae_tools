from enum import Enum
import sys, os
import itertools
import tensorflow as tf
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from vae_tools import sampling, setfun, custom_variational_layer
from keras import metrics
from keras import backend as K
from keras.models import Model
from keras.models import model_from_json
try:
    from keras.layers.merge import concatenate as concat
except:
    from tensorflow.python.keras.layers.merge import concatenate as concat

class ReconstructionLoss(Enum):
    '''The reconstruction losses'''
    MSE = 1
    BCE = 2
    def __str__(self):
        return str(self._member_names_)
    

class WarmupMethod(Enum):
    '''The Warmup methods'''
    LINEAR = 1
    RELU = 2
    def __str__(self):
        return str(self._member_names_)

class GenericVae():
    """A generic VAE model
    
    Member Variables:
        name            string: The model name used for identifying storage and loading
        # x               List of model inputs
        # y               List of model outputs
        # z_mean          Latent mean
        # z_logvar        Latent log variances
        z_dim           Latent dimension
        encoder         List of M uni-modal encoder networks e: [e_1, e_2, ..., e_M]
                        where e_m is a list having of L' layers: [l_1, l_2, ..., l_L]
        decoder         List of M uni-modal decoder networks d: [d_1, d_2, ..., d_M]
                        where d_m is a list having of L* layers: [l_1, l_2, ..., l_L]
        encoder_inputs  List of encoder inputs [l_1_e_1, l_1_e_2, ..., l_1_e_M]
        decoder_outputs List of decoder outputs [l_L_d_1, l_L_d_2, ..., l_L_d_M]
    """
    
    def __init__(self, z_dim, encoder, decoder, encoder_inputs_dim, name='GenericVae', \
                 reconstruction_loss_metrics = [ReconstructionLoss.MSE]):
        # Sanity checks
        # Single reconstruction loss was given
        if type(reconstruction_loss_metrics) is not list:
            reconstruction_loss_metrics = [reconstruction_loss_metrics]
        if (len(reconstruction_loss_metrics) == 1) and (len(reconstruction_loss_metrics) != len(encoder)):
            reconstruction_loss_metrics = [reconstruction_loss_metrics[0] for idx in range(len(encoder))]
        if (len(reconstruction_loss_metrics) != 1) and (len(reconstruction_loss_metrics) != len(encoder)):
            raise Exception("reconstruction_loss_metrics needs to match the size of the modality inputs")
        # Single encoder input dimension was given
        if type(encoder_inputs_dim) is not list:
            encoder_inputs_dim = [encoder_inputs_dim]
        if (len(encoder_inputs_dim) == 1) and (len(encoder_inputs_dim) != len(encoder)):
            encoder_inputs_dim = [encoder_inputs_dim[0] for idx in range(len(encoder))]
        if (len(encoder_inputs_dim) != 1) and (len(encoder_inputs_dim) != len(encoder)):
            raise Exception("encoder_inputs_dim needs to match the size of the modality inputs")
        # Copy the parameters
        self.name = name
        self.z_dim = z_dim
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_inputs_dim = encoder_inputs_dim
        self.reconstruction_loss_metrics = reconstruction_loss_metrics
        # Build corresponding powersets
        self.encoder_inputs = [encoder[0] for encoder in self.encoder]
        self.decoder_outputs = [decoder[-1] for decoder in self.decoder]
        self.encoder_powerset = setfun.powerset(self.encoder)
        self.decoder_powerset = setfun.powerset(self.decoder)
        self.decoder_outputs = [decoder[-1] for decoder in self.decoder]
        self.encoder_inputs_powerset = setfun.powerset(self.encoder_inputs, minimum_elements_per_set = 1, sets_as_list = True)
        self.encoder_inputs_dim_powerset = setfun.powerset(encoder_inputs_dim, minimum_elements_per_set = 1, sets_as_list = True)
        self.reconstruction_loss_metrics_powerset = setfun.powerset(reconstruction_loss_metrics, minimum_elements_per_set = 1, sets_as_list = True)

    #def get_model(self, get_new_model = False):
    #    if 'model' not in dir(self) or get_new_model:
    #        self.model = Model(self.x, self.y)
    #    return self.model


    def get_encoder_mean(self):
        ''' Get the encoder model for mean values'''
        pass
        #return self(self.x, self.z_mean)

    def get_encoder_logvar(self):
        ''' Get the encoder model for logvar values'''
        pass
        #eturn self(self.x, self.z_log_var)
    
    def store_model(self, name = None, model = None, overwrite = False):
        ''' Store any model'''
        if model is None:
            raise Exception('Specify a model to store')
        if name is None: # take the name of the class
            name = self.name
        filename_json = name + ".json"
        filename_h5 = name + ".h5"
        # serialize model to JSON
        if not os.path.isfile(filename_json) or overwrite:
            model_json = model.to_json()
            with open(filename_json, "w") as json_file:
                json_file.write(model_json)
            print("Saved model " + name + " to disk")
        if not os.path.isfile(filename_h5) or overwrite:
            # serialize weights to HDF5
            model.save_weights(filename_h5, save_format='h5')
            print("Saved weights of model " + name + " to disk")

    def load_model(self, name = None):
        ''' Load any model'''
        if name is None: # take the name of the class
            name = self.name
        filename_json = name + ".json"
        filename_h5 = name + ".h5"
        # load json and create model
        json_file = open(filename_json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(filename_h5)
        print("Loaded model " + name + " from disk")
        return loaded_model
    
    def store_model_powerset(self, prefix, model_inputs, get_model_callback = None):
        ''' Stores the models of a powerset model given it's model inputs
        This function stores the models with the corresponding input as bitmask:
        e.g. store_model_powerset('enc_mean_xw_', vae_obj.encoder_inputs, vae_obj.get_encoder_mean)
        stores the models:
        enc_mean_xw_10.* : Encoder with input x
        enc_mean_xw_01.* : Encoder with input w
        enc_mean_xw_11.* : Encoder with input x and w

        prefix              (str): Some prefix name for storing json and h5 (e.g. enc_mean_xw_)
        model_inputs       (list): List of keras input layers of the model in corresponding order of bitmask [input_x,input_w]
        get_model_callback   (cb): Callback function which returns a graph model given a subset of model inputs

        '''
        bitmask_powerset, bitmask_powerset_str = setfun.get_bitmask_powerset(num_elements = len(model_inputs))
        for bitmask_set, bitmask_set_str in zip(bitmask_powerset[1:], bitmask_powerset_str[1:]):
            model_input = list(itertools.compress(model_inputs, bitmask_set))
            self.store_model(name = prefix + bitmask_set_str, 
                                model = get_model_callback(model_input), overwrite = True)

    def load_model_powerset(self, prefix, num_elements):
        ''' Load models of a powerset model given the number of elements
        This function loads the models with the corresponding:
        e.g. load_model_powerset('enc_mean_xw_', num_elements = 2) loads the models:
        enc_mean_xw_10: Encoder as the first list element with bitmask 10
        enc_mean_xw_01: Encoder as the second list element with bitmask 01
        enc_mean_xw_11: Encoder as the third list element with bitmask 11

        returns list of loaded models and coresponding bitmask
        '''
        encoder_powerset = []
        bitmask_powerset, bitmask_powerset_str = setfun.get_bitmask_powerset(num_elements)
        for bitmask_set, bitmask_set_str in zip(bitmask_powerset[1:], bitmask_powerset_str[1:]):
            encoder_powerset.append(self.load_model(name = prefix + bitmask_set_str))
        return encoder_powerset, bitmask_powerset
    
class Warmup:
    '''The Warmup class for value definitions'''
    def __init__(self, v_init = 0.0, v_max = 1.0, v_max_epoch = 10, method = WarmupMethod.LINEAR):
        self.v_init = v_init
        self.v_max = v_max
        self.v_max_epoch = v_max_epoch
        self.method = method

    def __str__(self):
        return str(vars(self))

from keras.callbacks import LambdaCallback
class Losslayer(Layer):
    '''Generic loss layer'''
    def __init__(self, **kwargs):
        '''
        weight              : A static weight value which is used if warmup is None
        warmup              : A Warmup object defining the warmup function
        '''
        #self.weight = K.variable(value = weight if warmup is not None else warmup.v_init)
        self.weight = K.variable(value=kwargs.pop('weight', 1.0))
        self.warmup = kwargs.pop('warmup', None)
        if self.warmup is not None: # we use the values from warmup instead
            K.set_value(self.weight, warmup.v_init)
        self.is_placeholder = True
        super().__init__(**kwargs)
        
    def warmup_linear(self, epoch):
        slope = self.warmup.v_max - self.warmup.v_init
        # ramping up + const (we start with epoch=0 with on_epoch_begin)
        if epoch <= self.warmup.v_max_epoch:
            value = self.warmup.v_init + slope * (epoch/self.warmup.v_max_epoch)
        else: # epoch > self.warmup.v_max_epoch
            value = self.warmup.v_max
        #print("weight (", self.name, "): ", value)
        K.set_value(self.weight, value)
    
    def warmup_relu(self, epoch):
        raise Exception("TBD")
        
    def get_cb_warmup(self, method):
        ''' Returns the callback method for warmup
        method         : A WarmupMethod type
        '''
        cb = {
            WarmupMethod.LINEAR: self.warmup_linear,
            WarmupMethod.RELU: self.warmup_relu,
        }
        return LambdaCallback(on_epoch_begin=lambda epoch, log: cb.get(method)(epoch))
        
class LosslayerDistributionGaussianPrior(Losslayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # We assume always a single input of the latent layer consisting of mean and log_var
        z_mean = inputs[0]
        z_logvar = inputs[1]

        # Calculate the KL divergences wrt. the prior
        kl_prior_loss = K.mean(self.weight * K.sum(custom_variational_layer.kl_loss_n(z_mean, z_logvar), axis=-1))

        # Define the final loss
        self.add_loss(kl_prior_loss, inputs=inputs)

        # Return the loss value
        return kl_prior_loss
    

# Custom loss layer for vanilla VAE
class LosslayerDistributionGaussianMutual(Losslayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # We assume always a single input of the latent layer consisting of mean and log_var
        # the left (l) and right (r) argument aka DKL(N_l || N_r)
        z_mean_l = inputs[0]
        z_logvar_l = inputs[1]
        z_mean_r = inputs[2]
        z_logvar_r = inputs[3]

        # Calculate the mutual KL divergences
        kl_mutual_loss = K.mean(self.weight * K.sum(custom_variational_layer.kl_loss(z_mean_l, z_logvar_l, z_mean_r, z_logvar_r), axis=-1))
        
        # Define the final loss
        self.add_loss(kl_mutual_loss, inputs=inputs)

        # Return the loss value
        return kl_mutual_loss

class LosslayerReconstruction(Losslayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def metric(self, x, x_decoded):
        '''Calculation of the metric with x as input and x_decoded as output signal'''
        pass

    def call(self, inputs):
        '''We assume always a single input and ouput'''
        x = K.flatten(inputs[0]) # Inputs
        x_decoded_mean = K.flatten(inputs[1]) # Output
        
        reconstruction_loss = self.metric(x, x_decoded_mean)
        ## Define the final loss
        reconstruction_loss = K.sum(reconstruction_loss)
        self.add_loss(reconstruction_loss, inputs=inputs)
        # Return the loss value
        return reconstruction_loss

class LosslayerReconstructionMSE(LosslayerReconstruction):
    '''Loss layer for element-wise reconstruction with binary cross-entropy'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def metric(self, x, x_decoded):
        #print("K.get_value(self.weight): ", K.get_value(self.weight))
        return self.weight * metrics.mean_squared_error(x, x_decoded)

class LosslayerReconstructionBCE(LosslayerReconstruction):
    '''Loss layer for element-wise reconstruction with binary cross-entropy'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def metric(self, x, x_decoded):
        return self.weight * metrics.binary_crossentropy(x, x_decoded)
    
class MmVae(GenericVae, sampling.Sampling):

    def __init__(self, z_dim, encoder, decoder, encoder_inputs_dim, beta, reconstruction_loss_metrics, name='MmVae'):
        super().__init__(z_dim=z_dim, encoder=encoder, decoder=decoder,name=name,
                         reconstruction_loss_metrics = reconstruction_loss_metrics,
                         encoder_inputs_dim = encoder_inputs_dim)
        # loss parameter
        self.beta = beta
        #self.reconstruction_loss = reconstruction_loss
        # model parameter
        self.sampling_layer = Lambda(self.randn, output_shape=(self.z_dim,), name='sample')
        

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
            # Concat all elements of the current set
            # Note: It is "just" concat to enforce the parameter layers (z_mean & z_logvar) to learn linear
            #       combinations of the uni-modal cases
            # TODO: Extension to add a non-linear layer after the concatination
            if len(element_output) > 1:
                encoder_outputs_powerset.append(concat(element_output))
                #################################################################################################
                #encoder_outputs_powerset.append(Dense(int(intermediate_dim/2), activation='relu')(concat(element_output)))
                #################################################################################################
            else:
                encoder_outputs_powerset.append(element_output[0])

        # Add sampling for every permutation
        self.Z = []
        self.Z_mean = []
        self.Z_logvar = []
        for encoder_output in encoder_outputs_powerset:
            self.Z_mean.append(Dense(self.z_dim)(encoder_output))
            self.Z_logvar.append(Dense(self.z_dim)(encoder_output))
            self.Z.append(self.sampling_layer([self.Z_mean[-1], self.Z_logvar[-1]]))
            
        # Add a decoder output for every permutation
        self.decoder_outputs_powerset = []
        for z, decoder_set in zip(self.Z, self.decoder_powerset):
            element_output = []
            for decoder_element in decoder_set:
                element_output.append(decoder_element[0](z)) # Input the sample z into the first decoder layer
                # Append the next layer to the currently appended element in the set
                for decoder_layer in decoder_element[1:]:
                    #print(element_output[-1])
                    element_output[-1] = decoder_layer(element_output[-1])
            self.decoder_outputs_powerset.append(element_output)
        
        # collection of loss layers
        self.loss_layers = []
        
        # Calculate entropy losses for all sets in the powerset
        reconstruction_loss = []
        # Traverse the sets of the powerset
        for x_set, x_decoded_mean_set, encoder_inputs_dim, reconstruction_loss_metrics in \
                                                             zip(self.encoder_inputs_powerset, \
                                                                 self.decoder_outputs_powerset, \
                                                                 self.encoder_inputs_dim_powerset, \
                                                                 self.reconstruction_loss_metrics_powerset):
            reconstruction_loss_set = [] # Holds the loss for the whole powerset
            # Traverse the elements per set
            for x, x_decoded_mean, encoder_input_dim, reconstruction_loss_metric in zip(x_set, \
                                                                                        x_decoded_mean_set, \
                                                                                        encoder_inputs_dim, \
                                                                                        reconstruction_loss_metrics):

                # Choose the proper reconstruction loss metric
                #print("encoder_input_dim: ", encoder_input_dim)
                rl = {
                    ReconstructionLoss.MSE: LosslayerReconstructionMSE(weight=encoder_input_dim),
                    ReconstructionLoss.BCE: LosslayerReconstructionBCE(weight=encoder_input_dim),
                }
                #print("reconstruction_loss_metric: ", reconstruction_loss_metric)
                loss_layer = rl.get(reconstruction_loss_metric)
                self.loss_layers.append(loss_layer) # Backup the layer for callbacks, etc.
                loss = loss_layer([x, x_decoded_mean])
                reconstruction_loss_set.append(loss)
            reconstruction_loss.extend(reconstruction_loss_set)
        
        # Calculate the prior losses
        kl_prior_loss = []
        for z_mean, z_logvar, inputs in zip(self.Z_mean, self.Z_logvar, self.encoder_powerset):
            loss_layer = LosslayerDistributionGaussianPrior(weight=self.beta)
            self.loss_layers.append(loss_layer) # Backup the layer for callbacks, etc.
            loss = loss_layer([z_mean, z_logvar])
            kl_prior_loss.append(loss)
            
        # Calculate the mutual KL divergences for the to sets A and B of the powerset,
        # where |A|=|B|-1 and A is a proper subset of B (which is always valid for only one pair of sets)
        kl_mutual_loss = []
        subset_idx, superset_idx = setfun.find_proper_subsets(self.encoder_inputs_powerset, cardinality_difference = 1, debug = True)
        for A_idx, B_idx in zip(subset_idx, superset_idx):
            loss_layer = LosslayerDistributionGaussianMutual(weight=self.beta)
            self.loss_layers.append(loss_layer) # Backup the layer for callbacks, etc.
            loss = loss_layer([self.Z_mean[B_idx], self.Z_mean[A_idx], self.Z_logvar[B_idx], self.Z_logvar[A_idx]])
            kl_mutual_loss.append(loss)

        loss_list = reconstruction_loss + kl_prior_loss + kl_mutual_loss
        #print("\n +++++++++++++++++++++++ \n")
        #print("loss_list: ", loss_list)
        #print("reconstruction_loss: ", reconstruction_loss)
        #print("kl_prior_loss: ", kl_prior_loss)
        #print("kl_mutual_loss: ", kl_mutual_loss)
        #print("\n +++++++++++++++++++++++ \n")
        return loss_list
    
    def get_callback_functions(self):
        '''Returns the callback functions for warmup for now'''
        return [cb for cb in self.loss_layers.get_cb_warmup()]
    
    def get_model(self, get_new_model = False):
        self.y = self._configure()
        #print("\n ------------------------ \n")
        #print("self.encoder_inputs", self.encoder_inputs)
        #print("self.y", self.y)
        if 'model' not in dir(self) or get_new_model:
            self.model = Model(inputs = self.encoder_inputs, outputs = self.y)
        return self.model
    
    def get_encoder_mean(self, encoder_input_list):
        set_idx = setfun.get_set_idx_in_powerset(set(encoder_input_list),
                                                 setfun.powerset(self.encoder_inputs, sets_as_set = True))
        return Model(self.encoder_inputs_powerset[set_idx], self.Z_mean[set_idx])
    
    def get_encoder_logvar(self, encoder_input_list):
        set_idx = setfun.get_set_idx_in_powerset(set(encoder_input_list),
                                                 setfun.powerset(self.encoder_inputs, sets_as_set = True))
        return Model(self.encoder_inputs_powerset[set_idx], self.Z_logvar[set_idx])
    
    def get_decoder(self, latent_input = None, decoder_output_list = None):
        ''' Returns the decoder model
        latent_input               : Some own keras layers which should be used as input
        decoder_output_list  (list): List of decoder output layers for which the decoder should be build

        rerturns a decoder model with the desired decoder outputs
        '''
        if latent_input is None:
            latent_input = Input(shape=(self.z_dim,))
        decoder_outputs = []
        if decoder_output_list is None:
            decoder = self.decoder
        else: # Compare all decoder outputs with the desired outputs and store the net
            decoder = []
            for decoder_net in self.decoder:
                for decoder_output in decoder_output_list:
                    if decoder_net[-1] == decoder_output:
                        decoder.append(decoder_net)
        # Build the final decoder model
        for current_decoder in decoder:
            decoder_outputs.append(current_decoder[0](latent_input))
            for decoder_element in current_decoder[1:]:
                decoder_outputs[-1] = decoder_element(decoder_outputs[-1])
        return Model(latent_input, decoder_outputs)
    
    def get_encoder_decoder(self, encoder_input_list):
        '''ATTENTION: This model applies sampling in the hidden layer'''
        model_encoder_mean = self.get_encoder_mean(encoder_input_list)
        model_encoder_logvar = self.get_encoder_logvar(encoder_input_list)
        input_encoder_mean = Input(shape=(self.z_dim,))
        input_encoder_logvar = Input(shape=(self.z_dim,))
        z = self.sampling_layer([input_encoder_mean, input_encoder_logvar])
        model_z = Model([input_encoder_mean, input_encoder_logvar], z)
        model_decoder = self.get_decoder()
        output = model_decoder(model_z([model_encoder_mean(encoder_input_list), model_encoder_logvar(encoder_input_list)]))
        return Model(encoder_input_list, output)
        
        
        
        set_idx = setfun.get_set_idx_in_powerset(set(encoder_input_list), setfun.powerset(self.encoder_inputs, sets_as_set = True))
        print("self.encoder_inputs_powerset[set_idx]", self.encoder_inputs_powerset[set_idx])
        print("self.decoder_outputs", self.decoder_outputs)
        return Model(self.encoder_inputs_powerset[set_idx], self.decoder_outputs_powerset[-1])
