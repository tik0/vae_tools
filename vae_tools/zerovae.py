from vae_tools.mmvae import MmVae
from vae_tools.vae import *

class ZeroVae(MmVae):

    def __init__(self,
                 z_dim,
                 x_target,
                 encoder,
                 decoder,
                 encoder_inputs_dim,
                 beta,
                 beta_is_normalized=False,
                 reconstruction_loss_metrics=[ReconstructionLoss.MSE],
                 latent_encoder=None,
                 name='ZeroVae'):
        if len(encoder) > 1:
            raise Exception("> 1 modalities are not supported by Zero VAE")

        self.x_target = x_target

        super().__init__(z_dim=z_dim, encoder=encoder, decoder=decoder,
                     encoder_inputs_dim=encoder_inputs_dim,
                     beta = beta,
                     beta_is_normalized=beta_is_normalized,
                     reconstruction_loss_metrics=reconstruction_loss_metrics,
                     latent_encoder=latent_encoder,
                     name=name)



    def _losses_reconstruction(self):
        ''' Returns the reconstruction for the bi-modal set'''
        reconstruction_loss = []
        # Traverse the uni-modal set (just boilerplate code from JmVae, with a replaced input)
        for x, x_decoded_mean, encoder_input_dim, reconstruction_loss_metric, idx_input in zip(self.encoder_inputs_powerset[-1], \
                                                                                            self.decoder_outputs_powerset[-1], \
                                                                                            self.encoder_inputs_dim_powerset[-1], \
                                                                                            self.reconstruction_loss_metrics_powerset[-1], \
                                                                                            range(len(self.encoder_inputs_powerset[-1]))):
            # Choose the proper reconstruction loss metric
            loss_layer = self.get_reconstruction_loss(reconstruction_loss_metric, weight=encoder_input_dim,
                                                       name="loss_reconstruction_" + str(idx_input))
            self.loss_layers.append(loss_layer)  # Backup the layer for callbacks, etc.
            loss = loss_layer([self.x_target, x_decoded_mean])
            reconstruction_loss.append(loss)
        return reconstruction_loss

