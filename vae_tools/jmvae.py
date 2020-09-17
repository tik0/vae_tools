from vae_tools.mmvae import MmVae
from vae_tools.vae import *

class JmVae(MmVae):

    def __init__(self,
                 z_dim,
                 encoder,
                 decoder,
                 encoder_inputs_dim,
                 beta,
                 alpha,
                 beta_is_normalized=False,
                 reconstruction_loss_metrics=[ReconstructionLoss.MSE],
                 latent_encoder=None,
                 shared_weights=True,
                 name='JmVae'):
        if len(encoder) > 2:
            raise Exception("> 2 modalities are not supported by the paper")

        self.alpha = alpha

        # This init configures already
        super().__init__(z_dim=z_dim, encoder=encoder, decoder=decoder,
                     encoder_inputs_dim=encoder_inputs_dim,
                     beta = beta,
                     beta_is_normalized=beta_is_normalized,
                     beta_mutual=alpha,
                     reconstruction_loss_metrics=reconstruction_loss_metrics,
                     latent_encoder=latent_encoder,
                     shared_weights=True,
                     name=name)


    def _losses_reconstruction(self):
        ''' Returns the reconstruction for the bi-modal set'''
        reconstruction_loss = []
        # Traverse the bi-modal set
        for x, x_decoded_mean, encoder_input_dim, reconstruction_loss_metric, idx_input in zip(self.encoder_inputs_powerset[-1], \
                                                                                            self.decoder_outputs_powerset[-1], \
                                                                                            self.encoder_inputs_dim_powerset[-1], \
                                                                                            self.reconstruction_loss_metrics_powerset[-1], \
                                                                                            range(len(self.encoder_inputs_powerset[-1]))):
            # Choose the proper reconstruction loss metric
            loss_layer = self.get_reconstruction_loss(reconstruction_loss_metric, weight=encoder_input_dim,
                                                       name="loss_reconstruction_" + str(idx_input))
            self.loss_layers.append(loss_layer)  # Backup the layer for callbacks, etc.
            loss = loss_layer([x, x_decoded_mean])
            reconstruction_loss.append(loss)
        return reconstruction_loss

    def _losses_prior(self):
        ''' Returns the prior losses for the bi-modal set'''
        kl_prior_loss = []
        loss_layer = LosslayerDistributionGaussianPrior(weight=self.get_beta(x_dim=sum(self.encoder_inputs_dim_powerset[-1])),
                                                        name="loss_prior")
        self.loss_layers.append(loss_layer)  # Backup the layer for callbacks, etc.
        loss = loss_layer([self.Z_mean[-1], self.Z_logvar[-1]])
        kl_prior_loss.append(loss)
        return kl_prior_loss
