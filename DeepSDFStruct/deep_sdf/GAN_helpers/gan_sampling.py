import torch
from DeepSDFStruct.SDF import SDFfromDeepSDF

class ConvGAN_SDF_Sampler():
    def __init__(self, SDF, decoder, n_nodes, n_samples, sdf_param_bounds, device):
        self.n_nodes = n_nodes
        self.n_samples = n_samples
        self.device = device
        self.meshgrid = self._create_meshgrid()
        self.SDF = SDF
        self.decoder = decoder
        self.sdf_param_bounds = sdf_param_bounds
        
        self._update_random_params()

    def fetch_samples(self, fake_only=False, decoder_clamp_val=None):

        real_samples = []
        fake_samples = []
        latent_vectors = []

        for i_param, param in enumerate(self.random_params):

            # ---------------------------------------------------------
            # Set parameter of the real SDF
            # ---------------------------------------------------------

            self.SDF.setRadius(param)

            # ---------------------------------------------------------
            # Create conditioning / latent vector
            # ---------------------------------------------------------

            latent = torch.tensor(
                [[param]],
                dtype=torch.float32,
                device=self.device
            )

            latent_vectors.append(latent.squeeze(0))

            # ---------------------------------------------------------
            # Generate fake SDF
            # ---------------------------------------------------------

            fake = self._sample_decoder_meshgrid(latent)

            fake_samples.append(fake)

            # ---------------------------------------------------------
            # Generate real SDF if required
            # ---------------------------------------------------------

            if not fake_only:
                real = self._sample_real_sdf_meshgrid()
                real_samples.append(real)

        # -------------------------------------------------------------
        # Generate new random parameters for the next call
        # -------------------------------------------------------------

        self._update_random_params()

        # -------------------------------------------------------------
        # Stack fake samples and latent vectors
        # -------------------------------------------------------------

        fake_samples = torch.stack(
            fake_samples,
            dim=0
        )

        latent_vectors = torch.stack(
            latent_vectors,
            dim=0
        )

        # -------------------------------------------------------------
        # Clamp fake SDF if requested
        # -------------------------------------------------------------

        if decoder_clamp_val is not None:
            fake_samples = torch.clamp(
                fake_samples,
                min=-decoder_clamp_val,
                max=decoder_clamp_val
            )

        # -------------------------------------------------------------
        # If only fake samples were requested, return them
        # -------------------------------------------------------------

        if fake_only:
            return fake_samples, latent_vectors

        # -------------------------------------------------------------
        # Otherwise stack real samples too
        # -------------------------------------------------------------

        real_samples = torch.stack(
            real_samples,
            dim=0
        )

        return real_samples, fake_samples, latent_vectors


    
    
    def _update_random_params(self):
            #creating random parameters for real sdf
            self.random_params = self.sdf_param_bounds[0] + (self.sdf_param_bounds[1] - self.sdf_param_bounds[0]) * torch.rand(1,int(self.n_samples/2), device=self.device).squeeze(0).detach()


    def _create_meshgrid(self):
        line = torch.linspace(-1.0, 1.0, self.n_nodes, device=self.device)

        X, Y, Z = torch.meshgrid(line, line, line, indexing="ij")

        xyz = torch.stack((X, Y, Z), dim=-1)

        # xyz.shape = (n³, 3)
        return xyz.reshape(-1, 3)

    def _sample_real_sdf_meshgrid(self):

        sdf_values = self.SDF.forward(self.meshgrid)

        sdf_values = sdf_values.reshape(self.n_nodes, self.n_nodes, self.n_nodes).unsqueeze(0) #convert from (n³, 1) to (1, n, n, n)

        return sdf_values

    def _sample_decoder_meshgrid(self, latent):

        sdf_values = self.decoder.forward_with_latent(latent, self.meshgrid)
        
        return sdf_values.reshape(self.n_nodes, self.n_nodes, self.n_nodes).unsqueeze(0)

