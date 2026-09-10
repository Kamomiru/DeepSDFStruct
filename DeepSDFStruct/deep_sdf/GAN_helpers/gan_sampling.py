"""
GAN Sampler for Decoder Training
=================================

Builds real and fake SDF grid batches for adversarial + InfoGAN-style
regressor training.

Real batch: freshly drawn radii -> ground-truth CrossMsSDF grids. No labels
are attached; the regressor never touches real data.

Fake batch: freshly drawn (z, c) pairs -> decoder-generated grids. `z` is
free noise (irrelevant to the regressor loss); `c` is the control code the
regressor is trained to reconstruct from the generated grid.

`fetch_real_batch()` and `fetch_fake_batch()` are fully independent -- there
is no pairing between which real radius and which fake (z, c) get drawn in
a given training step. (The previous version of this sampler tied them
together, because the old supervised classifier needed a real label to
match a generated sample against. The regressor doesn't need that: real
data is only ever used for the discriminator's realism loss.)
"""

import torch
from DeepSDFStruct.sdf_primitives import CrossMsSDF
from DeepSDFStruct.deep_sdf.GAN_helpers.chi3d_SDF import Chi3DPrismSDF


class ConvGAN_SDF_Sampler():
    def __init__(
        self,
        decoder,
        specs,
        device,
    ):
        self.decoder = decoder
        self.device = device
        self.n_nodes = specs["DiscriminatorSpecs"]["n_nodes"]
        self.n_samples = specs["SamplesPerBatch"]
        self.meshgrid = self._create_meshgrid()
        self.SDFName = specs["SDFName"]
        if self.SDFName == "CrossMsSDF":
            self.SDF = CrossMsSDF(0.0)
        elif self.SDFName == "Chi3DPrismSDF":
            self.SDF = Chi3DPrismSDF(torch.ones((5,5), device=device))
        else:
            raise KeyError(f"The SDF Type {specs["SDFName"]} does not exist, or has not been implemented yet.")


        self.sdf_param_bounds = specs["SdfParameterBounds"] # bounds for real sdf

        # latent split: latent_size is the decoder's TOTAL input width
        # (z_dim + control_code_dim); control_code_dim is dim(c), the InfoGAN control code.
        
        self.control_code_dim = specs["ControlCodeDim"]
        self.latent_dim = specs["CodeLength"]
        self.z_dim = self.latent_dim - self.control_code_dim
        if self.z_dim < 0:
            raise RuntimeError(
                f"z_dim cannot be < 0!"
            )
        if self.control_code_dim < 1:
                    raise RuntimeError(
                        f"control_code_dim cannot be < 1!"
                    )
        self.z_distribution = specs["ZDistribution"]
        self.code_bounds = specs["ControlCodeBounds"] # bounds of latent code

        # variables for random meshgrid offset:
        self.rnd_mesh_offset = specs["RandomMeshgridOffset"]
        self.eps_max = 2 / self.n_nodes
        self.offset_meshgrid = self._get_offset_meshgrid()

        self.random_params = torch.empty((len(self.sdf_param_bounds), int(self.n_samples / 2)), device=self.device)
        self._update_random_params()

    def resample_mesh_offset(self):
        if self.rnd_mesh_offset:
            self.offset_meshgrid = self._get_offset_meshgrid()
 
    def fetch_real_batch(self):
        """
        Returns a (n_samples//2, 1, n, n, n) batch of ground-truth SDF grids
        at freshly drawn radii, using whatever query grid is currently set
        (see resample_mesh_offset()).
        """
        real_samples = []
        if self.SDFName == "CrossMsSDF":
            for param in self.random_params[0]:
                self.SDF.setParameter(param) #type: ignore
                real_samples.append(self._sample_real_sdf_meshgrid())
        elif self.SDFName == "Chi3DPrismSDF":
            for params in self.random_params.T:
                #print(params)
                #print(params.shape)
                params = params.expand(5,-1)
                self.SDF._set_param(params)
                real_samples.append(self._sample_real_sdf_meshgrid())
            
 
        self._update_random_params()
 
        return torch.stack(real_samples, dim=0)
 
    def fetch_fake_batch(self, batch_size=None, decoder_clamp_val=None):
        """
        Returns (fake_samples, sampled_codes), using whatever query grid is
        currently set (see resample_mesh_offset()):
          fake_samples:  (batch_size, 1, n, n, n) decoder output grids
          sampled_codes: (batch_size, control_code_dim) the c drawn for each grid --
                          this is the regressor's training target.
        """
        if batch_size is None:
            batch_size = int(self.n_samples / 2)
 
        z, c = self._sample_latent_batch(batch_size)
        latent = torch.cat([z, c], dim=-1)  # (batch_size, latent_dim)
 
        fake_samples = []
        for i in range(batch_size):
            fake_samples.append(self._sample_decoder_meshgrid(latent[i : i + 1]))  # sample decoder with current meshgrid and latent vec. We use slicing here since in perserves batch dimension
        fake_samples = torch.stack(fake_samples, dim=0)
 
        if decoder_clamp_val is not None:
            fake_samples = torch.clamp(
                fake_samples, min=-decoder_clamp_val, max=decoder_clamp_val
            )
 
        return fake_samples, c
 
    def _sample_latent_batch(self, batch_size):
        """
        Sample random latent code and z for fake sdf sampling
        """
        # sample z 
        if self.z_dim > 0:
            if self.z_distribution == "normal":
                z = torch.randn(batch_size, self.z_dim, device=self.device)
            elif self.z_distribution == "uniform":
                z = (torch.rand(batch_size, self.z_dim, device=self.device) * 2.0) - 1.0
            else:
                raise ValueError(
                    f"Unknown z_distribution: {self.z_distribution!r} (expected 'normal' or 'uniform')"
                )
        else:
            # control_code_dim == latent_dim: no free noise dimensions at all.
            z = torch.empty(batch_size, 0, device=self.device)
 
        #sample latent code through uniform distribution
        c = self.code_bounds[0] + (self.code_bounds[1] - self.code_bounds[0]) * torch.rand(batch_size, self.control_code_dim, device=self.device)
 
        return z, c
 
    def _update_random_params(self):
        """
        Update random parameters for real sdf sampling using uniform distribution
        """
        for i, bounds in enumerate(self.sdf_param_bounds):

            if  len(bounds) == 1:
                 self.random_params[i] = torch.full((1, int(self.n_samples / 2)), bounds[0])
            elif len(bounds) == 2:
                max_b = max(bounds)
                min_b = min(bounds)
                self.random_params[i] = min_b + (
                            max_b - min_b
                        ) * torch.rand((1, int(self.n_samples / 2)), device=self.device).squeeze(0).detach()
            else:
                raise ValueError(f"Dimension of bounds must be 1 for constant parameter or 2 for randomly sampled parameter! Got bounds of dim: {len(bounds)}")

        #print(f"random params: {self.random_params}")
        #print(f"random params shape: {self.random_params.shape}")

        return self.random_params
 
    def _create_meshgrid(self):
        line = torch.linspace(-1.0, 1.0, self.n_nodes, device=self.device)
 
        X, Y, Z = torch.meshgrid(line, line, line, indexing="ij")
 
        xyz = torch.stack((X, Y, Z), dim=-1)
 
        # xyz.shape = (n³, 3)
        return xyz.reshape(-1, 3)
 
    def _get_offset_meshgrid(self):
        offsets = (torch.rand(3, device=self.device) - 0.5) * 2 * self.eps_max
        offset_meshgrid = self.meshgrid + offsets
        return offset_meshgrid
 
    def _sample_real_sdf_meshgrid(self):
 
        if self.rnd_mesh_offset:
            sdf_values = self.SDF.forward(self.offset_meshgrid)
        else:
            sdf_values = self.SDF.forward(self.meshgrid)
 
        sdf_values = sdf_values.reshape(self.n_nodes, self.n_nodes, self.n_nodes).unsqueeze(0) #convert from (n³, 1) to (1, n, n, n)
 
        return sdf_values
 
    def _sample_decoder_meshgrid(self, latent):
 
        if self.rnd_mesh_offset:
            sdf_values = self.decoder.forward_with_latent(latent, self.offset_meshgrid)
        else:
            sdf_values = self.decoder.forward_with_latent(latent, self.meshgrid)
 
        return sdf_values.reshape(self.n_nodes, self.n_nodes, self.n_nodes).unsqueeze(0)