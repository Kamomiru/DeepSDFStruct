import torch
from DeepSDFStruct.SDF import SDFfromDeepSDF

class ConvGAN_SDF_Sampler():
    def __init__(self, SDF, decoder, n_nodes, n_samples, sdf_param_bounds, device, add_latent = False, rnd_mesh_offset = False):
        self.n_nodes = n_nodes
        self.n_samples = n_samples
        self.device = device
        self.meshgrid = self._create_meshgrid()
        self.SDF = SDF
        self.decoder = decoder
        self.sdf_param_bounds = sdf_param_bounds
        self.add_latent = add_latent

        #variables for random mesghrid offset:
        self.rnd_mesh_offset = rnd_mesh_offset
        self.eps_max = 2 / self.n_nodes
        self.offset_meshgrid = self._get_offset_meshgrid()
        
        self._update_random_params()

    def fetch_samples(self, fake_only = False, decoder_clamp_val = None):

        real_samples = []
        fake_samples = []
        latent_parameters = []

        for i_param, param in enumerate(self.random_params):

            self.SDF.setRadius(param)
            if self.add_latent:
                latent_parameters.append(param)

            fake = self._sample_decoder_meshgrid(torch.tensor([[param]], device=self.device)) #For now we implement the latvec as fixed conditioning vector

            fake_samples.append(fake)

            if fake_only == False:
                real = self._sample_real_sdf_meshgrid()
                real_samples.append(real) 

            #print("i_param: ", i_param)
            #print(f"param is {param}")
            #print("\n")
            #print(f"real  {self.real}")
            #print("\n")
            #print(f"fake {self.fake}")
            #print("\n")

        self._update_random_params()
        if self.rnd_mesh_offset:
            self.offset_meshgrid = self._get_offset_meshgrid()

        fake_samples = torch.stack(fake_samples, dim=0)


        if decoder_clamp_val is not None:
            fake_samples = torch.clamp(
            fake_samples,
            min=-decoder_clamp_val,
            max=decoder_clamp_val
        )

        if fake_only:
            if self.add_latent:
                latent_parameters = torch.stack(latent_parameters)
                return fake_samples, latent_parameters
            else:
                return fake_samples

        real_samples = torch.stack(real_samples, dim=0)

        if self.add_latent:
            latent_parameters = torch.stack(latent_parameters)
            return real_samples, fake_samples, latent_parameters
        else:
            return real_samples, fake_samples
    
    
    def _update_random_params(self):
            #creating random parameters for real sdf
            self.random_params = self.sdf_param_bounds[0] + (self.sdf_param_bounds[1] - self.sdf_param_bounds[0]) * torch.rand(1,int(self.n_samples/2), device=self.device).squeeze(0).detach()


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

