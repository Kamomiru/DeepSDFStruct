import torch
from DeepSDFStruct.SDF import SDFfromDeepSDF

def create_meshgrid(n_nodes):
    line = torch.linspace(-1.0, 1.0, n_nodes)

    X, Y, Z = torch.meshgrid(line, line, line, indexing="ij")

    xyz = torch.stack((X, Y, Z), dim=-1)

    # xyz.shape = (n³, 3)
    return xyz.reshape(-1, 3)

def sample_real_sdf_meshgrid(sdf, xyz_grid, n_nodes):

    sdf_values = sdf.forward(xyz_grid)

    sdf_values = sdf_values.reshape(n_nodes, n_nodes, n_nodes).unsqueeze(0) #convert from (n³, 1) to (1, n, n, n)

    return sdf_values

def sample_decoder_meshgrid(decoder, latent, xyz_grid, n_nodes):

    sdf_values = decoder.forward_with_latent(latent, xyz_grid)
    
    #deep_sdf = SDFfromDeepSDF(decoder) #potentially implement to forward through decoder with latent vec together so we can skip this step?
    #deep_sdf.set_latent_vec(latent)

    #print("meshgrid: ", xyz_grid.shape)
    #sdf_values = deep_sdf.forward(xyz_grid)
    #print(f"sdf_values.shape = {sdf_values.shape}")
    
    return sdf_values.reshape(n_nodes, n_nodes, n_nodes).unsqueeze(0)

class ConvGAN_SDF_Sampler():
    def __init__(self, SDF, decoder, n_nodes, n_samples, sdf_param_bounds):
        self.n_nodes = n_nodes
        self.n_samples = n_samples
        self.meshgrid = create_meshgrid(n_nodes)
        self.SDF = SDF
        self.decoder = decoder

        self.real = torch.zeros([int(n_samples/2), 1, n_nodes, n_nodes, n_nodes])
        self.fake = torch.zeros([int(n_samples/2), 1, n_nodes, n_nodes, n_nodes])
        
        #creating random parameters for real sdf
        self.random_params = sdf_param_bounds[0] + (sdf_param_bounds[1] - sdf_param_bounds[0]) * torch.rand(1,int(n_samples/2)).squeeze(0)

    def fetch_samples(self):
        #print(f"random_params is {self.random_params}")
        for i_param, param in enumerate(self.random_params):

            self.SDF.setRadius(param)

            self.real[i_param, 0, :, :, :] = sample_real_sdf_meshgrid(self.SDF, self.meshgrid, self.n_nodes).unsqueeze(0)
            self.fake[i_param, 0, :, :, :] = sample_decoder_meshgrid(self.decoder, torch.tensor([[param]]), self.meshgrid, self.n_nodes).unsqueeze(0) #What latent vector should be used to sample fake samples???
        
            print("i_param: ", i_param)
            print(f"param is {param}")
            print("\n")
            print(f"real  {self.real}")
            print("\n")
            print(f"fake {self.fake}")
            print("\n")

        return self.real, self.fake


        

