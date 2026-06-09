import torch
import numpy as np


standard_bounds = torch.tensor([[-1.0, 1.0],
                                [-1.0, 1.0],
                                [-1.0, 1.0]])


def sample_rand(decoder, latent_vec, n, bounds = standard_bounds, eps = 1.0):
    """
    Samples random points across total SDF bounds
    Can be used for sampling close to the surface by adjusting eps, although this is highly inefficient and hence not advised
    """
    decoder.eval()
    
    iter = 1

    samples = torch.tensor(())
    latvec = torch.tensor([[0.1]])

    while samples.shape[0] < n:
        
        #implement sampling through bounds
        xyz_rand = 2 * torch.rand(1,3) - 1.0 
        
        

        #print("Network input:")
        #print(torch.cat((latvec, xyz_rand), dim = 1))

        #print("Sample SDF Value:")
        sdf_value = decoder(torch.cat((latvec, xyz_rand), dim = 1))
        #print(sdf_value)

        sample = torch.cat((latvec, xyz_rand, sdf_value), dim = 1)

        if abs(sample[0, 4]) <= eps:
            samples = torch.cat((samples, sample), 0)
        
        iter += 1
        
    decoder.train()
    print(f"Sampling n = {n} points took {iter} iterations with eps = {eps}")
    return samples

def sample_meshgrid(decoder, latent_vec, n, bounds = standard_bounds):
    """
    Creates Samples using an uniform grid of at least n samples.
    to get exactly n samples use only numbers n = 3^x where x are whole numbers. e.g. n = [81, 243, 729, 1000, 2187].
    otherwise the total sample number will be increased to create a uniform grid of at least n samples.
    """
    decoder.eval()
    latent_vec = torch.tensor([[0.1]], dtype= torch.float32)

    n_dim = np.ceil(np.cbrt(n)).astype(int)
    n_total = n_dim**3

    latent_vec = latent_vec.expand(n_total,1)

    linemesh = np.linspace(-1.0, 1.0, n_dim)

    X,Y,Z = np.meshgrid(linemesh, linemesh, linemesh)
    xyz_grid = torch.from_numpy(np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis = 1)).float()

    sdf_values = decoder(torch.cat((latent_vec, xyz_grid), dim = 1))

    print(latent_vec.shape)
    print(xyz_grid.shape)
    print(sdf_values.shape)

    samples = torch.cat((latent_vec, xyz_grid, sdf_values), dim = 1)

    return samples




def sample_surface_newton(decoder, latent_vec, n, bounds = standard_bounds, eps = 0.01, maxiter = None):
    """
    improved efficiency over sample_surface.
    Uses newton method to iterate towards surface
    """
    decoder.eval()
    #implement arbitrary latvec value
    latvec = torch.tensor([[0.1]])
    samples = torch.tensor(())
    
    sample_n = 0
    
    while sample_n < n:
        #implement sampling in arbitrary bounds
        xyz = 2 * torch.rand(1,3) - 1.0
        xyz.requires_grad_(True)
        sdf_value = decoder(torch.cat((latvec, xyz), dim = 1))
        #print(f"First iteration sdf_value: {sdf_value}")

        

        iter = 0
        while((abs(sdf_value) > eps) and ((maxiter is None) or (iter < maxiter)) ):
            iter += 1
            grad = torch.autograd.grad(sdf_value.sum(), xyz, create_graph = False)[0] #returns tuple of multiple grad, since it supports multiple inputs -> [0] extracts the gradient value of our single input

            xyz = xyz - sdf_value * grad / (grad.norm(dim = -1, keepdim= True)**2 + 1e-8) #dim = -1 gives the norm of our last input. keepdim keeps the singular dimension so we get (N,1). +1e-8 to avoid 0 division
            sdf_value = decoder(torch.cat((latvec, xyz), dim = 1))

            #print(f"{iter}-th iteration sdf_value: {sdf_value}")

        #skip samples that are out of bounds due to newton method
        lower_bounds = standard_bounds[:, 0]
        upper_bounds = standard_bounds[:, 1]

        if ((xyz < lower_bounds) | (xyz > upper_bounds)).any():
            continue

        sample = torch.cat((latvec, xyz, sdf_value), dim = 1)
        samples = torch.cat((samples, sample), dim = 0)
        sample_n += 1

    decoder.train()
    return samples
        
 
    
        
if __name__ == "__main__":

    from DeepSDFStruct.deep_sdf.workspace import load_trained_model
    import matplotlib.pyplot as plt

    path = "C:/Users/camil/Desktop/Bachelorarbeit/DeepSDFStruct/DeepSDFStruct/trained_models/test_experiment4"

    device = torch.device("cpu")
    decoder = load_trained_model(path, "latest", device)

    samples = sample_rand(decoder, 0.1, 4096)

    print(samples)


    #----------PLOTTING----------
    samples = samples.detach()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
            samples[:, 1],
            samples[:, 2],
            samples[:, 3],
            c = samples[:, 4],
            s=2,
            alpha=0.8,
        )


    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Surface Samples")

    plt.show()



