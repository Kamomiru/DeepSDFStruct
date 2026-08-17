from DeepSDFStruct.deep_sdf.training_gan import train_deep_sdf_gan
import torch
import os

experiment_name = "gan_experiment7"

test_experiment_dir = "C:/Users/camil/Desktop/Bachelorarbeit/DeepSDFStruct/DeepSDFStruct/trained_models/test_experiments/" + experiment_name
experiment_dir = "C:/Users/camil/Desktop/Bachelorarbeit/DeepSDFStruct/DeepSDFStruct/trained_models/experiments/" + experiment_name

if os.path.isdir(test_experiment_dir):
    path = test_experiment_dir
elif os.path.isdir(experiment_dir):
    path = experiment_dir
else:
    raise RuntimeError("ERROR:  No suitable path found!")


if __name__ == "__main__":
    assert(torch.cuda.is_available())

    train_deep_sdf_gan(path, device="cuda")
    
        