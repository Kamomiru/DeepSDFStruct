from DeepSDFStruct.deep_sdf.training import train_deep_sdf
import torch

if __name__ == "__main__":
    assert(torch.cuda.is_available())

    data_dir = "C:/Users/camil/Desktop/Bachelorarbeit/DeepSDFStruct/Mkofler Dataset"
    train_deep_sdf("C:/Users/camil/Desktop/Bachelorarbeit/DeepSDFStruct/DeepSDFStruct/trained_models/test_experiment3", data_dir, device="cuda")
