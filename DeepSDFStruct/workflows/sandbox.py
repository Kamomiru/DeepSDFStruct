from DeepSDFStruct.deep_sdf.plotting import plot_decoder_set
from DeepSDFStruct.deep_sdf.workspace import *
from DeepSDFStruct.pretrained_models import get_model, PretrainedModels
from matplotlib import pyplot as plt

device = torch.device("cpu")
path = "C:/Users/camil/Desktop/Bachelorarbeit/DeepSDFStruct/DeepSDFStruct/trained_models/test_experiment4"
decoder = load_trained_model(path, "latest", device)

fig, ax = plt.subplots((3))

plot_decoder_set(path, decoder, ax)

plt.savefig("C:/Users/camil/Desktop/Bachelorarbeit/DeepSDFStruct/DeepSDFStruct/workflows/plot.png")





