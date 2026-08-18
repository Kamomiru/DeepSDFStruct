from DeepSDFStruct.deep_sdf.GAN_helpers.gan_training_helpers import *
from DeepSDFStruct.deep_sdf.plotting import *
from matplotlib import pyplot as plt

experiment_name = "gan_test_experiment26"

test_experiment_dir = "C:/Users/camil/Desktop/Bachelorarbeit/DeepSDFStruct/DeepSDFStruct/trained_models/test_experiments/" + experiment_name

decoder, pretrain_loss = pretrain_decoder(test_experiment_dir, device= "cuda")

fig, ax = plt.subplots(1,3)

plot_decoder_set(decoder, ax, device="cuda")

plt.savefig(test_experiment_dir + "/pretrainedDecoder.png")






