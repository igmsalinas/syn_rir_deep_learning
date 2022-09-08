"""
Ignacio Martín 2022

Code corresponding to the Bachelor Thesis "Synthesis of Room Impulse Responses by Means of Deep Learning"

Authors:
    Ignacio Martín
    José Antonio Belloch
    Gema Piñero

University Carlos III de Madrid
"""

import os
import pickle
from utils.dl_models.autoencoder import Autoencoder
from utils.dl_models.vae import VAE
from utils.dl_models.res_ae import ResAE
from loader import Loader
import argparse
import datetime

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description='Select gpus to train, distance between RIRs, security gap and '
                                                 'what models to train.')
    parser.add_argument('--gpu', '-g', metavar='G', type=int, nargs='+', default=[0],
                        help='specify number of gpus to use')
    parser.add_argument('--dis', '-d', metavar='D', type=int, nargs=1, default=1,
                        help='multiplier of distance between RIRs at loading')
    parser.add_argument('--min_gap', '-m', metavar='M', type=float, nargs=1, default=0.,
                        help='minimum gap between RIRs and walls')
    parser.add_argument('--train', '-t', metavar='T', type=str, default="ae",
                        help='choose what models to train: "ae", "vae" or "res_ae"')
    args = parser.parse_args()

    if not (args.train in ["ae", "vae", "res_ae"]):
        raise argparse.ArgumentTypeError("Invalid option, try 'ae' 'vae' or 'res_ae'")

    devices_str = ""

    for gpus in args.gpu:
        if gpus == args.gpu[-1]:
            devices_str += str(gpus)
        else:
            devices_str += str(gpus) + ","
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = devices_str
    print(f"Using gpus: {devices_str}")

    training_mode = f"dis_m{args.dis[0]}_min_g{args.min_gap}"
    print("training mode: ", training_mode)

    # Load the datasets
    FR_DATA_DIR = "../data/features/fr_features"
    FR_SPECTROGRAM_DIR = "../data/features/fr_features/spectrogram"

    BATCH_SIZE = 128
    MAX_EPOCHS = 1000

    loader = Loader(FR_SPECTROGRAM_DIR, FR_DATA_DIR,
                    distance_adjuster=args.dis[0], min_value=args.min_gap)

    x_train_f, x_train_v, y_train_f, x_val_f, x_val_v, y_val_f = loader.load_data()

    N_TRAIN = x_train_f.shape[0]
    STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE

    print(f"Features train shape x: {x_train_f.shape}")
    print(f"Vector train shape x : {x_train_v.shape}")
    print(f"Features train shape y: {y_train_f.shape}")
    print(f"Features val shape x: {x_val_f.shape}")
    print(f"Vector val shape x: {x_val_v.shape}")
    print(f"Features val shape y: {y_val_f.shape}")

    if args.train in ["ae"]:
        size_historiesAE = {}
        autoencoder = Autoencoder(
            input_shape=(144, 128, 2),
            inf_vector_shape=(10, 2),
            conv_filters=(32, 64, 128, 256),
            conv_kernels=(3, 3, 3, 3),
            conv_strides=(2, 2, 2, 2),
            latent_space_dim=32,
            n_neurons=20 * 64,
            name="AE"
        )

        start = datetime.datetime.now()

        size_historiesAE['AE'] = autoencoder.compile_and_fit(x_train_f, x_train_v, y_train_f,
                                                             x_val_f, x_val_v, y_val_f,
                                                             BATCH_SIZE, MAX_EPOCHS, STEPS_PER_EPOCH)

        print(f"AE Training time: {start - datetime.datetime.now()}")

        autoencoder.save(f"../models/AE")

        save_path = os.path.join(f"../models/AE", "size_histories_AE.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(size_historiesAE, f)

    if args.train in ["vae"]:
        size_historiesVAE = {}
        vae = VAE(
            input_shape=(144, 128, 2),
            inf_vector_shape=(10, 2),
            conv_filters=(32, 64, 128, 256),
            conv_kernels=(3, 3, 3, 3),
            conv_strides=(2, 2, 2, 2),
            latent_space_dim=16,
            n_neurons=20,
            name="VAE"
        )

        start = datetime.datetime.now()

        size_historiesVAE['VAE'] = vae.compile_and_fit(x_train_f, x_train_v, y_train_f,
                                                       x_val_f, x_val_v, y_val_f,
                                                       BATCH_SIZE, MAX_EPOCHS, STEPS_PER_EPOCH)

        print(f"VAE Training time: {start - datetime.datetime.now()}")

        vae.save(f"../models/VAE")

        save_path = os.path.join(f"../models/VAE", "size_histories_VAE.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(size_historiesVAE, f)

    if args.train in ["res_ae"]:
        size_historiesResAE = {}
        res_ae = ResAE(
            input_shape=(144, 128, 2),
            inf_vector_shape=(10, 2),
            conv_filters=(32, 64, 128, 256),
            conv_kernels=(3, 3, 3, 3),
            conv_strides=(2, 2, 2, 2),
            latent_space_dim=32,
            n_neurons=20 * 64,
            name="ResAE"
        )

        start = datetime.datetime.now()

        size_historiesResAE['ResAE'] = res_ae.compile_and_fit(x_train_f, x_train_v, y_train_f,
                                                              x_val_f, x_val_v, y_val_f,
                                                              BATCH_SIZE, MAX_EPOCHS, STEPS_PER_EPOCH)

        print(f"AE Training time: {start - datetime.datetime.now()}")

        res_ae.save(f"../models/ResAE")

        save_path = os.path.join(f"../models/ResAE", "size_histories_ResAE.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(size_historiesResAE, f)
