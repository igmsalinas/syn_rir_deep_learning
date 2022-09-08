"""
Ignacio Martín 2022

Code corresponding to the Bachelor Thesis "Synthesis of Room Impulse Responses by Means of Deep Learning"

Authors:
    Ignacio Martín
    José Antonio Belloch
    Gema Piñero

University Carlos III de Madrid
"""

import os.path
from utils.dl_models.autoencoder import Autoencoder as AE
from utils.dl_models.vae import VAE as VAE
from utils.dl_models.res_ae import ResAE as ResAE
from utils.loader import Loader
import numpy as np
from utils.postprocess import PostProcess
import datetime


class RIRGenerator:

    def __init__(self, data_loader):
        self.loader = data_loader

        self.features = None
        self.vectors = None

        self.input_vectors = None
        self.output_vectors = None
        self.mm = None

        self.mean_std_f = None
        self.mean_std_p = None

        self.stft_predictions = []

        self._set_data()

    def generate_data(self, autoencoder, path):
        autoencoder.summary()
        self.generate(autoencoder, path)

    def generate(self, autoencoder, path):
        print("Generating...")
        self.stft_predictions = []

        input_feature = []
        input_vector = []
        for i in range(len(self.features)):
            input_feature.append(self.features[0])
            input_vector.append(self.vectors[0])

        input_v = np.stack((input_vector, self.output_vectors), axis=2)
        inputs = [input_feature, input_v]

        start = datetime.datetime.now()
        stft = autoencoder.predict_stft(inputs)
        print(f"{path} Execution time: {datetime.datetime.now() - start}")

        postp = PostProcess(path)
        start = datetime.datetime.now()
        for i in range(len(self.output_vectors)):
            postp.post_process(stft[i], self.output_vectors[i], self.mm[0])
        print(f"{name} Post process time: {datetime.datetime.now() - start}")

    def _set_data(self):
        self.features, self.vectors, self.mm = self.loader.return_raw_database()
        self.output_vectors = loader.shuffle_list(self.vectors)
        self.input_vectors = np.stack((self.vectors, self.output_vectors), axis=2)


if __name__ == "__main__":

    MODELS_DIR1 = "../models/"

    FR_DATA_DIR = "../data/features/fr_features"
    FR_SPECTROGRAM_DIR = "../data/features/fr_features/spectrogram"

    adjuster = 8500 
    min_gap = 0

    loader = Loader(FR_SPECTROGRAM_DIR, FR_DATA_DIR,
                    distance_adjuster=adjuster, min_value=min_gap)

    generator = RIRGenerator(loader)

    name = "AE"
    ae = AE.load(MODELS_DIR1 + name)
    generator.generate_data(ae, name)

    name = "ResAE"
    res_ae = ResAE.load(MODELS_DIR1 + name)
    generator.generate_data(res_ae, name)


    name = "VAE"
    vae = VAE.load(MODELS_DIR1 + name)
    generator.generate_data(vae, name)
