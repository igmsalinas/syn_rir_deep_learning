"""
Ignacio Martín 2022

Code corresponding to the Bachelor Thesis "Synthesis of Room Impulse Responses by Means of Deep Learning"

Authors:
    Ignacio Martín
    José Antonio Belloch
    Gema Piñero

University Carlos III de Madrid

Credit to Valerio Velardo for the general architecture of the preprocessingPipeline, available at:
https://github.com/musikalkemist

"""

import math
import os
import pathlib
import pickle
import librosa
import numpy as np
import scipy.io
import tensorflow as tf
from tqdm import tqdm


class MATLabLoader:

    def __init__(self):

        self.mic_data_list = []

    def load_mat_data(self, matlab_data_path):
        self._download_data(matlab_data_path)
        data_dir = pathlib.Path(matlab_data_path)
        files = tf.io.gfile.glob(str(data_dir) + '/*')
        temp_list = []
        for data in files:
            mat_contents = scipy.io.loadmat(data)
            mic_data = mat_contents['e_ir']
            temp_list.append(mic_data)

        self.mic_data_list = temp_list
        return self.mic_data_list

    def load_mic_list(self, mic_data_path):
        self.mic_data_list = np.load(mic_data_path)
        return self.mic_data_list

    def _download_data(self, matlab_data_path):
        if not matlab_data_path.exists():
            print('Directory not found, downloading')
            tf.keras.utils.get_file(
                'ficheros_mat.zip',
                origin="https://gtac.webs.upv.es/ficheros/software/AppAudios/ficheros_mat.zip",
                extract=True,
                cache_dir=".",
                cache_subdir="../data/databases/matlab_data")
            os.remove("../data/databases/matlab_data/ficheros_mat.zip")
            self._rename_ml_files(matlab_data_path)

    @staticmethod
    def _rename_ml_files(directory):
        fnames = os.listdir(directory)
        for fname in fnames:
            _, file_number_ext = fname.split("_")
            file_number, _ = file_number_ext.split(".")
            number = file_number.zfill(3)
            new_file_name = f"imp_{number}.mat"
            os.rename(os.path.join(directory, fname), os.path.join(directory, new_file_name))


class FRLoader:

    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, file_path):
        signal = librosa.load(file_path,
                              sr=self.sample_rate,
                              duration=self.duration,
                              mono=self.mono)[0]
        signal -= np.mean(signal)
        return signal


class Padder:

    def __init__(self, mod="constant", axis=None):
        self.mode = mod
        self.axis = axis

    def left_pad(self, array, num_missing_items):
        padded_array = np.pad(array,
                              (num_missing_items, 0),
                              mode=self.mode)
        return padded_array

    def right_pad(self, array, num_missing_items):
        padded_array = np.pad(array,
                              (0, num_missing_items),
                              mode=self.mode)
        return padded_array

    def left_snip(self, array, num_exceeding_items):
        snipped_array = np.delete(array,
                                  np.s_[:num_exceeding_items],
                                  axis=self.axis)
        return snipped_array

    def right_snip(self, array, num_exceeding_items):
        snipped_array = np.delete(array,
                                  np.s_[-num_exceeding_items:],
                                  axis=self.axis)
        return snipped_array


class ShapeTransform:

    def __init__(self, desired_shape):
        self.current_shape = None
        self.desired_shape = desired_shape
        self.c_rows = None
        self.c_columns = None
        self.n_rows = None
        self.n_columns = None

    def transform(self, array):
        if self.get_needed_transform(array):
            rp_array = self.row_transform(array)
            padded_array = self.col_transform(rp_array)
            return padded_array
        else:
            return array

    def get_needed_transform(self, array):
        self.current_shape = array.shape
        self.c_rows = self.current_shape[0]
        self.c_columns = self.current_shape[1]

        conditions = self.current_shape[0] > self.desired_shape[0] or self.current_shape[1] > self.desired_shape[1]
        if not conditions:
            self.n_rows = self.desired_shape[0] - self.current_shape[0]
            self.n_columns = self.desired_shape[1] - self.current_shape[1]
            return True
        else:
            return False

    def row_transform(self, array):
        rt_array = np.r_[array, np.zeros((self.n_rows, self.c_columns))]
        return rt_array

    def col_transform(self, array):
        temp_array = array
        for i in range(self.n_columns):
            temp_array = np.c_[temp_array, np.zeros(self.desired_shape[0])]
        ct_array = temp_array
        return ct_array


class FeatureExtractor:

    def __init__(self, n_fft, win_length, hop_length):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def extract(self, waveform):
        # Convert the waveform to a spectrogram via a STFT.
        spectrogram = librosa.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        stft = np.abs(spectrogram)
        phase = np.angle(spectrogram)
        log_spectrogram = librosa.amplitude_to_db(stft)
        return log_spectrogram, phase


class Normalizer:

    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val

    def normalize(self, array):
        norm_array = (array - array.min()) / (array.max() - array.min())
        norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array

    def denormalize(self, norm_array, original_min, original_max):
        array = (norm_array - self.min) / (self.max - self.min)
        array = array * (original_max - original_min) + original_min
        return array


class EmbeddingVector:

    def __init__(self, rdx, rdy, rdz, t60):
        self.rdx = rdx
        self.rdy = rdy
        self.rdz = rdz
        self.t60 = t60
        self.d_m_ml = 0.2
        self.d_s_ml = 0.2
        self.d_fr = 0.3
        self.vector = None

    def __repr__(self):
        return f"{self.vector}"

    def create_vector(self, md, filename):
        if md == "ml":
            _, spk, end = filename.split("-")
            mic, _ = end.split(".")
            self.vector = self._get_pos_ml(int(mic), int(spk))
            return self.vector

        elif md == "fr":
            _, wavnmb = filename.split("-")
            number, _ = wavnmb.split(".")
            self.vector = self._get_pos_fr(int(number))
            return self.vector

    def _get_pos_fr(self, nmb):
        z = 1.5
        x_mic = (nmb % (30 * 20 * 20 * 30)) / (30 * 20 * 20)
        y_mic = (nmb % (30 * 20 * 20)) / (30 * 20)
        x_spk = (nmb % (30 * 20)) / 20
        y_spk = nmb % 20

        x_mic = round(int(x_mic) * self.d_fr, 1)
        y_mic = round(int(y_mic) * self.d_fr, 1)
        x_spk = round(int(x_spk) * self.d_fr, 1)
        y_spk = round(int(y_spk) * self.d_fr, 1)

        return [x_mic, y_mic, z, x_spk, y_spk, z,
                self.rdx, self.rdy, self.rdz, self.t60]

    def _get_pos_ml(self, mic, spk):
        x_mic, y_mic, z_mic = self._mic_pos_ml(mic)
        x_spk, y_spk, z_spk = self._spk_pos_ml(spk)
        return [x_mic, y_mic, z_mic, x_spk, y_spk, z_spk, self.rdx, self.rdy, self.rdz, self.t60]

    def _mic_pos_ml(self, mic):
        mc = 24
        md0 = 1.7
        md1 = 1.05
        x_mic = self.rdx - md0
        y_mic = self.rdy - md1
        row = int(mic / mc)
        column = int(mic - (mc * row))
        x_mic -= column * self.d_m_ml
        y_mic -= row * self.d_m_ml
        z_mic = 1.5
        x_mic = round(x_mic, 2)
        y_mic = round(y_mic, 2)
        z_mic = round(z_mic, 2)

        return x_mic, y_mic, z_mic

    def _spk_pos_ml(self, spk):
        spk0 = [self.rdx - 7.48, self.rdy - 1.59]
        diag_dis = self.d_s_ml / math.sqrt(2)
        if spk == 0:
            x_spk = spk0[0]
            y_spk = spk0[1]
        elif spk < 8:
            x_spk = spk0[0] + spk * diag_dis
            y_spk = spk0[1] + spk * diag_dis
        elif spk < 32:
            x_spk = spk0[0] + 8 * diag_dis + ((spk - 8) * self.d_s_ml)
            y_spk = spk0[1] + 8 * diag_dis
        elif spk < 40:
            x_spk = spk0[0] + 9 * diag_dis + 23 * self.d_s_ml + (spk - 32) * diag_dis
            y_spk = spk0[1] + 7 * diag_dis - (spk - 32) * diag_dis
        elif spk < 48:
            x_spk = spk0[0] + 17 * diag_dis + 23 * self.d_s_ml
            y_spk = spk0[1] - self.d_s_ml + self.d_s_ml * diag_dis - ((spk - 40) * self.d_s_ml)
        elif spk < 56:
            x_spk = spk0[0] + 15 * diag_dis + 24 * self.d_s_ml - (spk - 48) * diag_dis
            y_spk = spk0[1] - 9 * self.d_s_ml + self.d_s_ml * diag_dis - (spk - 48) * diag_dis
        elif spk < 80:
            x_spk = spk0[0] + 7 * diag_dis + 24 * self.d_s_ml - ((spk - 56) * self.d_s_ml)
            y_spk = spk0[1] - 8 * self.d_s_ml - 9 * diag_dis
        elif spk < 88:
            x_spk = spk0[0] + 6 * diag_dis + self.d_s_ml - (spk - 80) * diag_dis
            y_spk = spk0[1] - 8 * self.d_s_ml - 8 * diag_dis + (spk - 80) * diag_dis
        elif spk < 96:
            x_spk = spk0[0] - 2 * diag_dis + self.d_s_ml
            y_spk = spk0[1] - 8 * self.d_s_ml + ((spk - 88) * self.d_s_ml)
        else:
            x_spk = spk0[0]
            y_spk = spk0[1]

        z_spk = 1.5
        x_spk = round(x_spk, 2)
        y_spk = round(y_spk, 2)
        z_spk = round(z_spk, 2)

        return x_spk, y_spk, z_spk


class Saver:
    """saver is responsible to save features, and the min max values."""

    def __init__(self, fr_feature_save_dir, fr_min_max_values_save_dir, fr_embedding_vector_save_dir,
                 ml_feature_save_dir, ml_min_max_values_save_dir, ml_embedding_vector_save_dir):

        self.fr_fsd = fr_feature_save_dir
        self.fr_mmsd = fr_min_max_values_save_dir
        self.fr_evsd = fr_embedding_vector_save_dir
        self.ml_fsd = ml_feature_save_dir
        self.ml_mmsd = ml_min_max_values_save_dir
        self.ml_evsd = ml_embedding_vector_save_dir
        self.mic_data_path = "./mic_data"

    def save_feature(self, feature, file_path="None", md="ml", name="None"):
        if md == "ml":
            file_name = name
            self.create_directory_if_none(self.ml_fsd)
            save_path = os.path.join(self.ml_fsd, file_name + ".npy")
        else:
            file_name = os.path.split(file_path)[1]
            rir, _ = file_name.split(".")
            self.create_directory_if_none(self.fr_fsd)
            save_path = os.path.join(self.fr_fsd, rir + ".npy")

        np.save(save_path, feature)
        return save_path

    def save_min_max_values(self, min_max_values_f, min_max_values_p, md="ml"):
        if md == "ml":
            save_dir = self.ml_mmsd
        else:
            save_dir = self.fr_mmsd
        self.create_directory_if_none(save_dir)
        save_path_f = os.path.join(save_dir, f"min_max_values_f_{md}.pkl")
        save_path_p = os.path.join(save_dir, f"min_max_values_p_{md}.pkl")
        self._save(min_max_values_f, save_path_f)
        self._save(min_max_values_p, save_path_p)

    def save_embedding_vector(self, embedding_vector, md="ml"):
        if md == "ml":
            save_dir = self.ml_evsd
        else:
            save_dir = self.fr_evsd
        self.create_directory_if_none(save_dir)
        save_path = os.path.join(save_dir, f"emb_vec_{md}.pkl")
        self._save(embedding_vector, save_path)

    def save_ml_waveform(self, mic_list):
        directory = pathlib.Path(self.mic_data_path)
        self.create_directory_if_none(directory)
        file_name = "mic_data"
        save_path = os.path.join(self.mic_data_path, file_name + ".npy")
        np.save(save_path, mic_list)

    @staticmethod
    def _save(data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def create_directory_if_none(dir_path):
        dir_path = dir_path
        directory = pathlib.Path(dir_path)
        if not directory.exists():
            os.makedirs(dir_path)


class Preprocess:

    def __init__(self):
        self.shape_t = None
        self.f_extractor = None
        self.normalizer = None
        self.saver = None
        self.min_max_values_f = {}
        self.min_max_values_p = {}
        self.emb_vec_values = {}
        self.MLloader = None
        self._FRloader = None
        self.emb_vector = None

        self.padder = None
        self._num_expected_samples = None

    @property
    def loader(self):
        return self._FRloader

    @loader.setter
    def loader(self, loader):
        self._FRloader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.duration)

    def process(self, fr_files_dir, ml_files_dir, md):
        if md == "fr" or md == "both":
            _md = "fr"
            self._reset_dictionaries()
            self._rename_fr_files(fr_files_dir)

            for root, _, files in os.walk(fr_files_dir):
                for file in tqdm(sorted(files)):
                    file_path = os.path.join(root, file)
                    self._process_file(file_path, _md)

            self.saver.save_min_max_values(self.min_max_values_f, self.min_max_values_p, md=_md)
            self.saver.save_embedding_vector(self.emb_vec_values, md=_md)

        if md == "ml" or md == "both":
            _md = "ml"
            self._reset_dictionaries()

            mic_list = self.MLloader.load_mat_data(ml_files_dir)
            self.saver.save_ml_waveform(mic_list)

            for mic_num, mic in enumerate(mic_list):
                for speaker_num, speaker_waveform in enumerate(mic):
                    spk_nm = str(speaker_num).zfill(2)
                    mic_string = str(mic_num).zfill(3)
                    file_name = f"RIR-{spk_nm}-{mic_string}.wav"
                    self._process_file(speaker_waveform, _md, name=file_name)
                    print(f"Processed file {file_name}")
            self.saver.save_min_max_values(self.min_max_values_f, self.min_max_values_p, md=_md)
            self.saver.save_embedding_vector(self.emb_vec_values, md=_md)

    def _process_file(self, file_path, md, name=None):
        if md == "fr":
            signal = self.loader.load(file_path)
            if self._is_padding_necessary(signal):

                signal = self._apply_right_padding(signal)

            name = os.path.basename(file_path)
            emb_vec = self.emb_vector.create_vector(md, name)

            feature, phase = self.f_extractor.extract(signal)
            norm_feature = self.normalizer.normalize(feature)
            pad_feature = self.shape_t.transform(norm_feature)
            norm_phase = self.normalizer.normalize(phase)
            pad_phase = self.shape_t.transform(norm_phase)

            spectrogram = tf.stack([pad_feature, pad_phase], axis=2)
            save_path = self.saver.save_feature(spectrogram, file_path=file_path, md=mode)
            self._store_min_max_value(save_path, feature.min(), feature.max(), phase.min(), phase.max())
            self._store_embedding_vector(save_path, emb_vec)

        elif md == "ml":
            signal = file_path
            if self._is_padding_necessary(signal):

                signal = self._apply_right_padding(signal)

            emb_vec = self.emb_vector.create_vector(md, name)

            feature, phase = self.f_extractor.extract(signal)
            norm_feature = self.normalizer.normalize(feature)
            pad_feature = self.shape_t.transform(norm_feature)
            norm_phase = self.normalizer.normalize(phase)
            pad_phase = self.shape_t.transform(norm_phase)

            spectrogram = tf.stack([pad_feature, pad_phase], axis=2)
            save_path = self.saver.save_feature(spectrogram, name=name, md=mode)
            self._store_min_max_value(save_path, feature.min(), feature.max(), phase.min(), phase.max())
            self._store_embedding_vector(save_path, emb_vec)

    def _reset_dictionaries(self):
        self.min_max_values_f = {}
        self.min_max_values_p = {}
        self.emb_vec_values = {}

    @staticmethod
    def _rename_fr_files(directory):
        fnames = sorted(os.listdir(directory))
        if fnames[0] != "RIR-000000.wav":
            print("Renaming files...")
            for fname in fnames:
                _, file_number_ext = fname.split("-")
                file_number, _ = file_number_ext.split(".")
                number = file_number.zfill(6)
                new_file_name = f"RIR-{number}.wav"
                os.rename(os.path.join(directory, fname), os.path.join(directory, new_file_name))

    def _is_padding_necessary(self, signal):
        if len(signal) < self._num_expected_samples:
            return True
        return False

    def _apply_right_padding(self, signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal

    def _apply_left_padding(self, signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.left_pad(signal, num_missing_samples)
        return padded_signal

    def _apply_right_snipping(self, signal):
        num_exceeding_samples = len(signal) - self._num_expected_samples
        padded_signal = self.padder.right_snip(signal, num_exceeding_samples)
        return padded_signal

    def _apply_left_snipping(self, signal):
        num_exceeding_samples = len(signal) - self._num_expected_samples
        padded_signal = self.padder.left_snip(signal, num_exceeding_samples)
        return padded_signal

    def _store_min_max_value(self, save_path, min_val_f, max_val_f, min_val_p, max_val_p):
        self.min_max_values_f[save_path] = {
            "min": min_val_f,
            "max": max_val_f
        }
        self.min_max_values_p[save_path] = {
            "min": min_val_p,
            "max": max_val_p
        }

    def _store_embedding_vector(self, save_path, emb_v):
        self.emb_vec_values[save_path] = {
            "lpx": emb_v[0],
            "lpy": emb_v[1],
            "lpz": emb_v[2],
            "spx": emb_v[3],
            "spy": emb_v[4],
            "spz": emb_v[5],
            "rdx": emb_v[6],
            "rdy": emb_v[7],
            "rdz": emb_v[8],
            "t60": emb_v[9]
        }


if __name__ == "__main__":

    N_FFT = 256 
    WINDOW_LENGTH = 64 
    HOP_LENGTH = 32
    DURATION = 0.25  # in seconds
    SAMPLE_RATE = 16000
    MONO = True

    RDX = 9
    RDY = 6
    RDZ = 2.5
    T60 = 0.2

    MATLAB_FILES_DIR = "../data/databases/matlab_data/RIR_files/Mat"
    WAV_FILES_DIR = "../data/databases/data_Fast_RIR"

    FR_SPECTROGRAM_SAVE_DIR = "../data/features/fr_features/spectrogram"
    FR_MIN_MAX_VALUES_SAVE_DIR = "../data/features/fr_features"
    FR_EMBEDDING_VECTOR_SAVE_DIR = "../data/features/fr_features"

    ML_SPECTROGRAM_SAVE_DIR = "../data/features/ml_features/spectrogram"
    ML_MIN_MAX_VALUES_SAVE_DIR = "../data/features/ml_features"
    ML_EMBEDDING_VECTOR_SAVE_DIR = "../data/features/ml_features"

    # instantiate all objects
    mode = "fr"  # ["both", "ml", "fr"]

    mlloader = MATLabLoader()
    frloader = FRLoader(SAMPLE_RATE, DURATION, MONO)
    shape_t = ShapeTransform((144, 128))
    padder = Padder()
    feature_extractor = FeatureExtractor(N_FFT, WINDOW_LENGTH, HOP_LENGTH)
    min_max_normalizer = Normalizer(0, 1)
    emb_vector = EmbeddingVector(RDX, RDY, RDZ, T60)
    saver = Saver(FR_SPECTROGRAM_SAVE_DIR, FR_MIN_MAX_VALUES_SAVE_DIR, FR_EMBEDDING_VECTOR_SAVE_DIR,
                  ML_SPECTROGRAM_SAVE_DIR, ML_MIN_MAX_VALUES_SAVE_DIR, ML_EMBEDDING_VECTOR_SAVE_DIR)

    preprocessing_pipeline = Preprocess()
    preprocessing_pipeline.MLloader = mlloader
    preprocessing_pipeline.loader = frloader
    preprocessing_pipeline.padder = padder
    preprocessing_pipeline.shape_t = shape_t
    preprocessing_pipeline.f_extractor = feature_extractor
    preprocessing_pipeline.normalizer = min_max_normalizer
    preprocessing_pipeline.saver = saver
    preprocessing_pipeline.emb_vector = emb_vector

    preprocessing_pipeline.process(WAV_FILES_DIR, MATLAB_FILES_DIR, mode)
