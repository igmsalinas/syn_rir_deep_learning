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
import pathlib
import pickle
import librosa
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from preprocess import FRLoader as FRLoader
from tqdm import tqdm
import time
from postprocess import PostProcess as P


def create_directory_if_none(dir_path):
    dir_path = dir_path
    directory = pathlib.Path(dir_path)
    if not directory.exists():
        os.makedirs(dir_path)


def plot_learning_curves(directory, model):
    abs_dir = directory + model
    path_dir = pathlib.Path(abs_dir)
    file = f"size_histories_{model}.pkl"
    file_dir = os.path.join(path_dir, file)
    with open(file_dir, "rb") as f:
        histories = pickle.load(f)
        ae_histories = histories[model]

    if model in ["ResAE", "AE"]:
        plt.plot(ae_histories['loss'])
        plt.plot(ae_histories['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        with open(abs_dir + f"/{model}_loss.txt", "w") as text_file:
            print(f"{model} losses at last epoch({len(ae_histories['loss'])}) of training\n"
                  f"Loss: {ae_histories['loss'][-1]}\n"
                  f"Validation loss: {ae_histories['val_loss'][-1]}", file=text_file)

    elif model in ["VAE"]:
        plt.plot(ae_histories['_calculate_reconstruction_loss'])
        plt.plot(ae_histories['val__calculate_reconstruction_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        with open(abs_dir + f"/{model}_loss.txt", "w") as text_file:
            print(f"{model} losses at last epoch({len(ae_histories['loss'])}) of training\n"
                  f"Total Loss: {ae_histories['loss'][-1]}\n"
                  f"Reconstruction Loss: {ae_histories['_calculate_reconstruction_loss'][-1]}\n"
                  f"KL Loss: {ae_histories['_calculate_kl_loss'][-1]}\n"
                  f"Total Validation Loss: {ae_histories['val_loss'][-1]}\n"
                  f"Reconstruction Validation loss: {ae_histories['val__calculate_reconstruction_loss'][-1]}\n"
                  f"KL Validation loss: {ae_histories['val__calculate_kl_loss'][-1]}", file=text_file)

    # plt.show()
    path = pathlib.Path(abs_dir + f"/{model}_learning_curve.png")
    plt.savefig(path)
    print(f"Learning curve and final loss of {model} saved at {abs_dir}")


def plot_spec_vs_wav(spectrogram, waveform):
    fig, ax = plt.subplots(2, figsize=(12, 8))
    plot_spectrogram_at_ax(spectrogram, ax[0], "Spectrogram")
    plot_waveform_at_ax(waveform, ax[1], "Waveform")


def plot_vector(vector):
    x = [vector[0], vector[3]]
    y = [vector[1], vector[4]]
    values = [0, 1]
    classes = ["lp", "sp"]
    colors = ListedColormap(['gray', 'black'])
    ax = plt.axes()
    scatter = ax.scatter(x, y, c=values, cmap=colors)
    ax.set_xlabel("$X$ [$m$]")
    ax.set_ylabel("$Y$ [$m$]")
    ax.set_xlim(0, vector[6])
    ax.set_ylim(0, vector[7])
    plt.legend(handles=scatter.legend_elements()[0], labels=classes, loc="best")

    plt.show()


def plot_spectrogram_at_ax(feature, ax, title):
    if len(feature.shape) > 2:
        assert len(feature.shape) == 3
        feature = np.squeeze(feature, axis=-1)

    height = feature.shape[0]
    width = feature.shape[1]
    x = np.linspace(0, np.size(feature), num=width, dtype=int)
    y = range(height)
    ax.pcolormesh(x, y, feature)
    ax.set_title(title)


def load_fr_spec(path, lpx, lpy, spx, spy):
    nmb = get_wav_nmb(lpx, lpy, spx, spy)
    file_name = "RIR-" + nmb + ".npy"
    file_path = os.path.join(path, file_name)
    fr_feature = np.load(file_path)
    p = P(path)
    log_spec, phase = p.get_stft_phase(fr_feature)
    log_spec, phase = p.de_shape(log_spec, phase, (129, 126))
    return log_spec, phase


def load_generated_spec(path, lpx, lpy, spx, spy):
    file_name = f"RIR-{str(lpx)}-{str(lpy)}-1.5-{str(spx)}-{str(spy)}-1.5-9.0-6.0-2.5-0.2.npy"
    file_path = os.path.join(path, file_name)
    feature = np.load(file_path)
    p = P(path)
    log_spec, phase = p.get_stft_phase(feature)
    log_spec, phase = p.de_shape(log_spec, phase, (129, 126))
    return log_spec, phase


def plot_comparison_specs(g_f, g_p, fr_f, fr_p, tit, file, model):
    fig, ax = plt.subplots(2, figsize=(12, 12))
    fig.suptitle(tit)
    plot_spectrogram_at_ax(fr_f, ax[0], "Real Log Magnitude")
    plot_spectrogram_at_ax(g_f, ax[1], "Generated Log Magnitude")
    directory = f"../../generated_rir/{model}/pngs/stfts/"
    create_directory_if_none(directory)
    name = file.rsplit(".", 1)[0]
    fig.savefig(f"../../generated_rir/{model}/pngs/stfts/log_mag_{name}.png")
    plt.close(fig)

    fig, ax = plt.subplots(2, figsize=(12, 12))
    fig.suptitle(tit)
    plot_spectrogram_at_ax(fr_p, ax[0], "Real Phase")
    plot_spectrogram_at_ax(g_p, ax[1], "Generated Phase")
    directory = f"../../generated_rir/{model}/pngs/stfts/"
    create_directory_if_none(directory)
    name = file.rsplit(".", 1)[0]
    fig.savefig(f"../../generated_rir/{model}/pngs/stfts/phase_{name}.png")
    plt.close(fig)


def plot_fr_vs_spec_gen(lpx, lpy, spx, spy, dir_path, file, model):
    tit = f"RIR at lpx: {str(lpx)} lpy: {str(lpy)} spx: {str(spx)} spy: {str(spy)}"
    fr_dir = "../../data/features/fr_features/spectrogram"
    g_f, g_p = load_generated_spec(dir_path, lpx, lpy, spx, spy)
    fr_f, fr_p = load_fr_spec(fr_dir, lpx, lpy, spx, spy)
    plot_comparison_specs(g_f, g_p, fr_f, fr_p, tit, file, model)


def plot_generated_spectrogram(g_dir, model):
    dir_path = g_dir + model + "/stft/"
    save_path = g_dir + model + "/png/stft"
    directory = pathlib.Path(dir_path)
    print("Plotting comparison between generated and requested spectrogram...")
    time.sleep(0.5)
    for root, _, files in os.walk(directory):
        for file in tqdm(files):
            _, lpx, lpy, _, spx, spy, _, _, _, _, _ = file.split("-")
            plot_fr_vs_spec_gen(float(lpx), float(lpy), float(spx), float(spy), dir_path, file, model)
    print(f"Comparison between generated and requested stft saved at {save_path} as PNG")


def get_wav_nmb(lpx, lpy, spx, spy):
    d_lx = int((lpx / 0.3) * 20 * 30 * 20)
    d_ly = int((lpy / 0.3) * 20 * 30)
    d_sx = int((spx / 0.3) * 20)
    d_sy = int(spy / 0.3)
    nmb = d_lx + d_ly + d_sx + d_sy
    nmb = str(nmb).zfill(6)
    return nmb


def return_fr_wav(loader, lpx, lpy, spx, spy):
    d_lx = int((lpx / 0.3) * 20 * 30 * 20)
    d_ly = int((lpy / 0.3) * 20 * 30)
    d_sx = int((spx / 0.3) * 20)
    d_sy = int(spy / 0.3)
    nmb = d_lx + d_ly + d_sx + d_sy
    nmb = str(nmb).zfill(6)
    wav = "../../data/databases/data_Fast_RIR/" + f"RIR-{nmb}.wav"
    signal = loader.load(wav)
    return signal, nmb


def plot_waveform_at_ax(waveform, ax, title):
    time_ax = np.linspace(0, len(waveform) / 16000, num=len(waveform))
    ax.plot(time_ax, waveform)
    ax.set_title(title)


def plot_r_wav_vs_g_wav(r_waveform, g_waveform, tit, file_name, model):
    fig, ax = plt.subplots(2, figsize=(12, 8))
    plot_waveform_at_ax(r_waveform, ax[0], "Real Waveform")
    plot_waveform_at_ax(g_waveform, ax[1], "Generated Waveform")
    fig.suptitle(tit)
    # plt.show()
    directory = f"../../generated_rir/{model}/png/wav"
    create_directory_if_none(directory)
    name = file_name.rsplit(".", 1)[0]
    fig.savefig(f"../../generated_rir/{model}/png/wav/{name}.png")
    plt.close(fig)


def load_wavs(g_dir, loader, lpx, lpy, spx, spy):
    wav_s, _ = return_fr_wav(loader, lpx, lpy, spx, spy)
    wav_g = librosa.load(g_dir + f"RIR-{str(lpx)}-{str(lpy)}"
                                 f"-1.5-{str(spx)}-{str(spy)}"
                                 f"-1.5-9.0-6.0-2.5-0.2.wav",
                         sr=16000,
                         duration=0.25,
                         mono=True)[0]
    return wav_s, wav_g


def plot_fr_vs_rir_gen(lpx, lpy, spx, spy, g_dir, file_name, model):
    loader = FRLoader(16000, 0.25, True)
    wav1, wav2 = load_wavs(g_dir, loader, lpx, lpy, spx, spy)
    tit = f"RIR at lpx: {str(lpx)} lpy: {str(lpy)} spx: {str(spx)} spy: {str(spy)}"
    plot_r_wav_vs_g_wav(wav1, wav2, tit, file_name, model)


def plot_generated_wav(g_dir, model):
    dir_path = g_dir + model + "/rir/"
    save_path = g_dir + model + "/png/wav"
    directory = pathlib.Path(dir_path)
    print("Plotting comparison between generated and requested wavs...")
    time.sleep(0.5)
    for root, _, files in os.walk(directory):
        for file in tqdm(files):
            _, lpx, lpy, _, spx, spy, _, _, _, _, _ = file.split("-")
            plot_fr_vs_rir_gen(float(lpx), float(lpy), float(spx), float(spy), dir_path, file, model)
    print(f"Comparison between generated and requested RIRs saved at {save_path} as PNG")


def get_mse(a, b):
    mse = (np.square(a - b)).mean()
    return mse


def get_mse_spectrogram(g_dir, model):
    dir_path = g_dir + model + "/stft/"
    directory = pathlib.Path(dir_path)
    fr_dir = "../../data/features/fr_features/spectrogram"
    mse_m = 0.0
    mse_p = 0.0
    number_stfts = 0
    for root, _, files in os.walk(directory):
        for file in files:
            _, lpx, lpy, _, spx, spy, _, _, _, _, _ = file.split("-")
            g_log_spec, g_phase = load_generated_spec(dir_path, float(lpx), float(lpy), float(spx), float(spy))
            s_log_spec, s_phase = load_fr_spec(fr_dir, float(lpx), float(lpy), float(spx), float(spy))
            mse_m += get_mse(s_log_spec, g_log_spec)
            mse_p += get_mse(s_phase, g_phase)
            number_stfts += 1

    mse_m = mse_m / number_stfts
    mse_p = mse_p / number_stfts
    with open(g_dir + model + f"/{model}_stft_loss.txt", "w") as text_file:
        print(f"{model} loss in {number_stfts} generated STFTs\n"
              f"MSE Log Magnitude: {mse_m}\n"
              f"MSE Phase: {mse_p}", file=text_file)
    print(f"{model} MSE loss in {number_stfts} generated RIRs saved at {g_dir + model}")


def get_mse_wav(g_dir, model):
    dir_path = g_dir + model + "/rir/"
    directory = pathlib.Path(dir_path)
    loader = FRLoader(16000, 0.25, True)
    mse = 0.0
    mse_50 = 0.0
    number_wavs = 0
    for root, _, files in os.walk(directory):
        for file in files:
            _, lpx, lpy, _, spx, spy, _, _, _, _, _ = file.split("-")
            wav_s, wav_g = load_wavs(dir_path, loader, float(lpx), float(lpy), float(spx), float(spy))
            mse += get_mse(wav_s, wav_g)
            mse_50 += get_mse(wav_s[:800], wav_g[:800])
            number_wavs += 1

    mse = mse / number_wavs
    mse_50 = mse_50 / number_wavs
    with open(g_dir + model + f"/{model}_wav_loss.txt", "w") as text_file:
        print(f"{model} loss in {number_wavs} generated RIRs\n"
              f"MSE: {mse}\n"
              f"MSE_50ms: {mse_50}", file=text_file)
    print(f"{model} MSE loss in {number_wavs} generated RIRs saved at {g_dir + model}")


if __name__ == "__main__":

    MODEL = "VAE"
    GENERATED_DIR = "../../generated_rir/"
    MODELS_DIR = "../../models/"

    plot_learning_curves(MODELS_DIR, MODEL)

    plot_generated_wav(GENERATED_DIR, MODEL)
    get_mse_wav(GENERATED_DIR, MODEL)

    plot_generated_spectrogram(GENERATED_DIR, MODEL)
    get_mse_spectrogram(GENERATED_DIR, MODEL)
