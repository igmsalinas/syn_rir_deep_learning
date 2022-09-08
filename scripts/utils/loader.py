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
import random
import numpy as np


class Loader:
    """
    The Loader class contains the methods to obtain the training data,
    by loading the STFTs, the vectors corresponding
    to them and the min max values for an upcoming denormalization.
    It is responsible for shuffling the data and splitting
    the training and validation sets.
    """

    def __init__(self, features_dir, vector_dir, distance_adjuster=1, min_value=0):
        self.features_dir = features_dir
        self.data_dir = vector_dir
        self.distance_adjuster = distance_adjuster
        self.min_value = min_value

        self.features_list = []
        self.vector_list = []
        self.mm_list = []

        self.train_f, self.train_v = None, None
        self.val_f, self.val_v = None, None
        self.test_f, self.test_v = None, None

        self.train_vectors, self.val_vectors, self.test_vectors = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None

    def load_data(self):
        """
        Main method for loading the data. Starts loading the features, shuffling
        them, splitting the validation and test sets and associates the corresponding
        STFTs with the target as indicated in the vector.

        :return:
        """
        self._load_features()
        self._shuffle()
        self._split_train_val_test()
        self._get_y_and_vector()

        return self._return_data()

    def return_raw_database(self):
        """
        Returns the database without being shuffled or split.

        :return: 3-element tuple (list[features], list[vectors], list[min_max])
        """
        self._load_features()
        return self.features_list, self.vector_list, self.mm_list

    @staticmethod
    def shuffle(list1, list2):
        """
        Shuffles two given lists.

        :param list1: Array 1
        :param list2: Array 2
        :return: Shuffled array 1 and array 2 with same indices
        """
        temp = list(zip(list1, list2))
        random.shuffle(temp)
        sh1, sh2 = zip(*temp)
        return np.array(sh1), np.array(sh2)

    @staticmethod
    def shuffle_list(i_list):
        """
        Shuffles a list.

        :param i_list: Input list
        :return: Shuffled list
        """
        random.shuffle(i_list)
        return i_list

    def _get_y_and_vector(self):
        """
        Associates the and stacks the target and the vectors for training and validation.
        """
        print("Obtaining vectors and labels")
        y_train_f, y_train_v = self.shuffle(self.train_f, self.train_v)
        x_train_vectors = np.stack((self.train_v, y_train_v), axis=2)

        y_val_f, y_val_v = self.shuffle(self.val_f, self.val_v)
        x_val_vectors = np.stack((self.val_v, y_val_v), axis=2)

        self.y_train, self.y_val = y_train_f, y_val_f
        self.train_vectors, self.val_vectors = x_train_vectors, x_val_vectors

    def _shuffle(self):
        """
        Shuffles the feature and the vector list.
        """
        print("Shuffling data")
        self.features_list, self.vector_list = self.shuffle(self.features_list, self.vector_list)

    def _split_train_val_test(self):
        """
        Splits the database into training and validation sets.
        """
        print("Splitting data")
        n = len(self.features_list)
        self.train_f, self.train_v = self.features_list[0:int(0.8 * n)], self.vector_list[0:int(0.8 * n)]
        self.val_f, self.val_v = self.features_list[int(n * 0.8):], self.vector_list[int(n * 0.8):]

    def _return_data(self):
        """
        Returns the training and validation sets

        :return: tuple[t_features, t_vectors, t_target, v_features, v_vectors, v_target]
        """
        return self.train_f, self.train_vectors, self.y_train, \
               self.val_f, self.val_vectors, self.y_val

    def _load_features(self):
        """
        Loads the features, vectors and min_max values into lists.
        """
        print("Loading data ...")
        vector_dic = self.load_vector()
        min_max_dic_f, min_max_dic_p = self.load_minmax()
        i = 1
        temp_f_list = []
        temp_v_list = []
        temp_mm_list = []
        for root, _, file_names in os.walk(self.features_dir):
            for file in sorted(file_names):
                file_path = os.path.join(root, file)
                dic_key = self.features_dir + "/" + file
                if i < self.distance_adjuster:
                    i += 1
                else:
                    if self._check_valid(vector_dic[dic_key]):
                       
                        vector = self._load_vec_into_list(vector_dic[dic_key])
                        mm_tuple = self._load_mm_into_tuple(min_max_dic_f[dic_key], min_max_dic_p[dic_key])
                        temp_mm_list.append(mm_tuple)
                        temp_v_list.append(vector)
                        feature = np.load(file_path)
                        temp_f_list.append(feature)
                        i = 1

        self.features_list = temp_f_list
        self.vector_list = temp_v_list
        self.mm_list = temp_mm_list

    def load_feature(self, feature_num):
        """
        Loads a single feature being given a number of feature.

        :param feature_num: int
        :return: np.ndarray
        """
        file_name = "RIR-" + feature_num + ".npy"
        file_path = os.path.join(self.features_dir, file_name)
        feature = np.load(file_path)
        return feature

    def load_vector(self):
        """
        Loads the whole vector values

        :return: dictionary of dictionaries
        """
        directory = pathlib.Path(self.data_dir)
        load_path = os.path.join(directory, "emb_vec_fr" + ".pkl")
        with open(load_path, "rb") as f:
            vector_dic = pickle.load(f)
        return vector_dic

    def load_minmax(self):
        """
        Loads the min_max values

        :return: dictionary of dictionaries
        """
        directory = pathlib.Path(self.data_dir)
        load_path = os.path.join(directory, "min_max_values_f_fr" + ".pkl")
        with open(load_path, "rb") as f:
            min_max_f = pickle.load(f)
        load_path = os.path.join(directory, "min_max_values_p_fr" + ".pkl")
        with open(load_path, "rb") as f:
            min_max_p = pickle.load(f)
        return min_max_f, min_max_p

    def _check_valid(self, vec_dictionary):
        """
        Checks if the gap is satisfied (parameter from loading)

        :param vec_dictionary: Dictionary of vectors
        :return: True or False
        """
        values = list(vec_dictionary.values())
        min_value = self.min_value
        max_value_x = values[6] - min_value
        max_value_y = values[7] - min_value
        condition = True
        if (values[0] < min_value or values[1] < min_value) or \
                (values[0] > max_value_x or values[1] > max_value_y):
            condition = False
        return condition

    @staticmethod
    def _load_vec_into_list(vec_dic):
        """
        Changes type from dictionary to list

        :param vec_dic: Vector dictionary
        :return: Vector list
        """
        vec_list = []
        for key, value in vec_dic.items():
            vec_list.append(float(value))
        return vec_list

    @staticmethod
    def _load_mm_into_tuple(mm_dic_f, mm_dic_p):
        """
        Changes type from dictionary to tuple

        :param mm_dic_f: Dictionary of features
        :param mm_dic_p: Dictionary of phases
        :return: Tuple of min_max
        """
        mm_list_f = []
        mm_list_p = []
        for key, value in mm_dic_f.items():
            mm_list_f.append(float(value))
        for key, value in mm_dic_p.items():
            mm_list_p.append(float(value))
        mm_tuple = (mm_list_f, mm_list_p)
        return mm_tuple


if __name__ == "__main__":
    FR_EMBEDDING_VECTOR_DIR = "../data/features/fr_features"
    FR_SPECTROGRAM_DIR = "../data/features/fr_features/spectrogram"
    loader = Loader(FR_SPECTROGRAM_DIR, FR_EMBEDDING_VECTOR_DIR, distance_adjuster=3)
