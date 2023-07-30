#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project: tbc_lstm_gan 
@File: utils_new.py
@Author: Zhe Zhang
@Email: zhe@nii.ac.jp
@Date: 2022/02/08
"""

from utils import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer

settings = {'settings_file': 'settings'}
settings = load_settings_from_file(settings)

locals().update(settings)

# Filter sb data
reasonable_p_rng = [0, 48]  # 4 octaves
reasonable_p_avg = [36, 84]  # C2 to C6
reasonable_p_var = [0, 100]  # no limits
reasonable_d_rng = [0, 8]
reasonable_d_avg = [0, 4]
reasonable_r_rng = [0, 8]


def filter_wrong_data(data, reasonable_p_rng, reasonable_p_avg, reasonable_p_var,
                      reasonable_d_rng, reasonable_d_avg, reasonable_r_rng):

    y = data[:, :SONG_LENGTH * NUM_SONG_FEATURES]              # 60 cols
    y = np.reshape(y, (-1, SONG_LENGTH, NUM_SONG_FEATURES))    # 60 = 20 x 3: 1 pitch, 2 duration, 3 rest

    y_p = y[:, :, 0]
    y_d = y[:, :, 1]
    y_r = y[:, :, 2]

    p_rng, p_avg, p_var = calculate_style_features(y_p, title_str='Pitch', plot=0)
    d_rng, d_avg, d_var = calculate_style_features(y_d, title_str='Duration', plot=0)
    r_rng, r_avg, r_var = calculate_style_features(y_r, title_str='Rest', plot=0)

    wrong_p_idx = []
    for idx, (r, a, v) in enumerate(zip(p_rng, p_avg, p_var)):
        if (r < reasonable_p_rng[0] or r > reasonable_p_rng[1]) or \
                (a < reasonable_p_avg[0] or a > reasonable_p_avg[1]) or \
                (v < reasonable_p_var[0] or v > reasonable_p_var[1]):
            wrong_p_idx.append(idx)

    wrong_d_idx = []
    for idx, (r, a) in enumerate(zip(d_rng, d_avg)):
        if (r < reasonable_d_rng[0] or r > reasonable_d_rng[1]) or \
                (a < reasonable_d_avg[0] or a > reasonable_d_avg[1]):
            wrong_d_idx.append(idx)

    wrong_r_idx = []
    for idx, r in enumerate(r_rng):
        if r > reasonable_r_rng[1]:
            wrong_r_idx.append(idx)

    wrong_idx = list(set(wrong_p_idx + wrong_d_idx + wrong_r_idx))

    print('Wrong Pitch Pieces: ', len(wrong_p_idx))
    print('Wrong Duration Pieces: ', len(wrong_d_idx))
    print('Wrong Rest Pieces: ', len(wrong_r_idx))
    print('Total Wrong Pieces: ', len(wrong_idx))

    data_filtered = np.delete(data, wrong_idx, axis=0)

    return data_filtered


def calculate_style_features(pdr, title_str=None, plot=0):

    rng = pdr.max(axis=1) - pdr.min(axis=1)
    avg = pdr.mean(axis=1)
    var = pdr.var(axis=1)

    if plot == 1:
        plt.figure(figsize=(20, 6))

        plt.subplot(1, 3, 1)
        plt.hist(rng, bins=20, density=True)
        plt.title(title_str + ' Range Distribution')

        plt.subplot(1, 3, 2)
        plt.hist(avg, bins=20, density=True)
        plt.title(title_str + ' Average Distribution')

        plt.subplot(1, 3, 3)
        plt.hist(var, bins=20, density=True)
        plt.title(title_str + ' Variance Distribution')

        plt.show()

    return rng, avg, var

def create_k_bins_encoder(feature, encoder_path, k):

    x1d = feature.reshape(-1, 1)

    encoder = KBinsDiscretizer(n_bins=k,  encode='onehot', strategy='uniform')

    encoder.fit(x1d)

    classes = encoder.n_bins_[0]

    pickle.dump(encoder, open(encoder_path, "wb"))

    return classes


def create_range_2d_encoder(rng, le_path):
    """
    :param x2d: an attribute of a note [pitch, duration or rest]
    :shape x2d: [None, SONG_LENGTH]
    """

    num_bins = int(rng.max() - rng.min())
    x1d = rng.reshape(-1, 1)
    le = KBinsDiscretizer(n_bins=num_bins, encode='onehot', strategy='uniform')

    le.fit(x1d)

    classes = le.n_bins_[0]
    print(classes)
    pickle.dump(le, open(le_path, "wb"))

    return classes


def create_average_2d_encoder(avg, le_path):
    """
    :param x2d: an attribute of a note [pitch, duration or rest]
    :shape x2d: [None, SONG_LENGTH]
    """

    # num_bins = int(np.ceil(avg.max()) - np.floor(avg.min()))
    num_bins = 20
    x1d = avg.reshape(-1, 1)
    le = KBinsDiscretizer(n_bins=num_bins, encode='onehot', strategy='uniform')

    le.fit(x1d)

    classes = le.n_bins_[0]
    print(classes)
    pickle.dump(le, open(le_path, "wb"))

    return classes

def create_variance_2d_encoder(var, le_path):
    """
    :param x2d: an attribute of a note [pitch, duration or rest]
    :shape x2d: [None, SONG_LENGTH]
    """

    # num_bins = int(np.ceil(avg.max()) - np.floor(avg.min()))
    num_bins = 20
    x1d = var.reshape(-1, 1)
    le = KBinsDiscretizer(n_bins=num_bins, encode='onehot', strategy='uniform')

    le.fit(x1d)

    classes = le.n_bins_[0]
    print(classes)
    pickle.dump(le, open(le_path, "wb"))

    return classes

def to_categorical_2d_style(x1d, le_path):
    """
    :param x2d: an attribute of a note [pitch, duration or rest]
    :shape x2d: [None, SONG_LENGTH]

    :return y3d: one hot representation of note attribute
    :shape  y3d: [None, SONG_LENGTH, NUM_[.]_TOKENS]
    """
    le = pickle.load(open(le_path, "rb"))
    x1d = x1d.reshape(-1, 1)
    x2d = le.transform(x1d)

    y2d = x2d.toarray()

    return y2d


def load_data_clean(path, le_paths, song_length, num_song_features, num_meta_features,
                    range_paths, average_paths, variance_paths,
                    is_unique=False, convert_to_tensor=False, return_style_values=False):

    data = np.load(path).astype(np.float32)

    if is_unique:
        data = np.unique(data, axis=0)

    data_filtered = filter_wrong_data(data, reasonable_p_rng, reasonable_p_avg, reasonable_p_var,
                                      reasonable_d_rng, reasonable_d_avg, reasonable_r_rng)

    x = data_filtered[:, song_length * num_song_features:]  # 400 cols : 60 => 460
    x = np.reshape(x, (-1, song_length, num_meta_features))  # 400 col => 20 x 20 cols

    y = data_filtered[:, :song_length * num_song_features]  # 60 cols
    y = np.reshape(y, (-1, song_length, num_song_features))  # 60 = 20 x 3: 1 pitch, 2 duration, 3 rest

    y_p = y[:, :, 0]
    y_d = y[:, :, 1]
    y_r = y[:, :, 2]

    y_p_ohe = to_categorical_2d(y_p, le_paths[0])
    y_d_ohe = to_categorical_2d(y_d, le_paths[1])
    y_r_ohe = to_categorical_2d(y_r, le_paths[2])

    p_rng, p_avg, p_var = calculate_style_features(y_p, title_str='Pitch', plot=0)
    d_rng, d_avg, d_var = calculate_style_features(y_d, title_str='Duration', plot=0)
    r_rng, r_avg, r_var = calculate_style_features(y_r, title_str='Rest', plot=0)

    p_rav = (p_rng, p_avg, p_var)
    d_rav = (d_rng, d_avg, d_var)
    r_rav = (r_rng, r_avg, r_var)

    p_rng_ohe = to_categorical_2d_style(p_rng, range_paths[0])
    p_avg_ohe = to_categorical_2d_style(p_avg, average_paths[0])
    p_var_ohe = to_categorical_2d_style(p_var, variance_paths[0])
    p_rav_ohe = np.concatenate([p_rng_ohe, p_avg_ohe, p_var_ohe], axis=-1)

    d_rng_ohe = to_categorical_2d_style(d_rng, range_paths[1])
    d_avg_ohe = to_categorical_2d_style(d_avg, average_paths[1])
    d_var_ohe = to_categorical_2d_style(d_var, variance_paths[1])
    d_rav_ohe = np.concatenate([d_rng_ohe, d_avg_ohe, d_var_ohe], axis=-1)

    r_rng_ohe = to_categorical_2d_style(r_rng, range_paths[2])
    r_avg_ohe = to_categorical_2d_style(r_avg, average_paths[2])
    r_var_ohe = to_categorical_2d_style(r_var, variance_paths[2])
    r_rav_ohe = np.concatenate([r_rng_ohe, r_avg_ohe, r_var_ohe], axis=-1)

    if convert_to_tensor:
        x = tf.convert_to_tensor(x)

        y_p_ohe = tf.convert_to_tensor(y_p_ohe)
        y_d_ohe = tf.convert_to_tensor(y_d_ohe)
        y_r_ohe = tf.convert_to_tensor(y_r_ohe)

        # p_rng_ohe = tf.convert_to_tensor(p_rng_ohe)
        # p_avg_ohe = tf.convert_to_tensor(p_avg_ohe)
        # p_var_ohe = tf.convert_to_tensor(p_var_ohe)
        #
        # d_rng_ohe = tf.convert_to_tensor(d_rng_ohe)
        # d_avg_ohe = tf.convert_to_tensor(d_avg_ohe)
        # d_var_ohe = tf.convert_to_tensor(d_var_ohe)
        #
        # r_rng_ohe = tf.convert_to_tensor(r_rng_ohe)
        # r_avg_ohe = tf.convert_to_tensor(r_avg_ohe)
        # r_var_ohe = tf.convert_to_tensor(r_var_ohe)

        p_rav_ohe = tf.convert_to_tensor(p_rav_ohe)
        d_rav_ohe = tf.convert_to_tensor(d_rav_ohe)
        r_rav_ohe = tf.convert_to_tensor(r_rav_ohe)

    if return_style_values:
        return x, (y_p, y_d, y_r), (y_p_ohe, y_d_ohe, y_r_ohe), p_rav_ohe, d_rav_ohe, r_rav_ohe, p_rav, d_rav, r_rav
    else:
        return x, (y_p, y_d, y_r), (y_p_ohe, y_d_ohe, y_r_ohe), p_rav_ohe, d_rav_ohe, r_rav_ohe


def get_num_tokens_from_le(le_path):
    # return len((pickle.load(open(le_path, "rb"))).classes_)
    return pickle.load(open(le_path, "rb")).n_bins_[0]


def map_parameter_to_value_range(input, range):
    assert 0 <= input <= 1
    return (range[0] + input * (range[1] - range[0])).astype(np.float32)


def get_style_ohe(pitch_rav, duration_rav, rest_rav, range_paths, average_paths, variance_paths, convert_to_tensor=False):

    p_rng, p_avg, p_var = pitch_rav
    p_rng_ohe = to_categorical_2d_style(p_rng, range_paths[0])
    p_avg_ohe = to_categorical_2d_style(p_avg, average_paths[0])
    p_var_ohe = to_categorical_2d_style(p_var, variance_paths[0])
    p_rav_ohe = np.concatenate([p_rng_ohe, p_avg_ohe, p_var_ohe], axis=-1)

    d_rng, d_avg, d_var = duration_rav
    d_rng_ohe = to_categorical_2d_style(d_rng, range_paths[1])
    d_avg_ohe = to_categorical_2d_style(d_avg, average_paths[1])
    d_var_ohe = to_categorical_2d_style(d_var, variance_paths[1])
    d_rav_ohe = np.concatenate([d_rng_ohe, d_avg_ohe, d_var_ohe], axis=-1)

    r_rng, r_avg, r_var = rest_rav
    r_rng_ohe = to_categorical_2d_style(r_rng, range_paths[2])
    r_avg_ohe = to_categorical_2d_style(r_avg, average_paths[2])
    r_var_ohe = to_categorical_2d_style(r_var, variance_paths[2])
    r_rav_ohe = np.concatenate([r_rng_ohe, r_avg_ohe, r_var_ohe], axis=-1)

    if convert_to_tensor:
        p_rav_ohe = tf.convert_to_tensor(p_rav_ohe)
        d_rav_ohe = tf.convert_to_tensor(d_rav_ohe)
        r_rav_ohe = tf.convert_to_tensor(r_rav_ohe)

    return (p_rav_ohe, d_rav_ohe, r_rav_ohe)

def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr
