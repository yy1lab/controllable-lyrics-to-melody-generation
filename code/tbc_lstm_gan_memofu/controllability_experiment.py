"""
@Project: tbc_lstm_gan_style
@File: controllability_experiment.py
@Author: Zhe Zhang
@Email: zhe@nii.ac.jp
@Date: 2022/09/27
"""
import matplotlib.pyplot as plt
import numpy as np

from generator_memofu import *
from discriminator import *
from drivers_memofu import *
from utils import *

import argparse
from gensim.models import Word2Vec
import time

from utils_new import *
import sys


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

def main():
    # Set seed for reproducibility

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    """# Data"""

    print("Data: \n")

    NUM_P_TOKENS = len((pickle.load(open(P_LE_PATH, "rb"))).classes_)
    NUM_D_TOKENS = len((pickle.load(open(D_LE_PATH, "rb"))).classes_)
    NUM_R_TOKENS = len((pickle.load(open(R_LE_PATH, "rb"))).classes_)

    num_pitch_range_tokens = get_num_tokens_from_le(P_RNG_PATH)
    num_duration_range_tokens = get_num_tokens_from_le(D_RNG_PATH)
    num_rest_range_tokens = get_num_tokens_from_le(R_RNG_PATH)

    num_pitch_average_tokens = get_num_tokens_from_le(P_AVG_PATH)
    num_duration_average_tokens = get_num_tokens_from_le(D_AVG_PATH)
    num_rest_average_tokens = get_num_tokens_from_le(R_AVG_PATH)

    num_pitch_variance_tokens = get_num_tokens_from_le(P_VAR_PATH)
    num_duration_variance_tokens = get_num_tokens_from_le(D_VAR_PATH)
    num_rest_variance_tokens = get_num_tokens_from_le(R_VAR_PATH)

    NUM_TOKENS = [NUM_P_TOKENS, NUM_D_TOKENS, NUM_R_TOKENS]
    LE_PATHS = [P_LE_PATH, D_LE_PATH, R_LE_PATH]

    num_pitch_style_tokens = num_pitch_range_tokens + num_pitch_average_tokens + num_pitch_variance_tokens
    num_duration_style_tokens = num_duration_range_tokens + num_duration_average_tokens + num_duration_variance_tokens
    num_rest_style_tokens = num_rest_range_tokens + num_rest_average_tokens + num_rest_variance_tokens
    num_style_tokens = [int(num_pitch_style_tokens), int(num_duration_style_tokens), int(num_rest_style_tokens)]

    range_paths = [P_RNG_PATH, D_RNG_PATH, R_RNG_PATH]
    average_paths = [P_AVG_PATH, D_AVG_PATH, R_AVG_PATH]
    variance_paths = [P_VAR_PATH, D_VAR_PATH, R_VAR_PATH]

    # load train (to compute STEPS_PER_EPOCH_TRAIN) and test data

    x_train, y_train_dat_attr, y_train, train_p_rav, train_d_rav, train_r_rav, style_p_rav, style_d_rav, style_r_rav = \
        load_data_clean(
            TRAIN_DATA_PATH,
            LE_PATHS,
            SONG_LENGTH,
            NUM_SONG_FEATURES,
            NUM_META_FEATURES,
            range_paths,
            average_paths,
            variance_paths,
            return_style_values=True)

    x_test, y_test_dat_attr, y_test, test_p_rav, test_d_rav, test_r_rav = load_data_clean(TEST_DATA_PATH,
                                                                                          LE_PATHS,
                                                                                          SONG_LENGTH,
                                                                                          NUM_SONG_FEATURES,
                                                                                          NUM_META_FEATURES,
                                                                                          range_paths,
                                                                                          average_paths,
                                                                                          variance_paths,
                                                                                          convert_to_tensor=True)

    TRAIN_LEN = len(x_train)
    TEST_LEN = len(x_test)

    STEPS_PER_EPOCH_TRAIN = np.ceil(TRAIN_LEN / BATCH_SIZE)
    print('Steps per epoch train: ', STEPS_PER_EPOCH_TRAIN)

    g_model_memofu_style = GeneratorWithMemoryFusionAndStyle(
        G_EMB_UNITS, G_PROJ_UNITS, G_EMB_DROPOUT_RATE, G_PROJ_DROPOUT_RATE,
        G_RNN_DEPTH, G_RNN_UNITS, NUM_TOKENS, NUM_META_FEATURES, num_style_tokens, fusion_layer=FUSION_LAYER)

    # Initialise discriminator
    d_model_style = DiscriminatorWithStyle(
        D_EMB_UNITS, D_PROJ_UNITS, D_EMB_DROPOUT_RATE, D_PROJ_DROPOUT_RATE,
        D_RNN_DEPTH, D_RNN_UNITS, NUM_TOKENS, NUM_META_FEATURES, num_style_tokens)

    adv_train_g_opt = tf.keras.optimizers.Adam(ADV_TRAIN_G_LR, beta_1=0.9, beta_2=0.999)
    adv_train_d_opt = tf.keras.optimizers.Adam(ADV_TRAIN_D_LR, beta_1=0.9, beta_2=0.999)
    seq_train_opt = tf.keras.optimizers.Adam(SEQ_TRAIN_LR, beta_1=0.9, beta_2=0.999)

    adv_train_driver_memofu_style_seq = AdversarialDriverWithStyleAndSeqLoss(g_model_memofu_style,
                                                                             d_model_style,
                                                                             adv_train_g_opt,
                                                                             adv_train_d_opt,
                                                                             seq_train_opt,
                                                                             TEMP_MAX,
                                                                             STEPS_PER_EPOCH_TRAIN,
                                                                             ADV_TRAIN_EPOCHS,
                                                                             NUM_TOKENS,
                                                                             MAX_GRAD_NORM,
                                                                             LE_PATHS)
    # memofu + style + seq
    memofu_style_seq_adv_train_ckpt = tf.train.Checkpoint(g_model=g_model_memofu_style,
                                                          d_model=d_model_style,
                                                          adv_train_g_opt=adv_train_g_opt,
                                                          adv_train_d_opt=adv_train_d_opt)

    memofu_style_seq_adv_train_ckpt_manager = tf.train.CheckpointManager(
        memofu_style_seq_adv_train_ckpt, MEMOFU_STYLE_SEQ_ADV_TRAIN_CKPT_PATH, max_to_keep=ADV_TRAIN_EPOCHS)

    if memofu_style_seq_adv_train_ckpt_manager.latest_checkpoint:
        memofu_style_seq_adv_train_ckpt.restore(memofu_style_seq_adv_train_ckpt_manager.latest_checkpoint).expect_partial()
        print('Latest checkpoint restored from {}'.format(memofu_style_seq_adv_train_ckpt_manager.latest_checkpoint))

    # update the temperature
    adv_train_driver_memofu_style_seq.update_temp(ADV_TRAIN_EPOCHS - 1, STEPS_PER_EPOCH_TRAIN - 1)
    print('Temperature: {}'.format(adv_train_driver_memofu_style_seq.temp.numpy()))

    # Prepare for RSEs

    style_p_rav_range = [[style_p_rav[0].min(), style_p_rav[0].max()],
                         [style_p_rav[1].min(), style_p_rav[1].max()],
                         [style_p_rav[2].min(), style_p_rav[2].max()]]

    style_d_rav_range = [[style_d_rav[0].min(), style_d_rav[0].max()],
                         [style_d_rav[1].min(), style_d_rav[1].max()],
                         [style_d_rav[2].min(), style_d_rav[2].max()]]

    style_r_rav_range = [[style_r_rav[0].min(), style_r_rav[0].max()],
                         [style_r_rav[1].min(), style_r_rav[1].max()],
                         [style_r_rav[2].min(), style_r_rav[2].max()]]

    rse_values = np.linspace(0.2, 0.8, 4, endpoint=True)

    # pitch control experiment
    style_pitch_range_list = [map_parameter_to_value_range(rse_value, style_p_rav_range[0])
                              for rse_value in rse_values]
    style_pitch_average_list = [map_parameter_to_value_range(rse_value, style_p_rav_range[1])
                                for rse_value in rse_values]
    style_pitch_variance_list = [map_parameter_to_value_range(rse_value, style_p_rav_range[2])
                                 for rse_value in rse_values]

    style_pitch_range_ohe_list = [to_categorical_2d_style(p_rng, P_RNG_PATH)
                                  for p_rng in style_pitch_range_list]
    style_pitch_average_ohe_list = [to_categorical_2d_style(p_avg, P_AVG_PATH)
                                    for p_avg in style_pitch_average_list]
    style_pitch_variance_ohe_list = [to_categorical_2d_style(p_var, P_VAR_PATH)
                                     for p_var in style_pitch_variance_list]

    style_pitch_range_fixed = map_parameter_to_value_range(0.5, style_p_rav_range[0])
    style_pitch_range_fixed_ohe = to_categorical_2d_style(style_pitch_range_fixed, P_RNG_PATH)
    style_pitch_average_fixed = map_parameter_to_value_range(0.6, style_p_rav_range[1])
    style_pitch_average_fixed_ohe = to_categorical_2d_style(style_pitch_average_fixed, P_AVG_PATH)
    style_pitch_variance_fixed = map_parameter_to_value_range(0.2, style_p_rav_range[2])
    style_pitch_variance_fixed_ohe = to_categorical_2d_style(style_pitch_variance_fixed, P_VAR_PATH)

    ctrl_pitch_range_restuls = []
    for style_pitch_range_ohe in style_pitch_range_ohe_list:
        # test_pitch_range_ohe = np.repeat(style_pitch_range_ohe, TEST_LEN, axis=0)
        # test_p_rav_ctrl = test_p_rav.numpy()
        # test_p_rav_ctrl[:, :num_pitch_range_tokens] = test_pitch_range_ohe
        #
        # test_style = (tf.convert_to_tensor(test_p_rav_ctrl), test_d_rav, test_r_rav)

        test_p_rav_ctrl = np.concatenate([style_pitch_range_ohe, style_pitch_average_fixed_ohe, style_pitch_variance_fixed_ohe], axis=-1)
        test_p_rav_ctrl = np.repeat(test_p_rav_ctrl, TEST_LEN, axis=0)
        test_style = tf.convert_to_tensor(test_p_rav_ctrl), test_d_rav, test_r_rav

        # generated test song attr.
        test_g_out = adv_train_driver_memofu_style_seq.generate(x_test, test_style)

        # infer attributes of generated songs
        y_test_gen_attr = infer(test_g_out, LE_PATHS, is_tune=False)

        y_p, y_d, y_r = y_test_gen_attr

        p_rng, p_avg, p_var = calculate_style_features(y_p, title_str='Pitch', plot=0)

        ctrl_pitch_range_restuls.append(y_p.reshape(-1))

    #plt.boxplot(ctrl_pitch_range_restuls, showfliers=True)
    #plt.show()

    ctrl_pitch_average_restuls = []
    for style_pitch_average_ohe in style_pitch_average_ohe_list:
        # test_pitch_range_ohe = np.repeat(style_pitch_range_ohe, TEST_LEN, axis=0)
        # test_p_rav_ctrl = test_p_rav.numpy()
        # test_p_rav_ctrl[:, :num_pitch_range_tokens] = test_pitch_range_ohe
        #
        # test_style = (tf.convert_to_tensor(test_p_rav_ctrl), test_d_rav, test_r_rav)

        test_p_rav_ctrl = np.concatenate([style_pitch_range_fixed_ohe, style_pitch_average_ohe, style_pitch_variance_fixed_ohe], axis=-1)
        test_p_rav_ctrl = np.repeat(test_p_rav_ctrl, TEST_LEN, axis=0)
        test_style = tf.convert_to_tensor(test_p_rav_ctrl), test_d_rav, test_r_rav

        # generated test song attr.
        test_g_out = adv_train_driver_memofu_style_seq.generate(x_test, test_style)

        # infer attributes of generated songs
        y_test_gen_attr = infer(test_g_out, LE_PATHS, is_tune=False)

        y_p, y_d, y_r = y_test_gen_attr

        p_rng, p_avg, p_var = calculate_style_features(y_p, title_str='Pitch', plot=0)

        ctrl_pitch_average_restuls.append(y_p.reshape(-1))

    #plt.boxplot(ctrl_pitch_average_restuls, showfliers=True)
    #plt.show()

    ctrl_pitch_variance_restuls = []
    for style_pitch_variance_ohe in style_pitch_variance_ohe_list:
        # test_pitch_range_ohe = np.repeat(style_pitch_range_ohe, TEST_LEN, axis=0)
        # test_p_rav_ctrl = test_p_rav.numpy()
        # test_p_rav_ctrl[:, :num_pitch_range_tokens] = test_pitch_range_ohe
        #
        # test_style = (tf.convert_to_tensor(test_p_rav_ctrl), test_d_rav, test_r_rav)

        test_p_rav_ctrl = np.concatenate([style_pitch_range_fixed_ohe, style_pitch_average_fixed_ohe, style_pitch_variance_ohe], axis=-1)
        test_p_rav_ctrl = np.repeat(test_p_rav_ctrl, TEST_LEN, axis=0)
        test_style = tf.convert_to_tensor(test_p_rav_ctrl), test_d_rav, test_r_rav

        # generated test song attr.
        test_g_out = adv_train_driver_memofu_style_seq.generate(x_test, test_style)

        # infer attributes of generated songs
        y_test_gen_attr = infer(test_g_out, LE_PATHS, is_tune=False)

        y_p, y_d, y_r = y_test_gen_attr

        p_rng, p_avg, p_var = calculate_style_features(y_p, title_str='Pitch', plot=0)

        ctrl_pitch_variance_restuls.append(y_p.reshape(-1))

    #plt.boxplot(ctrl_pitch_variance_restuls, showfliers=True)
    #plt.show()

    # subplots
    plt.figure(figsize=(12, 4), dpi=300)
    ticks = [0.2, 0.4, 0.6, 0.8]

    plt.subplot(131)
    plt.boxplot(ctrl_pitch_range_restuls, showfliers=True)
    plt.ylabel('Pitch Distribution')
    plt.xlabel('Pitch Range RSE')
    plt.xticks(range(1, len(ticks) + 1), ticks)

    plt.subplot(132)
    plt.boxplot(ctrl_pitch_average_restuls, showfliers=True)
    plt.ylabel('Pitch Distribution')
    plt.xlabel('Pitch Average RSE')
    plt.xticks(range(1, len(ticks) + 1), ticks)

    plt.subplot(133)
    plt.boxplot(ctrl_pitch_variance_restuls, showfliers=True)
    plt.ylabel('Pitch Distribution')
    plt.xlabel('Pitch Variance RSE')
    plt.xticks(range(1, len(ticks) + 1), ticks)

    plt.tight_layout()
    plt.show()
    # plt.savefig('boxcompare.png')

    # duration control experiment
    style_duration_range_list = [map_parameter_to_value_range(rse_value, style_d_rav_range[0])
                                 for rse_value in rse_values]
    style_duration_average_list = [map_parameter_to_value_range(rse_value, style_d_rav_range[1])
                                   for rse_value in rse_values]
    style_duration_variance_list = [map_parameter_to_value_range(rse_value, style_d_rav_range[2])
                                    for rse_value in rse_values]

    style_duration_range_ohe_list = [to_categorical_2d_style(d_rng, D_RNG_PATH)
                                     for d_rng in style_duration_range_list]
    style_duration_average_ohe_list = [to_categorical_2d_style(d_avg, D_AVG_PATH)
                                       for d_avg in style_duration_average_list]
    style_duration_variance_ohe_list = [to_categorical_2d_style(d_var, D_VAR_PATH)
                                        for d_var in style_duration_variance_list]

    style_duration_range_fixed = map_parameter_to_value_range(0.5, style_d_rav_range[0])
    style_duration_range_fixed_ohe = to_categorical_2d_style(style_duration_range_fixed, D_RNG_PATH)
    style_duration_average_fixed = map_parameter_to_value_range(0.2, style_d_rav_range[1])
    style_duration_average_fixed_ohe = to_categorical_2d_style(style_duration_average_fixed, D_AVG_PATH)
    style_duration_variance_fixed = map_parameter_to_value_range(0.0, style_d_rav_range[2])
    style_duration_variance_fixed_ohe = to_categorical_2d_style(style_duration_variance_fixed, D_VAR_PATH)

    ctrl_duration_range_restuls = []
    for style_duration_range_ohe in style_duration_range_ohe_list:

        test_d_rav_ctrl = np.concatenate([style_duration_range_ohe, style_duration_average_fixed_ohe, style_duration_variance_fixed_ohe], axis=-1)
        test_d_rav_ctrl = np.repeat(test_d_rav_ctrl, TEST_LEN, axis=0)
        test_style = test_p_rav, tf.convert_to_tensor(test_d_rav_ctrl), test_r_rav

        # generated test song attr.
        test_g_out = adv_train_driver_memofu_style_seq.generate(x_test, test_style)

        # infer attributes of generated songs
        y_test_gen_attr = infer(test_g_out, LE_PATHS, is_tune=False)

        y_p, y_d, y_r = y_test_gen_attr

        p_rng, p_avg, p_var = calculate_style_features(y_p, title_str='Pitch', plot=0)

        ctrl_duration_range_restuls.append(y_d.reshape(-1))

    #plt.boxplot(ctrl_duration_range_restuls, showfliers=True)
    #plt.show()

    ctrl_duration_average_restuls = []
    for style_duration_average_ohe in style_duration_average_ohe_list:

        test_d_rav_ctrl = np.concatenate([style_duration_range_fixed_ohe, style_duration_average_ohe, style_duration_variance_fixed_ohe], axis=-1)
        test_d_rav_ctrl = np.repeat(test_d_rav_ctrl, TEST_LEN, axis=0)
        test_style = test_p_rav, tf.convert_to_tensor(test_d_rav_ctrl), test_r_rav

        # generated test song attr.
        test_g_out = adv_train_driver_memofu_style_seq.generate(x_test, test_style)

        # infer attributes of generated songs
        y_test_gen_attr = infer(test_g_out, LE_PATHS, is_tune=False)

        y_p, y_d, y_r = y_test_gen_attr

        p_rng, p_avg, p_var = calculate_style_features(y_d, title_str='Pitch', plot=0)

        ctrl_duration_average_restuls.append(y_d.reshape(-1))

    #plt.boxplot(ctrl_duration_average_restuls, showfliers=True)
    #plt.show()

    ctrl_duration_variance_restuls = []
    for style_duration_variance_ohe in style_duration_variance_ohe_list:

        test_d_rav_ctrl = np.concatenate([style_duration_range_fixed_ohe, style_duration_average_fixed_ohe, style_duration_variance_ohe], axis=-1)
        test_d_rav_ctrl = np.repeat(test_d_rav_ctrl, TEST_LEN, axis=0)
        test_style = test_p_rav, tf.convert_to_tensor(test_d_rav_ctrl), test_r_rav

        # generated test song attr.
        test_g_out = adv_train_driver_memofu_style_seq.generate(x_test, test_style)

        # infer attributes of generated songs
        y_test_gen_attr = infer(test_g_out, LE_PATHS, is_tune=False)

        y_p, y_d, y_r = y_test_gen_attr

        p_rng, p_avg, p_var = calculate_style_features(y_p, title_str='Pitch', plot=0)

        ctrl_duration_variance_restuls.append(y_d.reshape(-1))

    #plt.boxplot(ctrl_duration_variance_restuls, showfliers=True)
    #plt.show()

    # subplots
    plt.figure(figsize=(12, 4), dpi=300)
    ticks = [0.2, 0.4, 0.6, 0.8]

    plt.subplot(131)
    plt.boxplot(ctrl_duration_range_restuls, showfliers=True)
    plt.ylabel('Duration Distribution')
    plt.xlabel('Duration Range RSE')
    plt.xticks(range(1, len(ticks) + 1), ticks)

    plt.subplot(132)
    plt.boxplot(ctrl_duration_average_restuls, showfliers=True)
    plt.ylabel('Duration Distribution')
    plt.xlabel('Duration Average RSE')
    plt.xticks(range(1, len(ticks) + 1), ticks)

    plt.subplot(133)
    plt.boxplot(ctrl_duration_variance_restuls, showfliers=True)
    plt.ylabel('Duration Distribution')
    plt.xlabel('Duration Variance RSE')
    plt.xticks(range(1, len(ticks) + 1), ticks)

    plt.tight_layout()
    plt.show()
    # plt.savefig('boxcompare.png')

    # pitch control experiment
    style_rest_range_list = [map_parameter_to_value_range(rse_value, style_r_rav_range[0])
                             for rse_value in rse_values]
    style_rest_average_list = [map_parameter_to_value_range(rse_value, style_r_rav_range[1])
                               for rse_value in rse_values]
    style_rest_variance_list = [map_parameter_to_value_range(rse_value, style_r_rav_range[2])
                                for rse_value in rse_values]

    style_rest_range_ohe_list = [to_categorical_2d_style(r_rng, R_RNG_PATH)
                                 for r_rng in style_rest_range_list]
    style_rest_average_ohe_list = [to_categorical_2d_style(r_avg, R_AVG_PATH)
                                   for r_avg in style_rest_average_list]
    style_rest_variance_ohe_list = [to_categorical_2d_style(r_var, R_VAR_PATH)
                                    for r_var in style_rest_variance_list]

    style_rest_range_fixed = map_parameter_to_value_range(0.3, style_r_rav_range[0])
    style_rest_range_fixed_ohe = to_categorical_2d_style(style_rest_range_fixed, R_RNG_PATH)
    style_rest_average_fixed = map_parameter_to_value_range(0.3, style_r_rav_range[1])
    style_rest_average_fixed_ohe = to_categorical_2d_style(style_rest_average_fixed, R_AVG_PATH)
    style_rest_variance_fixed = map_parameter_to_value_range(0.5, style_r_rav_range[2])
    style_rest_variance_fixed_ohe = to_categorical_2d_style(style_rest_variance_fixed, R_VAR_PATH)

    ctrl_rest_range_restuls = []
    for style_rest_range_ohe in style_rest_range_ohe_list:

        test_r_rav_ctrl = np.concatenate([style_rest_range_ohe, style_rest_average_fixed_ohe, style_rest_variance_fixed_ohe], axis=-1)
        test_r_rav_ctrl = np.repeat(test_r_rav_ctrl, TEST_LEN, axis=0)
        test_style = test_p_rav, test_d_rav, tf.convert_to_tensor(test_r_rav_ctrl)

        # generated test song attr.
        test_g_out = adv_train_driver_memofu_style_seq.generate(x_test, test_style)

        # infer attributes of generated songs
        y_test_gen_attr = infer(test_g_out, LE_PATHS, is_tune=False)

        y_p, y_d, y_r = y_test_gen_attr

        p_rng, p_avg, p_var = calculate_style_features(y_p, title_str='Pitch', plot=0)

        ctrl_rest_range_restuls.append(y_r.reshape(-1))

    #plt.boxplot(ctrl_rest_range_restuls, showfliers=True)
    #plt.show()

    ctrl_rest_average_restuls = []
    for style_rest_average_ohe in style_rest_average_ohe_list:

        test_r_rav_ctrl = np.concatenate([style_rest_range_fixed_ohe, style_rest_average_ohe, style_rest_variance_fixed_ohe], axis=-1)
        test_r_rav_ctrl = np.repeat(test_r_rav_ctrl, TEST_LEN, axis=0)
        test_style = test_p_rav, test_d_rav, tf.convert_to_tensor(test_r_rav_ctrl)

        # generated test song attr.
        test_g_out = adv_train_driver_memofu_style_seq.generate(x_test, test_style)

        # infer attributes of generated songs
        y_test_gen_attr = infer(test_g_out, LE_PATHS, is_tune=False)

        y_p, y_d, y_r = y_test_gen_attr

        p_rng, p_avg, p_var = calculate_style_features(y_d, title_str='Pitch', plot=0)

        ctrl_rest_average_restuls.append(y_r.reshape(-1))

    #plt.boxplot(ctrl_rest_average_restuls, showfliers=True)
    #plt.show()

    ctrl_rest_variance_restuls = []
    for style_rest_variance_ohe in style_rest_variance_ohe_list:

        test_r_rav_ctrl = np.concatenate([style_rest_range_fixed_ohe, style_rest_average_fixed_ohe, style_rest_variance_ohe], axis=-1)
        test_r_rav_ctrl = np.repeat(test_r_rav_ctrl, TEST_LEN, axis=0)
        test_style = test_p_rav, test_d_rav, tf.convert_to_tensor(test_r_rav_ctrl)

        # generated test song attr.
        test_g_out = adv_train_driver_memofu_style_seq.generate(x_test, test_style)

        # infer attributes of generated songs
        y_test_gen_attr = infer(test_g_out, LE_PATHS, is_tune=False)

        y_p, y_d, y_r = y_test_gen_attr

        p_rng, p_avg, p_var = calculate_style_features(y_p, title_str='Pitch', plot=0)

        ctrl_rest_variance_restuls.append(y_r.reshape(-1))

    #plt.boxplot(ctrl_rest_variance_restuls, showfliers=True)
    #plt.show()

    # subplots
    plt.figure(figsize=(12, 4), dpi=300)
    ticks = [0.2, 0.4, 0.6, 0.8]

    plt.subplot(131)
    plt.boxplot(ctrl_rest_range_restuls, showfliers=True)
    plt.ylabel('Rest Distribution')
    plt.xlabel('Rest Range RSE')
    plt.xticks(range(1, len(ticks) + 1), ticks)

    plt.subplot(132)
    plt.boxplot(ctrl_rest_average_restuls, showfliers=True)
    plt.ylabel('Rest Distribution')
    plt.xlabel('Rest Average RSE')
    plt.xticks(range(1, len(ticks) + 1), ticks)

    plt.subplot(133)
    plt.boxplot(ctrl_rest_variance_restuls, showfliers=True)
    plt.ylabel('Rest Distribution')
    plt.xlabel('Rest Variance RSE')
    plt.xticks(range(1, len(ticks) + 1), ticks)

    plt.tight_layout()
    plt.show()
    # plt.savefig('boxcompare.png')

if __name__ == '__main__':

    settings = {'settings_file': 'settings'}
    settings = load_settings_from_file(settings)

    print("Settings: \n")
    for (k, v) in settings.items():
        print(v, '\t', k)

    locals().update(settings)

    print("================================================================ \n ")

    main()

