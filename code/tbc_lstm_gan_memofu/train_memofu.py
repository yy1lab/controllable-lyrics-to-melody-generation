"""
@Project: tbc_lstm_gan_style
@File: train_memofu.py
@Author: Zhe Zhang
@Email: zhe@nii.ac.jp
@Date: 2022/09/19
"""

import pandas as pd

from generator_memofu import *
from discriminator import *
from drivers_memofu import *
from utils import *
from utils_new import *

import time
import os

tf.config.run_functions_eagerly(False)


def main():
    """# Data"""

    print("Data: \n")

    # create label encoders for each song attribute

    data = np.load(FULL_DATA_PATH)

    print("Shape of full data: ", data.shape)

    data_filtered = filter_wrong_data(data, reasonable_p_rng, reasonable_p_avg, reasonable_p_var,
                                      reasonable_d_rng, reasonable_d_avg, reasonable_r_rng)

    # y = data[:, :SONG_LENGTH * NUM_SONG_FEATURES]              # 60 cols
    y = data_filtered[:, :SONG_LENGTH * NUM_SONG_FEATURES]
    y = np.reshape(y, (-1, SONG_LENGTH, NUM_SONG_FEATURES))  # 60 = 20 x 3: 1 pitch, 2 duration, 3 rest

    print("Shape of melody data:", y.shape)

    y_p = y[:, :, 0]
    y_d = y[:, :, 1]
    y_r = y[:, :, 2]

    print("Shape of pitch data:", y_p.shape)
    print("Shape of duration data:", y_d.shape)
    print("Shape of rest data:", y_d.shape)

    p_rng, p_avg, p_var = calculate_style_features(y_p, title_str='Pitch', plot=0)
    d_rng, d_avg, d_var = calculate_style_features(y_d, title_str='Duration', plot=0)
    r_rng, r_avg, r_var = calculate_style_features(y_r, title_str='Rest', plot=0)

    # plt.figure()
    # p, num_p = np.unique(y_p, return_counts=True)
    # plt.bar(p, num_p / sum(num_p))
    # plt.xlabel('MIDI Number')
    # plt.ylabel('Occurence Probability')
    # plt.show()

    print("Number of unique pitches, durations and rests present in data:")
    NUM_P_TOKENS = create_categorical_2d_encoder(y_p, P_LE_PATH)
    NUM_D_TOKENS = create_categorical_2d_encoder(y_d, D_LE_PATH)
    NUM_R_TOKENS = create_categorical_2d_encoder(y_r, R_LE_PATH)

    num_pitch_range_tokens = create_k_bins_encoder(p_rng, P_RNG_PATH, k=int(p_rng.max() - p_rng.min()))
    num_duration_range_tokens = create_k_bins_encoder(d_rng, D_RNG_PATH, k=int(d_rng.max() - d_rng.min()))
    num_rest_range_tokens = create_k_bins_encoder(r_rng, R_RNG_PATH, k=int(r_rng.max() - r_rng.min()))

    num_pitch_average_tokens = create_k_bins_encoder(p_avg, P_AVG_PATH, k=int(np.ceil(p_avg.max() - p_avg.min())))
    num_duration_average_tokens = create_k_bins_encoder(d_avg, D_AVG_PATH, k=10)
    num_rest_average_tokens = create_k_bins_encoder(r_avg, R_AVG_PATH, k=10)

    num_pitch_variance_tokens = create_k_bins_encoder(p_var, P_VAR_PATH, k=20)
    num_duration_variance_tokens = create_k_bins_encoder(d_var, D_VAR_PATH, k=20)
    num_rest_variance_tokens = create_k_bins_encoder(r_var, R_VAR_PATH, k=20)

    NUM_TOKENS = [NUM_P_TOKENS, NUM_D_TOKENS, NUM_R_TOKENS]
    LE_PATHS = [P_LE_PATH, D_LE_PATH, R_LE_PATH]

    # num_range_tokens = [num_pitch_range_tokens, num_duration_range_tokens, num_rest_range_tokens]
    # num_average_tokens = [num_pitch_average_tokens, num_duration_average_tokens, num_rest_average_tokens]
    # num_variance_tokens = [num_pitch_variance_tokens, num_duration_variance_tokens, num_rest_variance_tokens]

    num_pitch_style_tokens = num_pitch_range_tokens + num_pitch_average_tokens + num_pitch_variance_tokens
    num_duration_style_tokens = num_duration_range_tokens + num_duration_average_tokens + num_duration_variance_tokens
    num_rest_style_tokens = num_rest_range_tokens + num_rest_average_tokens + num_rest_variance_tokens
    num_style_tokens = [int(num_pitch_style_tokens), int(num_duration_style_tokens), int(num_rest_style_tokens)]

    range_paths = [P_RNG_PATH, D_RNG_PATH, R_RNG_PATH]
    average_paths = [P_AVG_PATH, D_AVG_PATH, R_AVG_PATH]
    variance_paths = [P_VAR_PATH, D_VAR_PATH, R_VAR_PATH]

    # load train, validation and test data

    # lyrics, pdr(value), pdr(one-hot)
    x_train, y_train_dat_attr, y_train, train_p_rav, train_d_rav, train_r_rav = load_data_clean(TRAIN_DATA_PATH,
                                                                                                LE_PATHS,
                                                                                                SONG_LENGTH,
                                                                                                NUM_SONG_FEATURES,
                                                                                                NUM_META_FEATURES,
                                                                                                range_paths,
                                                                                                average_paths,
                                                                                                variance_paths)

    x_valid, y_valid_dat_attr, y_valid, valid_p_rav, valid_d_rav, valid_r_rav = load_data_clean(VALID_DATA_PATH,
                                                                                                LE_PATHS,
                                                                                                SONG_LENGTH,
                                                                                                NUM_SONG_FEATURES,
                                                                                                NUM_META_FEATURES,
                                                                                                range_paths,
                                                                                                average_paths,
                                                                                                variance_paths,
                                                                                                convert_to_tensor=True)

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
    VALID_LEN = len(x_valid)
    TEST_LEN = len(x_test)

    STEPS_PER_EPOCH_TRAIN = np.ceil(TRAIN_LEN / BATCH_SIZE)
    print('Steps per epoch train: ', STEPS_PER_EPOCH_TRAIN)

    # create train dataset object

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(TRAIN_LEN)
    train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False)

    print("================================================================ \n ")

    """# Training"""

    # Set seed for reproducibility

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    ## Initialise Model

    # Initialise generator model
    g_model = GeneratorWithMemoryFusion(
        G_EMB_UNITS, G_PROJ_UNITS, G_EMB_DROPOUT_RATE, G_PROJ_DROPOUT_RATE,
        G_RNN_DEPTH, G_RNN_UNITS, NUM_TOKENS, NUM_META_FEATURES, fusion_layer=FUSION_LAYER)

    # Initialise discriminator
    d_model = Discriminator(
        D_EMB_UNITS, D_PROJ_UNITS, D_EMB_DROPOUT_RATE, D_PROJ_DROPOUT_RATE,
        D_RNN_DEPTH, D_RNN_UNITS, NUM_TOKENS, NUM_META_FEATURES)

    ## Initialise Optimizer

    # Initialise optimizer for pretraining
    pre_train_g_opt = tf.keras.optimizers.Adam(
        PRE_TRAIN_LR, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # Initialise optimizer for adversarial training
    adv_train_g_opt = tf.keras.optimizers.Adam(ADV_TRAIN_G_LR, beta_1=0.9, beta_2=0.999)
    adv_train_d_opt = tf.keras.optimizers.Adam(ADV_TRAIN_D_LR, beta_1=0.9, beta_2=0.999)

    ## Initialise Driver

    # Initialise pre-train driver
    pre_train_driver = PreTrainDriver(g_model,
                                      pre_train_g_opt,
                                      NUM_TOKENS,
                                      MAX_GRAD_NORM)

    # Initialise adversarial driver
    adv_train_driver = AdversarialDriver(g_model,
                                         d_model,
                                         adv_train_g_opt,
                                         adv_train_d_opt,
                                         TEMP_MAX,
                                         STEPS_PER_EPOCH_TRAIN,
                                         ADV_TRAIN_EPOCHS,
                                         NUM_TOKENS,
                                         MAX_GRAD_NORM)

    ## Setup Checkpoint

    # Setup checkpoint for pretraining
    pre_train_ckpt = tf.train.Checkpoint(g_model=g_model,
                                         pre_train_g_opt=pre_train_g_opt)

    pre_train_ckpt_manager = tf.train.CheckpointManager(
        pre_train_ckpt, MEMOFU_PRE_TRAIN_CKPT_PATH, max_to_keep=PRE_TRAIN_EPOCHS)

    # Setup checkpoint for adversarial training
    adv_train_ckpt = tf.train.Checkpoint(g_model=g_model,
                                         d_model=d_model,
                                         adv_train_g_opt=adv_train_g_opt,
                                         adv_train_d_opt=adv_train_d_opt)

    adv_train_ckpt_manager = tf.train.CheckpointManager(
        adv_train_ckpt, MEMOFU_ADV_TRAIN_CKPT_PATH, max_to_keep=ADV_TRAIN_EPOCHS)

    ## Setup Logging

    os.makedirs(MEMOFU_LOG_DIR, exist_ok=True)
    previous_runs = os.listdir(MEMOFU_LOG_DIR)
    if len(previous_runs) == 0:
        run_number = 1
    else:
        run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1
    logdir = 'run_%02d' % run_number
    # summary_writer = tf.train.SummaryWriter(os.path.join('experiments', logdir), sess.graph)
    memofu_summary_writer = tf.summary.create_file_writer(os.path.join(MEMOFU_LOG_DIR, logdir))

    if REUSE_TEST_LOGS:
        # load test logs from previous run
        try:
            test_logs = pd.read_csv(TEST_LOGS_FILENAME).to_dict('records')
        except FileNotFoundError:
            test_logs = []
    else:
        test_logs = []

    # """## Tensorboard setup"""

    # ## Setup Tensorboard

    # %tensorboard --logdir LOG_DIR

    # # Share tensorboard for remote vizualizaton

    # !tensorboard dev upload --logdir LOG_DIR

    """## Training Loop"""

    # Training

    ## PreTraining

    print('\n ***** Starting PreTraining ***** \n')

    for epoch in range(PRE_TRAIN_EPOCHS):
        start = time.time()

        # log test metrics for current epoch
        logs = {'seed': SEED,
                'epoch': epoch,
                'method': 'TBC-LSTM-GAN-MEMOFU'}

        total_train_p_loss = 0
        total_train_d_loss = 0
        total_train_r_loss = 0

        for batch, inp in enumerate(train_dataset.take(STEPS_PER_EPOCH_TRAIN)):
            batch_p_loss, batch_d_loss, batch_r_loss = pre_train_driver.train_step(inp)
            total_train_p_loss += batch_p_loss
            total_train_d_loss += batch_d_loss
            total_train_r_loss += batch_r_loss

        train_p_loss = total_train_p_loss / STEPS_PER_EPOCH_TRAIN
        train_d_loss = total_train_d_loss / STEPS_PER_EPOCH_TRAIN
        train_r_loss = total_train_r_loss / STEPS_PER_EPOCH_TRAIN

        if epoch % EVAL_INTERVAL == 0:

            # log train summary
            with memofu_summary_writer.as_default():
                tf.summary.scalar('pre_train/p_loss', train_p_loss, step=epoch)
                tf.summary.scalar('pre_train/d_loss', train_d_loss, step=epoch)
                tf.summary.scalar('pre_train/r_loss', train_r_loss, step=epoch)

            # perform validation
            valid_p_loss, valid_d_loss, valid_r_loss = pre_train_driver.test_step((x_valid, y_valid))

            # generated val. song attr.
            valid_g_out = adv_train_driver.generate(x_valid)

            # infer val. song attr.
            y_valid_gen_attr = infer(valid_g_out, LE_PATHS, is_tune=True)

            # compute pitch, duration, rest & overall mmd score
            valid_p_mmd, valid_d_mmd, valid_r_mmd = compute_mmd_score(y_valid_dat_attr, y_valid_gen_attr)
            valid_o_mmd = valid_p_mmd + valid_d_mmd + valid_r_mmd

            # log validation summary
            with memofu_summary_writer.as_default():
                tf.summary.scalar('pre_train_valid/p_loss', valid_p_loss, step=epoch)
                tf.summary.scalar('pre_train_valid/d_loss', valid_d_loss, step=epoch)
                tf.summary.scalar('pre_train_valid/r_loss', valid_r_loss, step=epoch)

                tf.summary.scalar('MMD_valid/pMMD', valid_p_mmd, step=epoch)
                tf.summary.scalar('MMD_valid/dMMD', valid_d_mmd, step=epoch)
                tf.summary.scalar('MMD_valid/rMMD', valid_r_mmd, step=epoch)
                tf.summary.scalar('MMD_valid/oMMD', valid_o_mmd, step=epoch)

            # testing

            # generated test song attr.
            test_g_out = adv_train_driver.generate(x_test)

            # infer test song attr.
            y_test_gen_attr = infer(test_g_out, LE_PATHS, is_tune=True)

            # compute self-bleu score
            test_self_bleu = compute_self_bleu_score(y_test_gen_attr, N_GRAMS)

            # compute pitch, duration, rest & overall mmd score
            test_p_mmd, test_d_mmd, test_r_mmd = compute_mmd_score(y_test_dat_attr, y_test_gen_attr)
            test_o_mmd = test_p_mmd + test_d_mmd + test_r_mmd

            # log test summary
            with memofu_summary_writer.as_default():
                for n_gram in N_GRAMS:
                    tf.summary.scalar(f'SB_test/selfBLEU_{n_gram}', test_self_bleu[n_gram], step=epoch)

                tf.summary.scalar('MMD_test/pMMD', test_p_mmd, step=epoch)
                tf.summary.scalar('MMD_test/dMMD', test_d_mmd, step=epoch)
                tf.summary.scalar('MMD_test/rMMD', test_r_mmd, step=epoch)
                tf.summary.scalar('MMD_test/oMMD', test_o_mmd, step=epoch)

                # save test logs for current epoch

                for n_gram in N_GRAMS:
                    logs[f'selfBLEU_{n_gram}'] = test_self_bleu[n_gram]

                logs['pMMD'] = test_p_mmd
                logs['dMMD'] = test_d_mmd
                logs['rMMD'] = test_r_mmd
                logs['oMMD'] = test_o_mmd

            test_logs.append(logs)

            print('Epoch {} Pitch Loss: Train: {:.4f} Validation: {:.4f}'.format(
                epoch + 1, train_p_loss, valid_p_loss))

            print('Epoch {} Duration Loss: Train: {:.4f} Validation: {:.4f}'.format(
                epoch + 1, train_d_loss, valid_d_loss))

            print('Epoch {} Rest Loss: Train: {:.4f} Validation: {:.4f}'.format(
                epoch + 1, train_r_loss, valid_r_loss))

        # create a checkpoint
        pre_train_ckpt_save_path = pre_train_ckpt_manager.save()
        print('Saving pretrain checkpoint for epoch {} at {}'.format(
            epoch + 1, pre_train_ckpt_save_path))

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    ## Adversarial Training

    print('\n ***** Starting Adversarial Training ***** \n')

    # Temperature updates are done at step-level

    for epoch in range(ADV_TRAIN_EPOCHS):
        start = time.time()

        logs = {'seed': SEED,
                'epoch': epoch + PRE_TRAIN_EPOCHS,
                'method': 'TBC-LSTM-GAN-MEMOFU'}

        total_train_g_loss = 0
        total_train_d_loss = 0

        for step, inp in enumerate(train_dataset.take(STEPS_PER_EPOCH_TRAIN)):
            # update temperature
            adv_train_driver.update_temp(epoch, step)

            batch_g_loss, batch_d_loss, batch_g_out = adv_train_driver.train_step(inp)
            total_train_g_loss += batch_g_loss
            total_train_d_loss += batch_d_loss

            with memofu_summary_writer.as_default():
                tf.summary.scalar('temperature', tf.keras.backend.get_value(adv_train_driver.temp),
                                  step=epoch * STEPS_PER_EPOCH_TRAIN + step)

        train_g_loss = total_train_g_loss / STEPS_PER_EPOCH_TRAIN
        train_d_loss = total_train_d_loss / STEPS_PER_EPOCH_TRAIN

        if epoch % EVAL_INTERVAL == 0:

            # log train summary
            with memofu_summary_writer.as_default():
                tf.summary.scalar('train/g_loss', train_g_loss, step=epoch)
                tf.summary.scalar('train/d_loss', train_d_loss, step=epoch)

            # generated val. song attr.
            valid_g_loss, valid_d_loss, valid_g_out = adv_train_driver.test_step((x_valid, y_valid))

            # infer val. song attr.
            y_valid_gen_attr = infer(valid_g_out, LE_PATHS, is_tune=True)

            # compute pitch, duration, rest & overall mmd score
            valid_p_mmd, valid_d_mmd, valid_r_mmd = compute_mmd_score(y_valid_dat_attr, y_valid_gen_attr)
            valid_o_mmd = valid_p_mmd + valid_d_mmd + valid_r_mmd

            # log validation summary
            with memofu_summary_writer.as_default():
                tf.summary.scalar('valid/g_loss', valid_g_loss, step=epoch)
                tf.summary.scalar('valid/d_loss', valid_d_loss, step=epoch)

                tf.summary.scalar('MMD_valid/pMMD', valid_p_mmd, step=epoch+PRE_TRAIN_EPOCHS)
                tf.summary.scalar('MMD_valid/dMMD', valid_d_mmd, step=epoch+PRE_TRAIN_EPOCHS)
                tf.summary.scalar('MMD_valid/rMMD', valid_r_mmd, step=epoch+PRE_TRAIN_EPOCHS)
                tf.summary.scalar('MMD_valid/oMMD', valid_o_mmd, step=epoch+PRE_TRAIN_EPOCHS)

            # testing

            # generated test song attr.
            test_g_out = adv_train_driver.generate(x_test)

            # infer test song attr.
            y_test_gen_attr = infer(test_g_out, LE_PATHS, is_tune=True)

            # compute self-bleu score
            test_self_bleu = compute_self_bleu_score(y_test_gen_attr, N_GRAMS)

            # compute pitch, duration, rest & overall mmd score
            test_p_mmd, test_d_mmd, test_r_mmd = compute_mmd_score(y_test_dat_attr, y_test_gen_attr)
            test_o_mmd = test_p_mmd + test_d_mmd + test_r_mmd

            # log test summary
            with memofu_summary_writer.as_default():
                for n_gram in N_GRAMS:
                    tf.summary.scalar(f'selfBLEU_{n_gram}', test_self_bleu[n_gram], step=epoch + PRE_TRAIN_EPOCHS)

                tf.summary.scalar('MMD_test/pMMD', test_p_mmd, step=epoch+PRE_TRAIN_EPOCHS)
                tf.summary.scalar('MMD_test/dMMD', test_d_mmd, step=epoch+PRE_TRAIN_EPOCHS)
                tf.summary.scalar('MMD_test/rMMD', test_r_mmd, step=epoch+PRE_TRAIN_EPOCHS)
                tf.summary.scalar('MMD_test/oMMD', test_o_mmd, step=epoch+PRE_TRAIN_EPOCHS)

                # save test logs for current epoch

                for n_gram in N_GRAMS:
                    logs[f'selfBLEU_{n_gram}'] = test_self_bleu[n_gram]

                logs['pMMD'] = test_p_mmd
                logs['dMMD'] = test_d_mmd
                logs['rMMD'] = test_r_mmd
                logs['oMMD'] = test_o_mmd

            test_logs.append(logs)

            print('Epoch {} Train loss: G:{:.4f}, D:{:.4f}, Valid loss: G:{:.4f}, D:{:.4f}'.format(
                epoch + 1, train_g_loss, train_d_loss, valid_g_loss, valid_d_loss))

        # create a checkpoint
        adv_train_ckpt_save_path = adv_train_ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(
            epoch + 1, adv_train_ckpt_save_path))

        print('Temperature used for epoch {} : {}'.format(
            epoch + 1, tf.keras.backend.get_value(adv_train_driver.temp)))

        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    # save test logs as csv file

    pd.DataFrame.from_records(test_logs).to_csv(TEST_LOGS_FILENAME, index=False)


if __name__ == '__main__':

    settings = {'settings_file': 'settings'}
    settings = load_settings_from_file(settings)

    print("Settings: \n")
    for (k, v) in settings.items():
        print(v, '\t', k)

    locals().update(settings)

    print("================================================================ \n ")

    main()

    print("Training is complete.")

