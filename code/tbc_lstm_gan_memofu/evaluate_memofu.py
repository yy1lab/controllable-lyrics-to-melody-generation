"""
Script to evaluate the TBC-LSTM-MLE & TBC-LSTM-GAN model on test data.
TBC-LSTM-MLE is the model obtained at the end of pre-training."""
import os

import pandas as pd

from generator_memofu import *
from discriminator import *
from drivers_memofu import *
from utils import *

from utils_new import *


def main():
    
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
    
    NUM_TOKENS= [NUM_P_TOKENS, NUM_D_TOKENS, NUM_R_TOKENS]
    LE_PATHS  = [P_LE_PATH, D_LE_PATH, R_LE_PATH]

    num_pitch_style_tokens = num_pitch_range_tokens + num_pitch_average_tokens + num_pitch_variance_tokens
    num_duration_style_tokens = num_duration_range_tokens + num_duration_average_tokens + num_duration_variance_tokens
    num_rest_style_tokens = num_rest_range_tokens + num_rest_average_tokens + num_rest_variance_tokens
    num_style_tokens = [int(num_pitch_style_tokens), int(num_duration_style_tokens), int(num_rest_style_tokens)]

    range_paths = [P_RNG_PATH, D_RNG_PATH, R_RNG_PATH]
    average_paths = [P_AVG_PATH, D_AVG_PATH, R_AVG_PATH]
    variance_paths = [P_VAR_PATH, D_VAR_PATH, R_VAR_PATH]

    # load train (to compute STEPS_PER_EPOCH_TRAIN) and test data

    x_train, y_train_dat_attr, y_train, train_p_rav, train_d_rav, train_r_rav = load_data_clean(TRAIN_DATA_PATH,
                                                                                                LE_PATHS,
                                                                                                SONG_LENGTH,
                                                                                                NUM_SONG_FEATURES,
                                                                                                NUM_META_FEATURES,
                                                                                                range_paths,
                                                                                                average_paths,
                                                                                                variance_paths)

    x_test, y_test_dat_attr, y_test, test_p_rav, test_d_rav, test_r_rav = load_data_clean(TEST_DATA_PATH,
                                                                                          LE_PATHS,
                                                                                          SONG_LENGTH,
                                                                                          NUM_SONG_FEATURES,
                                                                                          NUM_META_FEATURES,
                                                                                          range_paths,
                                                                                          average_paths,
                                                                                          variance_paths,
                                                                                          convert_to_tensor=True)

    train_style = (train_p_rav, train_d_rav, train_r_rav)
    test_style = (test_p_rav, test_d_rav, test_r_rav)

    x_train_and_style = (x_train, train_style)
    x_test_and_style = (x_test, test_style)

    TRAIN_LEN = len(x_train)
    TEST_LEN  = len(x_test)

    STEPS_PER_EPOCH_TRAIN = np.ceil(TRAIN_LEN/BATCH_SIZE)
    print('Steps per epoch train: ', STEPS_PER_EPOCH_TRAIN)

    print("================================================================ \n " )

    # Set seed for reproducibility

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    ## Initialise Model

    # Initialise generator model
    g_model_memofu = GeneratorWithMemoryFusion(
        G_EMB_UNITS, G_PROJ_UNITS, G_EMB_DROPOUT_RATE, G_PROJ_DROPOUT_RATE,
        G_RNN_DEPTH, G_RNN_UNITS, NUM_TOKENS, NUM_META_FEATURES, fusion_layer=FUSION_LAYER)

    # Initialise discriminator
    d_model = Discriminator(
        D_EMB_UNITS, D_PROJ_UNITS, D_EMB_DROPOUT_RATE, D_PROJ_DROPOUT_RATE,
        D_RNN_DEPTH, D_RNN_UNITS, NUM_TOKENS, NUM_META_FEATURES)

    g_model_memofu_style = GeneratorWithMemoryFusionAndStyle(
        G_EMB_UNITS, G_PROJ_UNITS, G_EMB_DROPOUT_RATE, G_PROJ_DROPOUT_RATE,
        G_RNN_DEPTH, G_RNN_UNITS, NUM_TOKENS, NUM_META_FEATURES, num_style_tokens, fusion_layer=FUSION_LAYER)

    # Initialise discriminator
    d_model_style = DiscriminatorWithStyle(
        D_EMB_UNITS, D_PROJ_UNITS, D_EMB_DROPOUT_RATE, D_PROJ_DROPOUT_RATE,
        D_RNN_DEPTH, D_RNN_UNITS, NUM_TOKENS, NUM_META_FEATURES, num_style_tokens)

    # g_model_baseline = Generator(
    #     G_EMB_UNITS, G_PROJ_UNITS, G_EMB_DROPOUT_RATE, G_PROJ_DROPOUT_RATE,
    #     G_RNN_DEPTH, G_RNN_UNITS, NUM_TOKENS, NUM_META_FEATURES)
    #
    # d_model_baseline = Discriminator(
    #     D_EMB_UNITS, D_PROJ_UNITS, D_EMB_DROPOUT_RATE, D_PROJ_DROPOUT_RATE,
    #     D_RNN_DEPTH, D_RNN_UNITS, NUM_TOKENS, NUM_META_FEATURES)

    ## Initialise Optimizer

    # Initialise optimizer for pretraining
    pre_train_g_opt = tf.keras.optimizers.Adam(
        PRE_TRAIN_LR, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # Initialise optimizer for adversarial training
    adv_train_g_opt = tf.keras.optimizers.Adam(ADV_TRAIN_G_LR, beta_1=0.9, beta_2=0.999)
    adv_train_d_opt = tf.keras.optimizers.Adam(ADV_TRAIN_D_LR, beta_1=0.9, beta_2=0.999)
    seq_train_opt = tf.keras.optimizers.Adam(SEQ_TRAIN_LR, beta_1=0.9, beta_2=0.999)

    ## Initialise Driver

    adv_train_driver_memofu = AdversarialDriver(g_model_memofu,
                                                d_model,
                                                adv_train_g_opt,
                                                adv_train_d_opt,
                                                TEMP_MAX,
                                                STEPS_PER_EPOCH_TRAIN,
                                                ADV_TRAIN_EPOCHS,
                                                NUM_TOKENS,
                                                MAX_GRAD_NORM)

    adv_train_driver_memofu_seq = AdversarialDriverWithSeqLoss(g_model_memofu,
                                                               d_model,
                                                               adv_train_g_opt,
                                                               adv_train_d_opt,
                                                               seq_train_opt,
                                                               TEMP_MAX,
                                                               STEPS_PER_EPOCH_TRAIN,
                                                               ADV_TRAIN_EPOCHS,
                                                               NUM_TOKENS,
                                                               MAX_GRAD_NORM,
                                                               LE_PATHS)

    adv_train_driver_memofu_style = AdversarialDriverWithStyle(g_model_memofu_style,
                                                               d_model_style,
                                                               adv_train_g_opt,
                                                               adv_train_d_opt,
                                                               TEMP_MAX,
                                                               STEPS_PER_EPOCH_TRAIN,
                                                               ADV_TRAIN_EPOCHS,
                                                               NUM_TOKENS,
                                                               MAX_GRAD_NORM)

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



    ## Setup Checkpoint
    
    # Setup checkpoint for pretraining
    # pre_train_ckpt = tf.train.Checkpoint(g_model=g_model,
    #                                      pre_train_g_opt=pre_train_g_opt)
    #
    # pre_train_ckpt_manager = tf.train.CheckpointManager(
    #     pre_train_ckpt, MEMOFU_PRE_TRAIN_CKPT_PATH, max_to_keep=PRE_TRAIN_EPOCHS)

    # Setup checkpoint for adversarial training

    # memefu
    memofu_adv_train_ckpt = tf.train.Checkpoint(g_model=g_model_memofu,
                                                d_model=d_model,
                                                adv_train_g_opt=adv_train_g_opt,
                                                adv_train_d_opt=adv_train_d_opt)

    memofu_adv_train_ckpt_manager = tf.train.CheckpointManager(
        memofu_adv_train_ckpt, MEMOFU_ADV_TRAIN_CKPT_PATH, max_to_keep=ADV_TRAIN_EPOCHS)

    # memofu + seq
    memofu_seq_adv_train_ckpt = tf.train.Checkpoint(g_model=g_model_memofu,
                                                    d_model=d_model,
                                                    adv_train_g_opt=adv_train_g_opt,
                                                    adv_train_d_opt=adv_train_d_opt)

    memofu_seq_adv_train_ckpt_manager = tf.train.CheckpointManager(
        memofu_seq_adv_train_ckpt, MEMOFU_SEQ_ADV_TRAIN_CKPT_PATH, max_to_keep=ADV_TRAIN_EPOCHS)

    # memofu + style
    memofu_style_adv_train_ckpt = tf.train.Checkpoint(g_model=g_model_memofu_style,
                                                      d_model=d_model_style,
                                                      adv_train_g_opt=adv_train_g_opt,
                                                      adv_train_d_opt=adv_train_d_opt)

    memofu_style_adv_train_ckpt_manager = tf.train.CheckpointManager(
        memofu_style_adv_train_ckpt, MEMOFU_STYLE_ADV_TRAIN_CKPT_PATH, max_to_keep=ADV_TRAIN_EPOCHS)

    # memofu + style + seq
    memofu_style_seq_adv_train_ckpt = tf.train.Checkpoint(g_model=g_model_memofu_style,
                                                          d_model=d_model_style,
                                                          adv_train_g_opt=adv_train_g_opt,
                                                          adv_train_d_opt=adv_train_d_opt)

    memofu_style_seq_adv_train_ckpt_manager = tf.train.CheckpointManager(
        memofu_style_seq_adv_train_ckpt, MEMOFU_STYLE_SEQ_ADV_TRAIN_CKPT_PATH, max_to_keep=ADV_TRAIN_EPOCHS)

    # Setup checkpoint for baseline
    # baseline_ckpt = tf.train.Checkpoint(g_model=g_model_baseline,
    #                                     d_model=d_model_baseline,
    #                                     adv_train_g_opt=adv_train_g_opt,
    #                                     adv_train_d_opt=adv_train_d_opt)
    #
    # baseline_ckpt_manager = tf.train.CheckpointManager(
    #     adv_train_ckpt, ADV_TRAIN_CKPT_PATH, max_to_keep=ADV_TRAIN_EPOCHS)

    if REUSE_TEST_RESULT_LOGS:
        # load test result logs from previous run
        try: 
            test_result_logs = pd.read_csv(TEST_RESULT_LOGS_FILENAME).to_dict('records')
        except FileNotFoundError: 
            test_result_logs = []
    else:
        test_result_logs = []
            
    """# Evaluation"""

    # Compute ground truth data statistics and attribute information

    # gather song attributes
    test_dat_songs = gather_song_attr(y_test_dat_attr, (TEST_LEN, SONG_LENGTH, NUM_SONG_FEATURES))

    # compute stats
    test_dat_stats = gather_stats(y_test_dat_attr, (TEST_LEN, SONG_LENGTH, NUM_SONG_FEATURES))
    test_dat_mean_stats = get_mean_stats(test_dat_stats, TEST_LEN)

    print('Test data mean stats')
    for key, value in test_dat_mean_stats.items():
        print(f"{key} = {value}")

    logs = {'seed': SEED,
            'run_id': 'None',
            'method': 'Groundtruth'}

    for n_gram in N_GRAMS:
        logs[f'selfBLEU_{n_gram}'] = 0

    logs['pMMD'] = 0
    logs['dMMD'] = 0
    logs['rMMD'] = 0
    logs['oMMD'] = 0

    logs = {**logs, **test_dat_mean_stats}

    test_result_logs.append(logs)

    # save ground truth attributes, songs, mean stats 
    pickle.dump(y_test_dat_attr, open('../../results/test/y_test_dat_attr.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    np.save('../../results/test/test_dat_songs.npy', test_dat_songs)
    pickle.dump(test_dat_mean_stats, open('../../results/test/test_dat_mean_stats.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

    print("================================================================ \n " )

    """## TBC-LSTM-MLE-MEMOFU"""

    # # restore the checkpoint at the end of pre-training.
    #
    # if pre_train_ckpt_manager.latest_checkpoint:
    #     pre_train_ckpt.restore(pre_train_ckpt_manager.latest_checkpoint).expect_partial()
    #     print ('Latest pretrain checkpoint restored from {}'.format(pre_train_ckpt_manager.latest_checkpoint))
    #
    # # reset the temperature
    # adv_train_driver.reset_temp()
    #
    # print('Temperature: {}'.format(adv_train_driver.temp.numpy()))
    #
    # # Compute statistics & attribute info. of songs generated using C-Hybrid-MLE
    #
    # result = {}
    #
    # for run_id in range(EVAL_RUNS):
    #
    #     print(f"\n ***** --run-{run_id}-- ***** \n")
    #
    #     logs = {'seed': SEED,
    #             'run_id' : run_id,
    #             'method' : 'TBC-LSTM-MLE-MEMOFU'}
    #
    #     # generated test song attr.
    #     test_g_out = adv_train_driver.generate(x_test)
    #
    #     # infer attributes of generated songs
    #     y_test_gen_attr = infer(test_g_out, LE_PATHS, is_tune=False)
    #
    #     # gather song attributes
    #     test_gen_songs = gather_song_attr(
    #       y_test_gen_attr, (TEST_LEN, SONG_LENGTH, NUM_SONG_FEATURES))
    #
    #     # compute self bleu
    #     test_self_bleu = compute_self_bleu_score(y_test_gen_attr, N_GRAMS)
    #     print('Self-BLEU: {}'.format(test_self_bleu))
    #
    #     # compute mmd
    #     test_p_mmd, test_d_mmd, test_r_mmd = compute_mmd_score(y_test_dat_attr, y_test_gen_attr)
    #     test_o_mmd = test_p_mmd + test_d_mmd + test_r_mmd
    #     print('pMMD: {}\ndMMD: {}\nrMMD: {}\noMMD: {}\n'.format(test_p_mmd, test_d_mmd, test_r_mmd, test_o_mmd))
    #
    #     # compute stats
    #     test_gen_stats = gather_stats(
    #       y_test_gen_attr, (TEST_LEN, SONG_LENGTH, NUM_SONG_FEATURES))
    #
    #     test_gen_mean_stats = get_mean_stats(test_gen_stats, TEST_LEN)
    #     print('TBC-LSTM-MLE-MEMOFU mean stats')
    #     for key, value in test_gen_mean_stats.items():
    #         result[key] = result.get(key, []) + [value]
    #         print(f"{key}: {value}")
    #
    #     # save test result logs for current run
    #
    #     for n_gram in N_GRAMS:
    #         logs[f'selfBLEU_{n_gram}'] = test_self_bleu[n_gram]
    #
    #     logs['pMMD'] = test_p_mmd
    #     logs['dMMD'] = test_d_mmd
    #     logs['rMMD'] = test_r_mmd
    #     logs['oMMD'] = test_o_mmd
    #
    #     logs = {**logs, **test_gen_mean_stats}
    #
    #     test_result_logs.append(logs)
    #
    #     print()
    #
    # # print average stats across runs
    #
    # print(f'\nTBC-LSTM-MLE-MEMOFU mean stats across {EVAL_RUNS} runs.\n')
    # for key, value in result.items():
    #     print('{}: {} +/- {}'.format(key, round(np.mean(value), 4), round(np.std(value), 4)))
    #
    # # save songs, attr & mean stats generated using TBC-LSTM-MLE
    #
    # pickle.dump(y_test_gen_attr, open('../../results/tbc_lstm_gan_memofu/generated/tbc_lstm_mle_y_test_gen_attr.pkl', 'wb'),
    #             pickle.HIGHEST_PROTOCOL)
    # np.save('../../results/tbc_lstm_gan_memofu/generated/tbc_lstm_mle_test_gen_songs.npy', test_gen_songs)
    # pickle.dump(test_gen_mean_stats, open('../../results/tbc_lstm_gan_memofu/generated/tbc_lstm_mle_test_mean_stats.pkl', 'wb'),
    #             pickle.HIGHEST_PROTOCOL)
    #
    # print("================================================================ \n " )

    """## TBC-LSTM-GAN-MEMOFU"""

    # restore the checkpoint at the end of adversarial training.

    if memofu_adv_train_ckpt_manager.latest_checkpoint:
        memofu_adv_train_ckpt.restore(memofu_adv_train_ckpt_manager.latest_checkpoint).expect_partial()
        print ('Latest checkpoint restored from {}'.format(memofu_adv_train_ckpt_manager.latest_checkpoint))

    # update the temperature
    adv_train_driver_memofu.update_temp(ADV_TRAIN_EPOCHS-1, STEPS_PER_EPOCH_TRAIN-1)
    print('Temperature: {}'.format(adv_train_driver_memofu.temp.numpy()))

    # Compute statistics & attribute info. of songs generated using C-Hybrid-GAN

    result = {}

    for run_id in range(EVAL_RUNS):
      
        print(f"\n ***** --run-{run_id}-- ***** \n")

        logs = {'seed'   : SEED,
                'run_id' : run_id,
                'method' : 'TBC-LSTM-GAN-MEMOFU'}

        # generated test song attr.
        test_g_out = adv_train_driver_memofu.generate(x_test)

        # infer attributes of generated songs
        y_test_gen_attr = infer(test_g_out, LE_PATHS, is_tune=False)

        # gather song attributes
        test_gen_songs = gather_song_attr(
          y_test_gen_attr, (TEST_LEN, SONG_LENGTH, NUM_SONG_FEATURES))

        # compute self bleu
        test_self_bleu = compute_self_bleu_score(y_test_gen_attr, N_GRAMS)
        print('Self-BLEU: {}'.format(test_self_bleu))

        # compute mmd
        test_p_mmd, test_d_mmd, test_r_mmd = compute_mmd_score(y_test_dat_attr, y_test_gen_attr)
        test_o_mmd = test_p_mmd + test_d_mmd + test_r_mmd
        print('pMMD: {}\ndMMD: {}\nrMMD: {}\noMMD: {}\n'.format(test_p_mmd, test_d_mmd, test_r_mmd, test_o_mmd))

        # compute stats
        test_gen_stats = gather_stats(
          y_test_gen_attr, (TEST_LEN, SONG_LENGTH, NUM_SONG_FEATURES))

        test_gen_mean_stats = get_mean_stats(test_gen_stats, TEST_LEN)
        print('TBC-LSTM-GAN-MEMOFU mean stats')
        for key, value in test_gen_mean_stats.items():
            result[key] = result.get(key, []) + [value]
            print(f"{key}: {value}")

        # save test result logs for current run
        
        for n_gram in N_GRAMS:
            logs[f'selfBLEU_{n_gram}'] = test_self_bleu[n_gram]

        logs['pMMD'] = test_p_mmd
        logs['dMMD'] = test_d_mmd
        logs['rMMD'] = test_r_mmd
        logs['oMMD'] = test_o_mmd

        logs = {**logs, **test_gen_mean_stats}

        test_result_logs.append(logs)

        print()

    # print average stats across runs

    print(f'\nTBC-LSTM-GAN-MEMOFU mean stats across {EVAL_RUNS} runs.\n')
    for key, value in result.items():
        print('{}: {} +/- {}'.format(key, round(np.mean(value), 4), round(np.std(value), 4)))

    # save songs, attr & mean stats generated using TBC-LSTM-GAN

    results_dir_memofu = '../../results/tbc_lstm_gan_memofu/generated/'
    os.makedirs(results_dir_memofu, exist_ok=True)

    pickle.dump(y_test_gen_attr, open(os.path.join(results_dir_memofu, 'tbc_lstm_gan_y_test_gen_attr.pkl'), 'wb'),
                pickle.HIGHEST_PROTOCOL)
    np.save(os.path.join(results_dir_memofu, 'tbc_lstm_gan_test_gen_songs.npy'), test_gen_songs)
    pickle.dump(test_gen_mean_stats, open(os.path.join(results_dir_memofu, 'tbc_lstm_gan_test_mean_stats.pkl'), 'wb'),
                pickle.HIGHEST_PROTOCOL)

    """## TBC-LSTM-GAN-MEMOFU-SEQ"""

    # restore the checkpoint at the end of adversarial training.

    if memofu_seq_adv_train_ckpt_manager.latest_checkpoint:
        memofu_seq_adv_train_ckpt.restore(memofu_seq_adv_train_ckpt_manager.latest_checkpoint).expect_partial()
        print('Latest checkpoint restored from {}'.format(memofu_seq_adv_train_ckpt_manager.latest_checkpoint))

    # update the temperature
    adv_train_driver_memofu_seq.update_temp(ADV_TRAIN_EPOCHS - 1, STEPS_PER_EPOCH_TRAIN - 1)
    print('Temperature: {}'.format(adv_train_driver_memofu_seq.temp.numpy()))

    # Compute statistics & attribute info. of songs generated using C-Hybrid-GAN

    result = {}

    for run_id in range(EVAL_RUNS):

        print(f"\n ***** --run-{run_id}-- ***** \n")

        logs = {'seed': SEED,
                'run_id': run_id,
                'method': 'TBC-LSTM-GAN-MEMOFU-SEQ'}

        # generated test song attr.
        test_g_out = adv_train_driver_memofu_seq.generate(x_test)

        # infer attributes of generated songs
        y_test_gen_attr = infer(test_g_out, LE_PATHS, is_tune=False)

        # gather song attributes
        test_gen_songs = gather_song_attr(
            y_test_gen_attr, (TEST_LEN, SONG_LENGTH, NUM_SONG_FEATURES))

        # compute self bleu
        test_self_bleu = compute_self_bleu_score(y_test_gen_attr, N_GRAMS)
        print('Self-BLEU: {}'.format(test_self_bleu))

        # compute mmd
        test_p_mmd, test_d_mmd, test_r_mmd = compute_mmd_score(y_test_dat_attr, y_test_gen_attr)
        test_o_mmd = test_p_mmd + test_d_mmd + test_r_mmd
        print('pMMD: {}\ndMMD: {}\nrMMD: {}\noMMD: {}\n'.format(test_p_mmd, test_d_mmd, test_r_mmd, test_o_mmd))

        # compute stats
        test_gen_stats = gather_stats(
            y_test_gen_attr, (TEST_LEN, SONG_LENGTH, NUM_SONG_FEATURES))

        test_gen_mean_stats = get_mean_stats(test_gen_stats, TEST_LEN)
        print('TBC-LSTM-GAN-MEMOFU-SEQ mean stats')
        for key, value in test_gen_mean_stats.items():
            result[key] = result.get(key, []) + [value]
            print(f"{key}: {value}")

        # save test result logs for current run

        for n_gram in N_GRAMS:
            logs[f'selfBLEU_{n_gram}'] = test_self_bleu[n_gram]

        logs['pMMD'] = test_p_mmd
        logs['dMMD'] = test_d_mmd
        logs['rMMD'] = test_r_mmd
        logs['oMMD'] = test_o_mmd

        logs = {**logs, **test_gen_mean_stats}

        test_result_logs.append(logs)

        print()

    # print average stats across runs

    print(f'\nTBC-LSTM-GAN-MEMOFU-SEQ mean stats across {EVAL_RUNS} runs.\n')
    for key, value in result.items():
        print('{}: {} +/- {}'.format(key, round(np.mean(value), 4), round(np.std(value), 4)))

    # save songs, attr & mean stats generated using TBC-LSTM-GAN

    results_dir_memofu_seq = '../../results/tbc_lstm_gan_memofu_seq/generated/'
    os.makedirs(results_dir_memofu_seq, exist_ok=True)

    pickle.dump(y_test_gen_attr, open(os.path.join(results_dir_memofu_seq, 'tbc_lstm_gan_y_test_gen_attr.pkl'), 'wb'),
                pickle.HIGHEST_PROTOCOL)
    np.save(os.path.join(results_dir_memofu_seq, 'tbc_lstm_gan_test_gen_songs.npy'), test_gen_songs)
    pickle.dump(test_gen_mean_stats, open(os.path.join(results_dir_memofu_seq, 'tbc_lstm_gan_test_mean_stats.pkl'), 'wb'),
                pickle.HIGHEST_PROTOCOL)

    """## TBC-LSTM-GAN-MEMOFU-STYLE"""

    # restore the checkpoint at the end of adversarial training.

    if memofu_style_adv_train_ckpt_manager.latest_checkpoint:
        memofu_style_adv_train_ckpt.restore(memofu_style_adv_train_ckpt_manager.latest_checkpoint).expect_partial()
        print('Latest checkpoint restored from {}'.format(memofu_style_adv_train_ckpt_manager.latest_checkpoint))

    # update the temperature
    adv_train_driver_memofu_style.update_temp(ADV_TRAIN_EPOCHS - 1, STEPS_PER_EPOCH_TRAIN - 1)
    print('Temperature: {}'.format(adv_train_driver_memofu_style.temp.numpy()))

    # Compute statistics & attribute info. of songs generated using C-Hybrid-GAN

    result = {}

    for run_id in range(EVAL_RUNS):

        print(f"\n ***** --run-{run_id}-- ***** \n")

        logs = {'seed': SEED,
                'run_id': run_id,
                'method': 'TBC-LSTM-GAN-MEMOFU-STYLE'}

        # generated test song attr.
        test_g_out = adv_train_driver_memofu_style.generate(x_test, test_style)

        # infer attributes of generated songs
        y_test_gen_attr = infer(test_g_out, LE_PATHS, is_tune=False)

        # gather song attributes
        test_gen_songs = gather_song_attr(
            y_test_gen_attr, (TEST_LEN, SONG_LENGTH, NUM_SONG_FEATURES))

        # compute self bleu
        test_self_bleu = compute_self_bleu_score(y_test_gen_attr, N_GRAMS)
        print('Self-BLEU: {}'.format(test_self_bleu))

        # compute mmd
        test_p_mmd, test_d_mmd, test_r_mmd = compute_mmd_score(y_test_dat_attr, y_test_gen_attr)
        test_o_mmd = test_p_mmd + test_d_mmd + test_r_mmd
        print('pMMD: {}\ndMMD: {}\nrMMD: {}\noMMD: {}\n'.format(test_p_mmd, test_d_mmd, test_r_mmd, test_o_mmd))

        # compute stats
        test_gen_stats = gather_stats(
            y_test_gen_attr, (TEST_LEN, SONG_LENGTH, NUM_SONG_FEATURES))

        test_gen_mean_stats = get_mean_stats(test_gen_stats, TEST_LEN)
        print('TBC-LSTM-GAN-MEMOFU-STYLE mean stats')
        for key, value in test_gen_mean_stats.items():
            result[key] = result.get(key, []) + [value]
            print(f"{key}: {value}")

        # save test result logs for current run

        for n_gram in N_GRAMS:
            logs[f'selfBLEU_{n_gram}'] = test_self_bleu[n_gram]

        logs['pMMD'] = test_p_mmd
        logs['dMMD'] = test_d_mmd
        logs['rMMD'] = test_r_mmd
        logs['oMMD'] = test_o_mmd

        logs = {**logs, **test_gen_mean_stats}

        test_result_logs.append(logs)

        print()

    # print average stats across runs

    print(f'\nTBC-LSTM-GAN-MEMOFU-STYLE mean stats across {EVAL_RUNS} runs.\n')
    for key, value in result.items():
        print('{}: {} +/- {}'.format(key, round(np.mean(value), 4), round(np.std(value), 4)))

    # save songs, attr & mean stats generated using TBC-LSTM-GAN

    results_dir_memofu_style = '../../results/tbc_lstm_gan_memofu_style/generated/'
    os.makedirs(results_dir_memofu_style, exist_ok=True)

    pickle.dump(y_test_gen_attr, open(os.path.join(results_dir_memofu_style, 'tbc_lstm_gan_y_test_gen_attr.pkl'), 'wb'),
                pickle.HIGHEST_PROTOCOL)
    np.save(os.path.join(results_dir_memofu_style, 'tbc_lstm_gan_test_gen_songs.npy'), test_gen_songs)
    pickle.dump(test_gen_mean_stats, open(os.path.join(results_dir_memofu_style, 'tbc_lstm_gan_test_mean_stats.pkl'), 'wb'),
                pickle.HIGHEST_PROTOCOL)

    """## TBC-LSTM-GAN-MEMOFU-STYLE-SEQ"""

    # restore the checkpoint at the end of adversarial training.

    if memofu_style_seq_adv_train_ckpt_manager.latest_checkpoint:
        memofu_style_seq_adv_train_ckpt.restore(memofu_style_seq_adv_train_ckpt_manager.latest_checkpoint).expect_partial()
        print('Latest checkpoint restored from {}'.format(memofu_style_seq_adv_train_ckpt_manager.latest_checkpoint))

    # update the temperature
    adv_train_driver_memofu_style_seq.update_temp(ADV_TRAIN_EPOCHS - 1, STEPS_PER_EPOCH_TRAIN - 1)
    print('Temperature: {}'.format(adv_train_driver_memofu_style_seq.temp.numpy()))

    # Compute statistics & attribute info. of songs generated using C-Hybrid-GAN

    result = {}

    for run_id in range(EVAL_RUNS):

        print(f"\n ***** --run-{run_id}-- ***** \n")

        logs = {'seed': SEED,
                'run_id': run_id,
                'method': 'TBC-LSTM-GAN-MEMOFU-STYLE-SEQ'}

        # generated test song attr.
        test_g_out = adv_train_driver_memofu_style_seq.generate(x_test, test_style)

        # infer attributes of generated songs
        y_test_gen_attr = infer(test_g_out, LE_PATHS, is_tune=False)

        # gather song attributes
        test_gen_songs = gather_song_attr(
            y_test_gen_attr, (TEST_LEN, SONG_LENGTH, NUM_SONG_FEATURES))

        # compute self bleu
        test_self_bleu = compute_self_bleu_score(y_test_gen_attr, N_GRAMS)
        print('Self-BLEU: {}'.format(test_self_bleu))

        # compute mmd
        test_p_mmd, test_d_mmd, test_r_mmd = compute_mmd_score(y_test_dat_attr, y_test_gen_attr)
        test_o_mmd = test_p_mmd + test_d_mmd + test_r_mmd
        print('pMMD: {}\ndMMD: {}\nrMMD: {}\noMMD: {}\n'.format(test_p_mmd, test_d_mmd, test_r_mmd, test_o_mmd))

        # compute stats
        test_gen_stats = gather_stats(
            y_test_gen_attr, (TEST_LEN, SONG_LENGTH, NUM_SONG_FEATURES))

        test_gen_mean_stats = get_mean_stats(test_gen_stats, TEST_LEN)
        print('TBC-LSTM-GAN-MEMOFU-STYLE mean stats')
        for key, value in test_gen_mean_stats.items():
            result[key] = result.get(key, []) + [value]
            print(f"{key}: {value}")

        # save test result logs for current run

        for n_gram in N_GRAMS:
            logs[f'selfBLEU_{n_gram}'] = test_self_bleu[n_gram]

        logs['pMMD'] = test_p_mmd
        logs['dMMD'] = test_d_mmd
        logs['rMMD'] = test_r_mmd
        logs['oMMD'] = test_o_mmd

        logs = {**logs, **test_gen_mean_stats}

        test_result_logs.append(logs)

        print()

    # print average stats across runs

    print(f'\nTBC-LSTM-GAN-MEMOFU-STYLE-SEQ mean stats across {EVAL_RUNS} runs.\n')
    for key, value in result.items():
        print('{}: {} +/- {}'.format(key, round(np.mean(value), 4), round(np.std(value), 4)))

    # save songs, attr & mean stats generated using TBC-LSTM-GAN

    results_dir_memofu_style_seq = '../../results/tbc_lstm_gan_memofu_style_seq/generated/'
    os.makedirs(results_dir_memofu_style_seq, exist_ok=True)

    pickle.dump(y_test_gen_attr, open(os.path.join(results_dir_memofu_style_seq, 'tbc_lstm_gan_y_test_gen_attr.pkl'), 'wb'),
                pickle.HIGHEST_PROTOCOL)
    np.save(os.path.join(results_dir_memofu_style_seq, 'tbc_lstm_gan_test_gen_songs.npy'), test_gen_songs)
    pickle.dump(test_gen_mean_stats, open(os.path.join(results_dir_memofu_style_seq, 'tbc_lstm_gan_test_mean_stats.pkl'), 'wb'),
                pickle.HIGHEST_PROTOCOL)

    # save test result logs as csv file

    pd.DataFrame.from_records(test_result_logs).to_csv(TEST_RESULT_LOGS_FILENAME, index=False)

if __name__ == '__main__':
    
    settings = {'settings_file': 'settings'}
    settings = load_settings_from_file(settings)
    
    print("Settings: \n")
    for (k, v) in settings.items():
        print(v, '\t', k)
    
    locals().update(settings)
    
    print("================================================================ \n " )
    
    main()
    
