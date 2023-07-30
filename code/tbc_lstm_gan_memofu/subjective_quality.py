"""
@Project: tbc_lstm_gan_style
@File: subjective_quality.py
@Author: Zhe Zhang
@Email: zhe@nii.ac.jp
@Date: 2022/10/25
"""

from generator_memofu import *
from discriminator import *
from drivers_memofu import *
from utils import *

import argparse
from gensim.models import Word2Vec
import time

from utils_new import *
import sys


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

    TRAIN_LEN = len(x_train)

    STEPS_PER_EPOCH_TRAIN = np.ceil(TRAIN_LEN / BATCH_SIZE)
    print('Steps per epoch train: ', STEPS_PER_EPOCH_TRAIN)

    style_p_rav_range = [[style_p_rav[0].min(), style_p_rav[0].max()],
                         [style_p_rav[1].min(), style_p_rav[1].max()],
                         [style_p_rav[2].min(), style_p_rav[2].max()]]

    style_d_rav_range = [[style_d_rav[0].min(), style_d_rav[0].max()],
                         [style_d_rav[1].min(), style_d_rav[1].max()],
                         [style_d_rav[2].min(), style_d_rav[2].max()]]

    style_r_rav_range = [[style_r_rav[0].min(), style_r_rav[0].max()],
                         [style_r_rav[1].min(), style_r_rav[1].max()],
                         [style_r_rav[2].min(), style_r_rav[2].max()]]

    style_pitch_range = map_parameter_to_value_range(STYLE_PITCH[0], style_p_rav_range[0])
    style_pitch_average = map_parameter_to_value_range(STYLE_PITCH[1], style_p_rav_range[1])
    style_pitch_variance = map_parameter_to_value_range(STYLE_PITCH[2], style_p_rav_range[2])

    style_duration_range = map_parameter_to_value_range(STYLE_DURATION[0], style_d_rav_range[0])
    style_duration_average = map_parameter_to_value_range(STYLE_DURATION[1], style_d_rav_range[1])
    style_duration_variance = map_parameter_to_value_range(STYLE_DURATION[2], style_d_rav_range[2])

    style_rest_range = map_parameter_to_value_range(STYLE_REST[0], style_r_rav_range[0])
    style_rest_average = map_parameter_to_value_range(STYLE_REST[1], style_r_rav_range[1])
    style_rest_variance = map_parameter_to_value_range(STYLE_REST[2], style_r_rav_range[2])

    style_pdr_rav_ohe = get_style_ohe(
        (style_pitch_range, style_pitch_average, style_pitch_variance),
        (style_duration_range, style_duration_average, style_duration_variance),
        (style_rest_range, style_rest_average, style_rest_variance),
        range_paths, average_paths, variance_paths,
        convert_to_tensor=True
    )

    # prepare meta data using syll_lyrics & word_lyrics

    syllModel = Word2Vec.load(SYLL_MODEL_PATH)
    wordModel = Word2Vec.load(WORD_MODEL_PATH)

    syll_lyrics = SYLL_LYRICS.split()
    word_lyrics = WORD_LYRICS.split()

    assert len(syll_lyrics) == len(word_lyrics), "length of syllable-lyrics & word-lyrics must equal"

    song_length = len(syll_lyrics)

    meta = []
    for syll, word in zip(syll_lyrics, word_lyrics):
        try:
            syll2Vec = syllModel.wv[syll]
            word2Vec = wordModel.wv[word]
            meta.append(np.concatenate((syll2Vec, word2Vec)))
        except:
            print(f"KeyError: ({syll}, {word}) not present in vocab.")

    meta = np.expand_dims(meta, 0)  # [1, song_length, NUM_META_FEATURES]

    print("================================================================ \n ")

    ## Initialise Model

    # Initialise generator model
    g_model_memofu_style = GeneratorWithMemoryFusionAndStyle(
        G_EMB_UNITS, G_PROJ_UNITS, G_EMB_DROPOUT_RATE, G_PROJ_DROPOUT_RATE,
        G_RNN_DEPTH, G_RNN_UNITS, NUM_TOKENS, NUM_META_FEATURES, num_style_tokens)

    # Initialise discriminator
    d_model_style = DiscriminatorWithStyle(
        D_EMB_UNITS, D_PROJ_UNITS, D_EMB_DROPOUT_RATE, D_PROJ_DROPOUT_RATE,
        D_RNN_DEPTH, D_RNN_UNITS, NUM_TOKENS, NUM_META_FEATURES, num_style_tokens)

    g_model_memofu = GeneratorWithMemoryFusion(
        G_EMB_UNITS, G_PROJ_UNITS, G_EMB_DROPOUT_RATE, G_PROJ_DROPOUT_RATE,
        G_RNN_DEPTH, G_RNN_UNITS, NUM_TOKENS, NUM_META_FEATURES)

    d_model = Discriminator(
        D_EMB_UNITS, D_PROJ_UNITS, D_EMB_DROPOUT_RATE, D_PROJ_DROPOUT_RATE,
        D_RNN_DEPTH, D_RNN_UNITS, NUM_TOKENS, NUM_META_FEATURES)

    ## Initialise Optimizer

    # Initialise optimizer for adversarial training
    adv_train_g_opt = tf.keras.optimizers.Adam(ADV_TRAIN_G_LR, beta_1=0.9, beta_2=0.999)
    adv_train_d_opt = tf.keras.optimizers.Adam(ADV_TRAIN_D_LR, beta_1=0.9, beta_2=0.999)
    seq_train_opt = tf.keras.optimizers.Adam(SEQ_TRAIN_LR, beta_1=0.9, beta_2=0.999)

    ## Initialise Driver

    # Initialise adversarial driver
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

    adv_train_driver_memofu_style = AdversarialDriverWithStyle(g_model_memofu_style,
                                                               d_model_style,
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

    adv_train_driver_memofu = AdversarialDriver(g_model_memofu,
                                                d_model,
                                                adv_train_g_opt,
                                                adv_train_d_opt,
                                                TEMP_MAX,
                                                STEPS_PER_EPOCH_TRAIN,
                                                ADV_TRAIN_EPOCHS,
                                                NUM_TOKENS,
                                                MAX_GRAD_NORM)

    ## Setup Checkpoint

    if MODEL == 'Memofu':

        # Setup checkpoint for adversarial training
        adv_train_ckpt = tf.train.Checkpoint(g_model=g_model_memofu,
                                             d_model=d_model,
                                             adv_train_g_opt=adv_train_g_opt,
                                             adv_train_d_opt=adv_train_d_opt)

        adv_train_ckpt_manager = tf.train.CheckpointManager(
            adv_train_ckpt, MEMOFU_ADV_TRAIN_CKPT_PATH, max_to_keep=ADV_TRAIN_EPOCHS)

        # restore the adversarial training checkpoint.

        if adv_train_ckpt_manager.latest_checkpoint:
            adv_train_ckpt.restore(adv_train_ckpt_manager.latest_checkpoint).expect_partial()
            print('Latest checkpoint restored from {}'.format(adv_train_ckpt_manager.latest_checkpoint))
        else:
            print('No checkpoint found')
        # update the temperature
        adv_train_driver_memofu.update_temp(ADV_TRAIN_EPOCHS - 1, STEPS_PER_EPOCH_TRAIN - 1)
        print('Temperature: {}'.format(adv_train_driver_memofu.temp.numpy()))

        out = adv_train_driver_memofu.generate(meta)

    elif MODEL == 'SeqMelody':

        seq_adv_train_ckpt = tf.train.Checkpoint(g_model=g_model_memofu,
                                                 d_model=d_model,
                                                 adv_train_g_opt=adv_train_g_opt,
                                                 adv_train_d_opt=adv_train_d_opt,
                                                 seq_train_opt=seq_train_opt)

        seq_adv_train_ckpt_manager = tf.train.CheckpointManager(
            seq_adv_train_ckpt, MEMOFU_SEQ_ADV_TRAIN_CKPT_PATH, max_to_keep=ADV_TRAIN_EPOCHS)

        if seq_adv_train_ckpt_manager.latest_checkpoint:
            seq_adv_train_ckpt.restore(seq_adv_train_ckpt_manager.latest_checkpoint).expect_partial()
            print('Latest checkpoint restored from {}'.format(seq_adv_train_ckpt_manager.latest_checkpoint))

        # update the temperature
        adv_train_driver_memofu_seq.update_temp(ADV_TRAIN_EPOCHS - 1, STEPS_PER_EPOCH_TRAIN - 1)
        print('Temperature: {}'.format(adv_train_driver_memofu_seq.temp.numpy()))

        out = adv_train_driver_memofu_seq.generate(meta)

    elif MODEL == 'CtrlMelody':

        style_seq_adv_train_ckpt = tf.train.Checkpoint(g_model=g_model_memofu_style,
                                                       d_model=d_model_style,
                                                       adv_train_g_opt=adv_train_g_opt,
                                                       adv_train_d_opt=adv_train_d_opt,
                                                       seq_train_opt=seq_train_opt)

        style_seq_adv_train_ckpt_manager = tf.train.CheckpointManager(
            style_seq_adv_train_ckpt, MEMOFU_STYLE_ADV_TRAIN_CKPT_PATH, max_to_keep=ADV_TRAIN_EPOCHS)

        if style_seq_adv_train_ckpt_manager.latest_checkpoint:
            style_seq_adv_train_ckpt.restore(style_seq_adv_train_ckpt_manager.latest_checkpoint).expect_partial()
            print('Latest checkpoint restored from {}'.format(style_seq_adv_train_ckpt_manager.latest_checkpoint))

        # update the temperature
        adv_train_driver_memofu_style.update_temp(ADV_TRAIN_EPOCHS - 1, STEPS_PER_EPOCH_TRAIN - 1)
        print('Temperature: {}'.format(adv_train_driver_memofu_style.temp.numpy()))

        out = adv_train_driver_memofu_style.generate(meta, style_pdr_rav_ohe)

    elif MODEL == 'SeqCtrlMelody':

        style_seq_adv_train_ckpt = tf.train.Checkpoint(g_model=g_model_memofu_style,
                                                       d_model=d_model_style,
                                                       adv_train_g_opt=adv_train_g_opt,
                                                       adv_train_d_opt=adv_train_d_opt,
                                                       seq_train_opt=seq_train_opt)

        style_seq_adv_train_ckpt_manager = tf.train.CheckpointManager(
            style_seq_adv_train_ckpt, MEMOFU_STYLE_SEQ_ADV_TRAIN_CKPT_PATH, max_to_keep=ADV_TRAIN_EPOCHS)

        if style_seq_adv_train_ckpt_manager.latest_checkpoint:
            style_seq_adv_train_ckpt.restore(style_seq_adv_train_ckpt_manager.latest_checkpoint).expect_partial()
            print('Latest checkpoint restored from {}'.format(style_seq_adv_train_ckpt_manager.latest_checkpoint))

        # update the temperature
        adv_train_driver_memofu_style_seq.update_temp(ADV_TRAIN_EPOCHS - 1, STEPS_PER_EPOCH_TRAIN - 1)
        print('Temperature: {}'.format(adv_train_driver_memofu_style_seq.temp.numpy()))

        out = adv_train_driver_memofu_style_seq.generate(meta, style_pdr_rav_ohe)

    else:
        print('Model selection error.')

    # generated
    # out = adv_train_driver.generate(meta, style_pdr_rav_ohe)

    # infer generated song attributes
    gen_attr = infer(out, LE_PATHS, is_tune=False)

    # gather song attributes
    gen_song = gather_song_attr(gen_attr, (1, song_length, NUM_SONG_FEATURES))
    gen_song = tf.squeeze(gen_song, 0).numpy()

    gen_midi = create_midi_pattern_from_discretized_data_with_lyrics(gen_song, syll_lyrics)

    gen_pitch = np.expand_dims(gen_song[:, 0], 0)
    gen_duration = np.expand_dims(gen_song[:, 1], 0)
    gen_rest = np.expand_dims(gen_song[:, 2], 0)

    gen_pitch_rng, gen_pitch_avg, gen_pitch_var = calculate_style_features(gen_pitch)
    gen_duration_rng, gen_duration_avg, gen_duration_var = calculate_style_features(gen_duration)
    gen_rest_rng, gen_rest_avg, gen_rest_var = calculate_style_features(gen_rest)

    print(f'Input melody style reference: \n'
          f'pitch range: {style_pitch_range}, pitch average: {style_pitch_average}, pitch variance: {style_pitch_variance} \n'
          f'duration range: {style_duration_range}, duration average: {style_duration_average}, duration variance: {style_duration_variance} \n'
          f'rest range: {style_rest_range}, rest average: {style_rest_average}, rest variance: {style_rest_variance} \n')

    print(f'Generated melody style: \n'
          f'pitch range: {gen_pitch_rng}, pitch average: {gen_pitch_avg}, pitch variance: {gen_pitch_var} \n'
          f'duration range: {gen_duration_rng}, duration average: {gen_duration_avg}, duration variance: {gen_duration_var} \n'
          f'rest range: {gen_rest_rng}, rest average: {gen_rest_avg}, rest variance: {gen_rest_var} \n')

    style_str = '_rav' + \
                '_p[' + ','.join(map(str, STYLE_PITCH)) + ']' + \
                '_d[' + ','.join(map(str, STYLE_DURATION)) + ']' + \
                '_r[' + ','.join(map(str, STYLE_REST)) + ']'

    if MIDI_NAME:
        gen_midi.write(f'../../subjective_evaluation/quality/{MODEL + MIDI_NAME + style_str}.mid')
        print(f"Melody can be found at ../../subjective_evaluation/quality/{MIDI_NAME}.mid")
    else:
        timestamp = time.time()
        gen_midi.write(f'../../subjective_evaluation/quality/{MODEL + str(timestamp) + style_str}.mid')
        print(f"Melody can be found at ../../subjective_evaluation/quality/{timestamp}.mid")


if __name__ == '__main__':

    settings = {'settings_file': 'settings'}

    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--SYLL_LYRICS", required=True, help="syllable lyrics for which melody is to be generated.",
                        type=str)
    parser.add_argument("--WORD_LYRICS", required=True, help="word lyrics for which melody is to be generated.",
                        type=str)
    # parser.add_argument("--CKPT_PATH",   help="path to the model checkpoints.", type=str,
    #                     default='../../checkpoints/tbc_lstm_gan/adv_train_tbc_lstm_gan')
    parser.add_argument("--MODEL", help="If model checkpoint corresponds to GAN or not.", type=str)
    parser.add_argument("--MIDI_NAME", help="name of the generated melody", type=str)
    parser.add_argument('--STYLE_PITCH', nargs=3, type=float, default=[0.5, 0.5, 0.5])
    parser.add_argument('--STYLE_DURATION', nargs=3, type=float, default=[0.5, 0.5, 0.5])
    parser.add_argument('--STYLE_REST', nargs=3, type=float, default=[0.5, 0.5, 0.5])

    # SYLL_LYRICS = "I know that when you look at me there's so much that you just don't see"
    # WORD_LYRICS = "I know that when you look at me there's so much that you just don't see"

    # SYLL_LYRICS = "She was cry    ing    and I was lone   ly     and the band was play    ing     some old love song"
    # WORD_LYRICS = "She was crying crying and I was lonely lonely and the band was playing playing some old love song"

    # SYLL_LYRICS = "I have the time so I will sing yeah I'm just a boy but I will win yeah"
    # WORD_LYRICS = "I have the time so I will sing yeah I'm just a boy but I will win yeah"

    # SYLL_LYRICS = "Late one night I heard a knock on my door No sur      prise    it was my land     lord"
    # WORD_LYRICS = "Late one night I heard a knock on my door No surprise surprise it was my landlord landlord"

    # SYLL_LYRICS = "You and me and a bot    tle    of wine gon   na    hold you to      night"
    # WORD_LYRICS = "You and me and a bottle bottle of wine gonna gonna hold you tonight tonight"

    # SYLL_LYRICS = "I talk to the face in the mir ror but he can't get through"
    # WORD_LYRICS = "I talk to the face in the mirror mirror but he can't get through"

    # SYLL_LYRICS = "When I am king you will be first a gainst the wall"
    # WORD_LYRICS = "When I am king you will be first against against the wall"

    # SYLL_LYRICS = "Eve ry night in my dreams I see you I feel you"
    # WORD_LYRICS = "Every Every night in my dreams I see you I feel you"

    # SYLL_LYRICS = "This mo ment this min ute and each sec ond in it will leave a glow up"
    # WORD_LYRICS = "This moment moment this minute minute and each second second in it will leave a glow up"

    SYLL_LYRICS = "I want to live I want to give I've been a mi ner for a heart of gold"
    WORD_LYRICS = "I want to live I want to give I've been a miner miner for a heart of gold"

    NAME = '_gold'

    MODEL_LIST = ['Memofu', 'SeqMelody', 'CtrlMelody', 'SeqCtrlMelody']

    # [range, average, variance]
    STYLE_PITCH = ['0.2', '0.5', '0.3']
    STYLE_DURATION = ['0.2', '0.4', '0.2']
    STYLE_REST = ['0.1', '0.1', '0.1']

    sys.argv = ['generate_style_seq.py', '--SYLL_LYRICS', SYLL_LYRICS, '--WORD_LYRICS', WORD_LYRICS,
                '--MIDI_NAME', NAME,
                '--STYLE_PITCH', STYLE_PITCH[0], STYLE_PITCH[1], STYLE_PITCH[2],
                '--STYLE_DURATION', STYLE_DURATION[0], STYLE_DURATION[1], STYLE_DURATION[2],
                '--STYLE_REST', STYLE_REST[0], STYLE_REST[1], STYLE_REST[2]]

    settings.update(vars(parser.parse_args()))
    settings = load_settings_from_file(settings)

    # print("Settings: \n")
    # for (k, v) in settings.items():
    #     print(v, '\t', k)

    for MODEL in MODEL_LIST:

        settings['MODEL'] = MODEL

        locals().update(settings)

        print("================================================================ \n ")

        main()

