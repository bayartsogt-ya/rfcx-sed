import torch
import audiomentations as AA


class args:
    DEBUG = False

    exp_name = "SED_E0_5F_BASE"
    is_train = True
    use_mixup = True

    pretrain_weights = None
    model_param = {
        'encoder': 'tf_efficientnet_b0_ns',
        'sample_rate': 48000,
        'window_size': 512 * 2,  # 512 * 2
        'hop_size': 345 * 2,  # 320
        'mel_bins': 128,  # 60
        'fmin': 20,
        'fmax': 48000 // 2,
        'classes_num': 24
    }

    loss_param = {
        "output_key": "clipwise_output",
        "weights": [1, 0.5],
    }
    n_folds = 5
    fold = 2  # current train fold

    period = 10
    seed = 42
    start_epcoh = 0
    epochs = 200
    lr = 1e-3
    batch_size = 16
    num_workers = 2
    early_stop = 20
    step_scheduler = True
    epoch_scheduler = False

    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = '/content/rfcx_data'
    train_csv = "train_folds.csv"
    test_csv = "test_df.csv"
    sub_csv = f"{data_dir}/sample_submission.csv"
    output_dir = "weights"
    train_data_path = f"{data_dir}/train"
    test_data_path = f"{data_dir}/test"

    # ----------------- Augmentation -----------------
    train_audio_transform = AA.Compose([
        AA.AddGaussianNoise(p=0.5),
        AA.AddGaussianSNR(p=0.5),
        # AA.AddBackgroundNoise("../input/train_audio/", p=1)
        # AA.AddImpulseResponse(p=0.1),
        # AA.AddShortNoises("../input/train_audio/", p=1)
        # AA.FrequencyMask(min_frequency_band=0.0,
        #                  max_frequency_band=0.2,
        #                  p=0.1),
        # AA.TimeMask(min_band_part=0.0, max_band_part=0.2, p=0.1),
        AA.PitchShift(min_semitones=-0.5, max_semitones=0.5, p=0.1),
        # AA.Shift(p=0.1),
        # AA.Normalize(p=0.1),
        # AA.ClippingDistortion(min_percentile_threshold=0,
        #                       max_percentile_threshold=1,
        #                       p=0.05),
        # AA.PolarityInversion(p=0.05),
        AA.Gain(p=0.2)
    ])
