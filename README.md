# Controllable Lyrics-to-Melody Generation

This repository is the official implementation of [Controllable Lyrics-to-Melody Generation](https://doi.org/10.1007/s00521-023-08728-1). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

To run the codes, first go to the code directory:

```setup
cd code/tbc_lstm_gan_memofu
```

## Training

To train the proposed model in the paper, run this command:

```train
python train_memofu_style_seq.py
```

Additionally, to train other ablation models mentioned in the papers, run the following commands:
```train
python train_memofu_style_seq.py
python train_memofu_style.py
python train_memofu_seq.py
python train_memofu.py
```

## Objective Evaluation

To evaluate the aforementioned trained models with objective evaluation, run:

```eval
evaluate_memofu.py
```

## Controllability Evaluation

To evaluate the controllability of the proposed model, run:

```controllability
controllability_experiment.py
```

## Melody Generation

To generate melodies from the lyrics with designated Reference Style Embeddings (RSE), run `generate_memofu.py` with necessary parameters.
For example:

```generate
python generate.py --SYLL_LYRICS "When I am king you will be first a gainst the wall" --WORD_LYRICS "When I am king you will be first against against the wall" 
--STYLE_PITCH [0.2, 0.2, 0.2] --STYLE_Duration [0.2, 0.2, 0.2] --STYLE_REST [0.2, 0.2, 0.2]
```

By default, the pre-trained checkpoint is located at `checkpoints/adv_train_tbc_lstm_gan_memofu_style_seq`, which is contained in this repository.
To use a different directory, pass the parameter `--CKPT_PATH`.


[//]: # (## Pre-trained Models)

[//]: # ()
[//]: # (You can download pretrained models here:)

[//]: # ()
[//]: # (- [My awesome model]&#40;https://drive.google.com/mymodel.pth&#41; trained on ImageNet using parameters x,y,z. )

[//]: # ()
[//]: # (>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained &#40;if applicable&#41;.  Alternatively you can have an additional column in your results table with a link to the models.)

[//]: # (## Results)

[//]: # ()
[//]: # (Our model achieves the following performance on :)

[//]: # ()
[//]: # (### [Image Classification on ImageNet]&#40;https://paperswithcode.com/sota/image-classification-on-imagenet&#41;)

[//]: # ()
[//]: # (| Model name         | Top 1 Accuracy  | Top 5 Accuracy |)

[//]: # (| ------------------ |---------------- | -------------- |)

[//]: # (| My awesome model   |     85%         |      95%       |)

[//]: # ()
[//]: # (>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. )


[//]: # (## Contributing)

[//]: # ()
[//]: # (>📋  Pick a licence and describe how to contribute to your code repository. )