# music_separation

# Installation

GPU strongly recommended to avoid very long training times.

### Option 1: Direct install (recommended)

System requirements:
* Python 3.6

* [libsndfile](http://mega-nerd.com/libsndfile/) 

* [ffmpeg](https://www.ffmpeg.org/)

Clone the repository:
```
https://github.com/cjzcjz666/music_separation.git
```

Recommended: Create a new virtual environment to install the required Python packages into, then activate the virtual environment:

```
conda create -n waveunet python=3.6
conda activate waveunet
```

Install all the required packages listed in the ``requirements.txt``:

```
pip install -r requirements.txt
```

# Download datasets

To directly use the pre-trained models we provide for download to separate your own songs, now skip directly to the [last section](#test), since the datasets are not needed in that case.

To start training your own models, download the [full MUSDB18HQ dataset](https://sigsep.github.io/datasets/musdb.html) and extract it into a folder of your choice. It should have two subfolders: "test" and "train" as well as a README.md file.

You can of course use your own datasets for training, but for this you would need to modify the code manually, which will not be discussed here. However, we provide a loading function for the normal MUSDB18 dataset as well.

You should use tools to generate 'bass.wav' first.

There are tools: https://github.com/sigsep/sigsep-mus-db

Use command: 
```
musdbconvert path/to/musdb-stems-root path/to/new/musdb-wav-root
```

# Training the models

To train a Wave-U-Net, the basic command to use is

```
python train.py --dataset_dir /PATH/TO/MUSDB18HQ 
```
where the path to MUSDB18_wav dataset needs to be specified, which contains the ``train`` and ``test`` subfolders.

Add more command line parameters as needed:
* ``--cuda`` to activate GPU usage
* ``--checkpoint_dir`` and ``--log_dir`` to specify where checkpoint files and logs are saved/loaded
* ``--load_model checkpoints/model_name/checkpoint_X`` to start training with weights given by a certain checkpoint

For more config options, see ``mytrain.py``.

Training progress can be monitored by using Tensorboard on the respective ``log_dir``.
After training, the model is evaluated on the MUSDB18HQ test set, and SDR/SIR/SAR metrics are reported for all instruments and written into both the Tensorboard, and in more detail also into a ``results.pkl`` file in the ``checkpoint_dir``

# <a name="test"></a> Test trained models on songs!

We provide the default model in a pre-trained form as download so you can separate your own songs right away.

## Downloading our pretrained models

Download our pretrained model [here](https://www.dropbox.com/s/r374hce896g4xlj/models.7z?dl=1).
Extract the archive into the ``checkpoints`` subfolder in this repository, so that you have one subfolder for each model (e.g. ``REPO/checkpoints/waveunet``)

## Run pretrained model

To apply our pretrained model to any of your own songs, simply point to its audio file path using the ``input_path`` parameter:

```
python predict.py --load_model checkpoints/waveunet/model --input "audio_examples/Cristina Vane - So Easy/mix.mp3"
```

* Add ``--cuda `` when using a GPU, it should be much quicker
* Point ``--input`` to the music file you want to separate

By default, output is written where the input music file is located, using the original file name plus the instrument name as output file name. Use ``--output`` to customise the output directory.

To run your own model:
* Point ``--load_model`` to the checkpoint file of the model you are using. If you used non-default hyper-parameters to train your own model, you must specify them here again so the correct model is set up and can receive the weights!
