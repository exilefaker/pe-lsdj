    ____    __   ___     __   __    __    ____ __       ____  _    _  ____  ____ ___   __  _    _  ____  __  _ ______   ____
    | _ \~~/  \~~| _\~~~/  \~~| |~~~| |~~~| _|~| |~~~~~~| _|~~\\~~//~~| _ \~| _|~| _\~~||~~|\~~/|~~| _|~~| \||~|__ __|~/  _ \
    ||_|| | /\ | ||_|| | /\ | | |   | |   ||_. | |      ||_.   \\//   ||_|| ||_. ||_|| ||  |.\/.|  ||_.  ||\ |   | |   | | \|
    | __|~| -- |~|  _|~| -- |~| |__~| |__~| -|~| |__~~~~| -|~~~//\\~~~| __|~| -|~|  _|~||~~||\/||~~| -|~~||~\|~~~| |~~~\  \_
    | |   ||  || ||\\  ||  || |   | |   | ||_. |   |    ||_.  //  \\  | |   ||_. ||\\  ||  ||  ||  ||_.  || ||   | |    \_  \
    |_|~~~||~~||~|| \\~||~~||~|___|~|___|~|__|~|___|~~~~|__|~//~~~~\\~|_|~~~|__|~||~\\~||~~||~~||~~|__|~~||~\|~~~|_|~~~~/___/

    .____         .____     ._____       .____
    .____       .________   ._______     .____     PARELLEL EXPERIMENTS LSDJ/
    .____       .___   .    .__  .__     .____     LITTLE SOUND DJ NEURAL NETWORK TOOLKIT
    .____          .___     .__  .__     .____     FOR PYTHON / JAX / NUMPY
    .________   .___.____   ._______    .____      exileFaker 2015 - 2026
    .________    .______    ._____    .____

# Introduction

This is a set of tools for training, exploring, and live-performing with machine learning models using Little Sound DJ (LSDJ) song file (.lsdsng) data.
It consists of:

- A tokenization pipeline, converts raw .lsdsng data into a mix of discrete tokens and continuous input signals
- A hand-designed hierarchical embedding scheme
- A multi-channel transformer-based baseline model with axial attention
- Tools for training and generation
- Tools for streaming generation into a playing LSDJ file with live tuning of gen parameters + recording sessions

# Usage


## Tokenization

Import/tokenize an `.lsdsng` file, do stuff, and render back to bytes:

```python
from pe_lsdj import SongFile

sf = SongFile("data/chiptune.lsdsng")

# ...do stuff to data...

sf.to_lsdsng("output.lsdsng")

```

## Training

Train the default model (too large for most CPUs -- see Colab notebook):

```python
import glob
import jax.random as jr
from pe_lsdj.training import load_songs, train
from pe_lsdj.models import LSDJTransformer

DATA_PATH = "..."
song_fns = glob.glob(DATA_PATH + /*.lsdsng")
songs = load_songs(song_fns)

key = jr.PRNGKey(33)
model_key, train_key = jr.split(key)

model = LSDJTransformer(model_key)

CHECKPOINT_PATH = "..."
model, opt_state = train(
    model, 
    songs,
    checkpoint_path=CHECKPOINT_PATH,
)
```

## Generation

### Streaming generation

Start a live-generation session (see `streaming/README.md` for further documentation):

```bash
python3 -m scripts.lsdj_stream --rom lsdj.gb --sav lsdj.sav --song data/tohou-final.lsdsng --weights data/weights/step_011950.eqx data/weights/step_000700.eqx --web-port 8765 --record session.npz  
```

### Play back a recorded session

```bash
python3 -m scripts.lsdj_replay session.npz --rom lsdj.gb --sav lsdj.sav
```

### Static generation

Generate 2 x 1024 sequencer steps based on a 64-step prompt from an existing track, using a specific model + temperature setting

```bash
python3 generate.py --weights data/weights/step110190.eqx --song data/songs/tohou-final.lsdsng --output data/gen_output --num=steps 1024 --prompt-steps 64 --num-samples 2 --temperature 1.5
```


# Credits

Code by Alex Kiefer / exileFaker and Claude Opus/Sonnet.

Byte parsing and packing builds on pylsdj and uses it for compression/decompression.

Tokenizer and hierarchical embedder designs by Alex. Transformer design brainstormed with Claude and Gemini.
