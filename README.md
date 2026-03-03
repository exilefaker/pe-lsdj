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

This is a set of tools for training machine learning models on Little Sound DJ (LSDJ) song file (.lsdsng) data.
It consists of:

- A tokenization pipeline, converts raw .lsdsng data into a mix of discrete tokens and continuous input signals
- A hand-designed hierarchical embedding scheme
- A multi-channel transformer-based baseline model with axial attention
- Tools for training and generation

## Credits

Code by Alex Kiefer / exileFaker and Claude Opus/Sonnet.

Byte parsing and packing builds on pylsdj and uses it for compression/decompression.

Tokenizer and hierarchical embedder designs by Alex. Transformer design brainstormed with Claude and Gemini.
