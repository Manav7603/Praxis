# Praxis Core
Praxis is an open AI research laboratory.
Build, train, and deploy custom transformer models from scratch.

Our mission is to understand intelligence deeply,
build it openly,
and share it freely.

## Project structure

```
praxis-core/
│
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
│
├── configs/                  # YAML/JSON configs (model, training, data, inference)
│   ├── model/
│   ├── training/
│   ├── data/
│   └── inference/
│
├── praxis/                   # Main library
│   ├── config/               # Config loading & dataclasses
│   ├── tokenizer/            # Tokenizer training & encoding
│   ├── datasets/             # Dataset loaders & preprocessing
│   ├── embeddings/           # Token & positional embeddings
│   ├── attention/            # Attention mechanisms (MHA, GQA, etc.)
│   ├── transformer/          # Transformer blocks & layers
│   ├── models/               # Full model architectures
│   ├── training/             # Training loops, checkpointing
│   │   ├── optimizers/
│   │   └── schedulers/
│   ├── inference/            # Generation & sampling
│   ├── evaluation/           # Metrics & benchmarks
│   ├── losses/               # Loss functions
│   ├── distributed/          # Multi-GPU / distributed training
│   ├── export/               # Model serialization (ONNX, etc.)
│   ├── utils/                # Logging, registry, helpers
│   └── visualization/        # Training curves, attention maps
│
├── scripts/                  # CLI entry points (train, eval, infer)
├── tests/
│   ├── unit/
│   └── integration/
├── notebooks/                # Experiments & exploration
├── examples/                 # End-to-end usage examples
│   ├── train_from_scratch/
│   ├── inference_demo/
│   └── preprocess_data/
└── docs/                     # Documentation
```

## Setup

```bash
pip install -e ".[dev]"
```
