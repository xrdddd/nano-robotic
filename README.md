
# nano-robotic
nano-robotic is a project aiming to train high efficient Vision-Language-Action model and deploy on edge robotic devices. It support the following:
- **Multiple modalities**: Train a model with text, image-captions, and robotics data without any external dependencies. 
- **Multi-node training**: nano-robotic supports [FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) and streams datasets with [WebDatasets](https://github.com/webdataset/webdataset). Multi-GPU training works well locally with `torchrun`, in the future on large clusters with AWS SageMaker (TODO)
- **Hugging Face support**: Modules can either be loaded using the native PyTorch implementation, or loaded using pre-trained weights from Hugging Face. This allows users to develop on top of state-of-the-art model releases for LLMs, VLMs, CLIP models, etc. (TODO)

## Contents
- [Installation](#installation)
- [Contributing Guidelines](#contributing-guidelines)
- [Quickstart](#quickstart)
- [Deployment Examples](#deployment-examples)
- [Repo Structure and Implementation](#repo-structure-and-implementation)
  <ol type="1">
    <li><a href="#1-paramargument-structure">Param</a></li>
    <li><a href="#2-data">data</a></li>
    <li><a href="#3-modules">modules</a></li>
    <li><a href="#4-training">Training</a></li>
    <li><a href="#5-evaluating">evaluating</a></li>
    <li><a href="#6-dataloading-pipeline">dataloading Pipeline</a></li>
    <li><a href="#7-logging">Logging</a></li>
    <li><a href="#8-tests">Tests</a></li>
  </ol>
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Installation
We recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/) for environment management. Please follow the uv documentation for installation. Once uv is installed, create a Python 3.12 virtual environment using uv and install the project dependencies with the command below:
```bash
uv sync
uv pip install -e .
```
To activate the virtual env you can then run `source .venv/bin/activate`

## Contributing Guidelines
Please see [CONTRIBUTING.md](CONTRIBUTING.md)

## Quickstart
The train entrypoint is `nano_robotic/train.py`.

An example command is something like this:
```bash
.venv/bin/torchrun --nproc_per_node=8 --nnodes=1 nano_robotic/train.py \
--config "include config/vlm_1b.yaml" \
--processor google/paligemma-3b-pt-224 \
--img_num_tokens 256 \
--per_gpu_batch_size 2 \
--global_batch_size 64 \
```

## Deployment
We use lightweight evaluation utilities and gRPC policy server templates.

## Repo Structure and Implementation
The sections below highlight several key design choices and functionalities of the repo.

### 1. Param
We use [draccus](https://github.com/dlwh/draccus) for argument parsing.

### 2. Data
Data are stored in shards. Each shard is a tar file. Within each tar file, each sample is distinguished by its unique prefix.
The structure of the directory is as follows:

```
dataset_name/
├── manifest.jsonl
├── shard_00000000.tar
│   ├── unique_name_or_hash_1_image1.jpg
│   ├── unique_name_or_hash_1_image2.jpg
│   ├── unique_name_or_hash_1_image3.jpg
│   ├── unique_name_or_hash_1_meta.json
│   ├── unique_name_or_hash_1_caption.json
│   ├── unique_name_or_hash_1_actions.npz
│   ├── unique_name_or_hash_1_otherstuff.json
│   ├── unique_name_or_hash_2_...json
│   ├── unique_name_or_hash_3_...json
├── shard_00000001.tar
│   ├── unique_name_or_hash_100_...json
├── shard_00000002.tar
├── shard_00000003.tar
└── ...
```

In the directory above, the `unique_name_or_hash_1_...` files make up the first sample, the `unique_name_or_hash_2_...` files make up the second sample, and so on. Each tar file can have hundreds or thousands of samples.

The `manifest.jsonl` provides an overview of the tar files as follows:
```
{"shard": "00000000", "num_sequences": 4518}
{"shard": "00000001", "num_sequences": 4617}
{"shard": "00000002", "num_sequences": 4625}
{"shard": "00000003", "num_sequences": 4701}
```

The dataset can be either local or on S3. An example is `s3://your-bucket/your-path/datasets/datacompdr_1b/`.

During dataloading, the code will read `manifest.jsonl`, shuffle the rows, then select the appropriate number of tar files for the given number of training steps.

### 3. Modules

### 4. Training

### 5. Evaluating

### 6. Dataloading Pipeline
We use [webdatasets](https://github.com/webdataset/webdataset) to load the data. Each modality (e.g., image+caption, interleaved, image+actions) has its own pipeline where all the processing steps are defined at a high-level. This involves steps like untarring, shuffling, batching, etc. 

### 7. Logging
Logging is done automatically to [wandb](wandb.ai).

### 8. Tests
Tests are implemented with [pytest](https://docs.pytest.org/en/stable/). To run these tests, you can call
```
uv run pytest tests
```
To run more verbose tests, you can add `-v` for detailed per-test breakdowns and `-s` to display print statement outputs.

## Citation

## Acknowledgements
Inspired by nanochat project and other VLA open source projects.
