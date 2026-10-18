# DEPF

DEPF is a few-shot, multi-label image classification pipeline built on COCO2014. It combines label-combination prototypical networks (LC-Protonets) with a cross-modal classifier (CoMC) based on CLIP text and image embeddings.

The repository is organized as a sequence of Jupyter notebooks:

1. `step1_data_preparation.ipynb` creates mutually exclusive base, validation, and novel class splits from COCO2014.
2. `step2_text_generation.ipynb` generates descriptions for novel and validation classes using Qwen Max and ImageNet-style templates.
3. `step3_lc_protonet.ipynb` trains and evaluates LC-Protonets with episodic few-shot learning.
4. `step4_comc_text_only_training.ipynb` trains the CoMC text classifier and runs image inference.
5. `step5_ensemble_evaluation0.1.ipynb` combines LC-Protonets and CoMC predictions and evaluates the ensemble.

`alpha_sweep_analysis.py` evaluates multiple ensemble weights and writes analysis figures and JSON/CSV reports. `run_multiseed_experiment.py` automates the notebook pipeline for multiple random seeds.

## Requirements

Use Python 3.10 or newer with a CUDA-enabled PyTorch installation for practical training runs. The notebooks use the following packages:

- PyTorch and torchvision
- Jupyter and nbconvert
- OpenAI CLIP
- NumPy, scikit-learn, Matplotlib, Seaborn, Pillow, tqdm, and requests

Install the Python packages in an environment appropriate for your CUDA version. For example:

```bash
python -m pip install jupyter nbconvert numpy scikit-learn matplotlib seaborn pillow tqdm requests
python -m pip install torch torchvision
python -m pip install git+https://github.com/openai/CLIP.git
```

The exact PyTorch and CUDA versions should match the target machine. The repository does not include model weights, the COCO dataset, or generated experiment artifacts.

## Dataset Layout

Download COCO2014 from the official COCO distribution and place it under the directory referenced by `COCO_ROOT`. The expected layout is:

```text
coco2014/
|-- annotations/
|   |-- instances_train2014.json
|   `-- instances_val2014.json
|-- train2014/
`-- val2014/
```

The default paths are defined in `.env.example` and can be overridden with environment variables. COCO images and annotations are not redistributed by this repository; follow the COCO license and terms when downloading or sharing them.

## Configuration

Copy `.env.example` to `.env` for local reference, or export the variables directly in your shell. The notebooks read environment variables and do not load `.env` automatically.

```bash
export QIANWEN_API_KEY="your-api-key"
export COCO_ROOT="coco2014"
export DEPF_DATA_DIR="coco2014_data"
export MULTISEED_ROOT="coco2014_multiseed_k5"
export K5_OUTPUT_DIR="coco2014_alpha_sweep_results"
```

`QIANWEN_API_KEY` is required only for API-based description generation in Step 2. Keep the key outside the repository. Never commit `.env`, credentials, checkpoints, or generated data.

## Running the Pipeline

Start Jupyter from the repository root:

```bash
jupyter notebook
```

Run the notebooks in this order. Execute all cells in Step 1 before starting Step 2, and complete Steps 2 through 4 before running Step 5. Each notebook creates its configured output directory when needed.

Step 1 writes the following files under `DEPF_DATA_DIR`:

- `coco_mutually_exclusive_split.json`
- `category_mapping.json`
- `split_report.txt`

Step 2 writes generated descriptions and a text-generation report under `DEPF_DATA_DIR`. API calls may incur usage costs and are controlled by the configured sample counts.

For a reproducible multi-seed run, use:

```bash
python run_multiseed_experiment.py
```

The default protocol uses seeds 42 through 51. You can select a different output root or force re-execution:

```bash
python run_multiseed_experiment.py --output-root coco2014_multiseed_k5 --seeds 42 43 44 45 46 47 48 49 50 51
python run_multiseed_experiment.py --force
```

The automation script expects the project-specific aggregation helper imported as `aggregate_multiseed_results`. If that helper is not present in your checkout, run the notebooks manually or provide the helper module before using the automation script.

## Outputs

With the default configuration, one seed produces:

```text
coco2014_multiseed_k5/seed_<seed>/
|-- lc_protonets/
|   |-- checkpoints/
|   `-- logs/
|-- comc/
`-- evaluation/
```

The ensemble notebook saves `ensemble_evaluation_results.json` and comparison figures. The alpha sweep script writes its reports under `K5_OUTPUT_DIR`:

```bash
python alpha_sweep_analysis.py
```

The script can use the optional `K5_*` variables in `.env.example` to select dataset paths, checkpoints, an output directory, and an episode manifest.

## Reproducibility

The notebooks read `EXPERIMENT_SEED` when it is provided. `run_multiseed_experiment.py` sets the seed for each run and records progress and run metadata under the seed-specific output directory. Training remains dependent on the installed PyTorch, torchvision, CLIP, CUDA, and hardware versions.

## Privacy and Repository Hygiene

- Store API keys only in environment variables or local `.env` files.
- Do not commit COCO images, annotations, model checkpoints, generated descriptions, logs, or experiment outputs.
- Review notebook outputs before publishing an executed notebook; outputs can contain local paths, metrics, or service responses.
- Check `git diff` and secret-scanning results before making the repository public.

## License

No project license file is included. Add a license before public distribution and verify the separate terms for COCO, CLIP, PyTorch, and any text-generation service used with this project.
