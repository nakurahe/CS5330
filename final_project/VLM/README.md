# VLM Jigsaw Puzzle Benchmark

This folder evaluates several Vision-Language Models (VLMs) on a 3×3 jigsaw puzzle reconstruction task.

Models evaluated:

- **InternVL3-8B** (`OpenGVLab/InternVL3-8B`)
- **Qwen3-VL-8B-Instruct** (`Qwen/Qwen3-VL-8B-Instruct`)
- **Visual-Jigsaw-Image-7B** (`craigwu/visual_jigsaw_image_7B`)

Each puzzle image is cut into a 3×3 grid (9 tiles). The model is given the shuffled tiles and asked to output the correct raster order (left-to-right, top-to-bottom). We then compute several accuracy metrics and compare models.

---

## Folder / File Overview
```text
.
├── infer_InternVL3.py
├── infer_qwen.py
├── infer_visual_jigsaw_image_7B.py
│
├── analyze_jigsaw_vlm_min.py
│
├── puzzle_transparent_results_InternVL3_8B.csv
├── puzzle_transparent_results_Qwen3-VL-8B-Instruct.csv
├── puzzle_transparent_results_visual_jigsaw_image_7B.csv
│
├── VLM_accuracy_summary.png
├── VLM_piece_type_performance.png
│
├── requirements.txt
└── dataset_transparent/test_set_transparent/
      ├── test_puzzle_000/pieces/piece_0.png ... piece_8.png
      ├── test_puzzle_001/pieces/...
      └── ...
```


## Environment Setup

This project assumes:

- Python 3.9+
- A GPU with CUDA support (recommended for running large VLMs)

Install dependencies:

```bash
pip install -r requirements.txt
```


## Running Inference

Run each model’s script:

```bash
python infer_InternVL3.py
python infer_qwen.py
python infer_visual_jigsaw_image_7B.py
```

Each script:
- Loads tiles from dataset_transparent/test_set_transparent/test_puzzle_XXX/pieces/
- Samples a random permutation of tiles
- Prompts the model to reconstruct the puzzle
- Saves a CSV with the following fields:
	- puzzle_id
	- accuracy
	- pred_perm
	- gt_perm
	- mapping_new2old
	- mapping_old2new

## Running Analysis

After all CSVs are generated:

```bash
python analyze_jigsaw_vlm.py
```

This script:
- Loads the three result files
- Computes, for each model:
	- Mean / std Direct Accuracy
	- Mean / std Neighbor Accuracy
	- Valid Permutation Rate
	- Computes Piece-Type Accuracy (Corner / Edge / Interior, mean ± std)
- Saves two plots:
	- VLM_accuracy_summary.png
	- VLM_piece_type_performance.png

## Metrics

- Direct Accuracy

	Exact-match accuracy between predicted and ground-truth permutations (length 9).

- Neighbor Accuracy

    Piece-adjacency accuracy: fraction of ground-truth adjacent tile pairs that remain adjacent (order-insensitive) in the predicted board.

- Valid Permutation Rate

    Percentage of model outputs that form a valid permutation of indices 0–8 (no duplicates / missing tiles).

- Piece-Type Accuracy

    Per-puzzle accuracy, grouped by tile type:

    - Corner: positions {0, 2, 6, 8}
    - Edge: positions {1, 3, 5, 7}
	- Interior: position {4}
