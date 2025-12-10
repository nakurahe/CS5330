import os
import re
import random
import torch
import pandas as pd

from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from transformers import AutoModel, AutoTokenizer

# 1. Load model: InternVL3-8B
MODEL_PATH = "OpenGVLab/InternVL3-8B"

print("Loading InternVL3-8B model...")

model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True
).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    use_fast=False
)

print("Model loaded.")

# 2. Tile-level Image Preprocessing (for pre-cut 3x3 patches)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_tile_transform(input_size=448):
    """
    Basic transforms for each jigsaw tile:
    - Convert to RGB
    - Resize to (input_size, input_size)
    - ToTensor
    - Normalize using ImageNet mean/std
    """
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform

tile_transform = build_tile_transform(input_size=448)


def load_tile_image(image_file, input_size=448):
    """
    Load a cut tile and return a tensor of shape (1, 3, H, W).
    """
    image = Image.open(image_file).convert("RGB")
    tensor = tile_transform(image)      # (3, H, W)
    tensor = tensor.unsqueeze(0)        # (1, 3, H, W)
    return tensor


# 3. Run one jigsaw
def run_one_jigsaw(image_paths, puzzle_name=""):
    """
    Run InternVL3-8B on a single 3x3 puzzle and return:
      - acc: position-level accuracy (0–1)
      - pred_perm: model-predicted permutation (0-based)
      - gt_perm: the randomly sampled ground-truth permutation π(0..8)
      - mapping_new2old: new position → original position (0-based)
      - mapping_old2new: original position → new position (0-based)

    image_paths: list of 9 paths in the correct original raster order
                 (left-to-right, top-to-bottom)
    """
    assert len(image_paths) == 9, "assert K=9"
    K = len(image_paths)  # 9

    # 1. Randomly sample a permutation π: {0..K-1} → {0..K-1}
    pi = list(range(K))
    random.shuffle(pi)
    mapping_old2new = {old: pi[old] for old in range(K)} 

    print(f"\n=== [{puzzle_name}] π(0..K-1) (original → new position) ===")
    for old_pos, new_pos in mapping_old2new.items():
        print(f"Original position {old_pos} → New position {new_pos}")

    # 2. Compute the inverse permutation π^{-1}
    pi_inv = [0] * K  # pi_inv[new] = old
    for old in range(K):
        new = pi[old]
        pi_inv[new] = old

    print("\n=== π^{-1}(0..K-1) (new → original position) ===")
    for new_pos, old_pos in enumerate(pi_inv):
        print(f"New position {new_pos} → Original position {old_pos}")

    # 3. Build shuffled path list using π^{-1}
    shuffled_paths = [image_paths[pi_inv[j]] for j in range(K)]

    print("\n=== Shuffled paths list (corresponding to Tile 1..9) ===")
    for j, p in enumerate(shuffled_paths, start=1):
        print(f"Tile {j}: {p}")

    mapping_new2old = {new: pi_inv[new] for new in range(K)}
    print("\nnew_idx(0-based) -> old_idx(0-based):", mapping_new2old)

    # 4. Load each tile and stack into InternVL3 pixel_values
    tile_tensors = []
    for path in shuffled_paths:
        t = load_tile_image(path).to(torch.bfloat16).cuda()  # (1, 3, H, W)
        tile_tensors.append(t)
    pixel_values = torch.cat(tile_tensors, dim=0)  # (9, 3, H, W)

    # 5. Build the textual instruction for InternVL3
    tile_desc_lines = []
    for i in range(1, K + 1):
        tile_desc_lines.append(
            f"Tile {i}: this corresponds to image #{i} in the input image list.\n"
        )
    tile_desc_text = "".join(tile_desc_lines)

    question = (
        f"{tile_desc_text}\n"
        "These nine tiles were created by cutting one original image into a 3x3 grid.\n"
        "You are given the nine tiles in a shuffled order (Tile 1 to Tile 9 as listed above).\n"
        "Your task:\n"
        "1. Mentally reassemble the original image.\n"
        "2. Output the tile indices (1-9) in the correct 3x3 raster order of the original image "
        "(left-to-right, top-to-bottom).\n"
        "3. Use a comma-separated list.\n"
        "Example: 5, 1, 3, 7, 9, 2, 4, 8, 6\n\n"
        "Important:\n"
        "- Only output the numbers in one line.\n"
        "- Wrap the numbers inside <answer> and </answer> tags.\n"
        "For example: <answer>5, 1, 3, 7, 9, 2, 4, 8, 6</answer>\n"
    )

    generation_config = dict(
        max_new_tokens=256,
        do_sample=False,
        temperature=0.0,
    )

    # 6. Call InternVL3-8B for multi-image chat inference
    response = model.chat(
        tokenizer=tokenizer,
        pixel_values=pixel_values,   # (9, 3, H, W)
        question=question,
        generation_config=generation_config,
    )

    full_text = response
    print("\n===== Full model output =====\n")
    print(full_text)

    # 7. Extract <answer>...</answer>
    m = re.search(r"<answer>(.*?)</answer>", full_text, re.DOTALL | re.IGNORECASE)
    if m:
        answer_text = m.group(1)
    else:
        answer_text = full_text

    print("\n===== Answer section =====\n")
    print(answer_text)

    # 8. Parse output numbers as 0-based permutation
    nums = re.findall(r"\d+", answer_text)
    pred_perm = [int(x) - 1 for x in nums if 1 <= int(x) <= K][:K]
    print("\nPredicted permutation π̂(0..K-1): ", pred_perm)

    # 9. Compute accuracy vs ground truth π
    gt_perm = pi
    print("Ground truth permutation π(0..K-1): ", gt_perm)

    if len(pred_perm) != K:
        print(f"⚠️ Predicted length {len(pred_perm)} != {K}, missing positions treated as incorrect.")

    correct = sum(
        (pred_perm[i] == gt_perm[i]) if i < len(pred_perm) else False
        for i in range(K)
    )
    acc = correct / K
    print(f"\n[{puzzle_name}] Position-level accuracy: {correct}/{K} = {acc:.3f}")

    if pred_perm == gt_perm:
        print("✅ Perfect match")
    else:
        print("❌ Partial match")

    return acc, pred_perm, gt_perm, mapping_new2old, mapping_old2new


# 4. Run all puzzles: puzzle_000 ~ puzzle_044
def main():
    base_folder = "./dataset_transparent/test_set_transparent"
    start_id = 0
    end_id = 45   # puzzle_000 to puzzle_044

    results = []

    for puzzle_num in range(start_id, end_id):
        folder_name = f"test_puzzle_{puzzle_num:03d}"
        folder_path = os.path.join(base_folder, folder_name, "pieces")

        image_paths = [os.path.join(folder_path, f"piece_{i}.png") for i in range(9)]

        # Check if all files exist
        if not all(os.path.exists(p) for p in image_paths):
            print(f"[WARN] Missing pieces in {folder_name}, skip.")
            continue

        print(f"\n\n########## Running {folder_name} ##########")
        acc, pred_perm, gt_perm, mapping_new2old, mapping_old2new = run_one_jigsaw(
            image_paths, puzzle_name=folder_name
        )

        results.append({
            "puzzle_id": folder_name,
            "accuracy": acc,
            "pred_perm": pred_perm,
            "gt_perm": gt_perm,
            "mapping_new2old": str(mapping_new2old),
            "mapping_old2new": str(mapping_old2new),
        })

    # save to csv
    if results:
        results_df = pd.DataFrame(results)
        out_csv = "./puzzle_transparent_results_InternVL3_8B.csv"
        results_df.to_csv(out_csv, index=False)
        print(f"\nAll done. Results saved to: {out_csv}")
    else:
        print("No puzzle processed, no CSV generated.")


if __name__ == "__main__":
    main()
