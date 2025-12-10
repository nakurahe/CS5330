import os
import torch
import random
import re
import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Visual Jigsaw model
model_id = "craigwu/visual_jigsaw_image_7B"

# Load model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    dtype="auto",  
    device_map="auto",   
)

# Load Processor
processor = AutoProcessor.from_pretrained(model_id)
print("Model loaded successfully")


def run_one_jigsaw(image_paths, puzzle_name=""):
    """
    Run visual jigsaw model on a single 3x3 puzzle and return:
      - acc: position-level accuracy (0~1)
      - pred_perm: model's predicted permutation (tile sequence)
      - gt_perm: ground truth permutation (our randomly sampled π)
    image_paths: list of 9 strings, patch paths in correct raster order (top-left to bottom-right)
    """
    assert len(image_paths) == 9, "assert K=9"

    K = len(image_paths)  # 9
    
    # 1. Sample permutation π: {0..K-1} → {0..K-1}
    # π maps an original position index i to its shuffled position π(i)
    pi = list(range(0, K))   # [0..8]
    random.shuffle(pi)
    mapping_old2new = {j: pi[j] for j in range(K)}   # 0-based: j=0..8, old=0..8
    print("\n=== Original to new ===")
    for old_pos, new_pos in mapping_old2new.items():
        print(f"Original position {old_pos} → New position {new_pos}")
    
    # 2. Compute π^{-1} to construct the shuffled sequence P_π = [p_{π^{-1}(0)},...,p_{π^{-1}(K-1)}]
    pi_inv = [0] * K         # pi_inv[j] = π^{-1}(j) where j is in the shuffled sequence
    for i in range(K):       # i = original position
        j = pi[i]            # j = π(i) = position in shuffled sequence
        pi_inv[j] = i        # pi_inv[j] = original position i

    mapping_new2old = {j: pi_inv[j] for j in range(K)}
    print("\n=== New to original ===")
    for new_pos, old_pos in enumerate(pi_inv):
        print(f"New position {new_pos} → Original position {old_pos}")
    
    # 3. Construct shuffled tiles (current order is P_π)
    # j-th element corresponds to the patch originally at position π^{-1}(j)
    shuffled_paths = [image_paths[pi_inv[j]] for j in range(K)]
    
    print("\n=== Shuffled paths list (corresponding to Tile 1..9) ===")
    for j, p in enumerate(shuffled_paths, start=1):
        print(f"Tile {j}: {p}")

    # 4. Construct prompt
    intro_text = (
        "You are given nine shuffled image tiles that were created by slicing one image into a 3x3 grid.\n"
        "Here are the tiles, each tagged with an index reflecting the current (shuffled) order in which they are shown:\n\n"
    )

    task_text = (
        "\nTask:\n"
        "Mentally reassemble the original image, arranging the tiles into the correct 3x3 layout and provide "
        "the tile indices in raster-scan order (left-to-right, top-to-bottom), separated by commas.\n"
        "Answer format example:\n"
        "5, 1, 3, 7, 9, 2, 4, 8, 6\n"
        "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        "The reasoning process MUST BE enclosed within <think></think> tags.\n"
        "The final answer MUST BE put within <answer></answer> tags."
    )

    content = []
    content.append({"type": "text", "text": intro_text})

    # Tile 1..9：order = shuffled_paths
    for i, path in enumerate(shuffled_paths, start=1):
        content.append({"type": "text", "text": f"Tile {i}: "})
        content.append({"type": "image", "image": f"file://{path}"})
        content.append({"type": "text", "text": "\n"})

    content.append({"type": "text", "text": task_text})

    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": content,
        },
    ]

    # 5. Prepare input using chat_template + process_vision_info
    prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    image_inputs, video_inputs = process_vision_info(conversation)

    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    ).to(model.device)

    # 6. Inference
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1028,
            temperature=0.2,
        )

    # Only keep newly generated part, remove prompt
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    full_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print("\n===== Full model output =====\n")
    print(full_text)

    # 7. Extract answer from <answer>...</answer>
    m = re.search(r"<answer>(.*?)</answer>", full_text, re.DOTALL)
    if m:
        answer_text = m.group(1)
    else:
        answer_text = full_text

    print("\n===== Answer section =====\n")
    print(answer_text)

    # 8. Parse into permutation ŷπ(0..K-1)（tile indices）
    nums = re.findall(r"\d+", answer_text)
    pred_perm = [int(x) - 1 for x in nums if 1 <= int(x) <= K][:K] 
    print("\nPredicted permutation π̂(0..K-1)：", pred_perm)

    # 9. Compare with ground truth π(0..K-1)
    gt_perm = pi
    print("Ground truth permutation π(0..K-1)：", gt_perm)

    correct = sum(p == g for p, g in zip(pred_perm, gt_perm))
    acc = correct / K
    print(f"\n[{puzzle_name}] Position-level accuracy: {correct}/{K} = {acc:.3f}")

    if pred_perm == gt_perm:
        print("✅ Perfect match")
    else:
        print("❌ Partial match")

    return acc, pred_perm, gt_perm, mapping_new2old, mapping_old2new


# ============Main==============
results = []

for puzzle_num in range(45):
    folder_name = f"test_puzzle_{puzzle_num:03d}"
    folder_path = f"./dataset_transparent/test_set_transparent/{folder_name}/pieces"

    image_paths = [os.path.join(folder_path, f"piece_{i}.png") for i in range(9)]

    acc, pred_perm, gt_perm, mapping_new2old, mapping_old2new = run_one_jigsaw(image_paths)

    # Save to the list
    results.append({
        "puzzle_id": folder_name,
        "accuracy": acc,
        "pred_perm": pred_perm,
        "gt_perm": gt_perm,
        "mapping_new2old": str(mapping_new2old),  
        "mapping_old2new": str(mapping_old2new)   
    })

# Save as csv
results_df = pd.DataFrame(results)
results_df.to_csv("./puzzle_transparent_results_visual_jigsaw_image_7B.csv", index=False)

