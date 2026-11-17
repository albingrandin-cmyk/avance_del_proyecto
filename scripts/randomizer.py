import json
from pathlib import Path
import numpy as np
from PIL import Image


def generate_puzzle(
    image_path: str,
    R: int,
    C: int,
    d: int,
    seed: int,
    input_dir: str,
    gt_dir: str,
    case_name: str = "case_01",
) -> None:
    """
    Generating the randomized image puzzle in a separate file and the ground truth image in other.
    First we need to centralize the image. The image (H_orig, W_orig) is most likely not evenly divisible
    by R x C we need to do integer division to get equally sized (p_h, p_w). To get the image centrum we
    crop an equal amount of pixels for the height and the width. After centralizing we divide the image
    into R x C puzzle pieces. We do this by enumarating the bits and creating indices k = [0,..,N-1], where
    N  = R x C. The rows r =[k/C] and columns c = k mod C. Every bit (patches) is created as a submatrix
    of the cropped image (patches = [cropped[r * s_h + q : r * s_h + q + p_h, c * s_w + q : c * s_w + q + p_w]).
    We than create a deterministic forward permutation where the original image can be recreated with inv_perm.
    The original image is saved in a separate map called gt_map.

    Parameters:
        image_path(str): The image we want to randomize into a puzzle
        R(int): The R rows the image is divided into
        C(int): The C columns the image is divided into
        d(int): The space of size d filled with zeroes
        seed(int): The randomizer seed for reproducibility
        input_dir(str): The input directory where we save the mixed image
        gt_dir(str): The grount truth directory where we save the original image
        case_name(str): name of the puzzle
    
        Return:
            None
    """
    if not (4 <= R <= 9 and 4 <= C <= 9):
        raise ValueError("R and C must be between 4 and 9.")
    if d not in {0, 1, 3, 5}:
        raise ValueError("Interleave d needs to be in {0, 1, 3, 5}.")

    # Reading in the 8-bit RGB-image
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    H_orig, W_orig, _ = arr.shape

    # Cells size, centralizecropping and controlling useful bit size
    s_h = H_orig // R
    s_w = W_orig // C
    p_h = s_h - d
    p_w = s_w - d

    if not (64 <= p_h <= 512 and 64 <= p_w <= 512):
        raise ValueError(
            f"Bit size ({p_h}x{p_w}) is outside the allowed [64, 512] pixles."
        )

    H_crop = R * s_h
    W_crop = C * s_w
    y0 = (H_orig - H_crop) // 2
    x0 = (W_orig - W_crop) // 2
    cropped = arr[y0 : y0 + H_crop, x0 : x0 + W_crop, :]

    # Extracting R x C useful bits (q = floor(d / 2) pixles in every cell)
    q = d // 2
    patches = [
        cropped[
            r * s_h + q : r * s_h + q + p_h,
            c * s_w + q : c * s_w + q + p_w,
            :,
        ]
        for r in range(R)
        for c in range(C)
     ]

    # Creating reproducible permutations with a seed
    N = R * C
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)  # perm[new_position] = original_index

    # Creating mixzed image with d sized space filled with zeroes
    scrambled = np.zeros((H_crop, W_crop, 3), dtype=np.uint8)

    for new_idx, orig_idx in enumerate(perm):
        r_new, c_new = divmod(new_idx, C)
        ys = r_new * s_h + q
        xs = c_new * s_w + q
        scrambled[ys : ys + p_h, xs : xs + p_w, :] = patches[orig_idx]

    # Saving mixed indata image and ground truth in separate maps
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    Path(gt_dir).mkdir(parents=True, exist_ok=True)

    # This image is sent to ensamblador
    Image.fromarray(scrambled).save(Path(input_dir) / f"{case_name}.png")

    # Ground truth saves isolated from the mixed image for later evaluation
    Image.fromarray(cropped).save(Path(gt_dir) / f"{case_name}_ref.png")
    gt_payload = {
        "R": R,
        "C": C,
        "d": d,
        "seed": seed,
        "patch_height": p_h,
        "patch_width": p_w,
        "perm": perm.tolist(),
    }
    with open(Path(gt_dir) / f"{case_name}_gt.json", "w", encoding="utf-8") as f:
        json.dump(gt_payload, f, indent=2)
