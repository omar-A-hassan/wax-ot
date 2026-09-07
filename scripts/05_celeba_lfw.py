"""Script 05 - Section VIII: differences between two large face datasets.

Embeds capped random samples of LFW and CelebA with a pretrained CLIP ViT-B/32.
The embeddings are cached, and the script uses MPS when it is available. It
models the shift by the Wasserstein distance with p = q = 2, then characterizes
that shift with U-WaX.

The characterization has two parts: a one-dimensional main transport trend
(r = 2), and a set of one-dimensional sub-shift subspaces (r = 3). Each
subspace is described by the English words whose CLIP embeddings align most
closely with it.

Downloads and caches: CLIP weights (~340 MB), LFW faces (~210 MB) and, unless
``--celeba-zip`` points to a local archive, the CelebA aligned images
(~1.3 GB).  Sub-samples are capped at ``--n`` faces per dataset (default 500).
Device is auto-selected: ``mps`` if available, otherwise ``cpu`` (override
with ``--device``).  CLIP inference uses ``float16`` autocast on MPS with a
``float32`` host transfer so WaX numerics are unchanged.
"""

from __future__ import annotations

import argparse
import io
import os
import zipfile

import numpy as np

from wax import attribute, eigen_subspace, exact, uwax_attribute, uwax_search
from wax.datasets import pkg_data_path
from wax.plotting import fig_face_subspace_words
from wax.words import load_words

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figs")
os.makedirs(FIGDIR, exist_ok=True)

CELEBA_URLS = [
    "https://cseweb.ucsd.edu/~weijian/static/datasets/celeba/img_align_celeba.zip",
    "https://ftp.mi.fu-berlin.de/pub/cmb-data/celeba/img_align_celeba.zip",
]


def get_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def load_clip(device: str = "cpu"):
    import open_clip

    # open_clip handles device internally; keep string form for .to() below
    model, _, transform = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device
    )
    model.eval()
    try:
        model = model.to(device)
    except Exception:
        pass
    # optional compile for fused kernels (no result change, ~10-20% on MPS)
    try:
        import torch

        if hasattr(torch, "compile") and device in ("mps", "cuda"):
            model = torch.compile(model, mode="reduce-overhead")  # type: ignore[assignment]
    except Exception:
        pass
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, transform, tokenizer


def _embed(images, transform, model, device: str = "cpu", step: int = 32):
    import torch

    embs: list[np.ndarray] = []
    use_autocast = device in ("mps", "cuda")
    for i in range(0, len(images), step):
        batch = images[i : i + step]
        # single stack instead of cat-loop (O(step) vs O(step^2) allocs)
        tensors = torch.stack([torch.as_tensor(transform(im)) for im in batch], dim=0)
        tensors = tensors.to(device, non_blocking=True)
        with torch.inference_mode():
            if use_autocast:
                with torch.autocast(device_type=device, dtype=torch.float16):
                    t = model.encode_image(tensors)
                    t = t.float()
                    t = t / t.norm(dim=1, keepdim=True)
            else:
                t = model.encode_image(tensors.float())
                t = t / t.norm(dim=1, keepdim=True)
        embs.append(t.float().cpu().numpy())
    return np.concatenate(embs).astype(np.float32)


def load_lfw(n, seed):
    from sklearn.datasets import fetch_lfw_people

    faces = fetch_lfw_people(color=True, funneled=True, resize=1.0, slice_=None)
    imgs = np.asarray(faces.images)  # (K, H, W, 3) uint8
    mask = faces.target != 0
    imgs = imgs[mask]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(imgs), size=min(n, len(imgs)), replace=False)
    from PIL import Image

    return [Image.fromarray(im.astype(np.uint8)) for im in imgs[idx]]


def load_celeba(n, seed, celeba_zip):
    import urllib.request

    from PIL import Image

    cache = pkg_data_path("cache")
    os.makedirs(cache, exist_ok=True)
    if celeba_zip is None:
        path = os.path.join(cache, "img_align_celeba.zip")
        if not (os.path.exists(path) and os.path.getsize(path) > 1e8):
            print("[05] downloading CelebA (1.3 GB, disk only) ...")
            ok = False
            for url in CELEBA_URLS:
                try:
                    urllib.request.urlretrieve(url, path)
                    ok = True
                    break
                except Exception as exc:  # noqa: BLE001
                    print(f"[05] mirror failed ({url}): {exc}")
            if not ok:
                raise RuntimeError(
                    "CelebA could not be downloaded; pass --celeba-zip with a local copy."
                )
        celeba_zip = path
    rng = np.random.default_rng(seed)
    with zipfile.ZipFile(celeba_zip) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".jpg")]
        if len(members) < n:
            raise RuntimeError(f"CelebA archive has only {len(members)} images")
        chosen = rng.choice(members, size=n, replace=False)
        imgs = []
        for name in chosen:
            with zf.open(name) as f:
                imgs.append(Image.open(io.BytesIO(f.read())).convert("RGB"))
    return imgs


def embed_or_load(key, n, seed, transform, model, device, step, loader, cache_path):
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        if key in data and len(data[key]) >= n:
            return data[key][:n].astype(np.float32)
    print(f"\n[05] embedding {key} ({n} images) on {device} ...", flush=True)
    imgs = loader(n, seed)
    embs = _embed(imgs, transform, model, device=device, step=step)
    extra = {}
    if os.path.exists(cache_path):
        old = np.load(cache_path)
        extra = {k: old[k] for k in old.files}
    extra[key] = embs  # type: ignore[assignment]
    np.savez(cache_path, **extra)
    return embs


def embed_words(words, model, tokenizer, device, cache_path):
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        if "text" in data and len(data["words"]) == len(words):
            return data["text"].astype(np.float32), words
    import torch

    print(f"[05] embedding {len(words)} words on {device} ...", flush=True)
    batch = 512 if device in ("mps", "cuda") else 256
    out: list[np.ndarray] = []
    use_autocast = device in ("mps", "cuda")
    with torch.inference_mode():
        for i in range(0, len(words), batch):
            tok = tokenizer(words[i : i + batch]).to(device)
            if use_autocast:
                with torch.autocast(device_type=device, dtype=torch.float16):
                    t = model.encode_text(tok)
                    t = t.float()
                    t = t / t.norm(dim=1, keepdim=True)
            else:
                t = model.encode_text(tok)
                t = t / t.norm(dim=1, keepdim=True)
            out.append(t.float().cpu().numpy())
    text = np.concatenate(out).astype(np.float32)
    np.savez(cache_path, text=text, words=np.asarray(words))
    return text, words


def cos_words(subspace_dir, text, words, k=10):
    scores = text @ subspace_dir
    # argpartition for O(N) top-k, then sort those k
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return {words[i]: float(scores[i]) for i in idx}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument(
        "--subspaces", type=int, default=24, help="number of 1-D sub-shift subspaces (r=3)"
    )
    ap.add_argument("--celeba-zip", default=None, help="local img_align_celeba.zip path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cached-only", action="store_true")
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
                    help="compute device for CLIP (auto picks mps if available)")
    ap.add_argument(
        "--batch-size-images", type=int, default=None,
        help="CLIP image batch (default 64 on mps, 32 on cpu)",
    )
    ap.add_argument(
        "--batch-size-text", type=int, default=None,
        help="CLIP text batch (default 512 on mps, 256 on cpu)",
    )
    args = ap.parse_args()

    device = get_device(args.device)
    default_images = 64 if device in ("mps", "cuda") else 32
    step_images = (
        args.batch_size_images if args.batch_size_images is not None else default_images
    )

    cache_dir = pkg_data_path("cache")
    os.makedirs(cache_dir, exist_ok=True)
    emb_path = os.path.join(cache_dir, "celeba_lfw_embeddings.npz")
    words_path = os.path.join(cache_dir, "text_out.npz")

    print(f"[05] device: {device} (requested {args.device}), image batch {step_images}", flush=True)
    model, transform, tokenizer = load_clip(device)
    words = load_words()

    if args.cached_only and not os.path.exists(emb_path):
        raise RuntimeError("--cached-only but no cached embeddings")

    E_lfw = embed_or_load(
        "lfw", args.n, args.seed, transform, model, device, step_images,
        lambda n, s: load_lfw(n, s), emb_path,
    )
    E_celeb = embed_or_load(
        "celeba", args.n, args.seed, transform, model, device, step_images,
        lambda n, s: load_celeba(n, s, args.celeba_zip), emb_path,
    )
    X, Y = E_lfw, E_celeb

    coupling = exact(X, Y, 2.0, 2.0)
    a = attribute(X, Y, coupling, 2.0, 2.0)
    print(f"\n=== CelebA vs LFW ({len(X)} vs {len(Y)} samples, d={X.shape[1]}) ===")
    print(f"W_2 = {a.W:.4f}, conservation: {a.conserved}")

    print("\nmain transport trend (U-WaX, C=1, r=2):")
    U1 = eigen_subspace(X, Y, coupling, [1])
    s1 = uwax_attribute(X, Y, coupling, U1, 2.0)
    print(f"  subspace score S = {s1.S_c[0]:.4f}, captured fraction = {s1.captured:.3f}")
    text, words_saved = embed_words(words, model, tokenizer, device, words_path)
    top = cos_words(U1[:, 0], text, words_saved, k=12)
    print("  most aligned words: " + ", ".join(top))

    print(f"\nsub-shift subspaces (U-WaX, C={args.subspaces}, r=3):")
    # PCA-128 projection accelerates the subspace search.
    Z = np.vstack([X, Y])
    mu = Z.mean(axis=0)
    Zn = Z - mu
    comps = np.linalg.svd(Zn, full_matrices=False)[2].T[:, : min(128, len(X))]  # (512, k)
    X128 = (X - mu) @ comps
    Y128 = (Y - mu) @ comps
    c128 = exact(X128, Y128, 2.0, 2.0)
    dims = [1] * args.subspaces
    U128, hist = uwax_search(X128, Y128, c128, r=3, dims=dims, seed=args.seed,
                             lr=0.3, max_iter=40, tol=1e-5)
    s = uwax_attribute(X128, Y128, c128, U128, 3.0)
    print(f"  captured fraction by {args.subspaces} subspaces: {s.captured:.3f} "
          f"(objective {hist[0]:.4f} -> {hist[-1]:.4f})")
    dirs = comps @ U128[:, : args.subspaces]  # back to 512-space (512, C)
    show = min(3, args.subspaces)
    subspace_words = []
    for c in range(show):
        top = cos_words(dirs[:, c] / np.linalg.norm(dirs[:, c]), text, words_saved, k=8)
        print(f"  subspace S{c + 1}: relevance {s.R_c[c]:.4f} | " + ", ".join(top))
        subspace_words.append((f"S{c + 1}", top))

    fig_face_subspace_words(
        os.path.join(FIGDIR, "fig_05_celeba_lfw.png"), subspace_words, {},
    )

    print(f"\nsaved figure: {os.path.join(FIGDIR, 'fig_05_celeba_lfw.png')}")


if __name__ == "__main__":
    main()
