
import tempfile
import time
from pathlib import Path

import numpy as np
import streamlit as st

# ---- Optional Numba ----
try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False
    def njit(*args, **kwargs):
        def deco(fn): return fn
        return deco

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    img { image-rendering: pixelated !important; }
    video { image-rendering: pixelated !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# idx 0 -> spin -1 (dark purple)
# idx 1 -> spin +1 (dark red)
PALETTE = np.array(
    [
        [40, 0, 80],
        [120, 0, 0],
    ],
    dtype=np.uint8,
)

# ---------- Numba kernels ----------
@njit(cache=True)
def _seed_numba(seed: int):
    np.random.seed(seed)

@njit(cache=True)
def metropolis_sweeps(spins: np.ndarray, T: float, n_sweeps: int,
                      ip1: np.ndarray, im1: np.ndarray) -> int:
    """
    n_sweeps full sweeps; 1 sweep = L^2 random single-spin flip attempts.
    Periodic BC via ip1/im1.
    Returns accepted flips (total).
    """
    L = spins.shape[0]
    accepted_total = 0

    if T > 0.0:
        w4 = np.exp(-4.0 / T)
        w8 = np.exp(-8.0 / T)
    else:
        w4 = 0.0
        w8 = 0.0

    for _ in range(n_sweeps):
        for _ in range(L * L):
            i = np.random.randint(0, L)
            j = np.random.randint(0, L)

            s = spins[i, j]
            nn = spins[im1[i], j] + spins[ip1[i], j] + spins[i, im1[j]] + spins[i, ip1[j]]
            dE = 2 * s * nn  # ΔE ∈ {-8,-4,0,4,8}

            if dE <= 0:
                spins[i, j] = -s
                accepted_total += 1
            else:
                r = np.random.random()
                if (dE == 4 and r < w4) or (dE == 8 and r < w8):
                    spins[i, j] = -s
                    accepted_total += 1

    return accepted_total

# ---------- Helpers ----------
def make_neighbors(L: int):
    ip1 = np.arange(L, dtype=np.int32) + 1
    ip1[-1] = 0
    im1 = np.arange(L, dtype=np.int32) - 1
    im1[0] = L - 1
    return ip1, im1

def init_lattice(L: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))

def spins_to_rgb(spins_view: np.ndarray) -> np.ndarray:
    idx = (spins_view > 0).astype(np.uint8)   # 0 or 1
    return PALETTE[idx]                       # (H, W, 3) uint8

def upscale_nearest(rgb: np.ndarray, cell_px: int) -> np.ndarray:
    if cell_px <= 1:
        return rgb
    return np.repeat(np.repeat(rgb, cell_px, axis=0), cell_px, axis=1)

def choose_cell_px(L_view: int, target_width_px: int) -> int:
    return max(1, int(target_width_px) // max(1, int(L_view)))

def make_frame(spins: np.ndarray, downsample: int, target_width_px: int):
    downsample = max(1, int(downsample))
    view = spins[::downsample, ::downsample]
    L_view = int(view.shape[0])
    cell_px = choose_cell_px(L_view, int(target_width_px))
    frame = spins_to_rgb(view)
    frame_big = upscale_nearest(frame, cell_px)
    width_px = int(frame_big.shape[1])
    return frame_big, L_view, cell_px, width_px

def write_mp4_clip_streaming(
    spins: np.ndarray,
    T: float,
    ip1: np.ndarray,
    im1: np.ndarray,
    frames: int,
    sweeps_between_frames: int,
    downsample: int,
    target_width_px: int,
    playback_fps: int,
    crf: int,
    progress,
):
    """
    Stream frames directly to ffmpeg via imageio. No big frame list in RAM.

    IMPORTANT for file size:
      - crf=0 is (near) lossless and can become large.
      - For two-color imagery, crf ~ 16–24 is usually visually perfect and much smaller.
    """
    import imageio  # requires imageio + imageio-ffmpeg

    frames = int(frames)
    sweeps_between_frames = int(sweeps_between_frames)
    downsample = max(1, int(downsample))

    view_L = int(spins[::downsample, ::downsample].shape[0])
    cell_px = choose_cell_px(view_L, int(target_width_px))
    width_px = view_L * cell_px

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    mp4_path = tmp.name


    ffmpeg_params = [
        "-pix_fmt", "yuv444p",
        "-crf", str(int(crf)),
        "-preset", "veryfast",
        "-movflags", "+faststart",
    ]

    accepted_total = 0
    with imageio.get_writer(
        mp4_path,
        fps=int(playback_fps),
        codec="libx264",
        ffmpeg_params=ffmpeg_params,
    ) as writer:
        for k in range(frames):
            accepted_total += int(metropolis_sweeps(spins, float(T), sweeps_between_frames, ip1, im1))

            view = spins[::downsample, ::downsample]
            frame_big = upscale_nearest(spins_to_rgb(view), cell_px)
            writer.append_data(frame_big)

            if progress is not None:
                progress.progress((k + 1) / frames)

    return mp4_path, width_px, accepted_total

# ---------- Session state ----------
if "seed" not in st.session_state:
    st.session_state.seed = 0
if "L" not in st.session_state:
    st.session_state.L = 128
    st.session_state.T = 2.2
    st.session_state.spins = init_lattice(st.session_state.L, st.session_state.seed)
    st.session_state.ip1, st.session_state.im1 = make_neighbors(st.session_state.L)
    st.session_state.sweep = 0
    st.session_state.last_mp4_path = None
    st.session_state.last_mp4_width = None
    st.session_state.last_clip_stats = None
    if NUMBA_OK:
        _seed_numba(st.session_state.seed)

# ---------- UI ----------
st.sidebar.header("Ising Magnetic Microstate Simulation")

L = st.sidebar.slider("Lattice size L", 16, 1024, int(st.session_state.L), step=16)
T = st.sidebar.slider("Temperature T", 0.1, 6.0, float(st.session_state.T), step=0.1)
seed = st.sidebar.number_input("RNG seed", min_value=0, max_value=2_000_000_000,
                               value=int(st.session_state.seed), step=1)

st.sidebar.divider()
st.sidebar.subheader("Display")
downsample = st.sidebar.selectbox("Downsample (visual only)", [1, 2, 4, 8, 16], index=0)
target_width_px = st.sidebar.slider("Target display width (px)", 300, 1600, 1000, step=50)

st.sidebar.divider()
st.sidebar.subheader("Clip settings")
clip_frames = st.sidebar.slider("Frames in clip", 30, 1200, 300)
clip_playback_fps = st.sidebar.slider("Playback FPS", 5, 60, 30)
clip_sweeps_between = st.sidebar.slider("Sweeps between frames", 1, 500, 10)

# File-size control (prevents MessageSizeError + long encodes)
crf = st.sidebar.slider("H.264 CRF (smaller ↔ higher quality)", 0, 30, 18)
st.sidebar.caption("Tip: CRF≈16–22 is typically sharp for two-color lattices; CRF=0 can get large.")

st.sidebar.divider()
colA, colB = st.sidebar.columns(2)
btn_init = colA.button("Generate initial config", use_container_width=True)
btn_clip = colB.button("Generate clip", use_container_width=True)

# Apply parameter changes cleanly when user requests init (or if L changed)
if int(L) != int(st.session_state.L):
    # If L changes, force new lattice (shape must match)
    st.session_state.L = int(L)
    st.session_state.spins = init_lattice(int(L), int(seed))
    st.session_state.ip1, st.session_state.im1 = make_neighbors(int(L))
    st.session_state.sweep = 0
    st.session_state.last_mp4_path = None
    st.session_state.last_mp4_width = None
    st.session_state.last_clip_stats = None
    st.session_state.seed = int(seed)
    if NUMBA_OK:
        _seed_numba(int(seed))

st.session_state.T = float(T)

if btn_init:
    st.session_state.seed = int(seed)
    st.session_state.spins = init_lattice(int(st.session_state.L), int(st.session_state.seed))
    st.session_state.sweep = 0
    st.session_state.last_mp4_path = None
    st.session_state.last_mp4_width = None
    st.session_state.last_clip_stats = None
    if NUMBA_OK:
        _seed_numba(int(st.session_state.seed))

# ---------- Layout ----------
left, right = st.columns([4, 1], vertical_alignment="top")


frame_big, L_view, cell_px, width_px = make_frame(
    st.session_state.spins,
    downsample=int(downsample),
    target_width_px=int(target_width_px),
)
left.image(frame_big, clamp=True, width=width_px)

# Clip generation
if btn_clip:
    try:
        progress = st.progress(0.0)
        with st.spinner("Simulating + encoding…"):
            t0 = time.perf_counter()
            mp4_path, mp4_width, acc_total = write_mp4_clip_streaming(
                st.session_state.spins,
                st.session_state.T,
                st.session_state.ip1,
                st.session_state.im1,
                frames=int(clip_frames),
                sweeps_between_frames=int(clip_sweeps_between),
                downsample=int(downsample),
                target_width_px=int(target_width_px),   # clip matches your crisp display scale
                playback_fps=int(clip_playback_fps),
                crf=int(crf),
                progress=progress,
            )
            elapsed = time.perf_counter() - t0
        progress.empty()

        st.session_state.last_mp4_path = mp4_path
        st.session_state.last_mp4_width = mp4_width
        st.session_state.sweep += int(clip_frames) * int(clip_sweeps_between)

        file_mb = Path(mp4_path).stat().st_size / (1024**2)
        st.session_state.last_clip_stats = dict(
            frames=int(clip_frames),
            fps=int(clip_playback_fps),
            sweeps_between=int(clip_sweeps_between),
            sweeps_added=int(clip_frames) * int(clip_sweeps_between),
            crf=int(crf),
            file_mb=file_mb,
            elapsed=elapsed,
            acc_total=int(acc_total),
        )

    except ModuleNotFoundError:
        right.error("Missing video deps. Install:\n\npython -m pip install imageio imageio-ffmpeg")
    except Exception as e:
        right.error(f"Clip generation failed: {e}")

# Show last clip (served as a file, avoids websocket giant payloads)
if st.session_state.last_mp4_path is not None:
    right.subheader("Last clip")
    right.video(st.session_state.last_mp4_path)

    # Download button 
    mp4_bytes = Path(st.session_state.last_mp4_path).read_bytes()
    right.download_button(
        "Download MP4",
        data=mp4_bytes,
        file_name="ising_clip.mp4",
        mime="video/mp4",
        use_container_width=True,
    )

    if st.session_state.last_clip_stats is not None:
        s = st.session_state.last_clip_stats
        right.markdown(
            f"""
**Clip stats**
- frames: {s['frames']}
- playback FPS: {s['fps']}
- sweeps between frames: {s['sweeps_between']}
- sweeps added: {s['sweeps_added']}
- CRF: {s['crf']}
- file size: {s['file_mb']:.1f} MB
- sim+encode time: {s['elapsed']:.2f} s
"""
        )

# Observables / info
N = int(st.session_state.L) * int(st.session_state.L)
m = float(st.session_state.spins.mean())

right.subheader("State")
right.markdown(
    f"""
- **L:** {int(st.session_state.L)}
- **T:** {st.session_state.T:.2f}
- **seed:** {int(st.session_state.seed)}
- **sweep counter:** {int(st.session_state.sweep)}
- **m = ⟨s⟩:** {m:.3f}

**Display**
- downsample: {int(downsample)} → view {L_view}×{L_view}
- cell size: {cell_px} px (integer NN)
- raster: {width_px}×{width_px} px
"""
)

if not NUMBA_OK:
    st.warning("Numba not found. Install: python -m pip install numba")

