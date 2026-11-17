# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: jupytext,-kernelspec,-widgets
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
# ---

# %% [markdown]
# # Decoding saccades from LIP: resolution vs. sensitivity
#
# In the last notebook, we analyzed functional ultrasound data from a human motor cortex and recovered task-related activity with a GLM. Here, we'll go one step further and ask whether we can read out *what* an animal is planning to do on a single trial, and what it is about functional ultrasound that makes that possible.
#
# We'll re-analyze data from [Griggs et al. (2025), *Functional ultrasound neuroimaging reveals mesoscopic organization of saccades in the lateral intraparietal area*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12488887/). Two macaques performed memory-guided saccades while the authors imaged the left posterior parietal cortex. The lateral intraparietal area (LIP) is a classic node for planning eye movements: neurons there fire during the delay between a flashed cue and the eye movement towards its remembered location. The paper finds that LIP is organized in small patches tuned to different saccade directions, and argues that fMRI lacks the resolution and sensitivity to resolve this organization.
#
# That raises a question we can test directly with the data: if we degrade functional ultrasound to look more like fMRI, what hurts decoding more—**coarser voxels** or **lower sensitivity**? fMRI differs from fUS in both:
#
# * **Resolution**: fUS voxels are ~100 μm × 100 μm in plane (× 400 μm thick); fMRI voxels are typically 1.5-3 mm on each side.
# * **Sensitivity**: the paper reports cerebral blood volume (CBV) changes of ~10-30% with fUS, compared to BOLD changes of ~0.4-2% for similar saccade tasks with fMRI.
#
# We'll first get to know the data, then build a single-trial decoder from scratch, and finally degrade the data along each axis separately.

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from scipy import linalg, stats
from scipy.ndimage import binary_erosion, convolve

import lip_data

# %% [markdown]
# # Getting one session out of a 56 GB archive
#
# The [dataset](https://data.caltech.edu/records/p5jan-02r60) is distributed as a single 56 GB zip file. Rather than downloading all of it, we can exploit the structure of zip files: they end with a table of contents (the *central directory*) listing where each file starts inside the archive. HTTP lets us request arbitrary byte ranges of a file, so we fetch the last few kB of the archive, read the table of contents, and then download only the bytes of the session we want. The details live in `lip_data.py`; it downloads in chunks so that a dropped connection resumes where it left off, and checks the result against the checksum stored in the archive.
#
# First, let's see what's in there.

# %%
data_dir = "../data/lip"
sessions = lip_data.session_table(data_dir)

for monkey in ["L", "P"]:
    rows = [s for s in sessions if s["Monkey"] == monkey and s["Project"] == "8-DIR"]
    print(
        f"Monkey {monkey}: {len(rows)} sessions, {sum(s['nTrials'] for s in rows)} trials, "
        f"imaging planes {sorted({s['Slot'] for s in rows})}"
    )

# %% [markdown]
# Each session images one coronal plane while the monkey performs a few hundred trials. We'll work with session 102 (~580 MB): it's the session the paper uses as its example for monkey P (recorded on 12 January 2022, "S20220112" in the paper), so we can compare our results with the paper's figures along the way. The session files keep only the successful trials.

# %%
print(next(s for s in sessions if s["Session"] == 102))
path = lip_data.fetch_session(102, 1, data_dir)
S = lip_data.load_session(path)

n_trials, n_frames, nz, nx = S.doppler.shape
print(f"{n_trials} successful trials x {n_frames} frames x {nz} x {nx} pixels")
print(f"Pixel size: {S.dx_mm:.3f} mm (lateral) x {S.dz_mm:.3f} mm (axial), {S.frame_rate:.0f} frame/s")

# %% [markdown]
# Each trial is a short Power Doppler movie at 1 frame per second, aligned to the start of the trial. Each frame was formed exactly like in notebook 3: plane waves at 5 angles were compounded at 500 Hz, 250 compound images (0.5 s) were clutter-filtered with the SVD, and their power was computed pixel by pixel. The probe has 128 elements with a 100 μm pitch, and the axial pixel size is one wavelength at 15.6 MHz (~0.1 mm).
#
# Here's the average image, the angiogram:

# %%
D = S.doppler.astype(np.float32)

# Motion correction pads the borders of the image with zeros; we'll ignore those pixels
valid = D.min(axis=(0, 1)) > 0
valid[:2, :] = valid[-2:, :] = valid[:, :2] = valid[:, -2:] = False

plt.figure(figsize=(6, 6))
plt.imshow(np.log(S.angiogram), cmap="gray", extent=S.extent)
plt.xlabel("Lateral position (mm)")
plt.ylabel("Depth (mm)")
plt.title("Angiogram (log Power Doppler), session 102")
plt.show()

# %% [markdown]
# The brain surface is at the top. The bright vessel running down through the middle of the image follows the intraparietal sulcus; LIP lines the banks of that sulcus.
#
# ## A note on units
#
# Power Doppler is a measure of blood volume, so we'll express responses as a percent change from baseline. That only makes sense if the stored values are proportional to the blood signal. A quick check: for a signal of the form $x = \text{CBV} \cdot (1 + \epsilon)$ with fluctuations $\epsilon$, each pixel's standard deviation should be proportional to its mean.

# %%
pixel_mean = D.mean(axis=(0, 1))[valid]
pixel_std = D.std(axis=(0, 1))[valid]
slope, intercept = np.polyfit(pixel_mean, pixel_std, 1)

plt.figure(figsize=(5, 4))
plt.plot(pixel_mean, pixel_std, ".", markersize=1, alpha=0.3)
xs = np.linspace(0, pixel_mean.max(), 10)
plt.plot(xs, slope * xs + intercept, "r-", label=f"linear fit, intercept = {intercept:.2g}")
plt.xlabel("Pixel mean (a.u.)")
plt.ylabel("Pixel std over time (a.u.)")
plt.legend()
plt.title("Standard deviation grows with the mean")
plt.show()

# %% [markdown]
# The relationship goes roughly through the origin, so percent changes are meaningful here. This is worth checking whenever you pick up a new dataset: in some sessions of this dataset (e.g. session 39), every pixel carries a large constant offset, which makes percent changes look ~100 times smaller than they really are unless you subtract it first.
#
# # Trial structure
#
# The monkeys performed a *memory-guided saccade* task:
#
# <img src="images/task_overview.png" width="900">
#
# *The task, adapted from Griggs et al. (2025), Fig. 2A. Subject P is the monkey we'll analyze.*
#
# On each trial, the monkey fixates a central point. A peripheral target is flashed for 0.4 s at one of 8 locations (the *cue*), and the monkey has to keep fixating while remembering that location (the *memory* period, 5 ± 1 s for monkey P). When the fixation point disappears, it makes a saccade to the remembered location (*target acquire*), holds its gaze there, and gets a reward. Here are the event times recorded on every trial of our session:

# %%
events = ["fixationhold", "cue", "memory", "target_acquire", "reward", "iti"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 3.5), gridspec_kw={"width_ratios": [3, 1]})
for i, ev in enumerate(events):
    ax1.plot(S.events[ev], np.full(n_trials, i) + np.random.uniform(-0.2, 0.2, n_trials), ".", markersize=2)
ax1.set_yticks(range(len(events)))
ax1.set_yticklabels(events)
ax1.set_xlabel("Time from trial start (s)")
ax1.set_xlim(0, n_frames)
ax1.set_title("Task events on every trial")

directions, counts = np.unique(S.target_deg, return_counts=True)
ax2.bar([f"{d:.0f}°" for d in directions], counts)
ax2.set_title("Trials per saccade direction")
ax2.tick_params(axis="x", rotation=90)
plt.tight_layout()
plt.show()

for ev in ["cue", "target_acquire"]:
    print(f"{ev}: median {np.median(S.events[ev]):.1f} s after trial start")

# %% [markdown]
# The cue comes ~5 s after the trial starts and the saccade ~5 s later. Directions are given as angles on the screen: 0° is rightward, 90° is upward. Since the recordings are from the *left* hemisphere, rightward saccades are *contralateral*.
#
# # Event-locked responses
#
# The cue comes at a slightly different time on each trial, so we'll align frames to it. Frame 0 *after the cue* is the first full frame after the cue on that trial; on a typical trial, it starts about half a second after the cue.
#
# We'll measure responses as a percent change from a baseline just before the cue, as the paper does (Fig. 3B): the 4 frames before frame 0. The last of them overlaps the cue, but blood volume takes a couple of seconds to respond, so it is still a clean baseline. Why not use the start of the trial as a baseline? In this session, blood volume drifts down by several percent while the monkey fixates, so everything after the cue would look like a decrease.

# %%
first_after_cue = np.ceil(S.events["cue"] * S.frame_rate).astype(int)
n_after = n_frames - first_after_cue.max()  # frames after the cue that every trial has


def after_cue(A, j):
    """Frame j after the cue on every trial: A[trial, first_after_cue[trial] + j]."""
    return A[np.arange(n_trials), first_after_cue + j]


baseline = range(-4, 0)  # the 4 frames before frame 0 after the cue


def to_pct(A):
    """Percent change from each trial's baseline, relative to the pixel's average baseline."""
    base = np.mean([after_cue(A, j) for j in baseline], axis=0)[:, None]  # (trials, 1, ...)
    return (A - base) / base.mean(axis=0, keepdims=True) * 100


with np.errstate(divide="ignore", invalid="ignore"):
    pct = to_pct(D)
pct[:, :, ~valid] = np.nan

# %% [markdown]
# Let's watch the response unfold, separately for contralateral (rightward) and ipsilateral (leftward) saccades. To make the maps less noisy, we smooth them slightly, with a 1-pixel-radius disk (the paper uses a Gaussian about one pixel wide).

# %%
# On a typical trial: when frame 0 starts, and when the memory period ends (the saccade), after the cue
frame_offset = np.median(first_after_cue / S.frame_rate - S.events["cue"])
saccade = np.median(S.events["target_acquire"] - S.events["cue"])


def frame_time(j):
    """Middle of frame j after the cue, in seconds after the cue (typical trial)."""
    return frame_offset + (np.asarray(j) + 0.5) / S.frame_rate


disk = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=float)
disk /= disk.sum()


def smooth(A):
    """Smooth images (the last two axes) with the disk, keeping only valid pixels."""
    out = convolve(np.nan_to_num(A), disk.reshape((1,) * (A.ndim - 2) + disk.shape), mode="nearest")
    out[..., ~valid] = np.nan
    return out


right = np.isin(S.target_deg, [315, 0, 45])  # contralateral
left = np.isin(S.target_deg, [135, 180, 225])  # ipsilateral

shown = [-1, 1, 3, 5, 7, 9]  # frames after the cue
fig, axes = plt.subplots(3, len(shown), figsize=(15, 7.5))
for col, j in enumerate(shown):
    frame = after_cue(pct, j)
    contra, ipsi = smooth(frame[right].mean(axis=0)), smooth(frame[left].mean(axis=0))
    for row, image in enumerate([contra, ipsi, contra - ipsi]):
        im = axes[row, col].imshow(image, cmap="RdBu_r", vmin=-10, vmax=10, extent=S.extent)
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
    axes[0, col].set_title(f"{round(float(frame_time(j)))} s after the cue")
for row, label in enumerate(["Contralateral\n(rightward)", "Ipsilateral\n(leftward)", "Contralateral\nminus ipsilateral"]):
    axes[row, 0].set_ylabel(label)
fig.colorbar(im, ax=axes, shrink=0.6, label="CBV change (%)")
plt.show()

# %% [markdown]
# Around the cue, the maps are close to zero, as they should be: that's our baseline. About 2 s after the cue, blood volume drops over much of the upper left of the image; at 4 s, it rises by more than 10% in a patch at the top of the sulcus; and after the saccade (~5.4 s after the cue), it rises across most of the upper part of the image, by 10% and more. So whether a given pixel seems to gain or lose blood depends very much on when we look. And nearly all of these changes are the same for both directions: they reflect the task as a whole (fixating, seeing the cue, making a saccade, getting a reward) rather than where the monkey is going to look.
#
# The bottom row isolates what *is* specific to the direction. During the memory period, a band along the sulcus responds more to contralateral saccades, and it isn't where the largest overall responses are. That's the signal we're after.
#
# ## When are responses tuned to direction?
#
# To map direction tuning, we'll average a few frames after the cue. Which ones? Let's measure, frame by frame, how many pixels respond differently to the 8 directions: for each pixel, a one-way ANOVA tests whether its response differs across directions. While we're at it, we summarize each pixel's preferred direction as the direction of the vector sum of its mean responses, $\sum_k r_k e^{i \theta_k}$; we'll need it shortly.

# %%
directions = np.unique(S.target_deg)


def direction_tuning(response):
    """Per-pixel direction ANOVA p-value and preferred direction (vector sum) of (trials, ...) responses."""
    by_dir = [response[S.target_deg == d] for d in directions]
    means = np.array([r.mean(axis=0) for r in by_dir])
    with np.errstate(invalid="ignore"):
        _, p = stats.f_oneway(*by_dir, axis=0)
    vector_sum = np.tensordot(np.exp(1j * np.deg2rad(directions)), means - means.mean(axis=0), axes=1)
    return p, np.degrees(np.angle(vector_sum)) % 360


frames = np.arange(-4, n_after)  # frames after the cue (negative: before the cue)
n_tuned = [np.sum(valid & (direction_tuning(smooth(after_cue(pct, j)))[0] < 1e-3)) for j in frames]
avg_window = range(3, 7)  # the frames we'll average: 3-6 after the cue


def shade_periods(ax, window=True):
    """Shade the memory period and, optionally, the averaging window, on an axis of time after the cue."""
    ax.axvspan(0, saccade, color="gray", alpha=0.15, label="memory period (cue to saccade, median)")
    if window:
        ax.axvspan(frame_offset + avg_window.start / S.frame_rate, frame_offset + avg_window.stop / S.frame_rate,
                   facecolor="none", edgecolor="C1", hatch="//", label="averaging window (typical trial)")


fig, ax = plt.subplots(figsize=(8, 4))
shade_periods(ax)
ax.plot(frame_time(frames), n_tuned, "o-", color="k")
ax.axhline(1e-3 * valid.sum(), color="k", linestyle="--", linewidth=1, label="expected by chance")
ax.set_xlabel("Time after the cue (s)")
ax.set_ylabel("Pixels tuned to direction (p < 0.001)")
ax.set_title("Direction tuning, frame by frame")
ax.legend(fontsize=8)
plt.show()

# %% [markdown]
# Direction tuning appears ~2 s after the cue, peaks about 5 s after it, around the time of the saccade, and then fades, even as the large post-saccade response builds up. The delay is the hemodynamic lag at work: blood volume follows neural activity with a delay of a couple of seconds, so the frames around the peak mostly reflect activity during the memory period. We'll average frames 3-6 after the cue (hatched), which cover the peak of the tuning. The paper also places its analysis window around the time of the saccade (Fig. 3B).
#
# # Direction tuning maps
#
# Now let's map, for every pixel, whether its averaged response is tuned to direction, and which direction it prefers.

# %%
response = np.mean([after_cue(pct, j) for j in avg_window], axis=0)  # (trials, nz, nx)
response_smooth = smooth(response)
p_anova, preferred = direction_tuning(response_smooth)
tuned = valid & (p_anova < 1e-3)
print(f"{tuned.sum()} pixels tuned to direction at p < 0.001 ({1e-3 * valid.sum():.0f} expected by chance)")


def direction_map(preferred, tuned, angiogram):
    """Color tuned pixels by their preferred direction, on top of the angiogram."""
    hue = np.nan_to_num(preferred) / 360
    rgb = hsv_to_rgb(np.dstack([hue, np.ones_like(hue), np.ones_like(hue)]))
    gray = np.log(np.clip(np.nan_to_num(angiogram), np.nanmin(angiogram[angiogram > 0]), None))
    gray = (gray - gray.min()) / np.ptp(gray) * 0.7
    return np.where(tuned[..., None], rgb, np.dstack([gray] * 3))


fig = plt.figure(figsize=(7.5, 6))
ax = fig.add_axes([0.05, 0.1, 0.65, 0.8])
ax.imshow(direction_map(preferred, tuned, S.angiogram), extent=S.extent)
ax.set_xlabel("Lateral position (mm)")
ax.set_ylabel("Depth (mm)")
ax.set_title("Preferred saccade direction (tuned pixels, p < 0.001)")

# Color wheel legend
wheel = fig.add_axes([0.79, 0.6, 0.15, 0.15], projection="polar")
theta = np.linspace(0, 2 * np.pi, 360)
wheel.scatter(theta, np.ones_like(theta), c=hsv_to_rgb(np.dstack([theta / (2 * np.pi), np.ones(360), np.ones(360)]))[0], s=40)
wheel.set_yticks([])
wheel.set_xticks(np.deg2rad([0, 90, 180, 270]))
wheel.set_xticklabels(["right\n(contra)", "up", "left", "down"], fontsize=7)
plt.show()

# %% [markdown]
# Most tuned pixels form one large cluster along the sulcus, and nearly all of them prefer contralateral (rightward) or upward saccades—consistent with the paper, which finds that over 85% of tuned LIP voxels prefer contralateral directions. Within the cluster, the preference changes from rightward at the top to upward further down, in patches. The paper's Fig. 3A shows this session differently, as one map of response strength per direction, but the same cluster along the sulcus stands out there too, responding most to contralateral and upward saccades.
#
# Let's look at tuning curves in two small (0.4 × 0.4 mm) regions of the cluster with different preferences.

# %%
def tuning_curve(center, size=4):
    """Mean response per direction in a size x size pixel region around `center`."""
    iz, ix = center
    region = response[:, iz - size // 2 : iz + size // 2, ix - size // 2 : ix + size // 2]
    per_trial = np.nanmean(region, axis=(1, 2))
    return np.array([per_trial[S.target_deg == d].mean() for d in directions]), np.array(
        [per_trial[S.target_deg == d].std() / np.sqrt(np.sum(S.target_deg == d)) for d in directions]
    )


# Pick the most strongly tuned pixel, then the most strongly tuned pixel within 3 mm
# of it whose preferred direction differs by at least 45 degrees
order = np.argsort(np.where(tuned, p_anova, 1), axis=None)
first = np.unravel_index(order[0], p_anova.shape)
second = next(
    np.unravel_index(i, p_anova.shape)
    for i in order
    if abs((preferred.flat[i] - preferred[first] + 180) % 360 - 180) >= 45
    and np.hypot(*np.subtract(np.unravel_index(i, p_anova.shape), first)) * S.dx_mm <= 3
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
ax1.imshow(direction_map(preferred, tuned, S.angiogram), extent=S.extent)
for (iz, ix), color, label in [(first, "w", "A"), (second, "k", "B")]:
    x_mm, z_mm = ix * S.dx_mm, S.depth_mm[0] + iz * S.dz_mm
    ax1.add_patch(plt.Rectangle((x_mm - 0.2, z_mm - 0.2), 0.4, 0.4, fill=False, color=color, linewidth=2))
    ax1.text(x_mm + 0.3, z_mm, label, color=color, fontsize=12, fontweight="bold")
    curve, err = tuning_curve((iz, ix))
    ax2.errorbar(directions, curve, err, marker="o", capsize=3, label=f"region {label}")
ax1.set_title("Regions")
ax1.set_xlabel("Lateral position (mm)")
ax1.set_ylabel("Depth (mm)")
ax2.set_xticks(directions)
ax2.set_xlabel("Saccade direction (°)")
ax2.set_ylabel("CBV change (%)")
ax2.set_title("Tuning curves (mean ± s.e.m.)")
ax2.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# Even within a single cluster, regions less than 1.5 mm apart are tuned differently: region A is sharply tuned to upward saccades, while region B is more broadly tuned, peaking up and to the right and still responding to rightward saccades. Neither responds much to leftward or downward saccades. The paper's Fig. 3C finds the same mix of broadly and narrowly tuned regions in this session.
#
# How do these responses unfold over time? Let's plot the response of each region to each direction, aligned to the cue (compare with the paper's Fig. 3B). We'll also compare them with the canonical hemodynamic response function (HRF) we used in notebook 4: if neurons are active from the cue to the saccade, the predicted response is that period convolved with the HRF.

# %%
def canonical_hrf(t, peak_delay=6.0, undershoot_delay=16.0, peak_to_undershoot=6.0):
    """SPM-style double-gamma HRF (as in notebook 4), normalized to unit area."""
    h = stats.gamma.pdf(t, peak_delay) - stats.gamma.pdf(t, undershoot_delay) / peak_to_undershoot
    return h / h.sum()


# Predicted response on each trial: neural activity from the cue to the saccade, convolved with the HRF
t_fine = np.arange(0, n_frames, 0.1)
hrf = canonical_hrf(np.arange(0, 30, 0.1))
frame_centers = (np.arange(n_frames) + 0.5) / S.frame_rate
predicted = np.array([
    np.interp(frame_centers, t_fine, np.convolve(((t_fine >= cue) & (t_fine < sacc)).astype(float), hrf)[: len(t_fine)])
    for cue, sacc in zip(S.events["cue"], S.events["target_acquire"])
])  # (trials, frames)
predicted = np.array([after_cue(predicted, j).mean() for j in frames])  # cue-aligned, averaged over trials


def region_time_course(center, size=4):
    """Cue-aligned response of a size x size pixel region around `center`: (trials, frames)."""
    iz, ix = center
    per_trial = np.nanmean(pct[:, :, iz - size // 2 : iz + size // 2, ix - size // 2 : ix + size // 2], axis=(2, 3))
    return np.stack([after_cue(per_trial, j) for j in frames], axis=1)


fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
for ax, center, label in [(axes[0], first, "A"), (axes[1], second, "B")]:
    tc = region_time_course(center)
    shade_periods(ax)
    means = np.array([tc[S.target_deg == d].mean(axis=0) for d in directions])
    for d, mean in zip(directions, means):
        ax.plot(frame_time(frames), mean, "o-", markersize=3, color=hsv_to_rgb([d / 360, 1, 0.85]), label=f"{d:.0f}°")
    # Fit the predicted response (gain and offset) to the region's preferred direction
    best = means[np.argmax(means[:, np.isin(frames, avg_window)].mean(axis=1))]
    gain, offset = np.linalg.lstsq(np.vstack([predicted, np.ones_like(predicted)]).T, best, rcond=None)[0]
    r = np.corrcoef(predicted, best)[0, 1]
    ax.plot(frame_time(frames), gain * predicted + offset, "k--", label="canonical HRF\n(fit to the best direction)")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Time after the cue (s)")
    ax.set_title(f"Region {label}: response to each direction (HRF fit: r = {r:.2f})")
axes[0].set_ylabel("CBV change (%)")
axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
plt.tight_layout()
plt.show()

# %% [markdown]
# Both regions respond to their preferred directions within ~2 s of the cue, keep rising through the memory period, and peak 2-3 s after the saccade, at roughly 20-25% above baseline. Responses to the other directions stay close to baseline, or even dip below it: region A's blood volume *drops* by over 10% during rightward and down-rightward saccades. The canonical HRF captures the overall shape of the preferred responses, even though it was derived from BOLD fMRI and CBV responses have their own dynamics.
#
# Note the size of these responses: 20-25% is an order of magnitude larger than typical BOLD signal changes. Remember this, it will matter later.
#
# This is the kind of mesoscopic organization that the paper argues fMRI can't resolve. But being able to *see* fine-grained maps and being able to *use* the information they contain are two different things. Let's build a decoder.
#
# # Decoding single trials
#
# Our goal is to predict, from the images of a single trial, which direction the monkey is planning to look. We'll follow the paper's recipe, with a few simplifications:
#
# 1. **Smooth** each image with a 2-pixel-radius pillbox, a bit more than for the maps.
# 2. **Features**: the responses of every pixel in every frame from the cue on, concatenated. With ~470 trials and ~16,000 pixels per frame, that's far more features than trials.
# 3. **Standardize** each feature (z-score). We use all trials for this: it doesn't involve the labels, so it can't leak information about the directions.
# 4. **PCA**: keep the principal components that explain 95% of the variance across the training trials.
# 5. **LDA**: model each class as a Gaussian with its own mean and a shared covariance $\Sigma$, and assign a trial to the class $c$ with the highest score $z^T \Sigma^{-1} \mu_c - \frac{1}{2}\mu_c^T \Sigma^{-1} \mu_c$.
#
# We evaluate the decoder with 10-fold cross-validation: each trial is predicted by a decoder that never saw it during training. The paper uses leave-one-out cross-validation instead, which trains on slightly more trials, and decodes the horizontal and vertical components of each saccade separately; we'll keep things simpler.
#
# **A trick to make this fast.** With far fewer trials ($n$) than features ($p$), PCA only needs the $n \times n$ *Gram matrix* $G = XX^T$ of products between trials, not the $p \times p$ covariance matrix. If the centered Gram matrix of the training trials is $U \Lambda U^T$, their PCA scores are $U \Lambda^{1/2}$, and a test trial whose (centered) products with the training trials are $g$ has scores $g^T U \Lambda^{-1/2}$. Better still, concatenating frames simply *adds* their Gram matrices. So we compute one Gram matrix per frame, once, and can then decode from any set of frames almost for free.

# %%
pillbox = (np.hypot(*np.mgrid[-2:3, -2:3]) <= 2).astype(np.float32)
pillbox /= pillbox.sum()
decode_valid = binary_erosion(valid, pillbox > 0)  # pixels whose whole pillbox is inside the valid image
D_smooth = np.where(decode_valid, convolve(D, pillbox[None, None], mode="nearest"), np.nan)


def gram(F):
    """Gram matrix of features F (trials x features), after z-scoring each feature across trials."""
    F = F.astype(np.float64)
    Z = (F - F.mean(axis=0)) / (F.std(axis=0) + 1e-12)
    return Z @ Z.T


pct_smooth = to_pct(D_smooth)
grams = np.array([gram(after_cue(pct_smooth, j)[:, decode_valid]) for j in range(n_after)])
print(f"{decode_valid.sum()} pixels x {n_after} frames after the cue = {decode_valid.sum() * n_after:,} features")


# %%
def make_folds(y, n_folds=10, seed=0):
    """Assign each trial to a fold, spreading every class evenly across folds."""
    rng = np.random.default_rng(seed)
    folds = np.zeros(len(y), dtype=int)
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        folds[idx] = np.arange(len(idx)) % n_folds
    return folds


def pca_scores(G, train, test, var_explained=0.95):
    """PCA scores of the training and test trials (boolean masks), from the Gram matrix G."""
    G_train, G_test = G[np.ix_(train, train)], G[np.ix_(test, train)]
    # Center the features on the mean of the training trials, in Gram-matrix form
    m = G_train.mean(axis=0)
    K_train = G_train - m[:, None] - m[None, :] + m.mean()
    K_test = G_test - G_test.mean(axis=1, keepdims=True) - m[None, :] + m.mean()
    evals, U = linalg.eigh(K_train)
    evals, U = np.clip(evals[::-1], 0, None), U[:, ::-1]
    k = np.searchsorted(np.cumsum(evals) / evals.sum(), var_explained) + 1
    k = min(k, np.sum(evals > 1e-9 * evals[0]))
    s = np.sqrt(evals[:k])
    return U[:, :k] * s, K_test @ U[:, :k] / s


def lda_predict(Z_train, y_train, Z_test):
    """Linear discriminant analysis with a shared covariance matrix."""
    classes = np.unique(y_train)
    # Estimating the covariance needs more trials than dimensions: keep at most n - n_classes components
    k = min(Z_train.shape[1], len(Z_train) - len(classes))
    Z_train, Z_test = Z_train[:, :k], Z_test[:, :k]
    means = np.array([Z_train[y_train == c].mean(axis=0) for c in classes])
    resid = Z_train - means[np.searchsorted(classes, y_train)]
    cov = resid.T @ resid / (len(Z_train) - len(classes))
    W = linalg.solve(cov, means.T, assume_a="pos")  # Sigma^-1 mu_c for every class
    scores = Z_test @ W - 0.5 * np.sum(means.T * W, axis=0)
    return classes[np.argmax(scores, axis=1)]


def decode(G, y, seeds=(0,), n_folds=10):
    """Cross-validated accuracy (averaged over random fold assignments) and predictions, from a Gram matrix."""
    accuracies, predictions = [], []
    for seed in seeds:
        folds = make_folds(y, n_folds, seed)
        pred = np.empty_like(y)
        for fold in range(n_folds):
            test = folds == fold
            Z_train, Z_test = pca_scores(G, ~test, test)
            pred[test] = lda_predict(Z_train, y[~test], Z_test)
        accuracies.append(np.mean(pred == y))
        predictions.append(pred)
    return np.mean(accuracies), predictions[0]


# %% [markdown]
# We'll decode two things: the full **8-direction** problem (chance = 1/8), and the easier **left vs. right** problem, using only the trials with a horizontal component (chance = 1/2). For now, we use all the frames from the cue on.

# %%
y8 = np.searchsorted(directions, S.target_deg)  # direction index, 0..7
y_lr = np.where(right, 1, np.where(left, -1, 0))
lateral = y_lr != 0

G_all = grams.sum(axis=0)  # all frames from the cue on
acc8, pred8 = decode(G_all, y8, seeds=(0, 1))
acc_lr, _ = decode(G_all[np.ix_(lateral, lateral)], y_lr[lateral], seeds=(0, 1))
print(f"8 directions: {acc8:.1%} correct (chance 12.5%)")
print(f"Left vs. right: {acc_lr:.1%} correct (chance 50%)")


# %% [markdown]
# Is this better than chance? A robust way to check is a permutation test: shuffle the labels, decode again, and repeat many times to see what accuracies we'd get if there was no relationship between images and directions. Since PCA doesn't look at the labels, we can compute it once per fold and only redo the cheap LDA step for each shuffle.

# %%
def permutation_null(G, y, n_permutations=100, n_folds=10, seed=0):
    """Accuracies obtained after shuffling the labels, reusing the label-independent PCA."""
    folds = make_folds(y, n_folds, 0)
    cached = [(folds == f, *pca_scores(G, folds != f, folds == f)) for f in range(n_folds)]
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_permutations):
        y_shuffled = rng.permutation(y)
        pred = np.empty_like(y)
        for test, Z_train, Z_test in cached:
            pred[test] = lda_predict(Z_train, y_shuffled[~test], Z_test)
        null.append(np.mean(pred == y_shuffled))
    return np.array(null)


null8 = permutation_null(G_all, y8)
null_lr = permutation_null(G_all[np.ix_(lateral, lateral)], y_lr[lateral])

fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
for ax, null, acc, title in [(axes[0], null8, acc8, "8 directions"), (axes[1], null_lr, acc_lr, "Left vs. right")]:
    ax.hist(null, bins=20, color="gray", label="shuffled labels")
    ax.axvline(acc, color="r", linewidth=2, label="real labels")
    ax.set_xlabel("Cross-validated accuracy")
    ax.set_title(f"{title}: p < {1 / (len(null) + 1):.3f}" if acc > null.max() else title)
    ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# Both are far outside the null distribution. Which mistakes does the 8-direction decoder make? (Compare with the paper's Fig. 4B.)

# %%
confusion = np.zeros((8, 8))
for true, guess in zip(y8, pred8):
    confusion[true, guess] += 1
confusion /= confusion.sum(axis=1, keepdims=True)

# Same order as the paper's Fig. 4B: clockwise from up-left
paper_order = [list(directions).index(d) for d in [135, 90, 45, 0, 315, 270, 225, 180]]
arrows = {0: "→", 45: "↗", 90: "↑", 135: "↖", 180: "←", 225: "↙", 270: "↓", 315: "↘"}
labels = [f"{arrows[directions[i]]} {directions[i]:.0f}°" for i in paper_order]

angular_error = np.abs((directions[pred8] - directions[y8] + 180) % 360 - 180)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
im = ax1.imshow(confusion[np.ix_(paper_order, paper_order)], cmap="viridis", vmin=0, vmax=1)
ax1.set_xticks(range(8))
ax1.set_xticklabels(labels, rotation=90)
ax1.set_yticks(range(8))
ax1.set_yticklabels(labels)
ax1.set_xlabel("Predicted direction")
ax1.set_ylabel("True direction")
ax1.set_title("Confusion matrix (fraction of trials)")
plt.colorbar(im, ax=ax1)

errors, counts = np.unique(angular_error, return_counts=True)
ax2.bar(errors, counts / n_trials, width=30)
ax2.axhline(1 / 8, color="k", linestyle="--", label="uniform guessing")
ax2.set_xticks([0, 45, 90, 135, 180])
ax2.set_xlabel("Angular error (°)")
ax2.set_ylabel("Fraction of trials")
ax2.set_title(f"Mean angular error: {angular_error.mean():.0f}° (chance: 90°)")
ax2.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# Most errors are to neighboring directions: the decoder knows roughly where the monkey will look even when it gets the exact target wrong. Upward, up-left and rightward saccades are decoded best, and leftward ones worst: they are mostly mistaken for down-left ones, as in the paper's Fig. 4B. The decoder also has a soft spot for 135° (up-left): when the evidence is weak, as for leftward and downward saccades, that's where many of its guesses go.
#
# The paper reports 59.6% correct and a mean angular error of 23.7° for this session (Fig. 4A). We're close. Our pipeline is simpler: the paper uses leave-one-out cross-validation, so each decoder trains on ~10% more trials, and its preprocessing differs in the details (for instance, it removes slow drifts by detrending, where we subtract each trial's baseline).
#
# ## When does the information appear?
#
# Following the paper (Fig. 4A), we'll decode from a growing window: at each time after the cue, from all the frames since the cue. Thanks to the Gram trick, that's just a cumulative sum. Accuracies that fall outside the shaded band are unlikely under chance (95% binomial interval).

# %%
cumulative = np.cumsum(grams, axis=0)
acc8_t = [decode(G, y8)[0] for G in cumulative]
acc_lr_t = [decode(G[np.ix_(lateral, lateral)], y_lr[lateral])[0] for G in cumulative]
# When the last frame used ends, relative to the cue (typical trial)
t_after = frame_offset + np.arange(1, n_after + 1) / S.frame_rate

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, acc, chance, n, title in [
    (axes[0], acc8_t, 1 / 8, n_trials, "8 directions"),
    (axes[1], acc_lr_t, 1 / 2, lateral.sum(), "Left vs. right"),
]:
    lo, hi = stats.binom.ppf([0.025, 0.975], n, chance) / n
    shade_periods(ax, window=False)
    ax.fill_between([0, t_after[-1]], lo, hi, color="C3", alpha=0.2, label="chance (95%)")
    ax.plot(t_after, acc, "o-", label="decoding accuracy")
    ax.set_xlabel("Time after the cue, at the end of the last frame used (s)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"{title}, from all frames since the cue")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# Decoding rises above chance about 3 s after the cue: neurons in LIP start encoding the target as soon as it's flashed, but blood volume takes a few seconds to follow. By the end of the memory period, the decoder is right on ~45% of trials (left vs. right: ~85%), and it keeps improving for a few more seconds as it takes in the frames around the saccade, before leveling off at ~58%. The paper also finds that decoding becomes significant within 3 s of the cue (Fig. 4A). Some of the late information is about the movement itself rather than the plan: by then, the monkey has made its saccade, and blood volume is still catching up. This lag is the price of measuring neural activity through hemodynamics—with fUS just as with fMRI.
#
# # The fMRI question
#
# We now have a decoder that works well on the native fUS data. Let's make the data progressively more fMRI-like, one property at a time.
#
# ## Spatial resolution
#
# We'll emulate larger voxels by averaging non-overlapping blocks of pixels. Note that we average the raw Power Doppler and *then* compute the percent change, as an fMRI voxel would sum signal across its volume. We start from the smoothed images we used for decoding, so that 0.1 mm "voxels" reproduce the decoder above, and drop blocks that fall mostly outside the valid image.

# %%
def block_average(A, f):
    """Average non-overlapping f x f pixel blocks over the last two axes, ignoring NaNs."""
    if f == 1:
        return A  # nothing to average
    Z, X = (A.shape[-2] // f) * f, (A.shape[-1] // f) * f
    blocks = A[..., :Z, :X].reshape(*A.shape[:-2], Z // f, f, X // f, f)
    n_valid = np.isfinite(blocks).sum(axis=(-3, -1))
    total = np.nansum(blocks, axis=(-3, -1))
    # Keep blocks that are mostly inside the valid image
    return np.where(n_valid > f * f / 2, total / np.maximum(n_valid, 1), np.nan)


def voxel_pct(f):
    """Percent change images (trials, frames, nz // f, nx // f) for voxels of f x f pixels."""
    return to_pct(block_average(D_smooth, f))


def voxel_responses(f):
    """Percent change of every valid voxel in every frame: (trials, frames, voxels)."""
    V = voxel_pct(f)
    return V[..., np.isfinite(V).all(axis=(0, 1))]


def voxel_gram(V):
    """Gram matrix of all the frames from the cue on, for voxel responses V (trials, frames, voxels)."""
    return sum(gram(after_cue(V, j)) for j in range(n_after))


block_sizes = [1, 2, 3, 5, 10, 15, 20]  # in pixels of ~0.1 mm
voxel_mm = np.array(block_sizes) * S.dx_mm

# %% [markdown]
# Here's what the angiogram and the direction map look like at increasingly coarse resolutions:

# %%
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for col, f in enumerate([1, 5, 15]):
    angio = block_average(np.where(valid, S.angiogram, np.nan), f)
    axes[0, col].imshow(np.log(angio), cmap="gray", extent=S.extent)
    axes[0, col].set_title(f"{f * S.dx_mm:.1f} mm voxels")

    Vp = voxel_pct(f)
    p, pref = direction_tuning(np.mean([after_cue(Vp, j) for j in avg_window], axis=0))
    good = np.isfinite(p) & (p < 1e-3)
    axes[1, col].imshow(direction_map(pref, good, angio), extent=S.extent)
    axes[1, col].set_title(f"Preferred direction, {f * S.dx_mm:.1f} mm voxels")
for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()

# %% [markdown]
# At 1.5 mm, the cluster has merged into about ten voxels. The coarse split between rightward-preferring voxels at the top and upward-preferring ones below survives, but the patchwork within it is gone. What about decoding?

# %%
acc8_res, acc_lr_res = [], []
for f in block_sizes:
    G = voxel_gram(voxel_responses(f))
    acc8_res.append(decode(G, y8, seeds=(0, 1))[0])
    acc_lr_res.append(decode(G[np.ix_(lateral, lateral)], y_lr[lateral], seeds=(0, 1))[0])
    print(f"{f * S.dx_mm:.1f} mm voxels: 8 directions {acc8_res[-1]:.1%}, left/right {acc_lr_res[-1]:.1%}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, acc, chance, title in [(axes[0], acc8_res, 1 / 8, "8 directions"), (axes[1], acc_lr_res, 1 / 2, "Left vs. right")]:
    ax.plot(voxel_mm, acc, "o-")
    ax.axhline(chance, color="k", linestyle="--", label="chance")
    ax.axvspan(1.5, 3.0, color="orange", alpha=0.2, label="typical fMRI voxels")
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(voxel_mm)
    ax.set_xticklabels([f"{v:.1f}" for v in voxel_mm])
    ax.set_xlabel("Voxel size (mm)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# Mild averaging (0.2-0.5 mm) slightly *helps*: it averages out noise faster than it destroys signal, and decoding holds up all the way to 1 mm voxels. Beyond that, it drops: at 1.5-2 mm, where the direction map has collapsed to a handful of voxels, 8-direction accuracy falls by 7-15 points, and left vs. right by 9-12 points. So resolution does matter for decoding here, but even 2 mm voxels keep much of the information, mixed together in ways a linear decoder can partly untangle.
#
# ## Temporal resolution
#
# fMRI volumes are typically acquired every 1-3 s (the repetition time, TR). Let's bin the frames after the cue into longer bins, and decode from each bin on its own.

# %%
frames_after = np.stack([after_cue(pct_smooth, j)[:, decode_valid] for j in range(n_after)], axis=1)

fig, ax = plt.subplots(figsize=(8, 4))
shade_periods(ax, window=False)
for TR in [1, 2, 3]:
    n_bins = n_after // TR
    acc = [decode(gram(frames_after[:, b * TR : (b + 1) * TR].mean(axis=1)), y8)[0] for b in range(n_bins)]
    ax.step(frame_offset + np.arange(n_bins + 1) * TR / S.frame_rate, acc + acc[-1:], where="post", label=f"TR = {TR} s")
ax.axhline(1 / 8, color="k", linestyle="--", label="chance")
ax.set_xlabel("Time after the cue (s)")
ax.set_ylabel("8-direction accuracy")
ax.set_title("Decoding from single time bins")
ax.legend(fontsize=8)
plt.show()
del frames_after

# %% [markdown]
# Longer bins actually help: averaging 2-3 frames averages out noise, and since the memory period lasts ~5 s and the hemodynamic response is slow anyway, little signal is lost. What longer TRs cost is *timing*: with 3 s bins, we can no longer tell precisely when the information appears. For decoding a sustained plan, sampling speed is not the bottleneck.
#
# ## Sensitivity
#
# Block-averaging doesn't just blur the image: it also averages out noise. To see how these two effects trade off, let's measure the **contrast-to-noise ratio** (CNR) of the direction signal. For each voxel, we take the difference between its mean responses to contralateral and ipsilateral saccades, and divide it by its trial-to-trial noise (a single-trial $d'$).

# %%
noise_by_size, dprime_by_size = [], []
for f in block_sizes:
    V = np.mean([after_cue(voxel_responses(f), j) for j in avg_window], axis=0)  # (trials, voxels), averaging window
    contrast = V[right].mean(axis=0) - V[left].mean(axis=0)
    noise = np.sqrt((V[right].var(axis=0) + V[left].var(axis=0)) / 2)
    dprime = np.sort(np.abs(contrast) / noise)
    noise_by_size.append(np.median(noise))
    dprime_by_size.append(dprime[-max(1, len(dprime) // 50) :].mean())  # best 2% of voxels

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(voxel_mm, noise_by_size, "o-")
ax1.set_ylabel("Single-trial noise (% CBV)")
ax1.set_title("Noise per voxel (median)")
ax2.plot(voxel_mm, dprime_by_size, "o-")
ax2.set_ylabel("d' (contralateral vs. ipsilateral)")
ax2.set_title("CNR of the most informative voxels (top 2%)")
for ax in (ax1, ax2):
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(voxel_mm)
    ax.set_xticklabels([f"{v:.1f}" for v in voxel_mm])
    ax.set_xlabel("Voxel size (mm)")
plt.tight_layout()
plt.show()

# %% [markdown]
# Up to 1 mm, the noise drops faster than the direction signal is diluted, so the CNR of the best voxels goes *up*. Beyond that, voxels start to mix patches that prefer different directions, and the CNR falls, just like decoding accuracy did. fMRI doesn't get the benefit of averaging for free: its noise at a given voxel size is set by the physics of MRI, and its signal is smaller. The paper quotes BOLD changes of ~0.4-2% for saccade tasks, versus ~10-30% CBV changes with fUS.
#
# So how much CNR can the decoder afford to lose? We can emulate a less sensitive modality by adding Gaussian noise to every voxel and frame, scaled so that the total noise is $k$ times the measured fUS noise (i.e. the added noise has standard deviation $\sqrt{k^2-1}$ times the original). We'll average over two independent noise draws, and decode across a grid of voxel sizes and noise factors.

# %%
rng = np.random.default_rng(0)
noise_factors = [1, 1.5, 2, 3, 5]
grid_sizes = [1, 3, 5, 10, 15, 20]


def add_noise(V, factor):
    """Add Gaussian noise so that each voxel's trial-to-trial noise is `factor` times larger."""
    if factor == 1:
        return V
    sigma = np.concatenate([V[y8 == c] - V[y8 == c].mean(axis=0) for c in range(8)]).std(axis=0)
    return V + rng.normal(size=V.shape) * sigma * np.sqrt(factor**2 - 1)


grid8 = np.zeros((len(noise_factors), len(grid_sizes)))
grid_lr = np.zeros_like(grid8)
for j, f in enumerate(grid_sizes):
    Vf = voxel_responses(f)
    V = np.stack([after_cue(Vf, t) for t in range(n_after)], axis=1)  # (trials, frames after the cue, voxels)
    for i, k in enumerate(noise_factors):
        for seed in (0, 1):
            noisy = add_noise(V, k)
            G = sum(gram(noisy[:, t]) for t in range(n_after))
            grid8[i, j] += decode(G, y8, seeds=(seed,))[0] / 2
            grid_lr[i, j] += decode(G[np.ix_(lateral, lateral)], y_lr[lateral], seeds=(seed,))[0] / 2

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, grid, chance, title in [(axes[0], grid8, 1 / 8, "8 directions"), (axes[1], grid_lr, 1 / 2, "Left vs. right")]:
    im = ax.imshow(grid, cmap="magma", vmin=chance, vmax=grid.max(), origin="lower", aspect="auto")
    for i in range(len(noise_factors)):
        for j in range(len(grid_sizes)):
            ax.text(j, i, f"{grid[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="w" if grid[i, j] < (chance + grid.max()) / 2 else "k")
    ax.set_xticks(range(len(grid_sizes)))
    ax.set_xticklabels([f"{f * S.dx_mm:.1f}" for f in grid_sizes])
    ax.set_yticks(range(len(noise_factors)))
    ax.set_yticklabels([f"{k}x" for k in noise_factors])
    ax.set_xlabel("Voxel size (mm)")
    ax.set_ylabel("Noise relative to fUS")
    ax.set_title(f"{title} (chance {chance:.1%})")
    plt.colorbar(im, ax=ax, label="Accuracy")
plt.tight_layout()
plt.show()

# %% [markdown]
# Reading the grid along each axis tells the story:
#
# * **Horizontally** (bigger voxels, same noise), accuracy holds up to 1 mm and then drops, by 15 points at 2 mm.
# * **Vertically** (same voxels, more noise), accuracy collapses: doubling the noise costs about twice as much as going from 0.1 mm to 2 mm voxels, and by 3-5x the 8-direction decoder approaches chance.
# * The two costs add up: with 1.5 mm voxels *and* twice the noise, 8-direction accuracy is down to about a quarter.
#
# Here's the column for fMRI-like 1.5 mm voxels:

# %%
j15 = grid_sizes.index(15)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(noise_factors, grid8[:, j15], "o-", label="8 directions")
ax.plot(noise_factors, grid_lr[:, j15], "s-", label="left vs. right")
ax.axhline(1 / 8, color="C0", linestyle="--", linewidth=1, label="chance (8 directions)")
ax.axhline(1 / 2, color="C1", linestyle="--", linewidth=1, label="chance (left vs. right)")
ax.set_xscale("log")
ax.minorticks_off()
ax.set_xticks(noise_factors)
ax.set_xticklabels([f"{k}x" for k in noise_factors])
ax.set_xlabel("Noise relative to fUS")
ax.set_ylabel("Accuracy")
ax.set_title("1.5 mm voxels: decoding vs. sensitivity")
ax.legend(fontsize=8)
plt.show()

# %% [markdown]
# ## Where would fMRI sit?
#
# Placing fMRI precisely on this axis would require fMRI data from the same task: what matters is the size of the *direction-specific* part of the BOLD response relative to fMRI noise at that voxel size, and neither number is well established for monkey LIP. What we can say is that the decoder's margin is thin. Losing a factor of 2 in CNR costs about twice as much as making voxels 20 times larger, and a factor of 3-5 brings 8-direction decoding close to chance. BOLD changes are roughly an order of magnitude smaller than the CBV changes measured with fUS, only partly offset by fMRI's noise also being smaller in percent terms, so it is plausible that fMRI sits in that regime.
#
# In this session, then, **fUS's edge for decoding seems to come more from its sensitivity than from its resolution**, though resolution matters too once voxels grow beyond ~1 mm. This complements the paper rather than contradicting it: resolving the *maps*—where each direction is represented—genuinely requires small voxels, as we saw above. Decoding asks a different question, and linear decoders are good at pulling information out of mixtures.
#
# # Caveats
#
# This is a deliberately simple analysis, and several of its simplifications favor fMRI:
#
# * **A single 2D slice.** Our fUS voxels are 400 μm thick, and we only averaged within the plane. A 1.5 mm fMRI voxel also averages across ~4 such slices, mixing in more differently-tuned tissue. On the other hand, fMRI covers the whole brain, and a decoder could pool information from many regions.
# * **CBV vs. BOLD.** BOLD signals are dominated by draining veins downstream of the active tissue, which blurs them beyond the nominal voxel size. Block-averaging Power Doppler doesn't capture that.
# * **Idealized noise.** We added independent Gaussian noise. Real fMRI noise is dominated by physiological fluctuations that are correlated across voxels and time, which is harder on a decoder than independent noise.
# * **One session.** All numbers come from one session of one monkey: the session the paper chose as its example. Other sessions can decode worse, for instance when the imaging plane only partly covers the active region. The paper also shows that decoders can be trained on one day and tested months later.
#
# # Exercises
#
# * Repeat the analysis with session 76, the paper's example session for monkey L (recorded on 24 June 2021, "S20210624"), and compare with the paper's Figs. 3D-F and 4D-F. One thing to watch: on 12 of the 555 trials, the cue comes after 6 s, so drop those trials to keep 10 frames after the cue in every trial.
# * For other sessions, check the "units" plot first. In session 39, every pixel carries a large constant offset that you'll need to estimate and subtract (hint: where does the line in that plot cross zero standard deviation?).
# * The paper decodes the horizontal (left, center, right) and vertical (down, center, up) components of each saccade with two separate LDAs, and combines their predictions. Try it. Does it change the accuracy or the pattern of errors?
# * Restrict the decoder to the frames before the saccade (accounting for the hemodynamic lag). How much of the information is about the *plan* rather than the movement?
# * Replace the independent noise with noise that is correlated across voxels (e.g. a shared global fluctuation). How does that change the picture?
#
# # References
#
# * [Griggs et al. 2025](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12488887/). Functional ultrasound neuroimaging reveals mesoscopic organization of saccades in the lateral intraparietal area. *Nature Communications*. [Dataset](https://data.caltech.edu/records/p5jan-02r60).
# * [Norman et al. 2021](https://pubmed.ncbi.nlm.nih.gov/33756104/). Single-trial decoding of movement intentions using functional ultrasound neuroimaging. *Neuron*.
# * [Macé et al. 2011](https://www.nature.com/articles/nmeth.1641). Functional ultrasound imaging of the brain. *Nature Methods*.
