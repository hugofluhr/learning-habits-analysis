"""
Two-panel TR-collision timeline: an early-phase trial (cue lands right after a TR
boundary -> plenty of room, events stay in the cue's TR bin) vs a late-phase trial
(cue lands right before the next boundary -> second-stim/response spill into the
next TR bin). Real trial timing from bbt.csv (sub-01, learning1), not illustrative
numbers -- see session note this generates for.

Run: conda run -n neuroim python make_tr_collision_figure.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TR = 2.33384  # multivariate/run_glmsingle.py

df = pd.read_csv('/Users/hugofluhr/phd_local/data/LearningHabits/dev_sample/bbt.csv')
sub = df[(df['sub_id'] == 'sub-01') & (df['block'] == 'learning1')].copy()
scan_start = sub['t_first_stim'].min()
sub['phase'] = (sub['t_first_stim'] - scan_start) % TR
resp = sub[sub['action'].notna()].copy()

early = resp.loc[resp['phase'].idxmin()]
# Late-phase example: picked from a mid-range phase (not idxmax) so the cue onset
# itself stays clearly readable mid-bin, away from the TR boundary line, while
# second-stim/response still visibly spill into the next bin -- the idxmax trial
# had the cue landing right on the boundary, making the panel hard to read.
late_band = resp[(resp['phase'] > 1.6) & (resp['phase'] < 1.9)]
late = late_band.loc[(late_band['phase'] - 1.72).abs().idxmin()]

EVENTS = [
    ('first_stim', 't_first_stim', 'cue onset'),
    ('second_stim', 't_second_stim', 'second stimulus'),
    ('action', 't_action', 'response'),
    ('purple_frame', 't_purple_frame', 'feedback frame'),
]
COLORS = {'first_stim': '#1f77b4', 'second_stim': '#ff7f0e',
          'action': '#2ca02c', 'purple_frame': '#9467bd'}

fig, axes = plt.subplots(2, 1, figsize=(13, 5.75), sharex=True)

for ax, trial, title in [(axes[0], early, 'Early-phase trial — cue lands just after a TR boundary'),
                          (axes[1], late, 'Late-phase trial — cue lands later in its TR bin')]:
    phase = trial['phase']
    t0 = trial['t_first_stim'] - phase  # start of the cue's TR bin, in absolute time
    # TR grid, in time-since-bin-start coordinates, covering 2 bins
    for k in range(3):
        ax.axvline(k * TR, color='grey', lw=1, ls='--', zorder=0)
    ax.axvspan(0, TR, color='#1f77b4', alpha=0.06)
    ax.axvspan(TR, 2 * TR, color='#d62728', alpha=0.06)

    # Per-event vertical offset (points): staggered so the two "below" labels
    # (second_stim, purple_frame) never collide even when their markers land
    # close together horizontally, regardless of exact font size.
    Y_OFFSET = {'first_stim': 20, 'second_stim': -26, 'action': 20, 'purple_frame': -54}
    for key, col, label in EVENTS:
        x = trial[col] - t0
        same_tr = x < TR
        ax.scatter([x], [0], s=110, color=COLORS[key], zorder=3,
                  marker='o' if same_tr else 'X',
                  edgecolor='k' if not same_tr else None, linewidth=1.2)
        ax.annotate(label, (x, 0), xytext=(0, Y_OFFSET[key]),
                   textcoords='offset points', ha='center', fontsize=17,
                   color=COLORS[key], fontweight='bold')

    ax.set_yticks([])
    ax.set_xlim(-0.15, 2 * TR + 0.15)
    ax.set_title(title, fontsize=19, loc='left')
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)

axes[1].set_xlabel('Time since start of cue\'s TR bin (s)', fontsize=17)
for ax in axes:
    ax.tick_params(axis='x', labelsize=15)
axes[0].text(TR / 2, 0.68, "cue's TR bin", ha='center', fontsize=15, color='#1f77b4', transform=axes[0].get_xaxis_transform())
axes[0].text(TR * 1.5, 0.68, 'next TR bin', ha='center', fontsize=15, color='#d62728', transform=axes[0].get_xaxis_transform())

fig.suptitle('Same-TR collision depends on cue phase within its TR (TR = 2.33 s, sub-01/learning1)',
            fontsize=20, y=1.04)
fig.tight_layout()
out = '/Users/hugofluhr/phd_local/repositories/learning-habits-analysis/multivariate/presentation_assets/32-tr-collision-timeline.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
print('saved', out)
print(f"early trial: phase={early['phase']:.3f}s, t_first_stim={early['t_first_stim']:.2f}")
print(f"late  trial: phase={late['phase']:.3f}s, t_first_stim={late['t_first_stim']:.2f}")
