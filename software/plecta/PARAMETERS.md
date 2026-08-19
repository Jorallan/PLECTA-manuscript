# PLECTA parameter guide

This document describes every parameter in the frozen PLECTA mask-only core,
using the values materialized in [`params.json`](params.json). It also records
which values are expected to depend on image sampling or mask quality and which
should normally remain fixed.

All values were fixed before the reported evaluation. The recommendations below
apply only when transferring PLECTA to a new acquisition or segmentation
system; they are not permission to tune on held-out or test data.

## Recommendation labels

| Label | Meaning |
|---|---|
| **System-dependent** | Rescale or calibrate on representative development masks when pixel size, mask noise, filament geometry, or expected gap size changes. |
| **Prefer fixed** | Keep the frozen value initially. Change it only for a materially different domain, using a declared development set. |
| **Keep fixed/off** | Part of the evaluated method definition or a deliberately disabled option. Changing it creates a method variant. |

Parameters beginning with `j_` govern links within a merged junction node.
Parameters beginning with `g_` govern links across missing mask evidence.
Angles are in radians. Frame windows use skeleton arclength, gap lengths use
Euclidean tip-to-tip distance, and the remaining pixel thresholds count pixels
on the one-pixel-wide skeleton.

## Graph construction and frame estimation

| Parameter | Current value | Recommendation | Affects | Notes and effect of increasing the value |
|---|---:|---|---|---|
| `spur_px` | 3 px | **System-dependent, constrained** | Removal of short skeleton hairs | Removes more short endpoint branches and may erase genuine short ends. The frozen implementation requires a value no greater than 3 to remain consistent with the evaluation fragment definition. |
| `bridge_px` | 5 px | **System-dependent, constrained** | Formation of crossing nodes | Absorbs more short connectors between junction clusters into one node. The absorbed connector ceases to be a separate arm. The implementation requires a value no greater than 5. |
| `absorb_free_px` | 2 px | **System-dependent, constrained** | Removal of junction debris | Absorbs more short free-ended arms into their adjacent junction. Excessive values remove genuine terminal pieces. The implementation requires a value no greater than 5. |
| `join_px` | 14 px | **System-dependent** | Size of jointly solved crossing regions | Makes nearby crossings part of one exact matching problem while retaining their connecting arm. Excessive values can couple unrelated crossings and increase matching cost. |
| `window_local` | 24 px arclength | **System-dependent** | Initial tangent and curvature estimates | More support gives smoother, less noisy directions but averages over tight bends. |
| `window_chain` | 55 px arclength | **System-dependent** | Frames after provisional chains form | More chain context stabilizes directions on short arms, but can smooth real curvature and propagate an incorrect provisional link. It should normally be at least `window_local`. |
| `min_quadratic` | 18 px arclength | **System-dependent** | Linear versus quadratic frame fitting | A quadratic fit, and therefore nonzero curvature, is used only when the available span reaches this value. Increasing it produces fewer and more conservative curvature estimates. |
| `n_rounds` | 8 | **Prefer fixed** | Number of frame-and-matching refinement rounds | More rounds increase runtime and allow more re-estimation. With the current annealing schedule, only round 8 reaches the full configured gates and is returned. |
| `anneal_start` | 0.85 | **Prefer fixed** | Conservativeness of early refinement rounds | Lower values admit fewer early links. The schedule scales the unmatched prices and the two gap-angle gates linearly from this fraction to 1.0. A value of 1.0 disables annealing. It does not scale link-cost weights or `gap_max_len`. |

## Junction-link parameters

Junction links have no physical gap, so their length weight is always zero.
There is no separate hard distance or angle gate at a junction; an edge is
offered to the exact partial matcher only when its cost is below the combined
unmatched price of its two stubs.

| Parameter | Current value | Recommendation | Affects | Notes and effect of increasing the value |
|---|---:|---|---|---|
| `j_w_direct` (`w_d`) | 0.6 | **Prefer fixed** | Reversal-angle term, `w_d * theta` | More strongly favors antiparallel outward tangents, corresponding to a straight continuation through the crossing. |
| `j_w_turn` (`w_t`) | 1.2 | **Prefer fixed** | Chord-turn term, `w_t * q(d) * (phi_a + phi_b)` | More strongly penalizes lateral offset and arms that do not point toward one another. |
| `j_chord_floor` (`d_0`) | 4 px | **System-dependent** | Short-range fading of chord-angle evidence | Here `q(d) = min(1, d / 4)`. Increasing the value distrusts the chord over a larger distance, reducing raster quantization effects but weakening lateral-alignment evidence. |
| `j_w_kappa` (`w_kappa`) | 1.0 | **Prefer fixed at the reference resolution** | Curvature-continuity term | Larger values more strongly favor `kappa_a + kappa_b` near zero. Its effective strength also depends on the frame windows, `min_quadratic`, pixel scale, and the 30 px curvature normalization. |
| `j_unmatched` (`p_j`) | 0.62 | **Prefer fixed after development calibration** | Willingness to leave a junction stub unpaired | Each free junction stub costs 0.62. For two ordinary junction stubs, an edge is admissible only when `C < 2 * 0.62 = 1.24`. Increasing the value makes junction linking more eager and genuine terminations less likely. |
| `j_gap_relief` | 0 (off) | **Keep fixed/off** | Competition between a junction link and a possible gap link | When enabled, a promising alternative gap continuation can reduce a stub's price for declining a junction partner. It was disabled in the evaluated method; enabling it creates a method variant. |

## Gap-link parameters

A gap edge must pass the distance gate, both hard angle gates, and the
cost-versus-unmatched-price gate. These conditions are cumulative.

| Parameter | Current value | Recommendation | Affects | Notes and effect of increasing the value |
|---|---:|---|---|---|
| `gap_max_len` | 85 px | **System-dependent** | Hard gap-candidate radius | Allows repair of longer mask breaks, but increases false candidates, false merges, and computation. This is the maximum candidate length; it is distinct from `g_len_scale` and is not annealed. |
| `g_w_direct` (`w_d`) | 0.2 | **Prefer fixed** | Gap reversal-angle cost | Increasing it requires the two endpoint tangents to be more nearly antiparallel. |
| `g_w_turn` (`w_t`) | 1.0 | **Prefer fixed** | Alignment of both endpoints with the joining chord | Increasing it favors straight, collinear bridges and more strongly rejects lateral displacement. |
| `g_w_len` (`w_l`) | 0.15 | **Prefer fixed jointly with `g_len_scale`** | Soft preference for shorter gaps | Multiplies `d / l_0`. Increasing it penalizes longer gaps more strongly. Only the ratio `g_w_len / g_len_scale` determines the slope of this term. |
| `g_len_scale` (`l_0`) | 60 px | **System-dependent** | Normalization of gap length | Makes the gap-length term dimensionless. Increasing it weakens the soft length penalty. It is a reference scale, not a maximum; rescale it with pixel density. |
| `g_chord_floor` (`d_0`) | 3 px | **System-dependent** | Short-range fading of chord-angle evidence | Increasing it suppresses chord evidence over a longer distance. With the current value, gaps of at least 3 px receive the full chord-turn cost. |
| `g_w_kappa` (`w_kappa`) | 0.5 | **Prefer fixed at the reference resolution** | Curvature continuity across gaps | Increasing it favors bridges that preserve fitted bending direction. It is sensitive to noisy quadratic fits and to pixel scale. |
| `g_unmatched` (`p_g`) | 0.30 | **Prefer fixed after development calibration** | Willingness to bridge a gap | Each free gap endpoint costs 0.30, so a gap edge must satisfy `C < 2 * 0.30 = 0.60`. Increasing the value creates more bridges; decreasing it preserves more endpoints. |
| `gap_max_theta` | 0.40 rad (22.9 degrees) | **Prefer fixed after development calibration** | Hard tangent-disagreement gate | Increasing it permits greater directional mismatch. In round 1, annealing reduces the limit to `0.85 * 0.40 = 0.34` rad. |
| `gap_max_phi` | 0.28 rad (16.0 degrees) at each end | **Prefer fixed after development calibration** | Hard endpoint-to-chord gate | Increasing it admits more bent or laterally displaced bridges. This is one of the strongest controls on false gap links. In round 1 the limit is 0.238 rad. |

## Effective constants outside `params.json`

These values and rules also affect the frozen execution path, but are not
user-facing fields in `params.json`.

| Constant or rule | Current value | Recommendation | Notes |
|---|---:|---|---|
| `kappa_scale` (curvature normalization) | 30 px | **Fixed for reproduction; scale for a new resolution** | Multiplies curvature in px^-1 so that it is comparable to the angular terms. If the same physical geometry is sampled at `s` times as many pixels per unit length, this factor should also be multiplied by `s`, or the masks should be resampled to the reference scale. |
| Reliable-frame support | At least 3 samples spanning at least 3 px | **Prefer fixed at the reference resolution** | Stubs below this support do not participate in matching because their tangents are marked unreliable. |
| `min_isolated_px` | 6 px | **Fixed for reported evaluation** | Omits only isolated, one-arm, no-junction components shorter than 6 px. It is tied to the evaluation fragment floor. |
| Junction definition | Skeleton degree at least 3 | **Keep fixed** | A topological definition rather than a fitted threshold. |
| `include_nodes` | `True` | **Keep fixed** | Every chain touching a junction receives that junction's pixels, allowing crossing pixels to belong to multiple instances. |
| Gap rendering | Identity link only; no pixels painted across the gap | **Keep fixed** | Gap links join instance identities but do not invent centreline evidence in the fixed mask-only output. |
| Cycle guard | Remove the longest link from a closed matched component | **Keep fixed** | Enforces the method's open-filament assumption. Closed or looping filaments are outside the current model. |

## Transfer to another imaging system

The safest approach is to resample new masks to the physical pixel scale used
for the frozen configuration and retain every parameter unchanged. If that is
not possible:

1. Let `s` be the new number of pixels per physical unit divided by the
   reference number of pixels per physical unit.
2. Initially multiply all length-valued parameters by `s`: `spur_px`,
   `bridge_px`, `absorb_free_px`, `join_px`, `window_local`, `window_chain`,
   `min_quadratic`, `j_chord_floor`, `gap_max_len`, `g_len_scale`, and
   `g_chord_floor`. Integer and implementation limits still apply.
3. Account for the hard-coded curvature normalization as described above.
4. Keep weights, unmatched prices, angle gates, `n_rounds`, and
   `anneal_start` unchanged for the first transfer evaluation.
5. If the segmentation failure distribution or filament morphology is
   materially different, calibrate any further changes on development data and
   freeze the complete configuration before evaluating it.
