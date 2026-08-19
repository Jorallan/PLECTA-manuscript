"""Shared shaded-tube 3-D rendering for the depth figures.

Swept-tube meshes come from `plecta.depth.tube_mesh` (stubmatch); this module
only draws them: Lambert-shaded quads, depth-coloured through viridis, one
Poly3DCollection per panel so matplotlib's average-z sort orders every quad
globally (painter's algorithm across instances, not per instance).
"""
import sys

import numpy as np

sys.path.insert(0, r"C:\Repos\stubmatch")
from plecta.depth import tube_mesh  # noqa: E402,F401  (re-exported)

import matplotlib  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

VIRIDIS = matplotlib.colormaps["viridis"]


def z_colour(t):
    """Depth colour ramp, clipped to the readable middle of viridis."""
    return VIRIDIS(0.10 + 0.85 * float(np.clip(t, 0, 1)))[:3]


def draw_tubes_shaded(ax, tubes, z_lo, span, extent=512,
                      light=(0.4, -0.45, 0.80), elev=34, azim=-56,
                      zoom=1.35, z_stretch=0.62):
    """Draw (verts, faces, z_ref) tubes as lit solids into a 3-D axes."""
    light = np.asarray(light, float)
    light = light / np.linalg.norm(light)
    quads, cols = [], []
    for verts, faces, z_ref in tubes:
        if not len(verts):
            continue
        quad = verts[faces]                       # (M, 4, 3)
        n = np.cross(quad[:, 1] - quad[:, 0], quad[:, 3] - quad[:, 0])
        norm = np.linalg.norm(n, axis=1)
        norm[norm < 1e-9] = 1.0
        n = n / norm[:, None]
        lam = np.clip(n @ light, 0.0, 1.0)
        shade = 0.42 + 0.58 * lam
        base = np.asarray(z_colour((z_ref - z_lo) / span))
        quads.append(quad)
        cols.append(shade[:, None] * base[None, :])
    if quads:
        all_cols = np.concatenate(cols)
        # Poly3DCollection anti-aliases each quad's border against the
        # background independently, so with edgecolor="none" every seam
        # between adjacent facets still shows as a faint checkerboard line --
        # visible as a fish-scale texture once the cross-section has enough
        # sides to look round. Painting each quad's edge in its OWN face
        # colour fills that seam rather than leaving it translucent, which
        # is what actually makes the surface read as a smooth solid.
        coll = Poly3DCollection(np.concatenate(quads),
                                facecolors=all_cols,
                                edgecolors=all_cols, linewidths=0.35)
        coll.set_zsort("average")
        ax.add_collection3d(coll)
    ax.set_xlim(0, extent)
    ax.set_ylim(extent, 0)
    ax.set_zlim(z_lo - 10, z_lo + span + 10)
    ax.set_box_aspect((1, 1, z_stretch), zoom=zoom)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
