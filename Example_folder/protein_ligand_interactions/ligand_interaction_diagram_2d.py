#!/usr/bin/env python3
"""
2D protein-ligand interaction diagram: a flat skeletal depiction of the
ligand surrounded by residue nodes, built independently of any commercial
package (OpenEye, Schrodinger, ...) -- an original layout/icon design, not a
copy of any of those tools' look.

Reuses the 3D interaction detection from protein_ligand_interactions.py (same
geometric criteria, same CCD-derived ligand bond model) and lays the result
out as a 2D diagram:
  - ligand drawn as a bold skeletal structure (RDKit 2D coordinates)
  - a grey outline (no fill) traces the ligand's own footprint only -- the
    exact silhouette of a union of small disks, one per ligand atom, styled
    solid/dashed per arc from each atom's real solvent-accessible surface
    area (buried vs. solvent-exposed), not drawn freehand
  - each contacting residue drawn as a rounded-rectangle node around the
    ligand (a subtle dark shade over the chip = backbone+side-chain contact,
    plain = side-chain only; a plain circle = water, a diamond = a metal ion)
      - node fill:   lavender = ligand is H-bond acceptor here
                     salmon   = ligand is H-bond donor here
                     tan      = metal chelation
                     green    = pi-stacking
                     white    = van der Waals / hydrophobic contact only
                     red outline = donor-donor / acceptor-acceptor clash
  - typed connecting lines, arrowed donor -> acceptor for H-bonds and
    ligand -> metal for chelation; salt bridges and pi-stacking are drawn as
    plain (non-directional) solid lines; hydrophobic-only contacts get no
    line at all, matching standard convention (the node itself signals the
    contact)

This is a simplified, geometry-only reconstruction of that diagram style, not
a pixel clone of any one tool: donor/acceptor roles come from static
side-chain chemistry tables and ligand H-counts (no explicit ligand
hydrogens are placed, so H-bond *angles* are not evaluated -- only atom
identity, distance, and role compatibility).

Usage:
    python3 ligand_interaction_diagram_2d.py structure.cif
    python3 ligand_interaction_diagram_2d.py structure.cif --ligand G4K --out-dir .
    python3 ligand_interaction_diagram_2d.py structure.cif --ligand-cif my_ligand.cif
"""

import argparse
import math
import sys
import warnings
from pathlib import Path

import numpy as np
with warnings.catch_warnings():  # benign numpy/scipy version-range mismatch notice
    warnings.simplefilter("ignore", UserWarning)
    from scipy.optimize import minimize

import matplotlib
if "--interactive" in sys.argv:
    # A real GUI backend is needed for the draggable window; picked before
    # pyplot is imported since matplotlib can't reliably switch afterwards.
    for _backend in ("MacOSX", "TkAgg", "Qt5Agg"):
        try:
            matplotlib.use(_backend)
            break
        except ImportError:
            continue
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
# A humanist sans-serif rather than matplotlib's default DejaVu Sans, which is
# what gives most auto-generated matplotlib figures (and, incidentally, a lot
# of OpenEye-style output) their recognizable look.
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Avenir Next", "Avenir", "Futura", "Verdana", "DejaVu Sans"]
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch
from rdkit import Chem
from rdkit.Chem import rdCoordGen
from rdkit.Geometry import Point3D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protein_ligand_interactions import (  # noqa: E402
    METAL_RESNAMES,
    WATER_RESNAMES,
    apply_ligand_protonation,
    compute_growth_room,
    compute_residue_pkas,
    find_ligand_residues,
    find_polar_and_hydrophobic_contacts,
    find_pi_stacking,
    find_salt_bridges,
    ligand_to_rdkit_mol,
    load_structure,
    protein_atoms,
    register_ligand_cif_override,
    residue_charge_sign,
    residue_label,
)
from Bio.PDB import NeighborSearch  # noqa: E402

# ---------------------------------------------------------------------------
# Chemistry tables (static, standard amino-acid H-bond donor/acceptor atoms)
# ---------------------------------------------------------------------------
PROTEIN_SIDECHAIN_DONORS = {
    "ARG": {"NE", "NH1", "NH2"}, "ASN": {"ND2"}, "GLN": {"NE2"},
    "HIS": {"ND1", "NE2"}, "LYS": {"NZ"}, "SER": {"OG"}, "THR": {"OG1"},
    "TYR": {"OH"}, "TRP": {"NE1"}, "CYS": {"SG"},
}
PROTEIN_SIDECHAIN_ACCEPTORS = {
    "ASN": {"OD1"}, "ASP": {"OD1", "OD2"}, "GLN": {"OE1"}, "GLU": {"OE1", "OE2"},
    "HIS": {"ND1", "NE2"}, "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"},
    "CYS": {"SG"}, "MET": {"SD"},
}
BACKBONE_DONOR_ATOM, BACKBONE_ACCEPTOR_ATOM = "N", "O"

METAL_COORDINATION_CUTOFF = 2.8

# Side-chain protonation state is not something a deposited PDB/mmCIF file
# records (no H atoms, no formal charge field for standard residues). Real
# per-residue pKa (PROPKA, via compute_residue_pkas) is used when available;
# residue_charge_sign() falls back to the standard textbook assumption
# (Asp/Glu deprotonated, Lys/Arg protonated -- Cys/Tyr/His only when a real
# pKa says so) otherwise. See protein_ligand_interactions.residue_charge_sign.
# Mathtext (${}^+$), not a bare Unicode superscript char -- matplotlib's math
# renderer always has these glyphs regardless of which text font is active,
# where a system sans-serif font (e.g. Avenir) can be missing them outright.
CHARGE_SIGN_CHAR = {1: "${}^+$", -1: "${}^-$", 0: ""}

CPK_COLORS = {
    "N": "#2b5fd9", "O": "#d32f2f", "S": "#c9a227", "F": "#3a9b3a",
    "CL": "#3a9b3a", "BR": "#8c2f2f", "P": "#c96f2f",
}

FILL = {
    "acceptor": "#c9c9f2", "donor": "#f4c2cc", "clash": "#ffffff",
    "chelator": "#f0c987", "pi": "#c9e6c2", "contact": "#ffffff", "water": "#eaf3fb",
    "metal": "#f0c987",
}
LINE_COLOR = {
    "acceptor": "#6f6fbf", "donor": "#c96f8a", "clash": "#c0392b",
    "salt": "#c0392b", "chelator": "#c98f2f", "pi": "#3f8c3f",
}

NODE_W, NODE_H = 1.55, 0.92
NODE_R = math.hypot(NODE_W, NODE_H) / 2  # effective radius, used for line trimming/layout spacing
PI_MARKER_R = 0.32
BOND_LW = 2.6
POCKET_LIGAND_DISK_R = 0.85
# Clearance from a residue's anchor point (on the ligand) to its node centre:
# needs to clear the pocket outline disk around that atom (POCKET_LIGAND_DISK_R)
# plus the chip's own half-diagonal (NODE_R) plus a visible gap, or the chip
# sits on top of the ligand skeleton/pocket outline instead of beside it.
NODE_ANCHOR_PAD = POCKET_LIGAND_DISK_R + NODE_R + 0.5

# --interactive window cap (inches) -- a large/spread-out complex can size
# the batch-mode figure well past a screen's usable area, making dragging
# impractical; the on-screen window is shrunk to fit (see run_interactive),
# while a save ('s') temporarily restores the full size so PNG/SVG quality
# matches batch mode exactly.
INTERACTIVE_MAX_W = 11.0


# ---------------------------------------------------------------------------
# Role classification
# ---------------------------------------------------------------------------
def protein_atom_roles(atom):
    res = atom.get_parent()
    name = atom.get_name()
    donor = (name == BACKBONE_DONOR_ATOM and res.resname != "PRO") or \
            name in PROTEIN_SIDECHAIN_DONORS.get(res.resname, set())
    acceptor = (name == BACKBONE_ACCEPTOR_ATOM) or \
               name in PROTEIN_SIDECHAIN_ACCEPTORS.get(res.resname, set())
    return donor, acceptor


def ligand_atom_roles(mol, atom_idx):
    atom = mol.GetAtomWithIdx(atom_idx)
    symbol = atom.GetSymbol().upper()
    if symbol not in ("N", "O", "S"):
        return False, False
    donor = atom.GetTotalNumHs() > 0
    acceptor = True
    return donor, acceptor


def classify_polar_contact(mol, lig_atom_idx, prot_atom):
    lig_donor, lig_acceptor = ligand_atom_roles(mol, lig_atom_idx)
    prot_donor, prot_acceptor = protein_atom_roles(prot_atom)
    valid_as_acceptor = lig_acceptor and prot_donor  # ligand accepts, protein donates
    valid_as_donor = lig_donor and prot_acceptor      # ligand donates, protein accepts
    if valid_as_acceptor and not valid_as_donor:
        return "acceptor"
    if valid_as_donor and not valid_as_acceptor:
        return "donor"
    if valid_as_donor and valid_as_acceptor:
        return "acceptor"  # ambiguous both-ways; show as one bond, acceptor by convention
    if (lig_donor and prot_donor) or (lig_acceptor and prot_acceptor and not lig_donor and not prot_donor):
        return "clash"
    return None  # role unknown for this protein atom name; skip rather than mislabel


# ---------------------------------------------------------------------------
# Metal / chelator detection
# ---------------------------------------------------------------------------
def find_chelation(structure, ligand_residue):
    lig_atoms = [a for a in ligand_residue if a.element != "H"]
    hits = []
    for res in structure[0].get_residues():
        if res.resname not in METAL_RESNAMES:
            continue
        metal_atom = next(iter(res), None)
        if metal_atom is None:
            continue
        for lig_atom in lig_atoms:
            dist = lig_atom - metal_atom
            if dist <= METAL_COORDINATION_CUTOFF:
                hits.append((lig_atom, metal_atom, res, dist))
    return hits


# ---------------------------------------------------------------------------
# Ligand 2D layout
# ---------------------------------------------------------------------------
def build_2d_ligand(ligand_residue, rotate_deg=0.0, flip_h=False, flip_v=False):
    mol, atoms = ligand_to_rdkit_mol(ligand_residue)
    if mol is None:
        raise RuntimeError(f"No usable bond model for ligand {ligand_residue.resname}; "
                            f"cannot build a 2D depiction.")
    mol2d = Chem.Mol(mol)
    mol2d.RemoveAllConformers()
    rdCoordGen.AddCoords(mol2d)
    conf = mol2d.GetConformer()
    if flip_h or flip_v or rotate_deg:
        theta = math.radians(rotate_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        for i in range(mol2d.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            x, y = (-p.x if flip_h else p.x), (-p.y if flip_v else p.y)
            conf.SetAtomPosition(i, Point3D(x * cos_t - y * sin_t, x * sin_t + y * cos_t, 0.0))
    positions = {i: (conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y) for i in range(mol2d.GetNumAtoms())}
    name_to_idx = {atoms[i].get_name(): i for i in range(len(atoms))}
    return mol2d, atoms, positions, name_to_idx


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def unit(v):
    n = math.hypot(*v)
    return (v[0] / n, v[1] / n) if n else (0.0, 0.0)


def trim_segment(p1, p2, r1=0.0, r2=0.0):
    ux, uy = unit((p2[0] - p1[0], p2[1] - p1[1]))
    a = (p1[0] + ux * r1, p1[1] + uy * r1)
    b = (p2[0] - ux * r2, p2[1] - uy * r2)
    return a, b


def bond_ring_membership(mol):
    """Map each ring bond index to the atom indices of (one of) its ring(s),
    so double bonds inside a ring can be offset toward the ring interior."""
    ri = mol.GetRingInfo()
    mapping = {}
    for atom_ring, bond_ring in zip(ri.AtomRings(), ri.BondRings()):
        for b in bond_ring:
            mapping.setdefault(b, atom_ring)
    return mapping


def draw_bond(ax, p1, p2, order, ring_centroid):
    ux, uy = unit((p2[0] - p1[0], p2[1] - p1[1]))
    perp = (-uy, ux)
    length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    # Primary line always spans the full (already label-trimmed) bond so
    # adjacent bonds visually meet at every vertex.
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="black", linewidth=BOND_LW,
             solid_capstyle="round", solid_joinstyle="round", zorder=2)

    if order == 1:
        return

    # Trim/offset scale with bond length so short (e.g. exocyclic C=O) bonds
    # don't get reduced to an oddly floating stub.
    trim = min(0.16, length * 0.22)

    if order == 3:
        offset = min(0.13, length * 0.16)
        for sign in (-1, 1):
            a = (p1[0] + perp[0] * offset * sign, p1[1] + perp[1] * offset * sign)
            b = (p2[0] + perp[0] * offset * sign, p2[1] + perp[1] * offset * sign)
            a, b = trim_segment(a, b, trim, trim)
            ax.plot([a[0], b[0]], [a[1], b[1]], color="black", linewidth=BOND_LW * 0.8,
                     solid_capstyle="round", zorder=2)
        return

    # order == 2: one short accent line, offset toward the ring interior when known
    offset = min(0.16, length * 0.2)
    sign = 1.0
    if ring_centroid is not None:
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        if (ring_centroid[0] - mid[0]) * perp[0] + (ring_centroid[1] - mid[1]) * perp[1] < 0:
            sign = -1.0
    o = offset * sign
    a = (p1[0] + perp[0] * o, p1[1] + perp[1] * o)
    b = (p2[0] + perp[0] * o, p2[1] + perp[1] * o)
    a, b = trim_segment(a, b, trim, trim)
    ax.plot([a[0], b[0]], [a[1], b[1]], color="black", linewidth=BOND_LW * 0.8,
             solid_capstyle="round", zorder=2)


def heteroatom_label(mol, idx):
    atom = mol.GetAtomWithIdx(idx)
    symbol = atom.GetSymbol()
    if symbol == "C":
        return None
    nh = atom.GetTotalNumHs()
    charge = atom.GetFormalCharge()
    label = symbol + ("H" if nh == 1 else f"H{nh}" if nh > 1 else "")
    if charge > 0:
        label += "+" * charge
    elif charge < 0:
        label += "-" * abs(charge)
    return label


def free_valence_direction(mol, idx, positions):
    """Direction pointing away from all of an atom's bonded neighbours --
    i.e. where its free valence (H, charge) actually sits in space, not
    just wherever a fixed left-to-right label string happens to put it."""
    atom = mol.GetAtomWithIdx(idx)
    pos = positions[idx]
    nbr_idxs = [n.GetIdx() for n in atom.GetNeighbors()]
    if not nbr_idxs:
        return (0.0, 1.0)
    vecs = [unit((pos[0] - positions[j][0], pos[1] - positions[j][1])) for j in nbr_idxs]
    sx = sum(v[0] for v in vecs) / len(vecs)
    sy = sum(v[1] for v in vecs) / len(vecs)
    return unit((sx, sy)) if math.hypot(sx, sy) > 1e-6 else (0.0, 1.0)


def draw_ligand_skeleton(ax, mol, positions):
    label_radius = {}
    for i in range(mol.GetNumAtoms()):
        lbl = heteroatom_label(mol, i)
        label_radius[i] = 0.34 if lbl else 0.0

    ring_membership = bond_ring_membership(mol)

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        p1, p2 = positions[i], positions[j]
        p1t, p2t = trim_segment(p1, p2, label_radius[i], label_radius[j])
        order = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3}.get(bond.GetBondType(), 1)
        ring_atoms = ring_membership.get(bond.GetIdx())
        ring_centroid = None
        if ring_atoms:
            ring_centroid = (sum(positions[a][0] for a in ring_atoms) / len(ring_atoms),
                              sum(positions[a][1] for a in ring_atoms) / len(ring_atoms))
        draw_bond(ax, p1t, p2t, order, ring_centroid)

    for i in range(mol.GetNumAtoms()):
        lbl = heteroatom_label(mol, i)
        if not lbl:
            continue
        atom = mol.GetAtomWithIdx(i)
        symbol = atom.GetSymbol()
        color = CPK_COLORS.get(symbol.upper(), "#333333")
        pos = positions[i]
        ax.text(pos[0], pos[1], symbol, color=color, fontsize=16,
                fontweight="bold", ha="center", va="center", zorder=4)

        nh, charge = atom.GetTotalNumHs(), atom.GetFormalCharge()
        suffix = ("H" if nh == 1 else f"H{nh}" if nh > 1 else "") + \
                 ("+" * charge if charge > 0 else "-" * abs(charge) if charge < 0 else "")
        if not suffix:
            continue
        # Drawn as its own text, offset toward the atom's free valence (away
        # from every bonded neighbour) and snapped to N/S/E/W-ish alignment --
        # not appended in-line after the symbol, which reads as if the H sat
        # collinear with (and so bonded to) whichever neighbour is opposite.
        dx, dy = free_valence_direction(mol, i, positions)
        off = 0.24
        ha = "center" if abs(dx) < 0.35 else ("left" if dx > 0 else "right")
        va = "center" if abs(dy) < 0.35 else ("bottom" if dy > 0 else "top")
        ax.text(pos[0] + dx * off, pos[1] + dy * off, suffix, color=color, fontsize=16,
                fontweight="bold", ha=ha, va=va, zorder=4)


def residue_chip(ax, center, w, h, fill, edge_color, backbone, lw=1.7):
    x, y = center
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                 boxstyle="round,pad=0,rounding_size=0.14",
                                 facecolor=fill, edgecolor=edge_color, linewidth=lw, zorder=5))
    if backbone:
        # Backbone+side-chain contact: a soft dark overlay tints the whole
        # chip rather than a second inset outline -- reads as "this one also
        # touches the backbone" without adding another hard line to the icon.
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                     boxstyle="round,pad=0,rounding_size=0.14",
                                     facecolor="#2b2b2b", edgecolor="none", alpha=0.15, zorder=5.5))


def water_node(ax, center, r, fill, edge_color):
    ax.add_patch(Circle(center, r, facecolor=fill, edgecolor=edge_color, linewidth=1.6, zorder=5))


def metal_node(ax, center, r, fill, edge_color):
    x, y = center
    d = r * 1.05
    verts = [(x, y + d), (x + d, y), (x, y - d), (x - d, y), (x, y + d)]
    ax.add_patch(PathPatch(MplPath(verts), facecolor=fill, edgecolor=edge_color, linewidth=1.6, zorder=5))


def add_residue_node(ax, node):
    center = node["pos"]
    edge_color = "#c0392b" if node["clash"] else "#2b2b2b"
    if node["kind"] == "water":
        water_node(ax, center, NODE_H / 2, FILL["water"], edge_color)
    elif node["kind"] == "metal":
        metal_node(ax, center, NODE_H / 2 + 0.08, FILL["metal"], edge_color)
    else:
        residue_chip(ax, center, NODE_W, NODE_H, node["fill"], edge_color, node["backbone"])
    label = node["label"]
    ax.text(center[0], center[1] + 0.02, label, ha="center", va="center", fontsize=9,
            fontweight="bold", color="#111111", zorder=7)
    if node.get("pka") is not None:
        ax.text(center[0], center[1] - NODE_H / 2 - 0.14, f"pKa {node['pka']:.1f}",
                ha="center", va="top", fontsize=7, color="#666666", zorder=7)


# ---------------------------------------------------------------------------
# Binding-pocket outline
#
# A close-fitting contour hugging the ligand's own footprint only (it does
# not reach out to residues) -- drawn as the exact silhouette of a union of
# small disks, one per ligand atom. Its texture is real structural data, not
# decoration: each arc is styled from the actual per-atom solvent-accessible
# surface area (SASA, Shrake-Rupley, computed on the full complex) of
# whichever ligand atom the arc belongs to -- solid where that atom is
# buried against the protein, dashed where it's solvent-exposed.
#
# The silhouette is computed per-disk (the part of each atom's own circle
# not covered by any neighbouring atom's circle -- standard circle-union
# boundary via pairwise circle/circle intersection angles), not by casting
# rays from a single interior centre point. Ray-casting from one centre only
# works for a star-shaped outline (every boundary point visible from that
# centre in a straight line); a bent/elongated ligand isn't star-shaped, so
# some ray directions could hit no disk at all and collapsed to the ligand's
# centroid -- a spurious straight line cutting across the structure. The
# per-disk approach has no such assumption and is exact for any shape.
# ---------------------------------------------------------------------------
SASA_BURIED_CUTOFF = 4.0  # A^2; below this an atom is considered buried


def compute_ligand_sasa(structure, ligand_residue):
    from Bio.PDB.SASA import ShrakeRupley
    ShrakeRupley().compute(structure[0], level="A")
    return {a.get_name(): a.sasa for a in ligand_residue if a.element != "H"}


def _complement_angle_intervals(intervals, full=2 * math.pi):
    """Given angular intervals (radians, hi >= lo, possibly extending past
    `full`), return the complementary set of intervals within [0, full)."""
    if not intervals:
        return [(0.0, full)]
    norm = []
    for lo, hi in intervals:
        lo_mod = lo % full
        hi_mod = lo_mod + (hi - lo)
        if hi_mod <= full:
            norm.append((lo_mod, hi_mod))
        else:
            norm.append((lo_mod, full))
            norm.append((0.0, hi_mod - full))
    norm.sort()
    merged = []
    for lo, hi in norm:
        if merged and lo <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    if len(merged) > 1 and merged[0][0] <= 1e-9 and merged[-1][1] >= full - 1e-9:
        merged[0][0] = merged[-1][0] - full
        merged.pop()
    if not merged:
        return [(0.0, full)]
    gaps = []
    n = len(merged)
    for k in range(n):
        lo, hi = merged[k]
        nxt_lo = merged[(k + 1) % n][0] + (full if k == n - 1 else 0.0)
        if nxt_lo - hi > 1e-6:
            gaps.append((hi, nxt_lo))
    return gaps


def _circle_uncovered_arcs(idx, disks, epsilon=1e-9):
    """Angular intervals (radians) of disks[idx]'s own circular boundary that
    are NOT covered by any other disk -- the piece of this atom's circle
    that contributes to the union's outer silhouette."""
    center, r = disks[idx]
    covered = []
    for j, (c2, r2) in enumerate(disks):
        if j == idx:
            continue
        dx, dy = c2[0] - center[0], c2[1] - center[1]
        d = math.hypot(dx, dy)
        if d < epsilon:
            continue  # coincident disk; the other copy accounts for it
        if d >= r + r2 - epsilon:
            continue  # too far apart to overlap
        if d <= r2 - r + epsilon:
            return []  # this disk sits entirely inside another one
        if d <= r - r2 + epsilon:
            continue  # the other disk sits entirely inside this one
        a = (r * r - r2 * r2 + d * d) / (2 * d)
        h2 = r * r - a * a
        if h2 <= 0:
            continue
        base_angle = math.atan2(dy, dx)
        offset = math.acos(max(-1.0, min(1.0, a / r)))
        covered.append((base_angle - offset, base_angle + offset))
    return _complement_angle_intervals(covered)


def draw_pocket_outline(ax, mol, positions, atom_names, sasa, arc_step=0.06):
    order = list(positions.keys())
    disks = [(positions[i], POCKET_LIGAND_DISK_R) for i in order]

    # Ring atoms alone don't cover their own ring's interior at this radius
    # (e.g. a hexagon's circumradius exceeds the disk radius), which would
    # otherwise leave a small spurious hole/loop drawn inside every ring.
    # Extra unlabelled disks at each ring's centroid plug that hole -- they
    # sit fully inside the ring, contribute no boundary of their own, and
    # only ever act as an occluder for the real atoms' arcs.
    n_real = len(disks)
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        ring_pos = [positions[i] for i in ring if i in positions]
        if not ring_pos:
            continue
        cx = sum(p[0] for p in ring_pos) / len(ring_pos)
        cy = sum(p[1] for p in ring_pos) / len(ring_pos)
        disks.append(((cx, cy), POCKET_LIGAND_DISK_R))

    # The same kind of gap shows up at any branch point that isn't part of a
    # single formal ring -- e.g. the concave notch where three separate rings
    # meet via two single bonds off one shared atom (an ortho-terphenyl-like
    # junction). General fix: for every atom with 2+ neighbours, plug the
    # local notch between every pair of its neighbours with a filler disk at
    # their shared centroid -- covers ring interiors too, but cheaply catches
    # every other branch shape ring centroids alone miss.
    for atom in mol.GetAtoms():
        j = atom.GetIdx()
        if j not in positions:
            continue
        neighbor_idxs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in positions]
        for a in range(len(neighbor_idxs)):
            for b in range(a + 1, len(neighbor_idxs)):
                i, k = neighbor_idxs[a], neighbor_idxs[b]
                cx = (positions[i][0] + positions[j][0] + positions[k][0]) / 3
                cy = (positions[i][1] + positions[j][1] + positions[k][1]) / 3
                disks.append(((cx, cy), POCKET_LIGAND_DISK_R))

    style = dict(color="#8a9099", linewidth=1.5, zorder=0.4)
    for idx in range(n_real):
        center, r = disks[idx]
        arcs = _circle_uncovered_arcs(idx, disks)
        if not arcs:
            continue
        atom_sasa = sasa.get(atom_names[order[idx]], 0.0)
        linestyle = "-" if atom_sasa < SASA_BURIED_CUTOFF else (0, (2, 2))
        for lo, hi in arcs:
            span = hi - lo
            if span < 1e-3:
                continue
            n = max(2, int(span / arc_step) + 1)
            pts = [(center[0] + r * math.cos(lo + span * t / (n - 1)),
                    center[1] + r * math.sin(lo + span * t / (n - 1))) for t in range(n)]
            ax.plot(*zip(*pts), linestyle=linestyle, **style)


GROWTH_ROOM_BASE_R = 0.5
GROWTH_ROOM_SCALE = 0.4  # Angstrom of bash distance -> extra data-unit radius


def draw_growth_room_outline(ax, positions, atom_names, growth_room, arc_step=0.06):
    """A second, distinct outline: 'how much room is there to grow a
    substituent here', not solvent accessibility -- see
    protein_ligand_interactions.compute_growth_room (Coot's Substitution
    Contour idea, pli/flev.cc, ported independently). A bigger bulge means
    more real 3D clearance before clashing with the protein; an atom with no
    free hydrogen (nothing to grow from) contributes no disk at all, so this
    contour only ever appears at genuine substitution points -- it can be
    fragmented around the ligand, unlike the continuous SASA pocket outline."""
    order = [i for i in positions if atom_names.get(i) in growth_room]
    if not order:
        return
    disks = [(positions[i], GROWTH_ROOM_BASE_R + GROWTH_ROOM_SCALE * growth_room[atom_names[i]]) for i in order]

    style = dict(color="#8e5fd1", linewidth=1.4, linestyle=(0, (2, 2)), zorder=0.35)
    for idx in range(len(disks)):
        arcs = _circle_uncovered_arcs(idx, disks)
        if not arcs:
            continue
        center, r = disks[idx]
        for lo, hi in arcs:
            span = hi - lo
            if span < 1e-3:
                continue
            n = max(2, int(span / arc_step) + 1)
            pts = [(center[0] + r * math.cos(lo + span * t / (n - 1)),
                    center[1] + r * math.sin(lo + span * t / (n - 1))) for t in range(n)]
            ax.plot(*zip(*pts), **style)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def _declutter(nodes, center, min_gap):
    """Deterministic collision cleanup: walk nodes closest-to-the-ligand
    first and, if a node is closer than min_gap to an already-placed one,
    push it directly outward along its own (center, node) ray until clear.
    Used as a final safety net after the force-directed relaxation below,
    whose soft repulsion discourages but never strictly forbids overlap."""
    placed = []
    for node in sorted(nodes, key=lambda n: math.hypot(n["pos"][0] - center[0], n["pos"][1] - center[1])):
        x, y = node["pos"]
        angle = math.atan2(y - center[1], x - center[0])
        r = math.hypot(x - center[0], y - center[1])
        while True:
            x = center[0] + r * math.cos(angle)
            y = center[1] + r * math.sin(angle)
            if all(math.hypot(x - px, y - py) >= min_gap - 1e-9 for px, py in placed):
                node["pos"] = (x, y)
                placed.append((x, y))
                break
            r += 0.05


# Force-directed layout weights, in the spirit of Coot's
# pli::optimise_residue_circles (pli/optimise-residue-circles.cc): a
# residue's node is pulled by a spring toward its own natural anchored
# position and pushed apart from every other node and from the ligand's own
# atoms by a soft exponential repulsion (stronger and shorter-range than the
# anchor spring, so it only acts when something is actually crowded).
LAYOUT_K_ANCHOR = 1.0
LAYOUT_K_RESIDUE_REPEL = 12.0
LAYOUT_RESIDUE_REPEL_SCALE = 0.85
LAYOUT_K_LIGAND_REPEL = 6.0
LAYOUT_LIGAND_REPEL_SCALE = 0.6


def _layout_energy_polar(flat_r_theta, center, natural_xy, ligand_pts):
    r, theta = flat_r_theta[0::2], flat_r_theta[1::2]
    n = len(r)
    xy = np.stack([center[0] + r * np.cos(theta), center[1] + r * np.sin(theta)], axis=1)
    energy = LAYOUT_K_ANCHOR * np.sum((xy - natural_xy) ** 2)
    if n > 1:
        diff = xy[:, None, :] - xy[None, :, :]
        d = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-9)
        iu = np.triu_indices(n, k=1)
        energy += LAYOUT_K_RESIDUE_REPEL * np.sum(np.exp(-d[iu] / LAYOUT_RESIDUE_REPEL_SCALE))
    if len(ligand_pts):
        dl = np.sqrt(np.sum((xy[:, None, :] - ligand_pts[None, :, :]) ** 2, axis=-1) + 1e-9)
        energy += LAYOUT_K_LIGAND_REPEL * np.sum(np.exp(-dl / LAYOUT_LIGAND_REPEL_SCALE))
    return energy


def resolve_layout(nodes, center, ligand_positions=(), min_gap=None):
    # Each node has its own natural radius -- anchor_dist + padding, i.e. how
    # far *its own* interacting ligand atom sits from the ligand centroid,
    # not a single shell shared by every residue (a shared shell, sized to
    # the ligand's single farthest atom, routinely put a residue's chip on
    # the far side of the molecule from the atom it actually touches, for an
    # elongated/bent ligand). That natural (radius, angle) position is used
    # as the rest position of a spring; a scipy.optimize relaxation (least-
    # squares energy minimisation, the same idea as Coot's flev residue-
    # circle layout -- see pli/optimise-residue-circles.cc -- ported to
    # Python rather than copied, it's GPL C++ using GSL) then finds the
    # nearby arrangement that best balances staying at that natural position
    # against not overlapping any other node or the ligand itself.
    #
    # Optimised in *polar* (radius, angle) coordinates around `center`, each
    # radius bounded below at that node's own natural radius: with free (x,y)
    # coordinates, relieving pressure in a crowded cluster could just as
    # cheaply move a node *inward*, past the clearance its own anchor
    # distance was calibrated to guarantee, right onto the ligand/pocket
    # outline. The angle is left free -- only inward radial drift is
    # actually a problem. A deterministic cleanup pass afterward guarantees
    # no two chips end up closer than min_gap, since the soft repulsion
    # terms only discourage that, never strictly forbid it.
    if min_gap is None:
        min_gap = 2 * NODE_R + 0.18
    n = len(nodes)
    if n == 0:
        return nodes
    if n == 1:
        node = nodes[0]
        node["pos"] = (center[0] + node["radius"] * math.cos(node["angle"]),
                        center[1] + node["radius"] * math.sin(node["angle"]))
        return nodes

    natural_r = np.array([nd["radius"] for nd in nodes])
    natural_theta = np.array([nd["angle"] for nd in nodes])
    natural_xy = np.stack([center[0] + natural_r * np.cos(natural_theta),
                            center[1] + natural_r * np.sin(natural_theta)], axis=1)
    ligand_pts = np.array(list(ligand_positions)) if len(ligand_positions) else np.empty((0, 2))

    x0 = np.empty(2 * n)
    x0[0::2], x0[1::2] = natural_r, natural_theta
    bounds = [b for r in natural_r for b in ((r, None), (None, None))]

    result = minimize(_layout_energy_polar, x0, args=(center, natural_xy, ligand_pts),
                       method="L-BFGS-B", bounds=bounds, options={"maxiter": 300})
    r, theta = result.x[0::2], result.x[1::2]
    for node, ri, ti in zip(nodes, r, theta):
        node["pos"] = (center[0] + ri * math.cos(ti), center[1] + ri * math.sin(ti))

    _declutter(nodes, center, min_gap)
    return nodes


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------
def compute_diagram_data(structure, ligand_residue, rotate_deg=0.0, flip_h=False, flip_v=False, ph=7.4,
                          compute_protonation=False, outline="sasa"):
    """Everything about the diagram that doesn't depend on how it's drawn:
    the ligand's 2D coordinates, every detected interaction, and an initial
    (automatic) layout for the residue nodes. Kept separate from rendering so
    the same data can be redrawn repeatedly -- e.g. after a node is dragged
    to a new position in --interactive mode -- without recomputing anything.

    compute_protonation=True (off by default) runs PROPKA/Dimorphite-DL for
    real per-residue/per-ligand pKa at `ph`; left False, the ligand is kept
    exactly as deposited and residues fall back to the standard textbook
    pKa assumption -- also the automatic fallback if propka/dimorphite-dl
    aren't installed.

    outline is one of "sasa" (default -- the ligand's own real solvent-
    accessible surface), "growth" (Coot-style substitution room, see
    protein_ligand_interactions.compute_growth_room), or "both"."""
    if compute_protonation:
        apply_ligand_protonation(ligand_residue, ph=ph)
    mol, atoms, positions, name_to_idx = build_2d_ligand(ligand_residue, rotate_deg, flip_h, flip_v)
    # so interaction lines/arrows stop beside a heteroatom label, not under it
    lig_label_radius = {atoms[i].get_name(): (0.34 if heteroatom_label(mol, i) else 0.0)
                         for i in range(mol.GetNumAtoms())}

    residue_pkas = compute_residue_pkas(structure, ph=ph) if compute_protonation else {}

    prot_atoms_list = protein_atoms(structure, exclude_residues=[ligand_residue])
    ns = NeighborSearch(prot_atoms_list)
    polar, hydrophobic = find_polar_and_hydrophobic_contacts(ligand_residue, prot_atoms_list, ns)
    salt = find_salt_bridges(ligand_residue, residue_pkas=residue_pkas, ph=ph)
    pistack = find_pi_stacking(ligand_residue)
    chelation = find_chelation(structure, ligand_residue)

    residues = {}  # (chain, resi) -> node dict

    def get_res_entry(res, kind="residue"):
        key = (res.get_parent().id, res.id[1], res.resname)
        if key not in residues:
            residues[key] = {
                "res": res, "kind": "water" if res.resname in WATER_RESNAMES else kind,
                "atom_names": set(), "anchor_pts": [], "edges": [],
                "clash": False, "fill": FILL["contact"],
            }
        return residues[key]

    for lig_atom, prot_atom, dist in polar:
        res = prot_atom.get_parent()
        entry = get_res_entry(res)
        entry["atom_names"].add(prot_atom.get_name())
        role = classify_polar_contact(mol, name_to_idx[lig_atom.get_name()], prot_atom)
        if role is None:
            continue
        lig_pos = positions[name_to_idx[lig_atom.get_name()]]
        entry["anchor_pts"].append(lig_pos)
        entry["edges"].append({"type": role, "lig_pos": lig_pos, "prot_atom": prot_atom.get_name(),
                                "lig_r": lig_label_radius.get(lig_atom.get_name(), 0.0)})
        if role == "clash":
            entry["clash"] = True
            entry["fill"] = FILL["clash"]
        elif not entry["clash"] and entry["fill"] == FILL["contact"]:
            entry["fill"] = FILL[role]

    for lig_atom, prot_atom, dist in hydrophobic:
        res = prot_atom.get_parent()
        entry = get_res_entry(res)
        entry["atom_names"].add(prot_atom.get_name())
        if lig_atom.get_name() in name_to_idx:
            entry["anchor_pts"].append(positions[name_to_idx[lig_atom.get_name()]])

    for lig_atom, res_atom, dist in salt:
        res = res_atom.get_parent()
        entry = get_res_entry(res)
        entry["atom_names"].add(res_atom.get_name())
        if lig_atom.get_name() in name_to_idx:
            pos = positions[name_to_idx[lig_atom.get_name()]]
            entry["anchor_pts"].append(pos)
            entry["edges"].append({"type": "salt", "lig_pos": pos, "prot_atom": res_atom.get_name(),
                                    "lig_r": lig_label_radius.get(lig_atom.get_name(), 0.0)})
        entry["fill"] = "#f5b7b1"

    pi_markers = []
    for lig_ring, prot_ring_atoms, res, dist, angle, geometry in pistack:
        entry = get_res_entry(res)
        ring_names = {a.get_name() for a in lig_ring}
        ring_idxs = [name_to_idx[n] for n in ring_names if n in name_to_idx]
        if not ring_idxs:
            continue
        cx = sum(positions[i][0] for i in ring_idxs) / len(ring_idxs)
        cy = sum(positions[i][1] for i in ring_idxs) / len(ring_idxs)
        entry["atom_names"].update(a.get_name() for a in prot_ring_atoms)
        entry["anchor_pts"].append((cx, cy))
        entry["edges"].append({"type": "pi", "lig_pos": (cx, cy), "lig_r": PI_MARKER_R})
        entry["fill"] = FILL["pi"]
        pi_markers.append({"pos": (cx, cy), "label": geometry[0].upper()})

    for lig_atom, metal_atom, res, dist in chelation:
        entry = get_res_entry(res, kind="metal")
        entry["kind"] = "metal"
        if lig_atom.get_name() in name_to_idx:
            pos = positions[name_to_idx[lig_atom.get_name()]]
            entry["anchor_pts"].append(pos)
            entry["edges"].append({"type": "chelator", "lig_pos": pos,
                                    "lig_r": lig_label_radius.get(lig_atom.get_name(), 0.0)})
        entry["fill"] = FILL["chelator"]

    if not residues:
        print("  [warn] no interactions detected; diagram will show the ligand only.", file=sys.stderr)

    lig_center = (sum(p[0] for p in positions.values()) / len(positions),
                  sum(p[1] for p in positions.values()) / len(positions))
    lig_radius = max(math.hypot(p[0] - lig_center[0], p[1] - lig_center[1]) for p in positions.values())

    nodes = []
    for key, entry in residues.items():
        chain, resi, resname = key
        anchor = entry["anchor_pts"][0] if entry["anchor_pts"] else lig_center
        if len(entry["anchor_pts"]) > 1:
            anchor = (sum(p[0] for p in entry["anchor_pts"]) / len(entry["anchor_pts"]),
                      sum(p[1] for p in entry["anchor_pts"]) / len(entry["anchor_pts"]))
        anchor_dist = math.hypot(anchor[0] - lig_center[0], anchor[1] - lig_center[1])
        angle = math.atan2(anchor[1] - lig_center[1], anchor[0] - lig_center[0])
        backbone = bool(entry["atom_names"] & {"N", "CA", "C", "O"}) and entry["kind"] != "metal"
        pka = residue_pkas.get((chain, resi, resname)) if entry["kind"] != "metal" else None
        charge_sign = CHARGE_SIGN_CHAR[residue_charge_sign(resname, pka, ph)] if entry["kind"] != "metal" else ""
        nodes.append({
            "key": key, "angle": angle, "radius": anchor_dist + NODE_ANCHOR_PAD,
            "kind": entry["kind"], "fill": entry["fill"],
            "clash": entry["clash"], "backbone": backbone, "pka": pka,
            "label": f"{resname}{charge_sign}\n{resi}{chain}",
            "edges": entry["edges"],
        })

    nodes = resolve_layout(nodes, lig_center, positions.values())

    idx_to_name = {v: k for k, v in name_to_idx.items()}
    ligand_sasa = compute_ligand_sasa(structure, ligand_residue)
    growth_room = compute_growth_room(mol, atoms, ns) if outline in ("growth", "both") else {}

    return {
        "mol": mol, "positions": positions, "idx_to_name": idx_to_name,
        "ligand_sasa": ligand_sasa, "growth_room": growth_room, "outline": outline,
        "nodes": nodes, "pi_markers": pi_markers,
        "lig_center": lig_center, "lig_radius": lig_radius,
    }


# Legend row/box sizing, shared between figure_size (which needs the total
# height before anything is drawn, to size the figure) and draw_legend
# (which lays content out top-down using these same increments) -- content
# is filtered to what this specific diagram actually uses (see
# _legend_content), so an unused style takes no space at all rather than
# leaving an empty legend row.
LEGEND_TITLE_H = 0.42
LEGEND_ROW_H = 0.62
LEGEND_NOTE_H = 0.34
LEGEND_BOX_PAD = 0.18
LEGEND_BOX_GAP = 0.18
LEGEND_MARGIN = 0.15


def _legend_content(nodes, pi_markers, outline):
    """Which legend items this specific diagram actually needs."""
    residue_kinds = set()
    for n in nodes:
        if n["kind"] == "water":
            residue_kinds.add("water")
        elif n["kind"] == "metal":
            residue_kinds.add("metal")
        else:
            residue_kinds.add("backbone" if n["backbone"] else "single")
    residue_items = [(k, t) for k, t in
                      [("single", "side-chain"), ("backbone", "backbone & side-chain"),
                       ("water", "water"), ("metal", "metal")] if k in residue_kinds]

    interaction_kinds = {e["type"] for n in nodes for e in n["edges"]}
    hydrophobic_present = any(n["kind"] not in ("water", "metal") and n["fill"] == FILL["contact"] for n in nodes)
    clash_present = any(n["clash"] for n in nodes)

    def present(k):
        if k == "contact":
            return hydrophobic_present
        if k == "clash":
            return clash_present
        return k in interaction_kinds

    interaction_items = [(k, t) for k, t in
                          [("acceptor", "ligand acceptor"), ("donor", "ligand donor"),
                           ("chelator", "chelator"), ("pi", "pi-stacking"),
                           ("salt", "salt bridge"), ("contact", "hydrophobic contact"),
                           ("clash", "clash")] if present(k)]
    interaction_rows = [interaction_items[i:i + 4] for i in range(0, len(interaction_items), 4)]

    return {
        "residue_items": residue_items,
        "interaction_rows": interaction_rows,
        "pi_present": bool(pi_markers),
        "directional_present": bool({"acceptor", "donor", "chelator"} & interaction_kinds),
        "sasa_present": outline in ("sasa", "both"),
        "growth_present": outline in ("growth", "both"),
    }


def _legend_note_rows(content):
    # pi-marker key and the donor->acceptor arrow key share one line; sasa
    # and growth captions each get their own line.
    return ((content["pi_present"] or content["directional_present"]) +
            content["sasa_present"] + content["growth_present"])


def _legend_height(content):
    h = 2 * LEGEND_MARGIN
    box_a = bool(content["residue_items"])
    box_b = bool(content["interaction_rows"]) or _legend_note_rows(content) > 0
    if box_a:
        h += LEGEND_TITLE_H + LEGEND_ROW_H + 2 * LEGEND_BOX_PAD
    if box_a and box_b:
        h += LEGEND_BOX_GAP
    if box_b:
        h += (LEGEND_TITLE_H + len(content["interaction_rows"]) * LEGEND_ROW_H
              + _legend_note_rows(content) * LEGEND_NOTE_H + 2 * LEGEND_BOX_PAD)
    return h


def figure_size(nodes, lig_center, lig_radius, pi_markers=(), outline="sasa"):
    max_extent = max((math.hypot(n["pos"][0] - lig_center[0], n["pos"][1] - lig_center[1]) for n in nodes),
                      default=lig_radius) + NODE_R + 0.8
    fig_w = max(9, max_extent * 1.55)
    legend_h = _legend_height(_legend_content(nodes, pi_markers, outline))
    return fig_w, fig_w * 0.82 + legend_h, legend_h


def render_diagram(ax, legend_ax, data, structure_label):
    """(Re)draws the full diagram -- ligand, pocket outline, residue nodes,
    interaction lines, legend -- from `data` onto already-created axes.
    Idempotent and cheap enough to call on every drag update in
    --interactive mode: it always clears both axes first."""
    mol, positions = data["mol"], data["positions"]
    idx_to_name, ligand_sasa = data["idx_to_name"], data["ligand_sasa"]
    growth_room, outline = data["growth_room"], data["outline"]
    nodes, pi_markers = data["nodes"], data["pi_markers"]
    lig_center = data["lig_center"]

    ax.clear()
    ax.set_aspect("equal")
    ax.axis("off")

    if outline in ("sasa", "both"):
        draw_pocket_outline(ax, mol, positions, idx_to_name, ligand_sasa)
    if outline in ("growth", "both"):
        draw_growth_room_outline(ax, positions, idx_to_name, growth_room)
    draw_ligand_skeleton(ax, mol, positions)

    for marker in pi_markers:
        ax.add_patch(Circle(marker["pos"], PI_MARKER_R, facecolor=FILL["pi"], edgecolor="#3f8c3f",
                             linewidth=1.6, zorder=6))
        ax.text(*marker["pos"], marker["label"], ha="center", va="center", fontsize=8,
                fontweight="bold", color="#2c5c2c", zorder=7)

    # Interaction lines: donor -> acceptor arrows for H-bonds and chelation
    # (directional interactions); plain lines for salt bridges/pi-stacking
    # (non-directional); clashes drawn as a plain dashed line (no valid
    # direction to point); hydrophobic-only contacts get no line at all
    # (the node itself signals the contact). The specific contacting atom
    # isn't labelled on the diagram -- the residue node already identifies
    # the side chain, and per-atom labels cluttered the diagram with plain
    # "O"/"N" backbone-atom letters.
    for node in nodes:
        for edge in node["edges"]:
            etype = edge["type"]
            lig_pt, node_pt = edge["lig_pos"], node["pos"]
            color = LINE_COLOR[etype]
            lig_end, node_end = trim_segment(lig_pt, node_pt, edge.get("lig_r", 0.0), NODE_R)

            if etype == "acceptor":  # protein donates -> ligand accepts
                ax.add_patch(FancyArrowPatch(node_end, lig_end, arrowstyle="-|>", mutation_scale=16,
                                              shrinkA=0, shrinkB=0, color=color, linewidth=1.8,
                                              linestyle=(0, (4, 3)), zorder=3))
            elif etype == "donor":  # ligand donates -> protein accepts
                ax.add_patch(FancyArrowPatch(lig_end, node_end, arrowstyle="-|>", mutation_scale=16,
                                              shrinkA=0, shrinkB=0, color=color, linewidth=1.8,
                                              linestyle=(0, (4, 3)), zorder=3))
            elif etype == "chelator":  # ligand lone pair -> metal
                ax.add_patch(FancyArrowPatch(lig_end, node_end, arrowstyle="-|>", mutation_scale=16,
                                              shrinkA=0, shrinkB=0, color=color, linewidth=1.8,
                                              linestyle=(0, (4, 3)), zorder=3))
            elif etype == "clash":
                ax.plot([lig_end[0], node_end[0]], [lig_end[1], node_end[1]], color=color,
                        linestyle=(0, (4, 3)), linewidth=1.8, zorder=3)
            else:  # salt bridge / pi-stacking: non-directional
                ax.plot([lig_end[0], node_end[0]], [lig_end[1], node_end[1]], color=color,
                        linestyle="-", linewidth=2.2, zorder=3)
        add_residue_node(ax, node)

    ax.set_title(structure_label, fontsize=20, fontweight="bold", pad=38)

    xs = [lig_center[0]] + [n["pos"][0] for n in nodes]
    ys = [lig_center[1]] + [n["pos"][1] for n in nodes]
    pad = NODE_R + 0.8
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)

    legend_ax.clear()
    draw_legend(legend_ax, data["nodes"], data["pi_markers"], data["outline"])


def build_diagram(structure, ligand_residue, out_dir, structure_label,
                   rotate_deg=0.0, flip_h=False, flip_v=False, ph=7.4, compute_protonation=False,
                   outline="sasa"):
    data = compute_diagram_data(structure, ligand_residue, rotate_deg, flip_h, flip_v, ph, compute_protonation,
                                 outline)
    fig_w, fig_h, legend_h = figure_size(data["nodes"], data["lig_center"], data["lig_radius"],
                                          data["pi_markers"], outline)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[fig_w * 0.82, legend_h], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    legend_ax = fig.add_subplot(gs[1])
    render_diagram(ax, legend_ax, data, structure_label)

    out_dir.mkdir(parents=True, exist_ok=True)
    resn, resi = ligand_residue.resname, ligand_residue.id[1]
    png_path = out_dir / f"{resn}_{resi}_2d_diagram.png"
    svg_path = out_dir / f"{resn}_{resi}_2d_diagram.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def run_interactive(structure, ligand_residue, out_dir, structure_label,
                     rotate_deg=0.0, flip_h=False, flip_v=False, ph=7.4, compute_protonation=False,
                     outline="sasa"):
    """Opens a draggable window: click and drag a residue chip to move it,
    press 's' to save the current arrangement as PNG+SVG, 'r' to snap back
    to the automatic layout. Closing the window ends the session."""
    data = compute_diagram_data(structure, ligand_residue, rotate_deg, flip_h, flip_v, ph, compute_protonation,
                                 outline)
    nodes, lig_center, lig_radius = data["nodes"], data["lig_center"], data["lig_radius"]
    save_size = figure_size(nodes, lig_center, lig_radius, data["pi_markers"], outline)  # full-size, used only when saving
    fig_w, fig_h, legend_h = save_size

    # A large/spread-out complex can size the batch-mode figure well past
    # what fits on a screen -- shrink the whole window (fig_w, fig_h and
    # legend_h together, so nothing's relative proportions change) to a
    # comfortable on-screen size for dragging; the full save_size is
    # temporarily restored in on_key so the saved PNG/SVG still comes out at
    # the same quality/proportions as batch mode, not the shrunk display.
    if fig_w > INTERACTIVE_MAX_W:
        scale = INTERACTIVE_MAX_W / fig_w
        fig_w, fig_h, legend_h = fig_w * scale, fig_h * scale, legend_h * scale

    matplotlib.rcParams["toolbar"] = "none"  # avoid the pan/zoom tool fighting with node-dragging
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[fig_w * 0.82, legend_h], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    legend_ax = fig.add_subplot(gs[1])

    out_dir.mkdir(parents=True, exist_ok=True)
    resn, resi = ligand_residue.resname, ligand_residue.id[1]
    png_path = out_dir / f"{resn}_{resi}_2d_diagram.png"
    svg_path = out_dir / f"{resn}_{resi}_2d_diagram.svg"

    state = {"dragging": None, "saved": False, "hint": None}

    def redraw():
        render_diagram(ax, legend_ax, data, structure_label)
        state["hint"] = fig.suptitle("drag a residue to reposition it   ·   's' save PNG/SVG   ·   'r' reset layout",
                                      fontsize=9, color="#777777", y=0.997)
        fig.canvas.draw_idle()

    redraw()

    def node_at(x, y):
        for node in nodes:
            cx, cy = node["pos"]
            if abs(x - cx) <= NODE_W / 2 + 0.1 and abs(y - cy) <= NODE_H / 2 + 0.1:
                return node
        return None

    def on_press(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        state["dragging"] = node_at(event.xdata, event.ydata)

    def on_motion(event):
        if state["dragging"] is None or event.inaxes is not ax or event.xdata is None:
            return
        state["dragging"]["pos"] = (event.xdata, event.ydata)
        redraw()

    def on_release(event):
        state["dragging"] = None

    def on_key(event):
        if event.key == "s":
            screen_size = fig.get_size_inches()
            fig.set_size_inches(save_size[0], save_size[1])
            state["hint"].set_visible(False)  # keyboard-shortcut hint is for the on-screen window only
            fig.savefig(png_path, dpi=220, bbox_inches="tight")
            fig.savefig(svg_path, bbox_inches="tight")
            state["hint"].set_visible(True)
            fig.set_size_inches(*screen_size)
            fig.canvas.draw_idle()
            state["saved"] = True
            print(f"  -> saved {png_path}")
            print(f"  -> saved {svg_path}")
        elif event.key == "r":
            resolve_layout(nodes, lig_center, data["positions"].values())
            redraw()

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)

    print(f"Interactive window open for {resn} {resi}{ligand_residue.get_parent().id} -- "
          f"drag residues, 's' to save, 'r' to reset, close the window when done.")
    plt.show()
    return (png_path, svg_path) if state["saved"] else (None, None)


def draw_legend(legend_ax, nodes, pi_markers, outline="sasa"):
    """Legend drawn in its own fixed-width (0..10) coordinate system, top
    down, showing only the styles this specific diagram actually uses --
    see _legend_content/_legend_height, which figure_size also calls to
    size the figure to match exactly what gets drawn here."""
    content = _legend_content(nodes, pi_markers, outline)
    total_h = _legend_height(content)
    legend_ax.set_xlim(0, 10)
    legend_ax.set_ylim(0, total_h)
    legend_ax.set_aspect("equal")
    legend_ax.axis("off")

    def row(items, y, icon_fn):
        n = len(items)
        step = 10.0 / n
        for i, (kind, text) in enumerate(items):
            cx = step * i + step / 2
            icon_fn((cx, y), kind)
            legend_ax.text(cx, y - 0.34, text, ha="center", va="top", fontsize=8.5)

    def residue_icon(c, kind):
        if kind == "water":
            water_node(legend_ax, c, 0.16, FILL["water"], "#2b2b2b")
        elif kind == "metal":
            metal_node(legend_ax, c, 0.19, FILL["metal"], "#2b2b2b")
        else:
            residue_chip(legend_ax, c, 0.42, 0.28, "#ffffff", "#2b2b2b", kind == "backbone", lw=1.2)

    def interaction_icon(c, kind):
        fill = FILL.get(kind, "#ffffff")
        edge = "#c0392b" if kind == "clash" else "#2b2b2b"
        legend_ax.add_patch(Circle(c, 0.14, facecolor=fill, edgecolor=edge, linewidth=1.2, zorder=5))

    box_style = dict(boxstyle="round,pad=0.02,rounding_size=0.06", facecolor="none",
                      edgecolor="#cccccc", linewidth=0.9, zorder=1)

    cursor = total_h - LEGEND_MARGIN

    if content["residue_items"]:
        box_top = cursor
        cursor -= LEGEND_BOX_PAD
        legend_ax.text(0.15, cursor, "Residue styles", fontsize=9.5, fontweight="bold", ha="left", va="top")
        cursor -= LEGEND_TITLE_H
        row(content["residue_items"], cursor - 0.20, residue_icon)
        cursor -= LEGEND_ROW_H
        cursor -= LEGEND_BOX_PAD
        legend_ax.add_patch(FancyBboxPatch((0.05, cursor), 9.9, box_top - cursor, **box_style))
        cursor -= LEGEND_BOX_GAP

    if content["interaction_rows"] or _legend_note_rows(content):
        box_top = cursor
        cursor -= LEGEND_BOX_PAD
        legend_ax.text(0.15, cursor, "Interaction styles", fontsize=9.5, fontweight="bold", ha="left", va="top")
        cursor -= LEGEND_TITLE_H
        for irow in content["interaction_rows"]:
            row(irow, cursor - 0.20, interaction_icon)
            cursor -= LEGEND_ROW_H

        if content["pi_present"] or content["directional_present"]:
            note_y = cursor - 0.16
            if content["pi_present"]:
                legend_ax.text(0.15, note_y, "pi-stacking marker: P parallel  ·  T T-shaped  ·  I intermediate",
                                ha="left", va="center", fontsize=8, color="#555555")
            if content["directional_present"]:
                ax_x = 6.9 if content["pi_present"] else 0.15
                legend_ax.add_patch(FancyArrowPatch((ax_x, note_y), (ax_x + 0.6, note_y), arrowstyle="-|>",
                                                     mutation_scale=12, color="#555555", linewidth=1.5, zorder=5))
                legend_ax.text(ax_x + 0.7, note_y, "donor -> acceptor", ha="left", va="center",
                                fontsize=8, color="#555555")
            cursor -= LEGEND_NOTE_H

        if content["sasa_present"]:
            note_y = cursor - 0.16
            legend_ax.plot([0.15, 0.75], [note_y, note_y], color="#8a9099", linewidth=1.5, linestyle="-")
            legend_ax.text(0.85, note_y, "ligand atom buried", ha="left", va="center", fontsize=8, color="#555555")
            legend_ax.plot([4.3, 4.9], [note_y, note_y], color="#8a9099", linewidth=1.5, linestyle=(0, (2, 2)))
            legend_ax.text(5.0, note_y, "ligand atom solvent-exposed", ha="left", va="center",
                            fontsize=8, color="#555555")
            cursor -= LEGEND_NOTE_H

        if content["growth_present"]:
            note_y = cursor - 0.16
            legend_ax.plot([0.15, 0.9], [note_y, note_y], color="#8e5fd1", linewidth=1.4, linestyle=(0, (2, 2)))
            legend_ax.text(1.0, note_y, "room to grow a substituent here (bigger = more real 3D clearance)",
                            ha="left", va="center", fontsize=8, color="#555555")
            cursor -= LEGEND_NOTE_H

        cursor -= LEGEND_BOX_PAD
        legend_ax.add_patch(FancyBboxPatch((0.05, cursor), 9.9, box_top - cursor, **box_style))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("structure", type=Path)
    ap.add_argument("--ligand", help="Ligand residue name (3-letter code); default: all hetero groups")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--ligand-cif", type=Path, default=None,
                     help="Local chem-component CIF (_chem_comp_atom/_chem_comp_bond format, e.g. from "
                          "CCP4 AceDRG/JLigand or Phenix eLBOW) to use instead of fetching from the public "
                          "PDB CCD -- needed for a ligand that isn't a standard/deposited PDB component.")
    ap.add_argument("--title", default=None, help="Diagram header text; default: the structure file's name")
    ap.add_argument("--rotate", type=float, default=0.0,
                     help="Rotate the ligand's 2D layout by this many degrees counter-clockwise "
                          "(applied after --flip-h/--flip-v); everything else (pocket outline, "
                          "residue placement) follows automatically.")
    ap.add_argument("--flip-h", action="store_true", help="Mirror the ligand layout left-right")
    ap.add_argument("--flip-v", action="store_true", help="Mirror the ligand layout top-bottom")
    ap.add_argument("--interactive", action="store_true",
                     help="Open a window where each residue can be dragged to a new position "
                          "before saving -- press 's' to save the current arrangement as PNG+SVG, "
                          "'r' to reset to the automatic layout.")
    ap.add_argument("--ph", type=float, default=7.4,
                     help="pH for protonation-state assignment: residue side chains via PROPKA "
                          "(real per-residue pKa from the 3D structure), ligand via Dimorphite-DL "
                          "(SMARTS-rule-based, exocyclic groups only). Falls back to standard "
                          "textbook pKa assumptions if either tool isn't installed. Default: 7.4")
    ap.add_argument("--protonation", action="store_true",
                     help="Compute real protonation states at --ph via PROPKA (residues) and "
                          "Dimorphite-DL (ligand), and show them on the diagram. Off by default -- "
                          "ligand kept exactly as deposited, residues use the standard textbook pKa "
                          "assumption (Asp/Glu/Lys/Arg only). Requires propka and dimorphite-dl "
                          "(pip install propka dimorphite-dl).")
    ap.add_argument("--outline", choices=("sasa", "growth", "both"), default="sasa",
                     help="Outline drawn around the ligand: 'sasa' (default) traces the ligand's own "
                          "real solvent-accessible surface (solid=buried, dashed=exposed); 'growth' "
                          "is Coot-style Substitution Contour -- how much real 3D room there is to "
                          "grow a substituent at each atom with a free hydrogen, from ray-marching "
                          "against nearby protein atoms; 'both' draws them together.")
    args = ap.parse_args()

    if args.ligand_cif:
        comp_id = register_ligand_cif_override(args.ligand_cif)
        print(f"Using local ligand dictionary {args.ligand_cif} for component '{comp_id}'")

    structure = load_structure(args.structure)
    ligands = find_ligand_residues(structure, args.ligand)
    if not ligands:
        print("No hetero (non-water) ligand residues found.", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir or args.structure.parent
    structure_label = args.title or args.structure.stem.upper()
    for ligand_residue in ligands:
        print(f"=== {ligand_residue.resname} {ligand_residue.id[1]}/{ligand_residue.get_parent().id} ===")
        fn = run_interactive if args.interactive else build_diagram
        png_path, svg_path = fn(structure, ligand_residue, out_dir, structure_label,
                                 rotate_deg=args.rotate, flip_h=args.flip_h, flip_v=args.flip_v, ph=args.ph,
                                 compute_protonation=args.protonation, outline=args.outline)
        if png_path is None:
            print("  (window closed without saving)")
        else:
            print(f"  -> {png_path}")
            print(f"  -> {svg_path}")


if __name__ == "__main__":
    main()
