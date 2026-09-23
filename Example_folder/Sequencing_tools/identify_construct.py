#!/usr/bin/env python3
"""
identify_construct.py

Match Sanger sequencing reads (.ab1) against a panel of candidate plasmid
constructs (GenBank, SnapGene .dna, or FASTA) to figure out which construct
each read actually came from.

Usage:
    python3 identify_construct.py --constructs constructs/ --reads reads/

    # plasmids are circular and the distinguishing region might span the
    # origin of the sequence file:
    python3 identify_construct.py --constructs constructs/ --reads reads/ --circular

    # also dump full pairwise alignments (with key features highlighted)
    # for the best hit of every read:
    python3 identify_construct.py --constructs constructs/ --reads reads/ --report alignments.txt

    # customise which features get highlighted in the alignment report:
    python3 identify_construct.py --constructs constructs/ --reads reads/ \\
        --report alignments.txt \\
        --motifs "TEV=GAAAACCTGTACTTCCAGGGA,8xHis=CATCACCATCACCATCACCATCAC" \\
        --restriction-enzymes "BamHI,XhoI,NdeI"

Results are written into a dated folder (results_YYYY-MM-DD/ by default, see
--outdir) so repeat runs don't clobber each other across days.

Requires: biopython >= 1.80 (for Alignment.counts()/Alignment.indices).
"""

import argparse
import csv
import glob
import os
import sys
from dataclasses import dataclass, field
from datetime import date

from Bio import Align, SeqIO
from Bio.Seq import Seq

CONSTRUCT_EXTS = {
    ".gb": "genbank",
    ".gbk": "genbank",
    ".gbff": "genbank",
    ".genbank": "genbank",
    ".dna": "snapgene",
    ".fasta": "fasta",
    ".fa": "fasta",
    ".fna": "fasta",
}

READ_EXTS = {
    ".ab1": "abi",
    ".fasta": "fasta",
    ".fa": "fasta",
    ".fna": "fasta",
    ".seq": "fasta",
    ".txt": "fasta",
}

SYMBOL_PALETTE = "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LINE_WIDTH = 60


def collect_files(paths, known_exts):
    """Expand a list of files/dirs/globs into a flat list of files with known extensions."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            for ext in known_exts:
                files.extend(sorted(glob.glob(os.path.join(p, f"*{ext}"))))
        elif os.path.isfile(p):
            files.append(p)
        else:
            matches = sorted(glob.glob(p))
            if not matches:
                print(f"warning: no files found for '{p}'", file=sys.stderr)
            files.extend(matches)
    return files


def load_constructs(paths):
    """Return {name: SeqRecord} for every candidate construct (features kept for highlighting)."""
    constructs = {}
    for path in collect_files(paths, CONSTRUCT_EXTS):
        ext = os.path.splitext(path)[1].lower()
        fmt = CONSTRUCT_EXTS.get(ext)
        if fmt is None:
            print(f"warning: skipping '{path}' (unrecognised construct format)", file=sys.stderr)
            continue
        base = os.path.splitext(os.path.basename(path))[0]
        try:
            records = list(SeqIO.parse(path, fmt))
        except Exception as exc:
            print(f"warning: could not parse '{path}' as {fmt}: {exc}", file=sys.stderr)
            continue
        if not records:
            print(f"warning: no sequences found in '{path}'", file=sys.stderr)
            continue
        for rec in records:
            name = base if len(records) == 1 else f"{base}:{rec.id}"
            constructs[name] = rec
    if not constructs:
        sys.exit("error: no candidate construct sequences were loaded")
    return constructs


def load_reads(paths, use_quality_trim=True, min_trimmed_len=20):
    """Return {name: sequence_str} for every sequencing read, quality-trimmed if .ab1."""
    reads = {}
    for path in collect_files(paths, READ_EXTS):
        ext = os.path.splitext(path)[1].lower()
        fmt = READ_EXTS.get(ext)
        if fmt is None:
            print(f"warning: skipping '{path}' (unrecognised read format)", file=sys.stderr)
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            if fmt == "abi":
                rec = None
                if use_quality_trim:
                    rec = SeqIO.read(path, "abi-trim")
                    if len(rec.seq) < min_trimmed_len:
                        print(
                            f"warning: '{path}' trimmed to {len(rec.seq)} bp, "
                            f"falling back to untrimmed read",
                            file=sys.stderr,
                        )
                        rec = None
                if rec is None:
                    rec = SeqIO.read(path, "abi")
            else:
                rec = next(SeqIO.parse(path, fmt))
        except Exception as exc:
            print(f"warning: could not parse '{path}': {exc}", file=sys.stderr)
            continue
        seq = str(rec.seq).upper()
        if len(seq) == 0:
            print(f"warning: '{path}' produced an empty sequence, skipping", file=sys.stderr)
            continue
        reads[name] = seq
    if not reads:
        sys.exit("error: no sequencing reads were loaded")
    return reads


def find_restriction_sites(seq_str, enzyme_names):
    """Return (start, end, label) 0-based half-open spans for each enzyme's recognition site."""
    import Bio.Restriction as Restriction

    hits = []
    for name in enzyme_names:
        name = name.strip()
        if not name:
            continue
        try:
            enzyme = getattr(Restriction, name)
        except AttributeError:
            print(f"warning: unknown restriction enzyme '{name}', skipping", file=sys.stderr)
            continue
        site = str(enzyme.site).upper()
        rc_site = str(Seq(site).reverse_complement())
        seen_starts = set()
        for query in {site, rc_site}:
            start = seq_str.find(query)
            while start != -1:
                if start not in seen_starts:
                    seen_starts.add(start)
                    hits.append((start, start + len(query), f"{name} site"))
                start = seq_str.find(query, start + 1)
    return hits


def find_motifs(seq_str, motif_pairs, construct_name):
    """Return (start, end, label) spans for exact NAME=SEQUENCE motifs (both strands)."""
    hits = []
    for name, motif_seq in motif_pairs:
        motif_seq = motif_seq.strip().upper()
        if not motif_seq:
            continue
        rc_motif = str(Seq(motif_seq).reverse_complement())
        seen_starts = set()
        for query in {motif_seq, rc_motif}:
            start = seq_str.find(query)
            while start != -1:
                if start not in seen_starts:
                    seen_starts.add(start)
                    hits.append((start, start + len(query), name))
                start = seq_str.find(query, start + 1)
        if not seen_starts:
            print(f"note: motif '{name}' ({motif_seq}) not found in construct '{construct_name}'", file=sys.stderr)
    return hits


def parse_motif_arg(arg):
    """Parse 'NAME=SEQ,NAME=SEQ' into [(name, seq), ...]."""
    pairs = []
    for chunk in arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            print(f"warning: ignoring malformed --motifs entry '{chunk}' (expected NAME=SEQUENCE)", file=sys.stderr)
            continue
        name, seq = chunk.split("=", 1)
        pairs.append((name.strip(), seq.strip()))
    return pairs


def find_annotated_features(record, keywords):
    """Return (start, end, label) spans for GenBank/SnapGene features matching any keyword."""
    if not keywords:
        return []
    keywords_lower = [k.strip().lower() for k in keywords if k.strip()]
    hits = []
    for feat in record.features:
        text_bits = []
        for key in ("label", "note", "gene", "product"):
            text_bits.extend(feat.qualifiers.get(key, []))
        haystack = " ".join(text_bits).lower()
        if not any(kw in haystack for kw in keywords_lower):
            continue
        label = (
            feat.qualifiers.get("label", [None])[0]
            or feat.qualifiers.get("gene", [None])[0]
            or feat.qualifiers.get("note", [None])[0]
            or feat.type
        )
        hits.append((int(feat.location.start), int(feat.location.end), label))
    return hits


def build_highlights(constructs, keywords, enzyme_names, motif_pairs):
    """Return {construct_name: [(start, end, label), ...]} sorted by start, deduplicated."""
    highlights = {}
    for name, record in constructs.items():
        seq_str = str(record.seq).upper()
        spans = (
            find_annotated_features(record, keywords)
            + find_restriction_sites(seq_str, enzyme_names)
            + find_motifs(seq_str, motif_pairs, name)
        )
        spans = sorted(set(spans), key=lambda s: s[0])
        highlights[name] = spans
        for enz in enzyme_names:
            enz = enz.strip()
            if enz and not any(label == f"{enz} site" for _, _, label in spans):
                print(f"note: no {enz} site found in construct '{name}'", file=sys.stderr)
    return highlights


def make_aligner(match, mismatch, open_gap, extend_gap):
    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = match
    aligner.mismatch_score = mismatch
    aligner.open_gap_score = open_gap
    aligner.extend_gap_score = extend_gap
    return aligner


@dataclass
class Hit:
    construct: str
    orientation: str
    identity_pct: float
    aligned_len: int
    coverage_pct: float
    construct_start: int
    construct_end: int
    aln: object = field(default=None, repr=False)
    construct_len: int = field(default=0, repr=False)


def best_hit_for_read(aligner, read_seq, construct_name, construct_seq, circular, overlap):
    """Try both orientations of read against construct, return the better Hit."""
    working_seq = construct_seq
    if circular:
        wrap = min(len(construct_seq), overlap)
        working_seq = construct_seq + construct_seq[:wrap]

    best = None
    for orientation, seq in (("fwd", read_seq), ("rev", str(Seq(read_seq).reverse_complement()))):
        alignments = aligner.align(seq, working_seq)
        aln = alignments[0]
        counts = aln.counts()
        aligned_len = counts.identities + counts.mismatches
        if aligned_len == 0:
            continue
        identity_pct = 100.0 * counts.identities / aligned_len
        coverage_pct = 100.0 * aligned_len / len(read_seq)
        c_start = int(aln.coordinates[1].min())
        c_end = int(aln.coordinates[1].max())
        if circular:
            c_start %= len(construct_seq)
            c_end %= len(construct_seq)
        hit = Hit(
            construct=construct_name,
            orientation=orientation,
            identity_pct=identity_pct,
            aligned_len=aligned_len,
            coverage_pct=coverage_pct,
            construct_start=c_start,
            construct_end=c_end,
            aln=aln,
            construct_len=len(construct_seq),
        )
        if best is None or (hit.identity_pct, hit.aligned_len) > (best.identity_pct, best.aligned_len):
            best = hit
    return best


def classify(ranked, min_identity, min_coverage, ambiguous_margin):
    if not ranked:
        return "no_match"
    top = ranked[0]
    if top.identity_pct < min_identity or top.coverage_pct < min_coverage:
        return "no_match"
    if len(ranked) > 1:
        second = ranked[1]
        if (top.identity_pct - second.identity_pct) <= ambiguous_margin and second.identity_pct >= min_identity:
            return "ambiguous"
    return "match"


def format_alignment_with_features(hit, feature_spans):
    """Build a wrapped alignment block (read / match / construct / feature-marker lines)
    plus a legend, using the feature spans that fall within the aligned region."""
    aln = hit.aln
    read_row = aln[0]
    construct_row = aln[1]
    idx_read = aln.indices[0]
    idx_construct = aln.indices[1]
    n = len(read_row)

    lo = min(hit.construct_start, hit.construct_end)
    hi = max(hit.construct_start, hit.construct_end)
    covering = [(s, e, label) for s, e, label in feature_spans if s < hi and e > lo]

    symbol_of = {}
    for s, e, label in covering:
        symbol_of[(s, e, label)] = SYMBOL_PALETTE[len(symbol_of) % len(SYMBOL_PALETTE)]

    def marker_char(col):
        pos = idx_construct[col]
        if pos == -1:
            return " "
        pos %= hit.construct_len
        candidates = [(e - s, sym) for (s, e, label), sym in symbol_of.items() if s <= pos < e]
        if not candidates:
            return " "
        # smallest (most specific) feature wins where features are nested/overlapping
        return min(candidates)[1]

    match_row = "".join(
        " " if (read_row[i] == "-" or construct_row[i] == "-")
        else ("|" if read_row[i] == construct_row[i] else ".")
        for i in range(n)
    )
    marker_row = "".join(marker_char(i) for i in range(n)) if covering else ""

    lines = []
    for block_start in range(0, n, LINE_WIDTH):
        block_end = min(block_start + LINE_WIDTH, n)
        rblock = idx_read[block_start:block_end]
        cblock = idx_construct[block_start:block_end]
        r_valid = rblock[rblock != -1]
        c_valid = cblock[cblock != -1]
        r_first = int(r_valid[0]) if len(r_valid) else 0
        r_last = int(r_valid[-1]) + 1 if len(r_valid) else 0
        c_first = int(c_valid[0]) % hit.construct_len if len(c_valid) else 0
        c_last = (int(c_valid[-1]) % hit.construct_len) + 1 if len(c_valid) else 0

        lines.append(f"read      {r_first:>6} {read_row[block_start:block_end]} {r_last}")
        lines.append(f"          {'':>6} {match_row[block_start:block_end]}")
        lines.append(f"construct {c_first:>6} {construct_row[block_start:block_end]} {c_last}")
        if marker_row:
            lines.append(f"features  {'':>6} {marker_row[block_start:block_end]}")
        lines.append("")

    if covering:
        lines.append("Feature legend:")
        for (s, e, label), sym in symbol_of.items():
            feat_len = e - s
            covered = sum(1 for i in range(n) if idx_construct[i] != -1 and s <= (idx_construct[i] % hit.construct_len) < e)
            lines.append(f"  {sym} = {label}  [construct {s + 1}-{e}]  ({covered}/{feat_len} bp covered by this read)")
        lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--constructs", nargs="+", required=True,
                     help="Directory, glob, or list of files with candidate constructs (.gb/.gbk/.dna/.fasta)")
    ap.add_argument("--reads", nargs="+", required=True,
                     help="Directory, glob, or list of sequencing read files (.ab1/.fasta/.seq)")
    ap.add_argument("--outdir", default=None,
                     help="Folder to write results into (default: results_YYYY-MM-DD, created if needed)")
    ap.add_argument("--out", default="construct_id_results.csv",
                     help="Output CSV filename (joined with --outdir unless it contains a path)")
    ap.add_argument("--report", default=None,
                     help="Filename to write full alignment text for each read's best hit "
                          "(joined with --outdir unless it contains a path)")
    ap.add_argument("--circular", action="store_true", help="Treat constructs as circular plasmids (wrap sequence to catch junctions spanning the origin)")
    ap.add_argument("--wrap-overlap", type=int, default=1000, help="bp of overlap to add when --circular is set (default: 1000)")
    ap.add_argument("--min-identity", type=float, default=90.0, help="Minimum %% identity to call a match (default: 90)")
    ap.add_argument("--min-coverage", type=float, default=30.0, help="Minimum %% of the read that must align (default: 30)")
    ap.add_argument("--ambiguous-margin", type=float, default=2.0, help="If top two %% identities are within this margin, call it ambiguous (default: 2.0)")
    ap.add_argument("--no-quality-trim", action="store_true", help="Disable Mott-algorithm quality trimming of .ab1 reads")
    ap.add_argument("--highlight-features", default="",
                     help="Comma-separated, case-insensitive keywords matched against GenBank/SnapGene "
                          "feature labels/notes/genes to highlight in the report (e.g. 'Strep,GB1'). "
                          "Off by default since annotated feature boundaries can be inflated/inconsistent "
                          "(see --motifs for exact-sequence highlighting instead).")
    ap.add_argument("--restriction-enzymes", default="BamHI,XhoI",
                     help="Comma-separated REBASE enzyme names (as known to Bio.Restriction) to locate "
                          "and highlight in the report (default: 'BamHI,XhoI'). Use '' to disable.")
    ap.add_argument("--motifs", default="TEV=GAAAACCTGTACTTCCAGGGA,8xHis=CATCACCATCACCATCACCATCAC",
                     help="Comma-separated NAME=SEQUENCE pairs of exact DNA motifs to locate and highlight "
                          "in the report, searched on both strands (default: the TEV protease site "
                          "ENLYFQG and the 8xHis tag). Use '' to disable.")
    ap.add_argument("--match-score", type=float, default=2.0)
    ap.add_argument("--mismatch-score", type=float, default=-1.0)
    ap.add_argument("--open-gap-score", type=float, default=-10.0)
    ap.add_argument("--extend-gap-score", type=float, default=-0.5)
    args = ap.parse_args()

    outdir = args.outdir or f"results_{date.today().isoformat()}"
    os.makedirs(outdir, exist_ok=True)
    out_path = args.out if os.path.dirname(args.out) else os.path.join(outdir, args.out)
    report_path = None
    if args.report:
        report_path = args.report if os.path.dirname(args.report) else os.path.join(outdir, args.report)

    constructs = load_constructs(args.constructs)
    reads = load_reads(args.reads, use_quality_trim=not args.no_quality_trim)

    keywords = [k for k in args.highlight_features.split(",") if k.strip()]
    enzyme_names = [e for e in args.restriction_enzymes.split(",") if e.strip()]
    motif_pairs = parse_motif_arg(args.motifs)
    highlights = build_highlights(constructs, keywords, enzyme_names, motif_pairs)

    construct_seqs = {name: str(rec.seq).upper() for name, rec in constructs.items()}

    print(f"Loaded {len(constructs)} candidate construct(s): {', '.join(constructs)}", file=sys.stderr)
    print(f"Loaded {len(reads)} sequencing read(s): {', '.join(reads)}", file=sys.stderr)
    print(f"Writing results to {outdir}/", file=sys.stderr)

    aligner = make_aligner(args.match_score, args.mismatch_score, args.open_gap_score, args.extend_gap_score)

    fieldnames = [
        "read", "call", "best_construct", "orientation", "identity_pct", "aligned_len",
        "read_len", "coverage_pct", "construct_start", "construct_end",
        "second_best_construct", "second_best_identity_pct",
    ]

    report_lines = []

    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for read_name, read_seq in reads.items():
            ranked = []
            for construct_name, construct_seq in construct_seqs.items():
                hit = best_hit_for_read(aligner, read_seq, construct_name, construct_seq,
                                         args.circular, args.wrap_overlap)
                if hit is not None:
                    ranked.append(hit)
            ranked.sort(key=lambda h: (h.identity_pct, h.aligned_len), reverse=True)

            call = classify(ranked, args.min_identity, args.min_coverage, args.ambiguous_margin)
            top = ranked[0] if ranked else None
            second = ranked[1] if len(ranked) > 1 else None

            row = {
                "read": read_name,
                "call": call,
                "best_construct": top.construct if top else "",
                "orientation": top.orientation if top else "",
                "identity_pct": f"{top.identity_pct:.2f}" if top else "",
                "aligned_len": top.aligned_len if top else "",
                "read_len": len(read_seq),
                "coverage_pct": f"{top.coverage_pct:.1f}" if top else "",
                "construct_start": top.construct_start if top else "",
                "construct_end": top.construct_end if top else "",
                "second_best_construct": second.construct if second else "",
                "second_best_identity_pct": f"{second.identity_pct:.2f}" if second else "",
            }
            writer.writerow(row)

            flag = {"match": "OK", "ambiguous": "??", "no_match": "XX"}[call]
            print(f"[{flag}] {read_name}: {row['best_construct']} "
                  f"({row['identity_pct']}% id, {row['coverage_pct']}% cov, {row['orientation']})")

            if report_path and top:
                report_lines.append(f"=== {read_name} -> {top.construct} ({top.orientation}, "
                                     f"{top.identity_pct:.2f}% identity, {top.coverage_pct:.1f}% coverage) ===")
                report_lines.append(format_alignment_with_features(top, highlights.get(top.construct, [])))
                report_lines.append("")

    if report_path:
        with open(report_path, "w") as fh:
            fh.write("\n".join(report_lines))
        print(f"Detailed alignments written to {report_path}", file=sys.stderr)

    print(f"Results written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
