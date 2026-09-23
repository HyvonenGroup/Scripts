# identify_construct

A command-line tool that matches Sanger sequencing reads (`.ab1` chromatograms) against a panel of candidate plasmid constructs to determine which construct/variant was actually cloned. Useful when one has designed several related plasmids (different inserts, tags, or point mutations) and need to figure out, from the sequencing data that comes back, which one you actually got.

Built on [Biopython](https://biopython.org/) — no external aligners (BLAST, EMBOSS) required.

## Features

- **Reads GenBank (`.gb`/`.gbk`), SnapGene (`.dna`), and FASTA** construct files directly; SnapGene maps are parsed natively via Biopython, no conversion needed.
- **Reads `.ab1` chromatograms** with automatic quality trimming (Mott's algorithm), or plain FASTA/`.seq` reads.
- **Local pairwise alignment** against each full-length candidate, tried in both orientations; works even when a read only covers part of the plasmid, and reports which strand/orientation matched.
- **Circular-plasmid aware** (`--circular`): wraps the sequence so a read spanning the origin of the reference file still aligns correctly.
- **Confidence-aware calling**: flags a read `match`, `ambiguous`, or `no_match` rather than always forcing a best guess — `ambiguous` specifically catches reads that don't cover the region that actually distinguishes two candidates (e.g. wrong/too-distant primer) instead of silently mis-assigning them.
- **Feature highlighting in the alignment report**: annotates the printed alignment with restriction sites (any enzyme known to `Bio.Restriction`, e.g. BamHI/XhoI) and exact sequence motifs you specify (e.g. a TEV protease site or a His-tag), so you can see at a glance whether a key feature is present and how much of it the read actually covers.
- **Dated output folders** (`results_YYYY-MM-DD/` by default) so repeat runs on different days don't overwrite each other.

## Requirements

- Python 3.8+
- [Biopython](https://biopython.org/) ≥ 1.80 (for `Alignment.counts()` / `Alignment.indices`)

```bash
pip install -r requirements.txt
```

## Quick start

```bash
python3 identify_construct.py \
  --constructs constructs/ \
  --reads reads/
```

`--constructs` and `--reads` each accept a directory, a glob pattern, or an explicit list of files. Results are written to `results_YYYY-MM-DD/construct_id_results.csv` by default, and progress/summary is printed to the console:

```
Loaded 3 candidate construct(s): pConstructA, pConstructB, pConstructC
Loaded 12 sequencing read(s): read01, read02, ...
[OK] read01: pConstructA (99.33% id, 100.0% cov, fwd)
[OK] read02: pConstructB (100.00% id, 100.0% cov, rev)
[??] read03: pConstructA (100.00% id, 100.0% cov, fwd)
[XX] read04: pConstructB (48.57% id, 93.3% cov, rev)
```

`OK` = confident match, `??` = ambiguous (read doesn't reach the distinguishing region), `XX` = no match above threshold.

### Circular plasmids

If the region that distinguishes your candidates might span the origin of the deposited sequence (position 0/end junction):

```bash
python3 identify_construct.py --constructs constructs/ --reads reads/ --circular
```

### Detailed alignment report with feature highlighting

```bash
python3 identify_construct.py \
  --constructs constructs/ --reads reads/ \
  --report alignments.txt \
  --motifs "TEV=GAAAACCTGTACTTCCAGGGA,8xHis=CATCACCATCACCATCACCATCAC" \
  --restriction-enzymes "BamHI,XhoI"
```

This writes a full pairwise alignment for each read's best hit, with a marker line showing exactly where each highlighted feature falls, plus a legend with coverage stats:

```
read           0 CGCTACCAAAACCTTCACGGTTACGGAAAGCGGTACCGAAAACCTGTACTTCCAGGGATC 60
                 ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
construct    300 CGCTACCAAAACCTTCACGGTTACGGAAAGCGGTACCGAAAACCTGTACTTCCAGGGATC 360
features                                              11111111111111111122222

Feature legend:
  1 = TEV  [construct 338-358]  (21/21 bp covered by this read)
  2 = BamHI site  [construct 356-361]  (6/6 bp covered by this read)
```

Nested/overlapping features (e.g. a restriction site sitting inside a larger annotated region) are resolved by showing the most specific (smallest) feature at each position.

## How matching works

For every read, both orientations (as-sequenced and reverse-complement) are locally aligned against every candidate construct with `Bio.Align.PairwiseAligner`. The best-scoring construct/orientation pair is reported. A read is called:

- **`match`** — top candidate's identity ≥ `--min-identity` and coverage ≥ `--min-coverage`, and it's clearly ahead of the runner-up.
- **`ambiguous`** — top candidate passes both thresholds, but the second-best candidate is within `--ambiguous-margin` percentage points of it. This is common when a read doesn't extend into the region that actually differs between candidates.
- **`no_match`** — no candidate passes the identity/coverage thresholds.

## Feature highlighting: annotations vs. exact motifs

Two independent ways to highlight regions of interest in `--report` output:

- **`--motifs "NAME=SEQUENCE,..."`** (default: `TEV=GAAAACCTGTACTTCCAGGGA,8xHis=CATCACCATCACCATCACCATCAC`) exact literal DNA sequence search, both strands. To use for short, well-defined motifs (protease sites, tags, primer-binding sites) where one knows the precise sequence. Recommended default, since annotated feature boundaries in SnapGene/GenBank files are sometimes inflated or inconsistent.
- **`--highlight-features "keyword,..."`** (off by default) case-insensitive keyword match against GenBank/SnapGene feature labels/notes/genes (e.g. `"Strep,GB1"`). Use this when the annotated span is trustworthy and one doesn't want to type out the sequence.
- **`--restriction-enzymes "NAME,..."`** (default: `BamHI,XhoI`) locates recognition sites for any enzyme known to `Bio.Restriction`.

## CLI reference

| Option | Default | Description |
|---|---|---|
| `--constructs` | *(required)* | Directory/glob/files with candidate constructs (`.gb`/`.gbk`/`.dna`/`.fasta`) |
| `--reads` | *(required)* | Directory/glob/files with sequencing reads (`.ab1`/`.fasta`/`.seq`) |
| `--outdir` | `results_YYYY-MM-DD` | Output folder (created if needed) |
| `--out` | `construct_id_results.csv` | CSV output filename |
| `--report` | *(off)* | Filename for the detailed alignment report |
| `--circular` | off | Wrap construct sequences to catch junctions spanning the origin |
| `--wrap-overlap` | `1000` | bp of wrap-around overlap when `--circular` is set |
| `--min-identity` | `90.0` | Minimum %identity to call a match |
| `--min-coverage` | `30.0` | Minimum % of the read that must align |
| `--ambiguous-margin` | `2.0` | %identity gap below which the top two hits are called ambiguous |
| `--no-quality-trim` | off | Disable Mott-algorithm quality trimming of `.ab1` reads |
| `--motifs` | `TEV=...,8xHis=...` | Exact `NAME=SEQUENCE` motifs to highlight (both strands) |
| `--highlight-features` | *(off)* | Keyword(s) matched against annotated feature labels |
| `--restriction-enzymes` | `BamHI,XhoI` | Enzyme names (per `Bio.Restriction`) to highlight |
| `--match-score`, `--mismatch-score`, `--open-gap-score`, `--extend-gap-score` | `2.0`, `-1.0`, `-10.0`, `-0.5` | Local alignment scoring parameters |

Run `python3 identify_construct.py --help` for the full list.

## Output CSV columns

| Column | Description |
|---|---|
| `read` | Read filename (without extension) |
| `call` | `match` / `ambiguous` / `no_match` |
| `best_construct` | Best-matching candidate construct |
| `orientation` | `fwd` or `rev` (relative to the read as provided) |
| `identity_pct` | % identity over the aligned region |
| `aligned_len` | Length of the aligned region (bp) |
| `read_len` | Total read length after quality trimming |
| `coverage_pct` | % of the read that aligned |
| `construct_start` / `construct_end` | Aligned region's position in the construct |
| `second_best_construct` / `second_best_identity_pct` | Runner-up, for judging ambiguous calls |

## Limitations

- Restriction-site and motif search do not currently account for sites spanning the circular origin junction.
- Designed for verifying which of several *known, designed* candidates a read matches — it is not a general-purpose variant caller and won't identify novel/unexpected mutations beyond reporting reduced %identity.

## License

MIT
