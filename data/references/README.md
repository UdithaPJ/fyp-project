# Validation reference databases

## Two classes of reference

| Class | Databases | Circularity | Used by |
|-------|-----------|-------------|---------|
| **Interaction** (same evidence type as the network) | TRRUST, BioGRID, miRTarBase | Circular if the input graph came from the same DB | `BiologicalValidator` |
| **Orthogonal** (evidence independent of topology) | DisGeNET, DEG/OGEE, DrugBank, Gene Ontology | Immune to circularity | `OrthogonalValidator`, `GOEnrichmentValidator` |

The **hold-out** validator (`HoldoutValidator`) needs no files at all — it
hides a random subset of the graph's own edges. Prefer it plus the orthogonal
references for defensible, non-circular validation.

---

## Download table

| Kind | Database | Where | Filename must contain | Env var | CLI flag |
|------|----------|-------|-----------------------|---------|----------|
| grn (interaction) | TRRUST v2 | https://www.grnpedia.org/trrust/ → `trrust_rawdata.human.tsv` | `trrust` | `FYP_TRRUST_PATH` | `--trrust-path` |
| ppi (interaction) | BioGRID | https://downloads.thebiogrid.org/BioGRID → `BIOGRID-ALL-*.tab3.txt` | `biogrid` | `FYP_BIOGRID_PATH` | `--biogrid-path` |
| mirna (interaction) | miRTarBase | https://mirtarbase.cuhk.edu.hk/ → `hsa_MTI.xlsx` (save as CSV/TSV) | `mirtarbase` / `hsa_mti` | `FYP_MIRTARBASE_PATH` | `--mirtarbase-path` |
| disease (orthogonal) | DisGeNET | https://www.disgenet.org/downloads → `curated_gene_disease_associations.tsv` | `disgenet` / `gene_disease` | `FYP_DISGENET_PATH` | `--disgenet-path` |
| essential (orthogonal) | DEG / OGEE | http://origin.tubic.org/deg/ or https://v3.ogee.info/ (human essential genes) | `deg` / `ogee` / `essential` | `FYP_DEG_PATH` | `--deg-path` |
| drug_target (orthogonal) | DrugBank | https://go.drugbank.com/releases → target polypeptide CSV (`Gene Name` column) | `drugbank` / `drug_target` | `FYP_DRUGBANK_PATH` | `--drugbank-path` |
| GO | Gene Ontology | http://current.geneontology.org/annotations/goa_human.gaf.gz (gunzip it) | `goa` / `.gaf` | `FYP_GO_PATH` | `--go-path` |

Search order per reference: explicit CLI path → env var → `data/raw/` →
`data/processed/` → `data/references/`.

---

## Column expectations (loaders are tolerant)

The gene-set loaders extract **one gene symbol per row** using candidate
column names, falling back to the first alphabetic cell:

- **DisGeNET** → `geneSymbol` (or `gene_symbol`, `symbol`, `gene`)
- **DEG/OGEE** → `gene_symbol` (or `symbol`, `gene`, `locus`)
- **DrugBank** → `Gene Name` (or `gene_name`, `symbol`, `HGNC`)
- **GO (GAF 2.x)** → tab-separated; column 3 = gene symbol, column 5 = GO ID,
  column 9 = aspect (`P`/`F`/`C`). Lines starting with `!` are comments.

---

## ⚠ ID format matters

The loaders match on **gene symbols** (e.g. `TP53`, `MYC`). If your input
graph uses opaque IDs (Ensembl `ENSP…`, UniProt, integer indices), every
overlap will be zero and validation will silently report `status=skipped`.

The bundled STRING file (`9606.protein.links.v12.0.txt`) uses Ensembl protein
IDs, so it must be mapped to gene symbols (via STRING's
`9606.protein.info.v12.0.txt` `preferred_name` column) **before** the
`node_index_map` is built, or use the hold-out validator (which is ID-agnostic
— it works on the graph's own edge structure).

---

## Running

```bash
# Self-contained, no downloads — start here:
python experiments/validation/run_validation.py --network-type ppi --holdout

# Orthogonal gene-set overlap (needs the disease/essential/drug files above):
python experiments/validation/run_validation.py --network-type ppi --orthogonal

# GO term enrichment (needs goa_human.gaf):
python experiments/validation/run_validation.py --network-type ppi --go --go-aspects P

# Everything at once:
python experiments/validation/run_validation.py --network-type ppi \
    --biological --orthogonal --holdout --go
```

Outputs land in `experiments/outputs/reports/*.csv` and
`experiments/outputs/plots/*.png`.
