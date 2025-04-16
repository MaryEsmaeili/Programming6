
# Molecular Clustering Pipeline using Snakemake

This project implements a fully automated and parallelized molecular clustering pipeline using Snakemake. The aim is to identify potential natural substitutes for synthetic compounds based on molecular descriptors and fingerprints.

## Project Structure

```
.
├── README.md
├── main.smk
├── scripts
│   ├── load_data.py
│   ├── clean.py
│   ├── features.py
│   ├── generate_fingerprint.py
│   ├── merge_feature.py
│   ├── merge_fingerprints.py
│   ├── clustering.py
│   ├── cluster_data.py
│   ├── cluster_analysis.py
│   ├── main.py
│   ├── cluster_kmean.py
│   ├── descriptor_fingerprint_processor.py
│   ├── cluster_analysis.py
│   ├── molecular_pipeline.py
│   ├── evaluation.py
│   ├── molecular_visualisation.py
│   ├── visualisation.py
│   └── substitute_analysis.py
├── data
│   ├── non_natural_products.txt
│   ├── subset_natural_products.txt
│   └── ...
├── output
│   ├── report.txt
│   ├── molecule_images/
│   ├── outlier_detection.png
│   ├── combined_clusters.png
│   ├── mixed_clusters.png
│   ├── substitutes_report.csv
│   └── tanimoto.png
```

## How to Run the Workflow

1. **Activate environment** (if using conda):
   ```bash
   conda activate your_env
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Snakemake with 4 cores**:
   ```bash
   snakemake -s main.smk --cores 4
   ```

## Generating the Workflow DAG

To visualize the workflow structure:
```bash
snakemake -s main.smk --dag | dot -Tpng > images/dag.png
```

## Outputs

- `output/report.txt`: Contains HAC clustering metrics and summary of natural substitutes
- `output/combined_clusters.png`: Clustering plot
- `output/tanimoto.png`: Histogram of Tanimoto similarities
- `output/property_differences.png`: Visual diff in molecular properties
- `data/substitutes_report.csv   `: Substitutes compounds data
- `data/outlier_detection.png`: Outlier plot
- `data/mixed_clusters `: Natural/Synthetic compounds plot

## Authors
- Maryam Esmaeili


