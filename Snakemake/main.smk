rule all:
    input:
        "data/all_nps.txt",
        expand("data/splitted/part_{i}", i=range(7)),
        expand("data/cleaned/cleaned_part_{i}.txt", i=range(7)),
        expand("data/featurized/featurized_part_{i}.csv", i=range(7)),
        expand("data/fingerprint/fingerprint_part_{i}.csv", i=range(7)),
        "data/featurized/merged_features.csv",
        "data/fingerprint/merged_fingerprints.csv",

        "output/report.txt",
        "output/combined_clusters.png",
        "output/outlier_detection.png",
        "output/tanimoto.png",
        "data/clustered_data.csv",
        "data/processed_data.csv",
        "data/natural_substitutes.csv"


rule load_data:
    input:
        synthetic="data/non_natural_products.txt",
        natural="data/subset_natural_products.txt"
    output:
        "data/all_nps.txt"
    shell:
        "python3 scripts/load_data.py {input.synthetic} {input.natural} {output}"


rule split_dataset:
    input:
        combined="data/all_nps.txt" 
    output:
        expand("data/splitted/part_{i}", i=range(7))  # Expected output
    params:
        output_dir="data/splitted",
        lines_per_chunk=200
    shell:
        """
        mkdir -p {params.output_dir}
        rm -f {params.output_dir}/part_*

        header_file="{params.output_dir}/header.txt"
        head -n 1 {input.combined} > "$header_file"

        tail -n +2 {input.combined} | split -l {params.lines_per_chunk} - {params.output_dir}/part_

        i=0
        for file in {params.output_dir}/part_*; do
            cat "$header_file" "$file" > {params.output_dir}/part_$i
            rm "$file"
            i=$((i+1))
        done
        """
rule clean_chunk:
    input:
        "data/splitted/part_{i}"
    output:
        "data/cleaned/cleaned_part_{i}.txt"
    shell:
        "/homes/mesmaeili/miniconda3/bin/python scripts/clean.py {input} {output}"
rule featurize_chunk:
    input:
        "data/cleaned/cleaned_part_{i}.txt"
    output:
        "data/featurized/featurized_part_{i}.csv"
    shell:
        "/homes/mesmaeili/miniconda3/bin/python scripts/features.py {input} {output}"

rule fingerprint_chunk:
    input:
        "data/featurized/featurized_part_{i}.csv"
    output:
        "data/fingerprint/fingerprint_part_{i}.csv"
    shell:
        "/homes/mesmaeili/miniconda3/bin/python scripts/generate_fingerprint.py {input} {output}"
rule merge_features:
    input:
        expand("data/featurized/featurized_part_{i}.csv", i=range(7))
    output:
        "data/featurized/merged_features.csv"
    shell:
        "/homes/mesmaeili/miniconda3/bin/python scripts/merge_feature.py"
rule merge_fingerprints:
    input:
        expand("data/fingerprint/fingerprint_part_{i}.csv", i=range(7))
    output:
        "data/fingerprint/merged_fingerprints.csv"
    shell:
        "/homes/mesmaeili/miniconda3/bin/python scripts/merge_fingerprints.py"

rule run_analysis:
    input:
        features="data/featurized/merged_features.csv",
        fingerprints="data/fingerprint/merged_fingerprints.csv"
    output:
        report="output/report.txt",
        plot="output/combined_clusters.png",
        outliers="output/outlier_detection.png",
        tanimoto="output/tanimoto.png",
        clustered="data/clustered_data.csv",
        processed="data/processed_data.csv",
        substitutes="data/natural_substitutes.csv"
    shell:
        "/homes/mesmaeili/miniconda3/bin/python scripts/main.py"
