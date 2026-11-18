"""Rule-based diagnosis of common RNA-seq pipeline errors.

A curated pattern library gives instant, reliable answers for well-known failures
(DESeq2, STAR/HISAT2, featureCounts, memory). Anything unmatched, or any case where the
user wants a deeper explanation, is passed to Gemini together with these matches
(see ``genai.explain_error``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    id: str
    tool: str
    pattern: str
    cause: str
    fix: str
    code: str = ""


RULES: list[Rule] = [
    Rule("deseq_sizefactors", "DESeq2",
         r"every gene contains at least one zero",
         "Median-of-ratios size-factor estimation needs genes with non-zero counts in every sample; sparse "
         "matrices (e.g. single-cell-like data, many low-depth samples) have none.",
         "Use the 'poscounts' estimator, or filter very low-count genes first.",
         "dds <- DESeq(dds, sfType = \"poscounts\")"),
    Rule("deseq_rowname_mismatch", "DESeq2",
         r"rownames\(colData\)|colnames\(countData\)|not in the same order|samples? .* (missing|not found)",
         "Sample names (or their order) differ between the count-matrix columns and the metadata rows.",
         "Check for typos/whitespace, transposed matrices, or a stray annotation column. Reorder metadata to match.",
         "all(colnames(counts) %in% rownames(meta))\nmeta <- meta[colnames(counts), , drop = FALSE]"),
    Rule("deseq_nonint", "DESeq2",
         r"(not integer|non-negative integer|some values in assay are not integers)",
         "DESeq2 needs raw integer counts. TPM/FPKM/normalised values (or RSEM expected counts) were supplied.",
         "Supply raw counts from featureCounts/HTSeq/STAR --quantMode GeneCounts, or import RSEM/Salmon via tximport.",
         "dds <- DESeqDataSetFromMatrix(round(as.matrix(counts)), colData, ~ condition)  # last resort"),
    Rule("deseq_fullrank", "DESeq2",
         r"(not full rank|full rank|linearly dependent)",
         "The design matrix has linearly dependent columns: a covariate is perfectly confounded with the "
         "condition (or two covariates encode the same thing).",
         "Remove the redundant covariate, or add samples so each batch contains more than one condition. "
         "Inspect with table(meta$batch, meta$condition).",
         "table(meta$batch, meta$condition)\ndesign(dds) <- ~ condition"),
    Rule("deseq_nolevels", "DESeq2",
         r"(contrast|factor).*(level|not found)|levels? .* not (present|found)|less than 2 levels",
         "The reference/treatment level does not exist in the condition column (case, spelling, or the "
         "column is not a factor).",
         "Print levels(factor(meta$condition)) and use the exact strings in contrast=c('condition', treat, ref).",
         "levels(factor(meta$condition))\nres <- results(dds, contrast = c(\"condition\", \"treated\", \"control\"))"),
    Rule("deseq_dup", "DESeq2",
         r"duplicate\(?s?\)? (row|gene)|duplicated? (row ?names|gene)|non-unique|'row.names' .* duplicate",
         "Duplicate gene identifiers in the count matrix.",
         "Collapse duplicates (sum counts per gene) before building the dataset.",
         "counts <- rowsum(counts, group = rownames(counts))"),
    Rule("deseq_na_foreign", "DESeq2",
         r"NA/NaN/Inf in foreign function call|missing values? (in|are)|NA values",
         "Missing, infinite, or non-numeric values in the counts or in a design variable.",
         "Check anyNA(counts), non-numeric columns, and empty cells in the metadata. Replace or remove them.",
         "anyNA(as.matrix(counts)); sapply(meta, function(x) sum(is.na(x)))"),
    Rule("deseq_norep", "DESeq2",
         r"(no replicates|same number of samples and coefficients|design matrix has .* rank|replicates)",
         "There are no residual degrees of freedom: too few samples per group to estimate dispersion.",
         "Add biological replicates (>=3 per group is recommended). Without them DE results are not reliable.",
         ""),
    Rule("memory", "General",
         r"cannot allocate vector of size|MemoryError|Killed|out of memory|std::bad_alloc|Unable to allocate",
         "The process ran out of RAM.",
         "Filter low-count genes, work on fewer samples at a time, increase memory, or use low-memory settings.",
         "dds <- dds[rowSums(counts(dds)) >= 10, ]"),
    Rule("star_genome", "STAR",
         r"EXITING because of FATAL (ERROR|INPUT ERROR).*(genome|SA|Genome)|could not open genome file|"
         r"genomeParameters\.txt",
         "STAR cannot find or read the genome index (wrong --genomeDir, incomplete index, or version mismatch).",
         "Rebuild the index with the same STAR version (--runMode genomeGenerate) and check the path/permissions.",
         "STAR --runMode genomeGenerate --genomeDir idx --genomeFastaFiles genome.fa --sjdbGTFfile genes.gtf --sjdbOverhang 99"),
    Rule("star_ram", "STAR",
         r"not enough memory for BAM sorting|limitBAMsortRAM|Fatal LIMIT error",
         "STAR ran out of the memory reserved for BAM sorting.",
         "Set --limitBAMsortRAM (bytes) larger, e.g. 10000000000, or use fewer threads.",
         "STAR ... --outSAMtype BAM SortedByCoordinate --limitBAMsortRAM 10000000000"),
    Rule("hisat_idx", "HISAT2",
         r"(Could not locate a HISAT2 index|Error: Encountered internal HISAT2 exception|\.ht2)",
         "HISAT2 cannot locate the index files (-x prefix wrong).",
         "Pass the index *prefix* (without .1.ht2 etc.) and confirm all 8 .ht2 files exist.",
         "ls genome_index*.ht2"),
    Rule("featurecounts_gtf", "featureCounts",
         r"(Failed to open the annotation file|No features were loaded|failed to find the gene identifier attribute|"
         r"Unable to find .* attribute)",
         "featureCounts cannot read the GTF, or the -g attribute (default gene_id) is missing; chromosome names may "
         "also differ between BAM and GTF (chr1 vs 1).",
         "Check the GTF path/format, use -g/-t appropriate for your annotation, and match chromosome naming.",
         "featureCounts -a genes.gtf -t exon -g gene_id -o counts.txt *.bam"),
    Rule("featurecounts_paired", "featureCounts",
         r"No paired-end reads were detected|paired-end reads are found but -p is not",
         "Paired-end data counted without -p (or single-end data counted with -p).",
         "Add -p --countReadPairs for paired-end data, drop -p for single-end.",
         "featureCounts -p --countReadPairs -a genes.gtf -o counts.txt *.bam"),
    Rule("low_assign", "featureCounts / STAR",
         r"(low|poor).*(assigned|mapping|alignment)|Unassigned_NoFeatures|uniquely mapped reads % \| *[0-3]?\d\.\d+%",
         "Very low assignment/mapping rate. Typical causes: wrong species/genome, rRNA contamination, wrong strandedness, "
         "adapters not trimmed.",
         "Verify the genome, try -s 0/1/2 for strandedness (infer_experiment.py), check rRNA and adapter content in FastQC.",
         "infer_experiment.py -r genes.bed -i sample.bam"),
    Rule("fastqc_adapter", "FastQC / trimming",
         r"(adapter content|overrepresented sequences|per base sequence content).*(fail|warn)",
         "Adapter read-through or biased composition at read ends (common in library prep; first bases bias is normal for RNA-seq).",
         "Trim adapters with fastp / Trim Galore / cutadapt and re-run FastQC.",
         "fastp -i R1.fq.gz -I R2.fq.gz -o R1.trim.fq.gz -O R2.trim.fq.gz --detect_adapter_for_pe"),
    Rule("python_pydeseq2", "PyDESeq2",
         r"(KeyError|ValueError).*(condition|design|contrast)|Design matrix|singular matrix",
         "A column named in the design/contrast is missing or the design is singular.",
         "Confirm metadata columns/levels and remove confounded covariates.",
         ""),
]


@dataclass
class Diagnosis:
    rule: Rule
    snippet: str


def diagnose(text: str) -> list[Diagnosis]:
    """Return all rules whose pattern matches ``text`` (case-insensitive), with the matched snippet."""
    out: list[Diagnosis] = []
    if not text:
        return out
    for rule in RULES:
        m = re.search(rule.pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            start = max(0, m.start() - 40)
            out.append(Diagnosis(rule, text[start:m.end() + 60].strip().replace("\n", " ")))
    return out


def format_diagnoses(diags: list[Diagnosis]) -> str:
    if not diags:
        return "No built-in pattern matched this message."
    parts = []
    for d in diags:
        r = d.rule
        block = f"**{r.tool} — {r.id}**\n\n- **Likely cause:** {r.cause}\n- **Fix:** {r.fix}"
        if r.code:
            block += f"\n\n```\n{r.code}\n```"
        parts.append(block)
    return "\n\n---\n\n".join(parts)
