#!/usr/bin/env Rscript
# DESeq2 differential expression, called from ngs_genius/de.py.
#
# Usage:
#   Rscript run_deseq2.R counts.csv meta.csv outdir condition reference treatment [covariates] [min_total]
#
# counts.csv : genes x samples raw integer counts (first column = gene id)
# meta.csv   : samples x variables (first column = sample id)
# covariates : optional comma-separated metadata columns added to the design (e.g. batch)
#
# Outputs in outdir: results.csv, normalized_counts.csv, sessionInfo.txt

suppressPackageStartupMessages(library(DESeq2))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
  stop("Usage: run_deseq2.R counts.csv meta.csv outdir condition reference treatment [covariates] [min_total]")
}
counts_path <- args[1]; meta_path <- args[2]; outdir <- args[3]
condition <- args[4]; reference <- args[5]; treatment <- args[6]
covariates <- if (length(args) >= 7 && nzchar(args[7])) strsplit(args[7], ",")[[1]] else character(0)
min_total  <- if (length(args) >= 8) as.numeric(args[8]) else 10

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

counts <- read.csv(counts_path, row.names = 1, check.names = FALSE)
meta   <- read.csv(meta_path,   row.names = 1, check.names = FALSE)

if (!all(colnames(counts) %in% rownames(meta))) {
  stop("rownames(colData) must match colnames(countData): sample names in the count matrix are missing from the metadata.")
}
meta <- meta[colnames(counts), , drop = FALSE]
if (!condition %in% colnames(meta)) stop(sprintf("Condition column '%s' not found in metadata.", condition))

meta[[condition]] <- relevel(factor(meta[[condition]]), ref = reference)
for (cv in covariates) {
  if (!is.numeric(meta[[cv]])) meta[[cv]] <- factor(meta[[cv]])
}

design <- as.formula(paste("~", paste(c(covariates, condition), collapse = " + ")))
message("Design: ", deparse(design))

dds <- DESeqDataSetFromMatrix(countData = round(as.matrix(counts)), colData = meta, design = design)
dds <- dds[rowSums(counts(dds)) >= min_total, ]

# Standard run; fall back to poscounts size factors for sparse matrices.
dds <- tryCatch(
  DESeq(dds, quiet = TRUE),
  error = function(e) {
    if (grepl("every gene contains at least one zero", conditionMessage(e))) {
      message("Falling back to sfType='poscounts' (", conditionMessage(e), ")")
      DESeq(dds, sfType = "poscounts", quiet = TRUE)
    } else stop(e)
  }
)

res <- results(dds, contrast = c(condition, treatment, reference), alpha = 0.05)
out <- data.frame(gene = rownames(res), as.data.frame(res), check.names = FALSE)

# Optional shrunken LFCs for ranking / visualisation (does not change p-values).
coef_name <- paste(condition, treatment, "vs", reference, sep = "_")
if (coef_name %in% resultsNames(dds) && requireNamespace("apeglm", quietly = TRUE)) {
  shr <- lfcShrink(dds, coef = coef_name, type = "apeglm", quiet = TRUE)
  out$log2FoldChange_shrunk <- as.data.frame(shr)[out$gene, "log2FoldChange"]
}

write.csv(out, file.path(outdir, "results.csv"), row.names = FALSE)
write.csv(counts(dds, normalized = TRUE), file.path(outdir, "normalized_counts.csv"))
writeLines(capture.output(sessionInfo()), file.path(outdir, "sessionInfo.txt"))
message("Done: ", nrow(out), " genes tested.")
