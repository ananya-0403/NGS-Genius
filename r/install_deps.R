# One-time R dependency install for NGS-Genius
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install(c("DESeq2", "apeglm"), ask = FALSE, update = FALSE)
