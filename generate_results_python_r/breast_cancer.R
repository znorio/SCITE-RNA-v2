library(tidyverse)
library(tidyestimate)
library(edgeR)  # for CPM normalization

# 1. Load your raw gene expression counts
counts <- read.csv("C:/Users/Norio/Documents/GitHub/SCITE-RNA-v2/data/input_data/bc03/gene_expression_counts.csv", row.names = 1)

# 2. Filter out genes with zero total expression (optional but recommended)
counts <- counts[rowSums(counts) > 0, ]

# 3. Convert raw counts to CPM (counts per million)
cpm_mat <- cpm(as.matrix(counts))

# 4. Apply log2 transformation with pseudocount
log_cpm <- log2(cpm_mat + 1)

log_cpm_df <- as.data.frame(log_cpm) %>%
  rownames_to_column("hgnc_symbol")

# 5. Filter to common ESTIMATE-compatible genes
filtered <- filter_common_genes(log_cpm_df,
                                 id = "hgnc_symbol",
                                 tidy = TRUE,      # since gene names are in the first column
                                 tell_missing = TRUE,
                                 find_alias = TRUE)

# 6. Run ESTIMATE scoring
scored <- estimate_score(filtered, is_affymetrix = FALSE)

scored$estimate <- ifelse(scored$estimate > 0, 1, 0)

# Select only relevant columns
output_df <- scored[, c("sample", "estimate")]

# Save to tab-delimited .txt file
write.table(output_df, file = "C:/Users/Norio/Documents/GitHub/SCITE-RNA-v2/data/input_data/bc03/estimate_classification.txt", sep = "\t", row.names = FALSE, quote = FALSE)

# 7. View result
print(head(scored))

# 8. Plot immune/stromal/ESTIMATE scores
scored %>%
  pivot_longer(cols = c(stromal, immune, estimate)) %>%
  ggplot(aes(x = sample, y = value, fill = name)) +
  geom_col(position = "dodge") +
  labs(x = "Sample", y = "Score", fill = "Score Type") +
  theme_minimal() +
  theme(axis.text.x = element_blank())