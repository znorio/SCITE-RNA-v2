# Script used to run DENDRO and SClineager on cancer datasets

library(DENDRO)
library(SClineager)

base_path <- file.path("data")

sample <- "mm34"
clones <- c(2,3,4,5)

read.matrix <- function(path){
  mat <- as.matrix(read.table(path, header=FALSE, sep=" "))
  dimnames(mat) <- NULL
  return(mat)
}


merge.to.parent <- function(merge.mat){
  n.leaves <- nrow(merge.mat) + 1
  result <- rep(-1, 2*n.leaves - 1)

  for (i in 1:nrow(merge.mat)){
    for (child in merge.mat[i,]){
      parent.id <- n.leaves + i - 1
      if (child < 0) result[-child] <- parent.id
      else result[n.leaves + child] <- parent.id
    }
  }

  return(result)
}


dir.create(file.path(base_path, "results", sample, "sclineager", "sclineager_vaf"), recursive = TRUE)
dir.create(file.path(base_path, "results", sample, "dendro", "dendro_parent_vec"), recursive = TRUE)
dir.create(file.path(base_path, "results", sample, "dendro", "dendro_clones"), recursive = TRUE)
dir.create(file.path(base_path, "results", sample, "sclineager", "sclineager_parent_vec"), recursive = TRUE)
dir.create(file.path(base_path, "results", sample, "sclineager", "sclineager_clones"), recursive = TRUE)
dir.create(file.path(base_path, "results", sample, "sclineager", "sclineager_selected"), recursive = TRUE)


sclineager_runtimes <- c()
dendro_runtimes <- c()

ref <- t(as.matrix(read.csv(file.path(base_path, "input_data", sample, "ref.csv"), row.names = 1)))
ref[is.na(ref)] <- 0
dimnames(ref) <- NULL
alt <- t(as.matrix(read.csv(file.path(base_path, "input_data", sample, "alt.csv"), row.names = 1)))
alt[is.na(alt)] <- 0
dimnames(alt) <- NULL

coverage <- ref + alt
mutations_mat <- alt / coverage

start_time_sclineager <- Sys.time()

keep <- apply(mutations_mat, 1, function(x) max(x, na.rm = T) -
               min(x, na.rm = T) > 0.1)
keep_numeric <- as.numeric(keep)
write.table(keep_numeric, file = file.path(base_path, "results", sample, "sclineager", "sclineager_selected", "sclineager_selected.txt"),
            row.names = FALSE, col.names = FALSE)
mutations_mat_sclineager <- mutations_mat[keep, ]
coverage_sclineager <- coverage[keep, ]

res_scl <-
  sclineager_internal(
    mutations_mat = mutations_mat_sclineager,
    coverage_mat = coverage_sclineager,
    max_iter = 2000,
    vaf_offset = 0.01,
    dfreedom = ncol(mutations_mat),
    psi = diag(10, ncol(mutations_mat)),
    save = F
  )
end_time_sclineager <- Sys.time()
runtime_sclineager <- end_time_sclineager - start_time_sclineager
sclineager_runtimes <- c(sclineager_runtimes, runtime_sclineager)

start_time_dendro <- Sys.time()

mut_indicator <- matrix(ifelse(round(mutations_mat, 1) > 0.3, 1, 0), nrow = nrow(mutations_mat), ncol = ncol(mutations_mat))
filtered <- FilterCellMutation(alt, coverage, mut_indicator, cut.off.VAF = 0.05, cut.off.sd = 10)

dist <- DENDRO.dist(filtered$X,filtered$N,filtered$Z,show.progress=TRUE)

end_time_dendro <- Sys.time()
runtime_dendro <- end_time_dendro - start_time_dendro
dendro_runtimes <- c(dendro_runtimes, runtime_dendro)

write.table(sclineager_runtimes, file.path(base_path, "results", sample, "sclineager", "sclineager_runtimes.txt"), row.names = FALSE, col.names = FALSE)
write.table(dendro_runtimes, file.path(base_path, "results", sample, "dendro", "dendro_runtimes.txt"), row.names = FALSE, col.names = FALSE)

for (clone in clones) {
  # Cluster the genotype matrix into k=clones
  dist_scl <- dist(t(res_scl[["genotype_mat"]])) # use same method for DENDRO and SClineager
  hc_scl <- hclust(dist_scl, method='ward.D')
  memb_pred_scl <- cutree(hc_scl, k = clone)
  cluster_scl <- DENDRO.cluster(dist_scl, plot=FALSE, type="phylogram")
  parent_vec_scl <- merge.to.parent(cluster_scl$merge)

  write.table(memb_pred_scl, file.path(base_path, "results", sample, "sclineager", "sclineager_clones", paste0("sclineager_clones_", clone, ".txt")), row.names=FALSE, col.names=FALSE)

  hc <- hclust(dist,method='ward.D')
  memb_pred <- cutree(hc, k = clone)
  cluster <- DENDRO.cluster(dist, plot=FALSE,type="phylogram")
  dendro_parent_vec <- merge.to.parent(cluster$merge)

  write.table(memb_pred, file.path(base_path, "results", sample, "dendro", "dendro_clones", paste0("dendro_clones_", clone, ".txt")), row.names=FALSE, col.names=FALSE)
}

write.table(t(res_scl[["genotype_mat"]]), file = file.path(base_path, "results", sample, "sclineager", "sclineager_vaf", paste0("sclineager_vaf.txt")), row.names = FALSE, col.names = FALSE)
write.table(parent_vec_scl, file.path(base_path, "results", sample, "sclineager", "sclineager_parent_vec", paste0("sclineager_parent_vec.txt")), row.names=FALSE, col.names=FALSE)
write.table(dendro_parent_vec, file.path(base_path, "results", sample, "dendro", "dendro_parent_vec", paste0("dendro_parent_vec.txt")), row.names=FALSE, col.names=FALSE)