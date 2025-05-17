# Script used to run DENDRO and SClineager on simulated datasets

library(DENDRO)
library(SClineager)
library(cardelino)

n_tests <- 100
n_cells_list <- c(50)
n_mut_list <- c(500)
clones_list <- c(5, 10, 20, "")

base_dir <- file.path("C:/Users/Norio/Documents/GitHub/SCITE-RNA-v2/data", "simulated_data")
# base_dir <- file.path("data", "simulated_data")
# base_dir <- file.path("/cluster/work/bewi/members/znorio/data", "simulated_data")


read.matrix <- function(path){
  mat <- as.matrix(read.table(path, header=FALSE, sep=" "))
  dimnames(mat) <- NULL
  return(mat)
}

read.str.matrix <- function(path) {
  mat <- as.matrix(read.table(path, header=FALSE, sep=" "))
  mat[mat == "A"] <- 1
  mat[mat == "H"] <- 0.5
  mat[mat == "R"] <- 0
  dimnames(mat) <- NULL
  return(mat)
}

merge.to.parent <- function(merge.mat){
  n.leaves <- nrow(merge.mat) + 1
  result <- rep(-1, 2*n.leaves - 1)

  for (i in seq_len(nrow(merge.mat))){
    for (child in merge.mat[i,]){
      parent.id <- n.leaves + i - 1
      if (child < 0) result[-child] <- parent.id
      else result[n.leaves + child] <- parent.id
    }
  }

  return(result)
}

generate.parent.vec <- function(base_path, n.tests=10, clones=5){
  dir.create(file.path(base_path, "sclineager", "sclineager_vaf"), recursive = TRUE)
  dir.create(file.path(base_path, "dendro", "dendro_parent_vec"), recursive = TRUE)
  dir.create(file.path(base_path, "dendro", "dendro_clones"), recursive = TRUE)
  dir.create(file.path(base_path, "sclineager", "sclineager_parent_vec"), recursive = TRUE)
  dir.create(file.path(base_path, "sclineager", "sclineager_clones"), recursive = TRUE)
  dir.create(file.path(base_path, "sclineager", "sclineager_selected"), recursive = TRUE)

  cat("Theoretical number of clones: ", clones)
  sclineager_runtimes <- c()
  dendro_runtimes <- c()

  for (i in 0:(n.tests-1)){
    ref <- t(read.matrix(file.path(base_path, sprintf("ref/ref_%d.txt", i))))
    alt <- t(read.matrix(file.path(base_path, sprintf("alt/alt_%d.txt", i))))
    coverage <- ref + alt
    mut_indicator <- read.matrix(file.path(base_path, sprintf("mut_indicator/mut_indicator_%d.txt", i)))
    mutations_mat <- alt / coverage

    unique_rows <- unique(t(mut_indicator))
    clones <- nrow(unique_rows) # actual number of clones
    cat(" Actual number of clones: ", clones)

    start_time_sclineager <- Sys.time()

    keep <- apply(mutations_mat, 1, function(x) max(x, na.rm = T) -
                   min(x, na.rm = T) > 0.01)
    keep_numeric <- as.numeric(keep)
    write.table(keep_numeric, file = file.path(base_path, sprintf("sclineager/sclineager_selected/sclineager_selected_%d.txt", i)),
                row.names = FALSE, col.names = FALSE)
    mutations_mat_sclineager <- mutations_mat[keep, ]
    coverage_sclineager <- coverage[keep, ]

    res_scl <-
      sclineager_internal(
        mutations_mat = mutations_mat_sclineager,
        coverage_mat = coverage_sclineager,
        max_iter = 200,
        vaf_offset = 0.01,
        dfreedom = ncol(mutations_mat),
        psi = diag(10, ncol(mutations_mat)),
        save = F
      )

    # Cluster the genotype matrix into k=clones
    dist_scl <- dist(t(res_scl[["genotype_mat"]])) # use same method for DENDRO and SClineager
    hc_scl <- hclust(dist_scl, method='ward.D')
    memb_pred_scl <- cutree(hc_scl, k = clones)
    cluster_scl <- DENDRO.cluster(dist_scl, plot=FALSE, type="phylogram")
    parent_vec_scl <- merge.to.parent(cluster_scl$merge)

    end_time_sclineager <- Sys.time()
    runtime_sclineager <- end_time_sclineager - start_time_sclineager
    sclineager_runtimes <- c(sclineager_runtimes, runtime_sclineager)

    write.table(t(res_scl[["genotype_mat"]]), file = file.path(base_path, sprintf("sclineager/sclineager_vaf/sclineager_vaf_%d.txt", i)), row.names = FALSE, col.names = FALSE)
    write.table(parent_vec_scl, file.path(base_path, sprintf("sclineager/sclineager_parent_vec/sclineager_parent_vec_%d.txt", i)), row.names=FALSE, col.names=FALSE)
    write.table(memb_pred_scl, file.path(base_path, sprintf("sclineager/sclineager_clones/sclineager_clones_%d.txt", i)), row.names=FALSE, col.names=FALSE)

    start_time_dendro <- Sys.time()

    filtered <- FilterCellMutation(alt, coverage, mut_indicator, cut.off.VAF = 0.01, cut.off.sd = 10)
    dist <- DENDRO.dist(filtered$X, filtered$N, filtered$Z, show.progress=FALSE)

    hc <- hclust(dist, method='ward.D')
    memb_pred <- cutree(hc, k = clones)
    cluster <- DENDRO.cluster(dist, plot=FALSE,type="phylogram")
    dendro_parent_vec <- merge.to.parent(cluster$merge)

    end_time_dendro <- Sys.time()
    runtime_dendro <- end_time_dendro - start_time_dendro
    dendro_runtimes <- c(dendro_runtimes, runtime_dendro)

    write.table(dendro_parent_vec, file.path(base_path, sprintf("dendro/dendro_parent_vec/dendro_parent_vec_%d.txt", i)), row.names=FALSE, col.names=FALSE)
    write.table(memb_pred, file.path(base_path, sprintf("dendro/dendro_clones/dendro_clones_%d.txt", i)), row.names=FALSE, col.names=FALSE)
  }
  write.table(sclineager_runtimes, file.path(base_path, "sclineager/sclineager_runtimes.txt"), row.names = FALSE, col.names = FALSE)
  write.table(dendro_runtimes, file.path(base_path, "dendro/dendro_runtimes.txt"), row.names = FALSE, col.names = FALSE)
}


paths <- NULL
clones <- NULL

for (i in seq_along(n_cells_list)) {
  n_cells <- n_cells_list[i]
  n_mut <- n_mut_list[i]

  for (clone in clones_list) {
    path <- paste(base_dir, "/", n_cells, "c", n_mut, "m", clone, sep = "")
    paths <- c(paths, path)
    clones <- c(clones, clone)

  }
}

print(paths)

for (i in seq_along(paths)) {
  n_clones <- clones[i]
  path <- paths[i]
  generate.parent.vec(path, n_tests, n_clones)
}

