# Script used to run DENDRO and SClineager on simulated datasets

library(DENDRO)
library(SClineager)

script_dir <- dirname(sys.frame(1)$ofile)

n_tests <- 100
n_cells_list <- c(50, 100, 100)
n_mut_list <- c(100, 50, 100)
clones_list <- c(5, 10, 20, "")


base_dir <- paste("../data/simulated_data/", sep="")


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

generate.parent.vec <- function(path=NA, n.tests=10, clones=5){
  if (!is.na(path)) setwd(path)

  dir.create("sclineager_selected", recursive = TRUE)
  dir.create("sclineager_vaf", recursive = TRUE)
  dir.create("dendro_parent_vec", recursive = TRUE)
  dir.create("dendro_clones", recursive = TRUE)
  
  cat("Theoretical number of clones: ", clones)
  
  pb <- txtProgressBar(min=0, max=n.tests, initial=0, style=3)
  for (i in 0:(n.tests-1)){
    ref <- read.matrix(sprintf("ref/ref_%d.txt", i))
    alt <- read.matrix(sprintf("alt/alt_%d.txt", i))
    coverage <- ref + alt
    mut_indicator <- read.matrix(sprintf("mut_indicator/mut_indicator_%d.txt", i))
    mutations_mat = alt / coverage
    
    unique_rows <- unique(t(mut_indicator))
    clones <- nrow(unique_rows) # actual number of clones
    cat("Actual number of clones: ", clones)
    
    # TRUE if the corresponding row's maximum and minimum values differ
    # by more than 0.1 and FALSE otherwise -> get rows with multiple genotypes
    keep = apply(mutations_mat, 1, function(x) max(x, na.rm = T) -
                   min(x, na.rm = T) > 0.1)
    keep_numeric <- as.numeric(keep)
    # save this in case rows are filtered out and not optimized by SClineager
    write.table(keep_numeric, file = sprintf("sclineager_selected/sclineager_selected_%d.txt", i),
                row.names = FALSE, col.names = FALSE)
    mutations_mat_sclineager = mutations_mat[keep, ]
    coverage_sclineager = coverage[keep, ]
    
    # run sclineager
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

    write.table(res_scl[["genotype_mat"]], file = sprintf("sclineager_vaf/sclineager_vaf_%d.txt", i), row.names = FALSE, col.names = FALSE)
    
    # run DENDRO
    filtered = FilterCellMutation(alt, coverage, mut_indicator, cut.off.VAF = 0.01, cut.off.sd = 5)
    dist = DENDRO.dist(filtered$X,filtered$N,filtered$Z,show.progress=FALSE)

    hc=hclust(dist,method='ward.D')
    memb_pred=cutree(hc, k = clones)
    cluster <- DENDRO.cluster(dist, plot=FALSE,type="phylogram")
    dendro_parent_vec <- merge.to.parent(cluster$merge)
    write.table(dendro_parent_vec, sprintf("dendro_parent_vec/dendro_parent_vec_%d.txt", i), row.names=FALSE, col.names=FALSE)
    write.table(memb_pred, sprintf("dendro_clones/dendro_clones_%d.txt", i), row.names=FALSE, col.names=FALSE)
  }
}


paths <- c()
clones <- c()

for (i in seq_along(n_cells_list)) {
  n_cells <- n_cells_list[i]
  n_mut <- n_mut_list[i]
  
  for (clone in clones_list) {
    path <- paste(base_dir, n_cells, "c", n_mut, "m", clone, sep = "")
    paths <- c(paths, path)
    clones <- c(clones, clone)
    
  }
}

print(paths)

for (i in seq_along(paths)) {
  n_clones <- clones[i]
  path <- paths[i]
  setwd(script_dir)
  generate.parent.vec(path, n_tests, n_clones)
}

