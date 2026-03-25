from propnetscore.node_selection import NodeSelectionTask
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm_multiply
from propnetscore.utils import standard_basis_vec
from juliacall import Main as jl
import time


class NodeSelectionTaskFast(NodeSelectionTask):
    def __init__(self, adjacency_matrix: np.ndarray, node_probs: np.ndarray, true_node:int):
        super().__init__(adjacency_matrix, node_probs, true_node)
        self.sparse_adjacency_matrix = None # compute only when necessary
        self.sparse_laplacian = None # compute only when necessary

    def resistance_score_fast(self):
        """
        Proper scoring rule based on resistance distance, but with a fast julia implementation of the laplacian inverse.
        """
        if self.sparse_adjacency_matrix is None:
            start = time.time()
            self.sparse_adjacency_matrix = csc_matrix(self.adjacency_matrix)
            print(f" --- Converted adjacency matrix to sparse format in {time.time() - start:.3f}s")

        # load julia packages
        start = time.time()
        jl.seval("using Laplacians")
        jl.seval("using LinearAlgebra")
        jl.seval("using SparseArrays")
        print(f" --- Loaded Julia packages in {time.time() - start:.3f}s")

        # transport sparse adjacency matrix to julia
        start = time.time()
        n = self.sparse_adjacency_matrix.shape[0]
        indptr_py = (self.sparse_adjacency_matrix.indptr + 1).astype(np.int64)  # julia uses 1-based indexing
        indices_py = (self.sparse_adjacency_matrix.indices + 1).astype(np.int64)  # julia uses 1-based indexing
        data_py = self.sparse_adjacency_matrix.data.astype(np.float64)
        indptr_jl = jl.Vector(indptr_py)  # convert to julia vector
        indices_jl = jl.Vector(indices_py)  # convert to julia vector
        data_jl = jl.Vector(data_py)  # convert to julia vector
        adjacency_jl = jl.SparseMatrixCSC(n, n, indptr_jl, indices_jl, data_jl)
        print(f" --- Transferred sparse adjacency matrix to Julia in {time.time() - start:.3f}s")

        # initialize laplacian solver and compute its action on vector
        start = time.time()
        lap_solver = jl.approxchol_lap(adjacency_jl)
        vector = standard_basis_vec(size=n, i=self.true_node) - self.node_probs # e_i - p
        score = vector @ lap_solver(vector) # resistance score is (e_i - p)^T L+ (e_i - p), convention = "positive"
        # NOTE: the corresponding kernel is L+ii + L+jj - 2L+ij, but the first two terms vanish when multiplied because
        # the entries of (e_i - p) sum to zero! (and standard factor -1/2 cancels with the -2 in the kernel definition)
        print(f" --- Computed resistance score in Julia in {time.time() - start:.3f}s")
        return float(score)

    def diffusion_score_fast(self, kappa2: float, convention: str | None = None):
        """
        Proper scoring rule based on the diffusion kernel but with a fast implementation based on expm_multiply
        (only matrix-vector products instead of full matrix exponentiation).

        Parameters
        ----------
        kappa2 : float (non-negative)
            Inverse length scale
        convention : str
            Kernel score convention (None or "positive")
        """
        if self.laplacian is None:
            self.laplacian = self._get_laplacian()
        if self.sparse_laplacian is None:
            self.sparse_laplacian = csc_matrix(self.laplacian)

        vector = standard_basis_vec(size=len(self.node_probs), i=self.true_node) - self.node_probs # e_i - p
        # the score with convention = "positive" is 0.5 * (e_i - p)^T H (e_i - p) where H = exp(-0.5 * kappa2 * L)
        # and expm_multiply computes e(-0.5 * kappa2 * L) @ vector efficiently without forming the full matrix
        score = 0.5 * vector @ expm_multiply(-0.5 * kappa2 * self.sparse_laplacian, vector)
        return float(score)

