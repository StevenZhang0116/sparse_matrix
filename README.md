# sparse_matrix

We provide a paradigm for conducting pairwise row operations (e.g., Pearson correlation) for large sparse matrices, such as connectome data or social network analysis. The classical `np.corrcoef()` function does not support the `csr_matrix` datatype, and transforming back to a dense matrix can be time-consuming or infeasible due to RAM limitations for large data. Our approach addresses this problem effectively by performing all operations on the sparse matrix directly. Users may also consider `dask` as a parallel computing add-on and/or alternative. We also provide an external datafile `microns_allW.npz` (90300×90300, density~0.001) for test purpose. 

Through tests in generating random sparse matrices, it is clear that our sparse correlation consistently has better time performance than the `numpy` counterpart for larger matrices. Still, it is recommended to truncate the matrix into smaller blocks (e.g., 100×N) and calculate each separately, either synchronously or asynchronously. It is recommanded to use `np.memmap()` to create memory-map for data storage. 

<img src="./images/side1_test.png" alt="Test comparison in varying dimensions" width="300"/>

![Test comparison in varying densities](./images/density_test.png)

