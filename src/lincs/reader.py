'''
Memory-efficient GCTX file reader for large LINCS L1000 datasets.

Examples
--------
Basic usage:
>>> from gctx_reader import devLINCSGCTXReader
>>> 
>>> # Load large GCTX file
>>> with devLINCSGCTXReader('GSE92742_Broad_LINCS_Level5_COMPZ_n476251x12328.gctx') as reader:
...     # Extract reference signature
...     ref_sig = reader.extract_knockdown_consensus('PRKCA', cell_lines=[b'A375'])
...     
...     # Compute concordance
...     results = reader.compute_signature_concordance(
...         reference_signature=ref_sig,
...         threshold=0.7,
...         group_by=['inchi_key', 'cell']
...     )
...     
...     # Results contain BAI scores and credible intervals
...     print(results[['inchi_key', 'bai_score', 'lower_credible', 'upper_credible']].head())
'''

import h5py
import numpy as np
import pandas as pd
from typing import List, Literal, Optional, Tuple, Dict, Any
import gc
from numba import njit, prange
from concurrent.futures import as_completed, ThreadPoolExecutor
from tqdm import tqdm
import os
import logging
from scipy.stats import beta as beta_dist

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Statistical constants
CREDIBLE_INTERVAL_COVERAGE = 0.95
EPSILON_VARIANCE = 1e-8
EPSILON_DIVISION = 1e-12
DEFAULT_AGREEMENT_SCALE = 0.5


def _process_chunk_with_grouping(
        data_chunk: np.ndarray,
        meta_chunk: pd.DataFrame,
        reference_signature: np.ndarray,
        group_by: List[str],
        threshold: float,
        n_workers: Optional[int] = None,
    ) -> pd.DataFrame:
    """
    Process a chunk by grouping replicates and computing CCC/BAI in parallel (threads).
    Does NOT pre-aggregate - passes raw replicates to CCC function.

    Args:
        data_chunk: Array of shape (n_samples, n_genes)
        meta_chunk: Metadata for samples in chunk
        reference_signature: Reference gene signature (1D array, shape: n_genes)
        group_by: Columns to group by
        threshold: Minimum CCC/BAI threshold
        n_workers: Max worker threads (default: CPUs-1)

    Returns:
        DataFrame with group metadata and CCC/BAI scores
    """
    # Defensive copies are not necessary; threads read shared arrays.
    meta_chunk = meta_chunk.reset_index(drop=True)

    grouped = meta_chunk.groupby(group_by, dropna=False)
    groups: Dict[Any, pd.Index] = grouped.groups  # key -> row index positions
    if not groups:
        return pd.DataFrame()

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    def _work(group_key, group_indices: pd.Index):
        # Get raw replicate data for this group (NOT aggregated)
        idx_list = group_indices.tolist()
        group_replicates = data_chunk[idx_list, :].astype(np.float32, copy=False)

        bai_score, lower_credible, upper_credible, mean_agreement = _compute_bai_with_credibility(
            sample_replicates=group_replicates,
            reference_signature=reference_signature.astype(np.float32, copy=False),
        )

        if bai_score < threshold:
            return None  # filtered

        # Use first replicate’s metadata as representative
        first_row = meta_chunk.iloc[idx_list[0]].to_dict()
        first_row['n_replicates'] = len(idx_list)
        # Collect replicate ids if available
        if 'id' in meta_chunk.columns:
            first_row['replicate_ids'] = ','.join(meta_chunk.iloc[idx_list]['id'].astype(str).tolist())
        else:
            first_row['replicate_ids'] = ','.join(map(str, idx_list))

        first_row['bai_score'] = bai_score
        first_row['lower_credible'] = lower_credible
        first_row['upper_credible'] = upper_credible
        first_row['mean_agreement'] = mean_agreement
        return first_row

    results: List[dict] = []

    # Submit all groups (safe for thousands; for 100k+, see bounded variant below)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_work, gk, gi) for gk, gi in groups.items()]
        for fut in as_completed(futures):
            try:
                out = fut.result()
                if out is not None:
                    results.append(out)
            except Exception as e:
                print(f"Error processing group: {e}")
                continue

    return pd.DataFrame(results) if results else pd.DataFrame()


@njit(fastmath=True, parallel=True)
def _compute_bai_with_credibility(
        sample_replicates: np.ndarray, 
        reference_signature: np.ndarray,
        agreement_scale: float = 0.5,
        prior_strength: float = 1.0
    ) -> tuple:
    """
    Compute Bayesian Agreement Index (BAI) with credible intervals.
    
    Uses a Beta-Bernoulli model where each gene's agreement is modeled
    as a Bernoulli trial with probability weighted by standardized
    difference between sample and reference.
    
    Parameters
    ----------
    sample_replicates : np.ndarray, shape (n_replicates, n_genes)
        Gene expression matrix for sample replicates
    reference_signature : np.ndarray, shape (n_replicates, n_genes)
        Reference gene expression signature
    agreement_scale : float, default=0.5
        Controls sensitivity of agreement probability to z-scores.
        Smaller values = more stringent agreement criterion.
    prior_strength : float, default=1.0
        Beta prior pseudocount (α₀ = β₀ = prior_strength).
        Higher values = more conservative estimates.
    
    Returns
    -------
    bai_score : float
        Bayesian Agreement Index (posterior mean), range [0, 1]
    lower_credible : float
        Lower bound of 95% credible interval
    upper_credible : float
        Upper bound of 95% credible interval
    mean_agreement : float
        Mean gene-wise agreement probability
    
    Notes
    -----
    The BAI is computed as:
    1. For each gene, compute standardized difference z-score
    2. Convert to agreement probability: p = exp(-z²/2σ²)
    3. Update Beta posterior: α = α₀ + Σp, β = β₀ + n_genes - Σp
    4. BAI = α/(α+β)
    
    References
    ----------
    .. [1] Your paper or method reference here
    
    Examples
    --------
    >>> replicates = np.random.randn(3, 978)  # 3 replicates, 978 genes
    >>> reference = np.random.randn(978)
    >>> bai, lower, upper, mean_agr = _compute_bai_with_credibility(replicates, reference)
    >>> print(f"BAI: {bai:.3f} [{lower:.3f}, {upper:.3f}]")
    """


    n_replicates, n_genes = sample_replicates.shape
    if n_replicates == 0 or n_genes == 0:
        return (0.0, 0.0, 0.0, 0.0)

    if agreement_scale <= 0.0:
        agreement_scale = 1e-6

    # Means
    sample_means = np.zeros(n_genes, dtype=np.float32)
    reference_means = np.zeros(n_genes, dtype=np.float32)
    for j in prange(n_genes):
        acc_s = 0.0
        acc_r = 0.0
        for i in range(n_replicates):
            acc_s += sample_replicates[i, j]
            acc_r += reference_signature[i, j]
        sample_means[j] = acc_s / n_replicates
        reference_means[j] = acc_r / n_replicates

    # Stds
    denom = n_replicates - 1
    if denom < 1:
        denom = 1

    sample_stds = np.zeros(n_genes, dtype=np.float32)
    reference_stds = np.zeros(n_genes, dtype=np.float32)
    for j in prange(n_genes):
        var_s = 0.0
        var_r = 0.0
        for i in range(n_replicates):
            ds = sample_replicates[i, j] - sample_means[j]
            dr = reference_signature[i, j] - reference_means[j]
            var_s += ds * ds
            var_r += dr * dr
        sample_stds[j] = np.sqrt(var_s / denom)
        reference_stds[j] = np.sqrt(var_r / denom)

    # Agreement probs
    agreement_probs = np.zeros(n_genes, dtype=np.float32)
    inv_scale2 = 1.0 / (agreement_scale * agreement_scale * 2.0)
    for j in prange(n_genes):
        diff = sample_means[j] - reference_means[j]
        pooled_std = np.sqrt(sample_stds[j]*sample_stds[j] + reference_stds[j]*reference_stds[j] + 1e-8)
        z = abs(diff) / pooled_std
        agreement_probs[j] = np.exp(- (z * z) * inv_scale2)

    # Posterior
    sum_agreements = np.sum(agreement_probs)
    alpha_post = prior_strength + sum_agreements
    beta_post  = prior_strength + (n_genes - sum_agreements)
    denom_post = alpha_post + beta_post
    if denom_post < 1e-12:
        denom_post = 1e-12

    bai_score = alpha_post / denom_post
    post_var = (alpha_post * beta_post) / (denom_post**2 * (denom_post + 1.0))
    post_std = np.sqrt(post_var)

    Z95 = 1.96
    lower_credible = max(0.0, bai_score - Z95 * post_std)
    upper_credible = min(1.0, bai_score + Z95 * post_std)
    mean_agreement = np.mean(agreement_probs)

    return (bai_score, lower_credible, upper_credible, mean_agreement)



class LINCSGCTXReader:
    """
    Memory-efficient reader for large GCTX files.
    Uses lazy loading and chunked processing.
    """

    def __init__(self, filepath: str, debug: bool = False):
        """
        Initialize reader. File stays open but data not loaded.
        
        Args:
            filepath: Path to .gctx file
        """
        self.filepath = filepath
        self.file = h5py.File(filepath, 'r')
        self.data = self.file['0']['DATA']['0']['matrix'] # type: ignore
        self.debug = debug

        # Load metadata (small, can fit in memory)
        self.row_meta = self._load_metadata('row')  # Genes
        self.col_meta = self._load_metadata('col')  # Samples/perturbations

        print(f"Loaded GCTX file: {filepath}") 
        print(f"Shape: {self.data.shape} ({self.n_genes} genes × {self.n_samples} samples)") # type: ignore
        print(f"Data type: {self.data.dtype}") # type: ignore
        print(f"Estimated size: {self.data.nbytes / 1e9:.2f} GB") # type: ignore


    @property
    def n_genes(self) -> int:
        return self.data.shape[1] # type: ignore
    
    @property
    def n_samples(self) -> int:
        return self.data.shape[0] # type: ignore
    
    def _load_metadata(self, axis: Literal['row', 'col']) -> pd.DataFrame:
        """Load row or column metadata."""
        meta_path = f'0/META/{axis.upper()}'
        meta_dict = {}
        
        for key in self.file[meta_path].keys():         # type: ignore
            values = self.file[meta_path][key][:]       # type: ignore
            # Decode bytes to strings if necessary
            if values.dtype.kind == 'S':                # type: ignore
                values = np.array(
                    [v.decode('utf-8') if isinstance(v, bytes) else v for v in values] # type: ignore
                )
            meta_dict[key] = values
        
        df = pd.DataFrame(meta_dict)
        return df
    
    def get_sample_by_index(self, indices: List[int]) -> np.ndarray:
        """
        Get specific samples by index (memory efficient).
        
        Args:
            indices: List of column indices to retrieve
            
        Returns:
            Array of shape (n_genes, len(indices))
        """
        # Read only specific columns from disk
        return self.data[indices, :] # type: ignore
    
    def get_sample_by_id(self, sample_ids: List[str]) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Get samples by perturbagen ID.
        
        Args:
            sample_ids: List of sample IDs (e.g., compound names, shRNA IDs)
            
        Returns:
            Tuple of (data array, metadata for selected samples)
        """
        # Find indices matching IDs
        mask = self.col_meta['id'].isin(sample_ids)
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            print(f"Warning: No samples found for IDs: {sample_ids}") if self.debug else None
            return np.array([]), pd.DataFrame()

        print(f"Found {len(indices)} samples matching {len(sample_ids)} IDs") if self.debug else None

        # Read only those columns
        data = self.get_sample_by_index(indices.tolist())
        metadata = self.col_meta.iloc[indices].copy()
        
        return data, metadata
    
    def search_perturbagens(self, query: str | List[str], field: str = 'pert_iname') -> pd.DataFrame:
        """
        Search for perturbagens (compounds/shRNAs) by name/target.
        
        Args:
            query: Search string (case-insensitive, partial match)
            field: Metadata field to search ('pert_iname', 'pert_id', 'gene_target', etc.)
            
        Returns:
            DataFrame of matching metadata
        """
        if field not in self.col_meta.columns:
            print(f"Available fields: {self.col_meta.columns.tolist()}") if self.debug else None
            return pd.DataFrame()
        
        # Mask if query is a string
        if isinstance(query, list):
            mask = self.col_meta[field].isin(query)
        else:
            mask = self.col_meta[field].astype(str).str.contains(query, case=False, na=False)
        
        # Return matching metadata
        results = self.col_meta[mask].copy()
        print(f"Found {len(results)} samples matching '{query}' in {field}") if self.debug else None
        return results
    

    def get_unique_perturbagens(self, field: str = 'pertname') -> List[str]:
        """
        Get unique perturbagen names/IDs from metadata.
        
        Args:
            field: Metadata field to extract unique values from
                - if using cpDataset, valid fields include:
                    - 'pertname': Perturbagen name
                    - 'smiles': SMILES string
                    - 'inchi_key': InChI Key
                    - 'dose': Dose level
                    - 'cell': Cell line
                    - 'timepoint': Timepoint
                    - 'id': Unique sample ID, related to plate and well location
                - if using shRNA dataset, valid fields include:
                    - 'pertname': Name of gene shRNA targets
                    - 'cell': Cell line
                    - 'timepoint': Timepoint
                    - 'id': Unique sample ID, related to plate and well location

        Returns:
            Series of unique perturbagen names/IDs
        """
        if field not in self.col_meta.columns:
            print(f"Available fields: {self.col_meta.columns.tolist()}") if self.debug else None
            return []
        
        unique_vals = self.col_meta[field].dropna().unique().tolist()
        print(f"Found {len(unique_vals)} unique values in field '{field}'") if self.debug else None
        return unique_vals
    
    def get_gene_signature(self, sample_ids: List[str], 
                          top_n: int = 978) -> pd.DataFrame:
        """
        Get L1000 landmark gene signature for specific samples.
        
        Args:
            sample_ids: List of sample IDs
            top_n: Number of top differential genes (default 978 landmarks)
            
        Returns:
            DataFrame with genes as index, samples as columns
        """
        data, metadata = self.get_sample_by_id(sample_ids)
        
        if data.size == 0:
            return pd.DataFrame()

        # Create DataFrame
        sig_df = pd.DataFrame(
            data,
            index=metadata['id'].values,
            columns=self.row_meta['id'].values
        )
        
        return sig_df
    
    def iterate_chunks(self, chunk_size: int = 1000):
        """
        Iterator for processing file in chunks (memory efficient).
        
        Args:
            chunk_size: Number of samples per chunk
            
        Yields:
            Tuple of (data_chunk, metadata_chunk)
        """
        n_chunks = int(np.ceil(self.n_samples / chunk_size))
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, self.n_samples)
            
            # Read chunk from disk
            data_chunk = self.data[start_idx:end_idx, :]                # type: ignore
            meta_chunk = self.col_meta.iloc[start_idx:end_idx].copy()
            
            print(f"Processing chunk {i+1}/{n_chunks} "
                  f"(samples {start_idx}-{end_idx})") if self.debug else None

            yield data_chunk, meta_chunk
            
            # Explicit garbage collection
            del data_chunk
            gc.collect()
    
    def compute_signature_concordance(
            self, 
            reference_signature: np.ndarray,
            threshold: float = 0.5,
            chunk_size: int = 200,
            group_by: List[str] = ['inchi_key', 'dose', 'cell', 'timepoint'],
            cell_lines: Optional[List[bytes]] = None,
            timepoints: Optional[List[bytes]] = None,
            field: str = 'inchi_key',
            n_workers: Optional[int] = None
        ) -> pd.DataFrame:
        """
        Compute concordance scores between reference and all samples.
        Processes in chunks with parallel processing.
        
        Args:
            reference_signature: 1D array of gene expression values
            chunk_size: Samples per chunk
            threshold: Minimum score to consider concordant
            group_by: Columns to group by for replicate aggregation
            cell_lines: Optional filter for cell lines
            timepoints: Optional filter for timepoints
            field: Metadata field to index results by
                - if using cpDataset, use `'inchi_key'` for compounds
                - if using shRNA dataset, use `'pertname'` for gene names
            n_workers: Number of parallel workers (None = use all CPUs)
            
        Returns:
            DataFrame of concordance scores with metadata
        """

        sig_concordance_df = []

        # Find unique perturbagen IDs to process
        unique_data_ids = self.get_unique_perturbagens(field)
        # Iterate through unique_data_ids in chunks
        for i in tqdm(
            range(0, len(unique_data_ids), chunk_size),
            total=int(np.ceil(len(unique_data_ids) / chunk_size)),
            desc="Processing chunks"
        ):
            # Extract chunk data and metadata for these perturbagen IDs
            chunk_ids = unique_data_ids[i:i + chunk_size]
            data_chunk, meta_chunk = self._extract_bulk_condition_data_and_metadata(
                chunk_ids,
                cell_lines=cell_lines,
                timepoints=timepoints,
                field=field
            )

            # Skip empty chunks
            if data_chunk.size == 0:
                print(f"No data found for chunk {unique_data_ids[i:i + chunk_size]}") if self.debug else None
                continue

            # Process chunk with grouping and concordance computation, multi-threaded subtasks
            df = _process_chunk_with_grouping(
                data_chunk=data_chunk,
                meta_chunk=meta_chunk,
                reference_signature=reference_signature,  # safe: shared read-only
                group_by=group_by,
                threshold=threshold
            )

            if not df.empty:
                sig_concordance_df.append(df)

        return pd.concat(sig_concordance_df, ignore_index=True) if sig_concordance_df else pd.DataFrame()


    def _extract_bulk_condition_data_and_metadata(
            self,
            gene_symbol: str | List[str],
            cell_lines: Optional[List[bytes]] = None,
            timepoints: Optional[List[bytes]] = None,
            field: str = "pertname"
        ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Extract bulk data and metadata for a gene knockdown condition.
        Args:
            gene_symbol: Target gene symbol (e.g., 'PRKCA' for PKC-alpha)
            cell_lines: List of cell lines to include (None = all)
            timepoints: List of timepoints to include (None = all)
            field: Metadata field to search ('pertname', 'pert_id', etc.)
        Returns:
            Tuple of (data array, metadata DataFrame)
        """

        # Search for shRNA targeting this gene
        results = self.search_perturbagens(gene_symbol, field=field)
        
        if len(results) == 0:
            print(f"No knockdown data found for {gene_symbol}") if self.debug else None
            return np.array([]), pd.DataFrame()
        
        # Filter by cell line if specified
        if cell_lines:
            results = results[results['cell'].isin(cell_lines)]
        
        # Filter by timepoint if specified
        if timepoints:
            results = results[results['timepoint'].isin(timepoints)]

        # Check if any results remain
        if len(results) == 0:
            print(f"No knockdown data found for {gene_symbol} in specified cell lines/timepoints") if self.debug else None
            return np.array([]), pd.DataFrame()
        
        if self.debug:
            print(f"Found {len(results)} knockdown samples for {gene_symbol}")
            print(f"Cell lines: {results['cell'].unique()}")
            print(f"Timepoints: {results['timepoint'].unique()}")
            print(f"Perturbagens: {results['pertname'].unique()}")
            print(f"Dose levels: {results['dose'].unique()}")
        
        # Get data
        sample_ids = results['id'].tolist()
        data, _ = self.get_sample_by_id(sample_ids)

        return data, results
    

    def extract_knockdown_consensus(self, 
                                   gene_symbol: str,
                                   cell_lines: Optional[List[bytes]] = None,
                                   timepoints: Optional[List[bytes]] = None,
                                   field: str = "pertname",
                                   consensus_type: Literal["median", "mean", None] = "median") -> np.ndarray:
        """
        Extract consensus knockdown signature for a gene (for shRNA data).
        
        Args:
            gene_symbol: Target gene symbol (e.g., 'PRKCA' for PKC-alpha)
            cell_lines: List of cell lines to include (None = all)
            
        Returns:
            Consensus signature (median across replicates/cell lines)
        """
        # Search for shRNA targeting this gene
        data, _ = self._extract_bulk_condition_data_and_metadata(
            gene_symbol, 
            cell_lines=cell_lines,
            timepoints=timepoints,
            field=field
        )

        # Compute consensus
        if consensus_type is None:
            print("Returning individual signatures without consensus.") if self.debug else None
            return data
        
        # Compute consensus (median) row-wise
        consensus = np.median(data, axis=0) if consensus_type == "median" else np.mean(data, axis=0)
        
        return consensus
    
    def close(self):
        """Close the HDF5 file."""
        self.file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
