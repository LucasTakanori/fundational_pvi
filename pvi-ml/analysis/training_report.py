# src/analysis/training_report.py

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import json


@dataclass
class SubjectReport:
    """Container for single subject/dataset analysis results"""
    dataset_name: str
    num_train: Optional[int] = None
    num_test: Optional[int] = None
    epoch: Optional[int] = None

    # Raw data
    results: Optional[pd.DataFrame] = None
    history: Optional[pd.DataFrame] = None
    configs: Optional[dict] = None
    stats: Optional[pd.DataFrame] = None

    metrics: Optional[pd.DataFrame] = None

class TrainingReport:
    def __init__(self,
                 log_dir: str | Path,
                 tag: str = 'main') -> None:

        self._alias = type(self).__name__

        # Set up directory structure
        if tag == 'main':
            self.results_dir = log_dir / 'results'
            pass
        elif tag == 'holdout':
            self.results_dir = log_dir / 'results' / '_holdout'
            pass
        else:
            raise ValueError(f"Invalid tag '{tag}'. Expected 'main' or 'holdout'")

        self.history_dir = log_dir / 'history'
        self.configs_dir = log_dir / 'configs'
        self.stats_dir = log_dir / 'statistics'  # optional

        self.results_files = list(self.results_dir.glob('*_results.csv'))
        self.history_files = list(self.history_dir.glob('*_history.csv'))
        self.configs_files = list(self.configs_dir.glob('*_configs_*.json'))
        self.stats_files = list(self.stats_dir.glob('*_statistics.json'))

        self.reports = [
        SubjectReport(dataset_name=name) for name in self.dataset_names
        ]

        self.results: Optional[pd.DataFrame] = None
        self.history: Optional[pd.DataFrame] = None
        self.stats_: Optional[pd.DataFrame] = None
        self.metrics: Optional[pd.DataFrame] = None
        self.metrics_combined: Optional[pd.DataFrame] = None

        print(f"{self._alias}: Initialized for session '{self.session}'")
        print(f"\t Found {len(self.results_files)} results files")
        print(f"\t Found {len(self.history_files)} history files")
        print(f"\t Found {len(self.configs_files)} configs files")

    def read_results(self) -> None:
        """Load all results CSV files (predictions + targets)"""

        for idx, file_path in enumerate(self.results_files):
            # df = pd.read_csv(file_path)
            # self.reports[idx].results = df
            pass

    def read_history(self) -> None:
        """Load all history CSV files (training curves)"""

        for idx, file_path in enumerate(self.history_files):
            # df = pd.read_csv(file_path)

            # Adjust epoch to 1-indexed if needed (your MATLAB does epoch+1)
            # df['epoch'] = df['epoch'] + 1

            # self.reports[idx].history = df
            # self.reports[idx].epoch = len(df)  # final epoch count
            pass

    def read_configs(self) -> None:
        """Load all config JSON files"""

        for idx, file_path in enumerate(self.configs_files):
            # with open(file_path, 'r') as f:
            #     configs = json.load(f)

            # Extract train/test counts
            # counts = configs['dataset']['counts']
            # Clean comma separators if present: "1,234" -> 1234
            # self.reports[idx].num_train = int(counts['num_train'].replace(',', ''))
            # self.reports[idx].num_test = int(counts['num_test'].replace(',', ''))
            # self.reports[idx].configs = configs
            pass

    def read_stats(self) -> None:
        """Load statistics JSON files (optional)"""

        if not self.stats_files:
            print(f"{self._alias}: No statistics files found, skipping...")
            return

        for idx, file_path in enumerate(self.stats_files):
            # with open(file_path, 'r') as f:
            #     stats_dict = json.load(f)

            # Adjust epoch if needed
            # stats_dict['epoch'] = stats_dict.get('epoch', 0) + 1

            # Convert to DataFrame
            # stats_df = pd.DataFrame([stats_dict])
            # self.reports[idx].stats = stats_df
            pass

    def process_reports(self) -> None:
        """Read all artifact files for this session"""

        print(f"{self._alias}: Processing reports...")
        # t1 = time.time()

        # self.read_results()
        # self.read_history()
        # self.read_configs()
        # self.read_stats()

        # dt = time.time() - t1
        # print(f"\t Done! ({dt:.2f} seconds)")

    def stack_results(self) -> None:
        """Concatenate all results DataFrames"""

        # Collect all results DataFrames
        # results_list = [rp.results for rp in self.reports if rp.results is not None]

        # Concatenate vertically
        # self.results_stacked = pd.concat(results_list, axis=0, ignore_index=True)

        print(f"{self._alias}: Stacked {len(self.results_stacked)} samples from {len(self.reports)} subjects")

    def stack_metrics(self) -> None:
        """Concatenate metrics from all subjects into single DataFrame"""

        # Build DataFrame with columns: [num_train, num_test, epoch, ...metrics...]
        # Index = dataset names

        # rows = []
        # for rp in self.reports:
        #     if rp.metrics is None:
        #         continue
        #     
        #     row = {
        #         'num_train': rp.num_train,
        #         'num_test': rp.num_test,
        #         'epoch': rp.epoch,
        #     }
        #     # Merge with computed metrics (will handle later)
        #     # row.update(rp.metrics)
        #     rows.append(row)

        # self.metrics_stacked = pd.DataFrame(rows, index=self.dataset_names)

    def stack_statistics(self) -> None:
        """Concatenate statistics from all subjects"""

        # stats_list = [rp.stats for rp in self.reports if rp.stats is not None]

        # if not stats_list:
        #     return

        # self.stats_stacked = pd.concat(stats_list, axis=0, ignore_index=False)
        # self.stats_stacked.index = self.dataset_names

    def stack_reports(self) -> None:
        """Stack all data (results, metrics, statistics)"""

        # self.stack_results()
        # self.stack_metrics()
        # self.stack_statistics()

    def aggregate_reports(self) -> None:
        """
        Compute session-level aggregated metrics.
        Two strategies:
            1. 'aggregated': Metrics computed on all stacked results
            2. 'weighted': Subject metrics weighted by test set size
        """

        print(f"{self._alias}: Aggregating reports...")

        # Strategy 1: Compute metrics directly on stacked results
        # aggregated_metrics = compute_ml_metrics(self.results_stacked)
        # (We'll implement compute_ml_metrics later)

        # Strategy 2: Weighted average of subject-level metrics
        # weights = self.metrics_stacked['num_test'].values
        # metric_cols = [col for col in self.metrics_stacked.columns 
        #                if col not in ['num_train', 'num_test', 'epoch']]
        # weighted_metrics = {}
        # for col in metric_cols:
        #     values = self.metrics_stacked[col].values
        #     weighted_metrics[col] = np.average(values, weights=weights)

        # Combine both strategies into single DataFrame
        # self.metrics_combined = pd.DataFrame([
        #     aggregated_metrics,
        #     weighted_metrics
        # ], index=['aggregated', 'weighted'])

        # Add summary columns (total train/test samples, max epoch)
        # self.metrics_combined.insert(0, 'num_train', self.metrics_stacked['num_train'].sum())
        # self.metrics_combined.insert(1, 'num_test', self.metrics_stacked['num_test'].sum())
        # self.metrics_combined.insert(2, 'epoch', self.metrics_stacked['epoch'].max())

    def export_excel(self,
                     output_path: Optional[Path] = None,
                     include_stacked: bool = True) -> None:
        """
        Export analysis results to Excel with multiple sheets.

        Args:
            output_path: Where to save. If None, auto-generate in log_dir
            include_stacked: Whether to include individual subject results
        """

        if output_path is None:
            # Auto-generate filename: tbl_results_{tag}_{session_id}.xlsx
            # output_path = self.log_dir / f"tbl_results_{self.tag}_{self.id}.xlsx"
            pass

        # with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        #     # Sheet 1: Combined metrics (aggregated + weighted)
        #     self.metrics_combined.to_excel(writer, sheet_name='combined')
        #     
        #     # Sheet 2: Individual subject metrics
        #     if include_stacked:
        #         self.metrics_stacked.to_excel(writer, sheet_name='subjects')
        #     
        #     # Sheet 3: Statistics (if available)
        #     if self.stats_stacked is not None:
        #         self.stats_stacked.to_excel(writer, sheet_name='statistics')

        print(f"{self._alias}: Exported results to: {output_path}")

    @property
    def session(self) -> str:
        """Full session name from directory"""
        return self._session_name

    @property
    def id(self) -> str:
        """Short session ID (e.g., 's01')"""
        return self._session_id


# Example usage
if __name__ == '__main__':
    from pathlib import Path

    # Single session analysis
    log_dir = Path("artifacts/_final_ss/s01-linear-img-to-fiducials-20250124")

    report = TrainingReport(log_dir, tag='main')
    report.process_reports()

    # Later, after implementing metrics:
    # report.stack_reports()
    # report.aggregate_reports()
    # report.export_excel()