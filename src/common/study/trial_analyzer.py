# src/common/study/trial_analyzer.py
# src/common/study/trial_analyzer.py
from __future__ import annotations
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, TypedDict, Union, Optional
import optuna
from optuna.trial import TrialState

logger = logging.getLogger(__name__)

class ParamRange(TypedDict):
    min: float
    max: float
    param_type: str  # 'float', 'int', or 'bool'

class TrialAnalyzer:
    """Analyzes trial history and performance patterns"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
    
    def _get_param_type(self, value: Any) -> str:
        """Determine parameter type."""
        if isinstance(value, bool):
            return 'bool'
        elif isinstance(value, int):
            return 'int'
        else:
            return 'float'
    
    def _convert_value(self, value: Any, param_type: str) -> Union[float, int, bool]:
        """Convert value to the correct type."""
        try:
            if param_type == 'bool':
                return bool(value)
            elif param_type == 'int':
                return int(float(value))
            else:
                return float(value)
        except (ValueError, TypeError) as e:
            logger.error(f"Error converting value {value} to type {param_type}: {str(e)}")
            raise
    
    def _compute_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Safely compute correlation between two arrays."""
        try:
            # Remove any NaN or Inf values
            mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
            x_clean = x[mask]
            y_clean = y[mask]
            
            # Check if we have enough valid values
            if len(x_clean) < 2 or len(np.unique(x_clean)) < 2:
                return 0.0
            
            # Compute correlation
            return np.abs(np.corrcoef(x_clean, y_clean)[0, 1])
            
        except Exception as e:
            logger.warning(f"Error computing correlation: {str(e)}")
            return 0.0
    
    def analyze_trials(
        self,
        trials: List[optuna.Trial],
        batch_size: int = 5
    ) -> Dict[str, Any]:
        """Analyze trials in batches to manage memory."""
        logger.info("Starting historical trials analysis")
        
        completed_trials = [t for t in trials if t.state == TrialState.COMPLETE]
        if not completed_trials:
            logger.info("No historical trials to analyze")
            return {}
        
        # Validate trial values
        valid_trials = []
        for trial in completed_trials:
            if trial.value is not None and not (isinstance(trial.value, float) and 
                (trial.value != trial.value or abs(trial.value) == float('inf'))):
                valid_trials.append(trial)
        
        if not valid_trials:
            logger.info("No valid trial values to analyze")
            return {}
        
        prior_data = []
        for i in range(0, len(valid_trials), batch_size):
            batch = valid_trials[i:i + batch_size]
            logger.info(f"Analyzing batch {i//batch_size + 1}/{(len(valid_trials)-1)//batch_size + 1}")
            
            batch_data = []
            for trial in batch:
                duration = (trial.datetime_complete - trial.datetime_start).total_seconds() if trial.datetime_complete and trial.datetime_start else 0
                trial_data = {
                    'params': trial.params,
                    'value': trial.value,
                    'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                    'duration': duration
                }
                batch_data.append(trial_data)
            
            batch_stats = self._compute_batch_statistics(batch_data)
            prior_data.append(batch_stats)
        
        merged_stats = self._merge_batch_statistics(prior_data)
        logger.info("Historical trials analysis completed")
        return merged_stats
    
    def _compute_batch_statistics(
        self,
        batch_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute statistics for a batch of trials."""
        stats = {
            'param_ranges': {},
            'best_params': None,
            'best_value': float('inf'),
            'avg_duration': 0.0,
            'param_correlations': {}
        }
        
        for trial in batch_data:
            # Track parameter ranges
            for param, value in trial['params'].items():
                if param not in stats['param_ranges']:
                    param_type = self._get_param_type(value)
                    stats['param_ranges'][param] = {
                        'min': float('inf'),
                        'max': float('-inf'),
                        'param_type': param_type
                    }
                
                try:
                    value_float = float(value)
                    stats['param_ranges'][param]['min'] = min(
                        stats['param_ranges'][param]['min'],
                        value_float
                    )
                    stats['param_ranges'][param]['max'] = max(
                        stats['param_ranges'][param]['max'],
                        value_float
                    )
                    logger.debug(f"Updated range for {param}: {stats['param_ranges'][param]}")
                except (ValueError, TypeError) as e:
                    logger.error(f"Error processing parameter {param} with value {value}: {str(e)}")
                    raise
            
            # Track best parameters
            if trial['value'] < stats['best_value']:
                stats['best_value'] = trial['value']
                stats['best_params'] = trial['params'].copy()
            
            stats['avg_duration'] += trial['duration']
        
        if batch_data:
            stats['avg_duration'] /= len(batch_data)
            
        return stats
    
    def _merge_batch_statistics(
        self,
        batch_stats: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Merge statistics from multiple batches."""
        merged = {
            'param_ranges': {},
            'best_params': None,
            'best_value': float('inf'),
            'avg_duration': 0.0
        }
        
        total_batches = len(batch_stats)
        if total_batches == 0:
            return merged
            
        for stats in batch_stats:
            # Merge parameter ranges
            for param, ranges in stats['param_ranges'].items():
                if param not in merged['param_ranges']:
                    merged['param_ranges'][param] = ranges.copy()
                else:
                    merged['param_ranges'][param]['min'] = min(
                        merged['param_ranges'][param]['min'],
                        ranges['min']
                    )
                    merged['param_ranges'][param]['max'] = max(
                        merged['param_ranges'][param]['max'],
                        ranges['max']
                    )
                    # Keep the same param_type
                    if merged['param_ranges'][param]['param_type'] != ranges['param_type']:
                        logger.warning(
                            f"Parameter {param} has inconsistent types: "
                            f"{merged['param_ranges'][param]['param_type']} vs {ranges['param_type']}"
                        )
            
            # Track best overall parameters
            if stats['best_value'] < merged['best_value']:
                merged['best_value'] = stats['best_value']
                merged['best_params'] = stats['best_params']
            
            merged['avg_duration'] += stats['avg_duration']
        
        merged['avg_duration'] /= total_batches
        
        return merged
    
    def plot_trial_curves(
        self,
        trials: List[optuna.Trial],
        title: str
    ) -> None:
        """Plot training curves and parameter importance."""
        # Validate trials
        completed_trials = [t for t in trials if t.state == TrialState.COMPLETE]
        if not completed_trials:
            logger.info("No completed trials to plot")
            return
            
        valid_trials = []
        for trial in completed_trials:
            if trial.value is not None and not (isinstance(trial.value, float) and 
                (trial.value != trial.value or abs(trial.value) == float('inf'))):
                valid_trials.append(trial)
        
        if not valid_trials:
            logger.info("No valid trial values to plot")
            return
        
        # Create visualization directory
        vis_dir = self.output_dir / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Find best trial by validation accuracy
        best_trial = max(valid_trials, key=lambda t: t.user_attrs.get('best_val_acc', 0))
        logger.info(f"Best trial {best_trial.number} with val_acc: {best_trial.user_attrs.get('best_val_acc', 0):.4f}")
        
        # Plot metrics from best trial
        metrics = best_trial.user_attrs.get('epoch_metrics', [])
        if metrics:
            try:
                epochs = range(len(metrics))
                plt.figure(figsize=(15, 10))
                
                # Plot metrics that exist
                plot_idx = 1
                for metric_name in ['loss', 'accuracy', 'perplexity']:
                    try:
                        train_values = [m.get('train', {}).get(metric_name) for m in metrics]
                        val_values = [m.get('validation', {}).get(metric_name) for m in metrics]
                        
                        # Only plot if we have valid values
                        if any(v is not None for v in train_values + val_values):
                            plt.subplot(2, 2, plot_idx)
                            if any(v is not None for v in train_values):
                                plt.plot(epochs, train_values, 'b-', label='Train')
                            if any(v is not None for v in val_values):
                                plt.plot(epochs, val_values, 'r-', label='Val')
                            plt.title(metric_name.capitalize(), fontsize=12)
                            plt.xlabel('Epoch')
                            plt.ylabel(metric_name.capitalize())
                            plt.legend()
                            plt.grid(True, alpha=0.3)
                            plot_idx += 1
                    except Exception as e:
                        logger.warning(f"Failed to plot {metric_name}: {str(e)}")
                        continue
                
                plt.suptitle(f"{title} - Best Trial (#{best_trial.number}) Metrics", fontsize=14, y=1.02)
                plt.tight_layout()
                plt.savefig(vis_dir / 'best_trial_metrics.png', bbox_inches='tight')
                plt.close()
            except Exception as e:
                logger.warning(f"Failed to plot trial metrics: {str(e)}")
            
        # Plot loss curve across all trials
        plt.figure(figsize=(12, 6))
        trial_numbers = [t.number for t in valid_trials]
        losses = [t.value for t in valid_trials]
        val_accs = [t.user_attrs.get('best_val_acc', 0) for t in valid_trials]
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot loss
        ax1.plot(trial_numbers, losses, 'b-', marker='o', label='Loss')
        ax1.set_xlabel('Trial Number', fontsize=12)
        ax1.set_ylabel('Loss', color='b', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Plot validation accuracy on secondary y-axis
        ax2 = ax1.twinx()
        ax2.plot(trial_numbers, val_accs, 'r-', marker='s', label='Val Acc')
        ax2.set_ylabel('Validation Accuracy', color='r', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='r')
        
        plt.title(f"{title} - Trial Performance", fontsize=14, pad=20)
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(vis_dir / 'trial_performance.png')
        plt.close()
        
        # Plot parameter importance if we have enough trials
        if len(valid_trials) >= 3:  # Only calculate correlations with at least 3 trials
            param_importance = {}
            for trial in valid_trials:
                for param_name, param_value in trial.params.items():
                    if param_name not in param_importance:
                        param_importance[param_name] = []
                    try:
                        value_float = float(param_value)
                        param_importance[param_name].append((value_float, trial.value))
                    except (ValueError, TypeError):
                        logger.debug(f"Skipping non-numeric parameter {param_name} for importance plot")
            
            plt.figure(figsize=(12, 6))
            importance_values = []
            param_names = []
            
            for param_name, values in param_importance.items():
                if len(values) >= 2:  # Need at least 2 points for correlation
                    values = np.array(values)
                    correlation = self._compute_correlation(values[:, 0], values[:, 1])
                    importance_values.append(correlation)
                    param_names.append(param_name)
            
            if importance_values:  # Only plot if we have valid correlations
                sorted_indices = np.argsort(importance_values)
                plt.barh(np.array(param_names)[sorted_indices], 
                        np.array(importance_values)[sorted_indices])
                plt.title(f"{title} - Parameter Importance", fontsize=14, pad=20)
                plt.xlabel("Importance (Correlation with Loss)", fontsize=12)
                plt.tight_layout()
                plt.savefig(vis_dir / 'parameter_importance.png')
                plt.close()