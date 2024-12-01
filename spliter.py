# noqa
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, TimeSeriesSplit
from abc import ABC, abstractmethod
import polars as pl

class SplitStrategy(ABC):
    """Base class for split strategies"""
    def __init__(self, test_ratio: float = 0.2):
        self.test_ratio = test_ratio
    
    @abstractmethod
    def split(self, date_range: Tuple[int, int]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Split remaining data into train/val sets for cross validation
        Returns:
            List of (train, val) date range tuples
        """
        pass
    
    def get_holdout_test(self, partition_date_ranges: Dict[int, Tuple[int, int]]) -> Tuple[int, int]:
        """Calculate holdout test date range based on partition date ranges
        Args:
            partition_date_ranges: Dict[int, Tuple[int, int]]: {partition_id: (min_date, max_date)}
        Returns:
            Tuple[Tuple[int, int], Tuple[int, int]]: ((train_start_date, train_end_date), (test_start_date, test_end_date))
        """
        # Get all unique dates from partition ranges
        all_dates = set() # noqa
        for _, (min_date, max_date) in partition_date_ranges.items():
            all_dates.update(range(min_date, max_date + 1))
        
        unique_dates = sorted(list(all_dates))
        split_idx = int(len(unique_dates) * (1 - self.test_ratio))
        split_date = unique_dates[split_idx]
        
        print("\nHoldout Test Split Info:")
        print(f"Total unique dates: {len(unique_dates)}")
        print(f"Train dates range: {unique_dates[0]} - {split_date}")
        print(f"Test dates range: {unique_dates[split_idx + 1]} - {unique_dates[-1]}")
        
        return (unique_dates[0], split_date), (unique_dates[split_idx + 1], unique_dates[-1])
    
    def visualize_splits(self, date_range: Tuple[int, int], figsize: Tuple[int, int] = (15, 8)):
        """Visualize the splits
        Args:
            date_range: Tuple[int, int]: (start_date, end_date)
            figsize: Tuple[int, int]: Figure size for matplotlib
        """
        splits = self.split(date_range)
        n_splits = len(splits)
        
        plt.figure(figsize=figsize)
        
        # Plot splits
        for idx, ((train_start, train_end), (val_start, val_end)) in enumerate(splits):
            # Plot training period
            plt.barh(y=idx, 
                    width=train_end - train_start, 
                    left=train_start, 
                    height=0.3, 
                    color='royalblue', 
                    alpha=0.6,
                    label='Train' if idx == 0 else "")
            
            # Plot validation period
            plt.barh(y=idx, 
                    width=val_end - val_start, 
                    left=val_start, 
                    height=0.3, 
                    color='coral', 
                    alpha=0.6,
                    label='Validation' if idx == 0 else "")
            
            # If gap exists (for PurgedGroupTimeSeriesSplit)
            if hasattr(self, 'group_gap') and self.group_gap > 0:
                plt.barh(y=idx, 
                        width=val_start - train_end, 
                        left=train_end, 
                        height=0.3, 
                        color='lightgray', 
                        alpha=0.3,
                        label='Gap' if idx == 0 else "")
        
        # Customize plot
        plt.yticks(range(n_splits), [f'Split {i}' for i in range(n_splits)])
        plt.xlabel('Date ID')
        plt.ylabel('CV Iteration')
        plt.title(f'{self.__class__.__name__} - {n_splits} Splits')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True, alpha=0.3)
        
        # Add date range info
        plt.text(0.98, -0.15, 
                f'Date Range: {date_range[0]} to {date_range[1]}',
                horizontalalignment='right',
                transform=plt.gca().transAxes,
                fontsize=10,
                alpha=0.7)
        
        plt.tight_layout()
        plt.show()
    

class TimeBasedSplit(SplitStrategy):
    def __init__(self, train_ratio: float = 0.75, test_ratio: float = 0.2):
        super().__init__(test_ratio)
        self.train_ratio = train_ratio
    
    def split(self, date_range: Tuple[int, int]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Single split based on date range
        Args:
            date_range: Tuple[int, int]: (start_date, end_date)
        Returns:
            List[Tuple[Tuple[int, int], Tuple[int, int]]]: [((train_start_date, train_end_date), (val_start_date, val_end_date))]
        """
        start_date, end_date = date_range
        total_days = end_date - start_date + 1
        split_idx = start_date + int(total_days * self.train_ratio) - 1
        
        print("\nTime Based Split Info:")
        print(f"Train dates range: {start_date} - {split_idx}")
        print(f"Val dates range: {split_idx + 1} - {end_date}")
        
        return [((start_date, split_idx), (split_idx + 1, end_date))]

class TimeSeriesKFold(SplitStrategy):
    def __init__(self, n_splits: int = 5, test_ratio: float = 0.2):
        super().__init__(test_ratio)
        self.n_splits = n_splits
    
    def split(self, date_range: Tuple[int, int]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Multiple splits based on date range
        Args:
            date_range: Tuple[int, int]: (start_date, end_date)
        Returns:
            List[Tuple[Tuple[int, int], Tuple[int, int]]]: List of ((train_start_date, train_end_date), (val_start_date, val_end_date))
        """
        start_date, end_date = date_range
        total_days = end_date - start_date + 1
        splits = []
        
        # Calculate initial training size and increment
        initial_train_days = total_days // (self.n_splits + 1)
        remaining_days = total_days - initial_train_days
        val_size = remaining_days // self.n_splits
        
        print(f"\nTime Series {self.n_splits}-Fold Split Info:")
        print(f"Total days: {total_days}")
        print(f"Initial train size: {initial_train_days} days")
        print(f"Validation size: ~{val_size} days per fold")
        
        for i in range(self.n_splits):
            train_end = start_date + initial_train_days + (i * val_size) - 1
            val_end = train_end + val_size
            if i == self.n_splits - 1:  # Last fold uses all remaining dates
                val_end = end_date
            
            print(f"\nFold {i+1}:")
            print(f"Train dates range: {start_date} - {train_end}")
            print(f"Val dates range: {train_end + 1} - {val_end}")
            
            splits.append(((start_date, train_end), (train_end + 1, val_end)))
        
        return splits

class PurgedGroupTimeSeriesSplit(SplitStrategy):
    def __init__(self, n_splits: int = 5, 
                 max_train_group_size: int = np.inf,
                 max_test_group_size: int = np.inf,
                 group_gap: int = None,
                 test_ratio: float = 0.2):
        super().__init__(test_ratio)
        self.n_splits = n_splits
        self.max_train_group_size = max_train_group_size 
        self.max_test_group_size = max_test_group_size
        self.group_gap = group_gap if group_gap is not None else 0

    def split(self, date_range: Tuple[int, int]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Multiple splits based on date range with gap between train and validation
        Args:
            date_range: Tuple[int, int]: (start_date, end_date)
        Returns:
            List[Tuple[Tuple[int, int], Tuple[int, int]]]: List of ((train_start_date, train_end_date), (val_start_date, val_end_date))
        """
        start_date, end_date = date_range
        total_days = end_date - start_date + 1
        splits = []
        
        # Calculate number of groups and group size
        n_groups = total_days
        n_folds = self.n_splits + 1
        
        if n_folds > n_groups:
            raise ValueError(
                f"Cannot have number of folds={n_folds} greater than"
                f" the number of groups={n_groups}")
        
        # Calculate test group size
        group_test_size = min(n_groups // n_folds, self.max_test_group_size)
        
        # Calculate test start positions
        group_test_starts = range(n_groups - self.n_splits * group_test_size,
                                n_groups, group_test_size)
        
        print(f"\nPurged Group Time Series {self.n_splits}-Fold Split Info:")
        print(f"Total days: {total_days}")
        print(f"Group test size: {group_test_size}")
        print(f"Group gap: {self.group_gap}")
        
        for group_test_start in group_test_starts:
            # Calculate train period
            group_st = max(0, group_test_start - self.group_gap - self.max_train_group_size)
            train_start = start_date + group_st
            train_end = start_date + group_test_start - self.group_gap
            
            # Calculate test period
            test_start = start_date + group_test_start + self.group_gap
            test_end = min(start_date + group_test_start + group_test_size, end_date)
            
            print(f"\nFold:")
            print(f"Train dates range: {train_start} - {train_end}")
            print(f"Gap dates range: {train_end + 1} - {test_start - 1}")
            print(f"Test dates range: {test_start} - {test_end}")
            
            splits.append(((train_start, train_end), (test_start, test_end)))
        
        return splits