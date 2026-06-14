"""
Clinical endpoint harmonization across multi-cohort datasets.

Converts heterogeneous survival/recurrence endpoints into a unified format:
- time_to_event: float (days from diagnosis/baseline)
- event_occurred: int (1 = event, 0 = censored)
"""

import logging
from typing import List, Optional
import pandas as pd
import numpy as np

from ..config import CohortConfig, EndpointType

logger = logging.getLogger(__name__)


class EndpointHarmonizer:
    """
    Harmonizes heterogeneous clinical endpoints across cohorts.
    
    Supports conversion from:
    - Survival in days/months
    - Recurrence dates (absolute)
    - Progression-free survival
    - Various event status formats
    """
    
    DAYS_PER_MONTH = 30.44  # Average days per month
    
    def __init__(self):
        self.harmonization_log: List[dict] = []
    
    def harmonize(
        self, 
        df: pd.DataFrame, 
        cohort_config: CohortConfig
    ) -> pd.DataFrame:
        """
        Harmonize endpoints for a single cohort.
        
        Args:
            df: DataFrame with original endpoint columns
            cohort_config: Configuration specifying endpoint format
            
        Returns:
            DataFrame with added 'time_to_event' and 'event_occurred' columns
        """
        df = df.copy()
        endpoint_type = cohort_config.endpoint_type
        
        logger.info(f"Harmonizing {cohort_config.name}: {endpoint_type.value}")
        
        if endpoint_type == EndpointType.SURVIVAL_DAYS:
            df['time_to_event'] = df[cohort_config.endpoint_column].astype(float)
            
        elif endpoint_type == EndpointType.SURVIVAL_MONTHS:
            df['time_to_event'] = (
                df[cohort_config.endpoint_column].astype(float) * self.DAYS_PER_MONTH
            )
            
        elif endpoint_type == EndpointType.RECURRENCE_DATE:
            df = self._convert_recurrence_dates(df, cohort_config)
            
        elif endpoint_type == EndpointType.RECURRENCE_DAYS:
            df['time_to_event'] = df[cohort_config.endpoint_column].astype(float)
            
        elif endpoint_type == EndpointType.PFS_DAYS:
            df['time_to_event'] = df[cohort_config.endpoint_column].astype(float)
            
        else:
            raise ValueError(f"Unsupported endpoint type: {endpoint_type}")
        
        # Standardize event column
        df['event_occurred'] = df[cohort_config.event_column].apply(
            self._parse_event_status
        )
        
        # Add cohort identifier
        df['cohort'] = cohort_config.name
        
        # Validation
        self._validate_harmonized(df, cohort_config.name)
        
        return df
    
    def harmonize_multiple(
        self,
        dataframes: List[pd.DataFrame],
        cohort_configs: List[CohortConfig]
    ) -> pd.DataFrame:
        """
        Harmonize and combine multiple cohorts.
        
        Args:
            dataframes: List of DataFrames, one per cohort
            cohort_configs: List of CohortConfig, one per cohort
            
        Returns:
            Combined DataFrame with harmonized endpoints
        """
        harmonized_dfs = []
        
        for df, config in zip(dataframes, cohort_configs):
            harmonized = self.harmonize(df, config)
            harmonized_dfs.append(harmonized)
        
        combined = pd.concat(harmonized_dfs, ignore_index=True)
        
        logger.info(
            f"Combined {len(cohort_configs)} cohorts: "
            f"{len(combined)} total samples"
        )
        
        return combined
    
    def _convert_recurrence_dates(
        self, 
        df: pd.DataFrame, 
        cohort_config: CohortConfig
    ) -> pd.DataFrame:
        """Convert absolute recurrence dates to days from diagnosis."""
        if cohort_config.diagnosis_date_column is None:
            raise ValueError(
                f"Cohort {cohort_config.name} has RECURRENCE_DATE endpoint "
                "but no diagnosis_date_column specified"
            )
        
        # Parse dates
        diagnosis_dates = pd.to_datetime(
            df[cohort_config.diagnosis_date_column], errors='coerce'
        )
        recurrence_dates = pd.to_datetime(
            df[cohort_config.endpoint_column], errors='coerce'
        )
        
        # Calculate days to recurrence
        time_delta = (recurrence_dates - diagnosis_dates).dt.days
        df['time_to_event'] = time_delta.astype(float)
        
        # Handle missing recurrence dates (censored patients)
        if 'last_followup_date' in df.columns:
            last_fu = pd.to_datetime(df['last_followup_date'], errors='coerce')
            censored_mask = df['time_to_event'].isna()
            df.loc[censored_mask, 'time_to_event'] = (
                (last_fu - diagnosis_dates).dt.days
            )
        
        return df
    
    def _parse_event_status(self, value) -> int:
        """
        Parse various event status representations to binary 0/1.
        
        Handles:
        - Numeric: 0, 1, 0.0, 1.0
        - String: 'dead', 'alive', 'yes', 'no', 'recurred', 'censored'
        - Boolean: True, False
        """
        if pd.isna(value):
            return 0  # Assume censored if missing
        
        if isinstance(value, bool):
            return int(value)
        
        if isinstance(value, (int, float)):
            return int(value)
        
        if isinstance(value, str):
            value_lower = value.lower().strip()
            # Event occurred
            if value_lower in [
                '1', 'dead', 'deceased', 'yes', 'recurred', 
                'progressed', 'event', 'true'
            ]:
                return 1
            # Censored
            elif value_lower in [
                '0', 'alive', 'living', 'no', 'censored', 
                'stable', 'false'
            ]:
                return 0
        
        logger.warning(f"Unknown event status value: {value}, treating as censored")
        return 0
    
    def _validate_harmonized(self, df: pd.DataFrame, cohort_name: str):
        """Validate harmonized endpoints."""
        issues = []
        
        # Check for negative times
        neg_times = (df['time_to_event'] < 0).sum()
        if neg_times > 0:
            issues.append(f"{neg_times} negative time values")
        
        # Check for missing times
        missing_times = df['time_to_event'].isna().sum()
        if missing_times > 0:
            issues.append(f"{missing_times} missing time values")
        
        # Check event distribution
        event_rate = df['event_occurred'].mean()
        if event_rate < 0.05 or event_rate > 0.95:
            issues.append(f"Unusual event rate: {event_rate:.1%}")
        
        log_entry = {
            'cohort': cohort_name,
            'n_samples': len(df),
            'event_rate': event_rate,
            'median_time': df['time_to_event'].median(),
            'issues': issues
        }
        self.harmonization_log.append(log_entry)
        
        if issues:
            logger.warning(f"{cohort_name} validation issues: {issues}")
        else:
            logger.info(
                f"{cohort_name} validated: n={len(df)}, "
                f"event_rate={event_rate:.1%}"
            )
    
    def get_summary(self) -> pd.DataFrame:
        """Get harmonization summary as DataFrame."""
        return pd.DataFrame(self.harmonization_log)
    
    def reset(self):
        """Clear harmonization log."""
        self.harmonization_log = []
