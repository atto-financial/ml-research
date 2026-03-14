import yaml
import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class FeatureEngine:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.groups = self.config.get('groups', {})
        self.mappings = self.config.get('mappings', {})
        self.ratios = self.config.get('ratios', [])
        self.interactions = self.config.get('interactions', [])

    def apply_mappings(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for _, config in self.mappings.items():
            cols = [c for c in config['columns'] if c in df.columns]
            mapping = config['mapping']
            for col in cols:
                df[col] = df[col].map(mapping).fillna(0).astype(np.float64)
        return df

    def create_group_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for group_name, cols in self.groups.items():
            valid_cols = [c for c in cols if c in df.columns]
            if not valid_cols:
                continue
            
            df[f'{group_name}_score_sum'] = df[valid_cols].sum(axis=1)
            df[f'{group_name}_score_avg'] = df[valid_cols].mean(axis=1)
            df[f'{group_name}_high_score_count'] = (df[valid_cols] == 3).sum(axis=1).astype(np.float64)
        return df

    def create_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for ratio in self.ratios:
            name = ratio['name']
            num = ratio['numerator']
            denom = ratio['denominator']
            
            if num in df.columns and denom in df.columns:
                # Same logic as SQL: num / (denom + 1)
                df[name] = df[num] / (df[denom] + 1)
        return df

    def create_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for inter in self.interactions:
            name = inter['name']
            col1 = inter['col1']
            col2 = inter['col2']
            
            if col1 in df.columns and col2 in df.columns:
                df[name] = df[col1] * df[col2]
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point to replicate dbt logic in Python.
        """
        df = self.apply_mappings(df)
        df = self.create_group_features(df)
        df = self.create_ratios(df)
        df = self.create_interactions(df)
        return df
