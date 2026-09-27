import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier

pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

def learn_optimal_bins(
    train_df,
    column,
    target,
    max_bins=6,
    min_samples_leaf=0.05
):
    """
    Learn target-aware bin thresholds for a numeric feature using a Decision Tree.
    """
    X = train_df[[column]].dropna()
    y = train_df.loc[X.index, target]

    tree = DecisionTreeClassifier(
        max_leaf_nodes=max_bins,
        min_samples_leaf=min_samples_leaf,
        criterion="entropy",
        random_state=42
    )
    tree.fit(X, y)

    thresholds = tree.tree_.threshold
    thresholds = sorted(
        threshold for threshold in thresholds if threshold != -2
    )
    return thresholds


class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Leakage-safe feature engineering pipeline for customer churn prediction.
    """

    def __init__(self):
        print("Feature Engineering Pipeline Initialized")

    def create_transform_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df = self.create_customer_health_score(df)
        df = self.create_risk_segments(df)
        df = self.create_interaction_features(df)
        df = self.create_recency_features(df)
        df = self.create_binned_features(df)
        df = self.zone_cols(df)

        print("Feature Engineering Completed successfully.")
        return df

    def create_customer_health_score(self, df: pd.DataFrame) -> pd.DataFrame:
        if {"satisfaction_score", "support_calls_90d"}.issubset(df.columns):
            df["customer_health_score"] = (df["satisfaction_score"] * 20) - (df["support_calls_90d"] * 2)
        if {"satisfaction_score", "network_complaints"}.issubset(df.columns):
            df["satisfaction_network_ratio"] = df["satisfaction_score"] / (df["network_complaints"] + 1)
        return df

    def create_risk_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        if "payment_delay_days" in df.columns:
            df["payment_delay_risk"] = pd.cut(df["payment_delay_days"], bins=[-np.inf, 0.5, 2.5], labels=False)
            df['has_payment_delay'] = (df['payment_delay_days'] > 0).astype(int)

        if "support_calls_90d" in df.columns:
            df["support_risk"] = pd.cut(df["support_calls_90d"], bins=[-np.inf, 0, 1, 3], labels=False)
            df['high_support_risk'] = (df['support_calls_90d'] >= 3).astype(int)
            
        if "network_complaints" in df.columns:
            df["network_risk"] = pd.cut(df["network_complaints"], bins=[-np.inf, 0, 2, 5], labels=False)
            
        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if {"number_of_services", "monthly_charge"}.issubset(df.columns):
            df["charge_per_service"] = df["monthly_charge"] / (df["number_of_services"] + 1)
        if {"data_usage_gb", "streaming_hours_monthly"}.issubset(df.columns):
            df["data_streaming_interaction"] = df["data_usage_gb"] * df["streaming_hours_monthly"]
        if {"support_calls_90d", "network_complaints"}.issubset(df.columns):
            df["support_network_interaction"] = df["support_calls_90d"] * df["network_complaints"]
        if {"payment_delay_days", "late_payment_count"}.issubset(df.columns):
            df["payment_risk_interaction"] = df["payment_delay_days"] * df["late_payment_count"]
        return df

    def create_recency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if {"tenure_months", "device_age_months"}.issubset(df.columns):
            df["device_age_vs_tenure"] = df["device_age_months"] / (df["tenure_months"] + 1)
        if {"lifetime_spend", "tenure_months"}.issubset(df.columns):
            df["monthly_lifetime_value"] = df["lifetime_spend"] / (df["tenure_months"] + 1)
        return df

    def create_binned_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Fixed monotonic bins
        if "tenure_months" in df.columns:
            df["tenure_months_bin"] = pd.cut(df["tenure_months"], bins=[-np.inf, 2.5, 7.5, 11.5, 23.5, 38.5], labels=False)
            
        if "monthly_charge" in df.columns:
            df["monthly_charge_bin"] = pd.cut(df["monthly_charge"], bins=[-np.inf, 38.625, 40.705, 43.445, 58.235, 68.605], labels=False)
            df['monthly_charge_cat_lmh'] = pd.cut(df['monthly_charge'], bins=[-np.inf, 35, 65,np.inf], labels=['Low', 'Medium', 'High'])
        
        if "age" in df.columns:
            df["age_bin"] = pd.cut(df["age"], bins=[-np.inf, 19.5, 29.5, 64.5], labels=False)
            
        if "data_usage_gb" in df.columns:
            df["data_usage_gb_bin"] = pd.cut(df["data_usage_gb"], bins=[-np.inf, 7.225, 8.415, 15.545, 21.915, 34.745, ], labels=False)
            
        if "streaming_hours_monthly" in df.columns:
            df["streaming_hours_monthly_bin"] = pd.cut(df["streaming_hours_monthly"], bins=[-np.inf, 6.45, 15.65, 22.25, 32.65, 127.35, ], labels=False)
            
        if "device_age_months" in df.columns:
            df["device_age_months_bin"] = pd.cut(df["device_age_months"], bins=[-np.inf, 2.5, 3.5, 8.5, 31.5, 39.5, ], labels=False)
            
        if "app_sessions_weekly" in df.columns:
            df["app_sessions_weekly_bin"] = pd.cut(df["app_sessions_weekly"], bins=[-np.inf, 2.5, 11.5, 13.5, 18.5, 21.5, ], labels=False)

        if {"app_sessions_weekly", "satisfaction_score"}.issubset(df.columns):
            df['danger_zone'] = ((df['app_sessions_weekly'] < 5) & (df['satisfaction_score'] < 5)).astype(int)
            
        if {"lifetime_spend", "monthly_charge"}.issubset(df.columns):
            df['lf_calculated_tenure_months'] = df['lifetime_spend'] / (df['monthly_charge'] + 1e-5)

        if "late_payment_count" in df.columns:
            df["late_payment_group"] = np.minimum(df["late_payment_count"], 3).astype('Int64')
            
        if "discount_percentage" in df.columns:
            df['discount_tier'] = pd.cut(df['discount_percentage'], bins=[-1, 0, 10, 25, 100], 
                                         labels=['0%(NoDiscount)', '1-10%', '11-25%', '25%+'])
                   
        return df

    def zone_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        if "average_session_minutes" in df.columns:
            df["average_session_log"] = np.log1p(df["average_session_minutes"])
        if {"app_sessions_weekly", "average_session_minutes"}.issubset(df.columns):
            df['total_weekly_minutes'] = df['app_sessions_weekly'] * df['average_session_minutes']
        return df

    def drop_columns_init(self,df:pd.DataFrame,drop_cols:list):
        df = df.copy()
        print("Before Shape",df.shape)
        df = df.drop(columns=drop_cols)
        print("After Shape",df.shape)

        return df



columns_init_to_drop = ['customer_id','tenure_months','satisfaction_score','monthly_charge','age','data_usage_gb','streaming_hours_monthly',
           'payment_delay_days','late_payment_count','discount_percentage','support_calls_90d','app_sessions_weekly',
           'average_session_minutes']

as_it_is = ['number_of_services','average_session_minutes','network_complaints','referral_count','referral_count']
log = ['average_session_minutes']

