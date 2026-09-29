import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineering(BaseEstimator, TransformerMixin):
    """Leakage-safe feature engineering transformer for customer churn prediction.

    All statistics used for feature engineering are learned during fit() using
    training data only, then reused during transform().
    """

    def __init__(self, drop_cols=None):
        self.medians_ = {}
        self.drop_cols = drop_cols or []

    def fit(self, X, y=None):
        """Learn statistics required for feature engineering from training data only."""
        X = X.copy()

        median_columns = [
            "satisfaction_score",
            "monthly_charge",
            "data_usage_gb",
            "streaming_hours_monthly",
            "tenure_months",
            "device_age_months",
        ]

        for col in median_columns:
            if col in X.columns:
                self.medians_[col] = X[col].median()

        return self

    def transform(self, X):
        """Apply feature engineering using statistics learned during fit()."""
        df = X.copy()

        df = self.create_customer_health_score(df)
        df = self.create_risk_segments(df)
        df = self.create_interaction_features(df)
        df = self.create_recency_features(df)
        df = self.create_binned_features(df)
        df = self.create_usage_features(df)

        if self.drop_cols:
            existing_cols = [c for c in self.drop_cols if c in df.columns]
            df = df.drop(columns=existing_cols)

        return df

    # --------------------------------------------------
    # CUSTOMER HEALTH
    # --------------------------------------------------

    def create_customer_health_score(self, df):
        """Create customer health and satisfaction/network features."""
        if {"satisfaction_score", "support_calls_90d"}.issubset(df.columns):
            sat_median = self.medians_.get(
                "satisfaction_score", df["satisfaction_score"].median()
            )
            sat = df["satisfaction_score"].fillna(sat_median)
            calls = df["support_calls_90d"].fillna(0)

            df["customer_health_score"] = sat * 20 - calls * 2

        if {"satisfaction_score", "network_complaints"}.issubset(df.columns):
            sat_median = self.medians_.get(
                "satisfaction_score", df["satisfaction_score"].median()
            )
            sat = df["satisfaction_score"].fillna(sat_median)
            complaints = df["network_complaints"].fillna(0)

            df["satisfaction_network_ratio"] = sat / (complaints + 1)

        return df

    # --------------------------------------------------
    # RISK FEATURES
    # --------------------------------------------------

    def create_risk_segments(self, df):
        """Create payment, support, and network risk features.

        Missing source values remain missing instead of being silently converted
        into the lowest-risk category.
        """
        if "payment_delay_days" in df.columns:
            df["payment_delay_risk"] = pd.cut(
                df["payment_delay_days"],
                bins=[-np.inf, 0.5, 2.5, np.inf],
                labels=False,
            )

            df["has_payment_delay"] = np.where(
                df["payment_delay_days"].isna(),
                np.nan,
                (df["payment_delay_days"] > 0).astype(float),
            )

        if "support_calls_90d" in df.columns:
            df["support_risk"] = pd.cut(
                df["support_calls_90d"],
                bins=[-np.inf, 0, 1, 3, np.inf],
                labels=False,
            )

            df["high_support_risk"] = np.where(
                df["support_calls_90d"].isna(),
                np.nan,
                (df["support_calls_90d"] >= 3).astype(float),
            )

        if "network_complaints" in df.columns:
            df["network_risk"] = pd.cut(
                df["network_complaints"],
                bins=[-np.inf, 0, 2, 5, np.inf],
                labels=False,
            )

        return df

    # --------------------------------------------------
    # INTERACTION FEATURES
    # --------------------------------------------------

    def create_interaction_features(self, df):
        """Create business-related interaction features."""
        if {"number_of_services", "monthly_charge"}.issubset(df.columns):
            services = df["number_of_services"].fillna(0)
            charge_median = self.medians_.get(
                "monthly_charge", df["monthly_charge"].median()
            )
            charge = df["monthly_charge"].fillna(charge_median)

            df["charge_per_service"] = charge / (services + 1)

        if {"data_usage_gb", "streaming_hours_monthly"}.issubset(df.columns):
            usage_median = self.medians_.get(
                "data_usage_gb", df["data_usage_gb"].median()
            )
            usage = df["data_usage_gb"].fillna(usage_median)

            hours_median = self.medians_.get(
                "streaming_hours_monthly",
                df["streaming_hours_monthly"].median(),
            )
            hours = df["streaming_hours_monthly"].fillna(hours_median)

            df["data_streaming_interaction"] = usage * hours

        if {"support_calls_90d", "network_complaints"}.issubset(df.columns):
            calls = df["support_calls_90d"].fillna(0)
            complaints = df["network_complaints"].fillna(0)

            df["support_network_interaction"] = calls * complaints

        if {"payment_delay_days", "late_payment_count"}.issubset(df.columns):
            delay = df["payment_delay_days"].fillna(0)
            count = df["late_payment_count"].fillna(0)

            df["payment_risk_interaction"] = delay * count

        return df

    # --------------------------------------------------
    # RECENCY / LIFETIME FEATURES
    # --------------------------------------------------

    def create_recency_features(self, df):
        """Create device-age and lifetime-value related features."""
        if {"tenure_months", "device_age_months"}.issubset(df.columns):
            tenure_median = self.medians_.get(
                "tenure_months", df["tenure_months"].median()
            )
            tenure = df["tenure_months"].fillna(tenure_median)

            dev_median = self.medians_.get(
                "device_age_months", df["device_age_months"].median()
            )
            device_age = df["device_age_months"].fillna(dev_median)

            df["device_age_vs_tenure"] = device_age / (tenure + 1)

        if {"lifetime_spend", "tenure_months"}.issubset(df.columns):
            tenure_median = self.medians_.get(
                "tenure_months", df["tenure_months"].median()
            )
            tenure = df["tenure_months"].fillna(tenure_median)
            spend = df["lifetime_spend"].fillna(0)

            df["monthly_lifetime_value"] = spend / (tenure + 1)

        return df

    # --------------------------------------------------
    # BIN FEATURES
    # --------------------------------------------------

    def create_binned_features(self, df):
        """Create predefined business/data-driven bins.

        Missing values in the original feature remain missing in the
        corresponding bin feature.
        """
        if "tenure_months" in df.columns:
            df["tenure_months_bin"] = pd.cut(
                df["tenure_months"],
                bins=[-np.inf, 2.5, 7.5, 11.5, 23.5, 38.5, np.inf],
                labels=False,
            )

        if "monthly_charge" in df.columns:
            df["monthly_charge_bin"] = pd.cut(
                df["monthly_charge"],
                bins=[
                    -np.inf,
                    38.625,
                    40.705,
                    43.445,
                    58.235,
                    68.605,
                    np.inf,
                ],
                labels=False,
            )

            df["monthly_charge_cat_lmh"] = pd.cut(
                df["monthly_charge"],
                bins=[-np.inf, 35, 65, np.inf],
                labels=["Low", "Medium", "High"],
            )

        if "age" in df.columns:
            df["age_bin"] = pd.cut(
                df["age"],
                bins=[-np.inf, 19.5, 29.5, 64.5, np.inf],
                labels=False,
            )

        if "data_usage_gb" in df.columns:
            df["data_usage_gb_bin"] = pd.cut(
                df["data_usage_gb"],
                bins=[
                    -np.inf,
                    7.225,
                    8.415,
                    15.545,
                    21.915,
                    34.745,
                    np.inf,
                ],
                labels=False,
            )

        if "streaming_hours_monthly" in df.columns:
            df["streaming_hours_monthly_bin"] = pd.cut(
                df["streaming_hours_monthly"],
                bins=[-np.inf, 6.45, 15.65, 22.25, 32.65, 127.35, np.inf],
                labels=False,
            )

        if "device_age_months" in df.columns:
            df["device_age_months_bin"] = pd.cut(
                df["device_age_months"],
                bins=[-np.inf, 2.5, 3.5, 8.5, 31.5, 39.5, np.inf],
                labels=False,
            )

        if "app_sessions_weekly" in df.columns:
            df["app_sessions_weekly_bin"] = pd.cut(
                df["app_sessions_weekly"],
                bins=[-np.inf, 2.5, 11.5, 13.5, 18.5, 21.5, np.inf],
                labels=False,
            )

        if {"app_sessions_weekly", "satisfaction_score"}.issubset(df.columns):
            valid = (
                df["app_sessions_weekly"].notna()
                & df["satisfaction_score"].notna()
            )

            df["danger_zone"] = np.where(
                valid,
                (
                    (df["app_sessions_weekly"] < 5)
                    & (df["satisfaction_score"] < 5)
                ).astype(float),
                np.nan,
            )

        if {"lifetime_spend", "monthly_charge"}.issubset(df.columns):
            spend = df["lifetime_spend"].fillna(0)
            charge_median = self.medians_.get(
                "monthly_charge", df["monthly_charge"].median()
            )
            charge = df["monthly_charge"].fillna(charge_median)

            df["lf_calculated_tenure_months"] = spend / (charge + 1e-5)

        if "late_payment_count" in df.columns:
            df["late_payment_group"] = (
                df["late_payment_count"]
                .fillna(0)
                .clip(upper=3)
                .astype(int)
            )

        if "discount_percentage" in df.columns:
            df["discount_tier"] = pd.cut(
                df["discount_percentage"],
                bins=[-np.inf, 0, 10, 25, np.inf],
                labels=["0%(NoDiscount)", "1-10%", "11-25%", "25%+"],
            )

        return df

    # --------------------------------------------------
    # USAGE FEATURES
    # --------------------------------------------------

    def create_usage_features(self, df):
        """Create usage-related transformations."""
        if "average_session_minutes" in df.columns:
            df["average_session_log"] = np.log1p(
                df["average_session_minutes"].clip(lower=0)
            )

        if {"app_sessions_weekly", "average_session_minutes"}.issubset(
            df.columns
        ):
            sessions = df["app_sessions_weekly"].fillna(0)
            avg_minutes = df["average_session_minutes"].fillna(0)

            df["total_weekly_minutes"] = sessions * avg_minutes

        return df

    def drop_columns_init(self, df: pd.DataFrame, drop_cols: list) -> pd.DataFrame:
        df = df.copy()
        # Drop only columns that actually exist in the dataframe to prevent KeyError
        existing_drop_cols = [col for col in drop_cols if col in df.columns]
        print("Before Shape:", df.shape)
        df = df.drop(columns=existing_drop_cols)
        print("After Shape:", df.shape)
        return df



columns_init_to_drop = ['customer_id','tenure_months','satisfaction_score','monthly_charge','age','data_usage_gb','streaming_hours_monthly',
           'payment_delay_days','late_payment_count','discount_percentage','support_calls_90d','app_sessions_weekly',
           'average_session_minutes']

as_it_is = ['number_of_services','average_session_minutes','network_complaints','referral_count','referral_count']
log = ['average_session_minutes']



