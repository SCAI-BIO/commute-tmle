import pandas as pd
from psmpy import PsmPy
from matplotlib import pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight

import os
from pathlib import Path
from typing import List, Optional


class PsmPyMod(PsmPy):
    """Add a version of PsmPy that allows for XGBoost propensity score estimation (which works in the presence of missing values)."""

    def hist_gradient_boosting_ps(
        self,
        balance=False,
        grid_search=False,
        max_depth_suggestions=[3, 4, 5],
        learning_rate_suggestions=[0.01, 0.1, 0.2],
        max_iter_suggestions=[50, 100, 200],
    ):
        if self.treatmentn < self.controln:
            minority, majority = self.treatmentdf, self.controldf
        elif self.treatmentn > self.controln:
            minority, majority = self.controldf, self.treatmentdf
        else:
            minority, majority = self.controldf, self.treatmentdf

        joint_df = pd.concat([majority, minority])
        treatment = joint_df[self.treatment]
        df_cleaned = joint_df.drop(columns=[self.treatment])

        X_train, X_val, treatment_train, treatment_val = train_test_split(
            df_cleaned, treatment
        )
        if grid_search:
            # perform a grid search to find a suitable set of hyperparameters
            cv_model = HistGradientBoostingClassifier()
            # Define parameter grid for XGBoost
            xgboost_param_grid = {
                "max_depth": max_depth_suggestions,
                "learning_rate": learning_rate_suggestions,
                "max_iter": max_iter_suggestions,
            }

            # Set up grid search
            grid_search = GridSearchCV(
                estimator=cv_model,
                param_grid=xgboost_param_grid,
                scoring="accuracy",
                cv=5,
                verbose=1,
                n_jobs=-1,
            )

            # Fit grid search and initialize model with best parameters
            grid_search.fit(X_train, treatment_train)
            best_params = grid_search.best_params_
            self.model = HistGradientBoostingClassifier(
                **best_params,
            )
        else:
            self.model = HistGradientBoostingClassifier(
                max_depth=6,
                learning_rate=0.01,
                max_iter=100,
            )

        # fit with sample weights according to treatment frequencies
        self.model.fit(
            X_train,
            treatment_train,
            sample_weight=(
                compute_sample_weight("balanced", treatment_train) if balance else None
            ),
        )

        pscore = self.model.predict_proba(df_cleaned)[:, 1]
        df_cleaned["propensity_score"] = pscore
        df_cleaned["propensity_logit"] = df_cleaned["propensity_score"].apply(
            lambda p: np.log(p / (1 - p))
        )
        predicted_data_reset = df_cleaned.reset_index()

        # merge with treatment df
        treatment_dfonly = self.dataIDindx[[self.treatment]].reset_index()
        self.predicted_data = pd.merge(
            predicted_data_reset, treatment_dfonly, how="inner", on=self.indx
        )


def perform_propensity_score_matching(
    df: pd.DataFrame,
    treatment: str,
    indx: str,
    exclude: List[str] = [],
    caliper: float = 0.2,
    grid_search: bool = False,
    save_plots_to: Optional[str] = None,
) -> pd.Series:
    # initialize PsmPy with the DataFrame and treatment variable
    exclude = [col for col in exclude if col in df.columns and col != indx]
    psm = PsmPyMod(df, treatment=treatment, indx=indx, exclude=exclude)

    # compute propensity scores using logistic regression
    psm.hist_gradient_boosting_ps(balance=True, grid_search=grid_search)

    # perform the matching
    psm.knn_matched(caliper=caliper)

    if save_plots_to is not None:
        original_cwd = Path.cwd()
        try:
            Path(save_plots_to).mkdir(parents=True, exist_ok=True)
            os.chdir(save_plots_to)
            psm.plot_match(
                Title="Side-by-side matched controls",
                names=["exposed", "not exposed"],
                colors=["#c00000", "#699aaf"],
                save=True,
            )
            plt.close()
            psm.effect_size_plot(
                title="Standardized mean differences across covariates before and after matching",
                save=True,
            )
            plt.close()
        finally:
            os.chdir(original_cwd)

    return psm.df_matched[indx]
