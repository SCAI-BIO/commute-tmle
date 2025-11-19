import pandas as pd
from psmpy import PsmPy
from matplotlib import pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight

import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class PsmPyMod(PsmPy):
    """Add a version of PsmPy that allows for XGBoost propensity score estimation (which works in the presence of missing values)."""

    def hist_gradient_boosting_ps(
        self,
        balance=False,
        grid_search=False,
        max_depth_suggestions=[3, 4, 5],
        learning_rate_suggestions=[0.01, 0.1, 0.2],
        max_iter_suggestions=[50, 100, 200],
        calibrate_propensities: bool = False,
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

        # isotonic calibration
        clf_isotonic = CalibratedClassifierCV(self.model, cv=5, method="isotonic")
        clf_isotonic.fit(
            X_train,
            treatment_train,
        )
        pscore_isotonic = clf_isotonic.predict_proba(df_cleaned)[:, 1]

        # sigmoid calibration
        clf_sigmoid = CalibratedClassifierCV(self.model, cv=5, method="sigmoid")
        clf_sigmoid.fit(
            X_train,
            treatment_train,
        )
        pscore_sigmoid = clf_sigmoid.predict_proba(df_cleaned)[:, 1]

        df_cleaned["propensity_score_uncalibrated"] = pscore
        df_cleaned["propensity_score_isotonic"] = pscore_isotonic
        df_cleaned["propensity_score_sigmoid"] = pscore_sigmoid
        predicted_data_reset = df_cleaned.reset_index()

        # merge with treatment df
        treatment_dfonly = self.dataIDindx[[self.treatment]].reset_index()
        self.predicted_data = pd.merge(
            predicted_data_reset, treatment_dfonly, how="inner", on=self.indx
        )

        if calibrate_propensities:
            self.pick_best_calibrated()
        else:
            self.predicted_data["propensity_score"] = self.predicted_data[
                "propensity_score_uncalibrated"
            ]
        self.predicted_data["propensity_logit"] = self.predicted_data[
            "propensity_score"
        ].apply(lambda p: np.log(p / ((1 - p) + 1e-15)))

    def pick_best_calibrated(self):
        data = self.predicted_data.reset_index(drop=True)

        prob_pos_uncalibrated = data["propensity_score_uncalibrated"]
        brier_score_uncalibrated = brier_score_loss(
            data[self.treatment], prob_pos_uncalibrated
        )

        prob_pos_isotonic = data["propensity_score_isotonic"]
        brier_score_isotonic = brier_score_loss(data[self.treatment], prob_pos_isotonic)

        prob_pos_sigmoid = data["propensity_score_sigmoid"]
        brier_score_sigmoid = brier_score_loss(data[self.treatment], prob_pos_sigmoid)

        brier_scores = {
            "uncalibrated": brier_score_uncalibrated,
            "isotonic": brier_score_isotonic,
            "sigmoid": brier_score_sigmoid,
        }
        best_method = min(brier_scores, key=brier_scores.get)
        logger.info(
            f"Best calibration method: {best_method} with Brier score {brier_scores[best_method]:.4f}"
        )
        if best_method == "uncalibrated":
            self.predicted_data["propensity_score"] = data[
                "propensity_score_uncalibrated"
            ]
        elif best_method == "isotonic":
            self.predicted_data["propensity_score"] = data["propensity_score_isotonic"]
        elif best_method == "sigmoid":
            self.predicted_data["propensity_score"] = data["propensity_score_sigmoid"]

        # Clip propensity scores to avoid zeros
        self.predicted_data["propensity_score"] = np.clip(
            self.predicted_data["propensity_score"], 1e-6, 1.0 - 1e-6
        )

    def plot_propensity_calibration(
        self,
        title: str = "Predicted probabilities",
        save: bool = False,
    ):
        data = self.predicted_data.reset_index(drop=True)

        # non-calibrated propensity scores
        prob_pos_uncalibrated = data["propensity_score_uncalibrated"]
        brier_score_uncalibrated = brier_score_loss(
            data[self.treatment], prob_pos_uncalibrated
        )
        order = np.lexsort((prob_pos_uncalibrated,))
        plt.plot(
            np.arange(len(order)),
            prob_pos_uncalibrated[order],
            "r",
            label="No calibration (%1.3f)" % brier_score_uncalibrated,
        )

        # empirical treatment assignement rate
        plt.plot(
            np.arange(len(order)),
            data[self.treatment][order].rolling(window=50).mean(),
            "k",
            linewidth=3,
            label=r"Empirical",
        )

        # isotonic-calibrated propensity scores
        prob_pos_isotonic = data["propensity_score_isotonic"]
        brier_score_isotonic = brier_score_loss(data[self.treatment], prob_pos_isotonic)
        plt.plot(
            np.arange(len(order)),
            prob_pos_isotonic[order],
            "g",
            label="Isotonic calibration (%1.3f)" % brier_score_isotonic,
            alpha=0.4,
        )

        # sigmoid-calibrated propensity scores
        prob_pos_sigmoid = data["propensity_score_sigmoid"]
        brier_score_sigmoid = brier_score_loss(data[self.treatment], prob_pos_sigmoid)
        plt.plot(
            np.arange(len(order)),
            prob_pos_sigmoid[order],
            "b",
            label="Sigmoid calibration (%1.3f)" % brier_score_sigmoid,
            alpha=0.4,
        )

        plt.title(title)
        plt.xlabel("Instances sorted according to predicted probability (uncalibrated)")
        plt.ylabel("P(y=1)")
        plt.legend(loc="upper left", title="Calibration method (Brier score)")
        if save:
            plt.savefig("propensity_score_calibration.png", dpi=250)


def perform_propensity_score_matching(
    df: pd.DataFrame,
    treatment: str,
    indx: str,
    exclude: List[str] = [],
    caliper: float = 0.2,
    grid_search: bool = False,
    calibrate_propensities: bool = False,
    save_plots_to: Optional[str] = None,
) -> pd.Series:
    # initialize PsmPy with the DataFrame and treatment variable
    exclude = [col for col in exclude if col in df.columns and col != indx]
    psm = PsmPyMod(df, treatment=treatment, indx=indx, exclude=exclude)

    # compute propensity scores using logistic regression
    logger.info("Computing propensity scores using HistGradientBoostingClassifier...")
    psm.hist_gradient_boosting_ps(
        balance=True,
        grid_search=grid_search,
        calibrate_propensities=calibrate_propensities,
    )

    # perform the matching
    logger.info(
        f"Performing propensity score matching with NearestNeighbors and caliper={caliper}..."
    )
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

            psm.plot_propensity_calibration(save=True)
            plt.close()
        finally:
            os.chdir(original_cwd)

    return psm.df_matched[indx]
