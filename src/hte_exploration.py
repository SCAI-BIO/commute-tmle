import numpy as np
import os
import pandas as pd
from pytmle import PyTMLE

from .utils.plotting import (
    plot_hte_distribution, 
    plot_quantile_distributions_continuous, 
    plot_quantile_distributions_binary
)

def explore_effect_heterogeneity(fitted_tmle: PyTMLE,
                                 input_df: pd.DataFrame,
                                 save_dir_path: str,
                                 quantile: float = 0.05,
                                 ): 

    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    # extract heterogeneous treatment effect estimates
    times = fitted_tmle._updated_estimates[0].times
    haz_0 = fitted_tmle._updated_estimates[0].hazards[..., 0]
    haz_1 = fitted_tmle._updated_estimates[1].hazards[..., 0]
    surv_0 = fitted_tmle._updated_estimates[0].event_free_survival_function
    surv_1 = fitted_tmle._updated_estimates[1].event_free_survival_function
    lagged_surv_0 = np.column_stack(
            [
                np.ones((surv_0.shape[0], 1)),
                surv_0[:, :-1],
            ],
        )
    lagged_surv_1 = np.column_stack(
            [
                np.ones((surv_1.shape[0], 1)),
                surv_1[:, :-1],
            ],
        )
    cif_0 = np.cumsum(lagged_surv_0 * haz_0, axis=1)
    cif_1 = np.cumsum(lagged_surv_1 * haz_1, axis=1)
    hte = cif_1 - cif_0

    for target_time in fitted_tmle.target_times:
        # categorize individuals into quantiles based on their HTE at the last time point
        time_idx = np.where(times == target_time)[0][0]

        hte_df = pd.DataFrame({'hte': hte[..., time_idx]})
        hte_df.loc[hte_df["hte"] > 0.5, "hte"] = np.nan
        hte_df["group"] = 1
        q_lower = hte_df['hte'].quantile(quantile)
        q_upper = hte_df['hte'].quantile(1-quantile)
        hte_df['quantile'] = pd.cut(hte_df['hte'],
                                bins=[-np.inf, q_lower, q_upper, np.inf],
                                labels=[f'bottom {quantile*100}%', f'middle {100 - 2*quantile*100}%', f'top {quantile*100}%'])

        # plot the HTE distribution at given target time
        plot_hte_distribution(save_path=save_dir_path,
                               hte_df=hte_df,
                               quantile=quantile,
                               target_time=target_time)

        # plot distributions of baseline covariates within HTE quantiles
        # separate continuous and binary features
        continuous_features = list(input_df.columns[input_df.apply(lambda x: pd.api.types.is_numeric_dtype(x) and len(np.unique(x)) > 5)])
        if continuous_features:
            plot_quantile_distributions_continuous(
                input_df=input_df,
                hte_df=hte_df,
                quantile=quantile,
                continuous_features=continuous_features,
                save_path=save_dir_path,
                target_time=target_time
            )

        binary_features = list(input_df.columns[input_df.apply(lambda x: len(np.unique(x)))==2])
        if binary_features:
            plot_quantile_distributions_binary(
                input_df=input_df,
                hte_df=hte_df,
                quantile=quantile,
                binary_features=binary_features,
                save_path=save_dir_path,
                target_time=target_time
            )
