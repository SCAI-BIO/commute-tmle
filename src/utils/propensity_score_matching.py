import pandas as pd
from psmpy import PsmPy
from matplotlib import pyplot as plt

import os
from pathlib import Path
from typing import List, Optional


def perform_propensity_score_matching(
    df: pd.DataFrame,
    treatment: str,
    indx: str,
    exclude: List[str] = [],
    caliper: float = 0.2,
    save_plots_to: Optional[str] = None,
) -> pd.DataFrame:
    # initialize PsmPy with the DataFrame and treatment variable
    exclude = [col for col in exclude if col in df.columns and col != indx]
    psm = PsmPy(df, treatment=treatment, indx=indx, exclude=exclude)

    # compute propensity scores using logistic regression
    psm.logistic_ps(balance=True)

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
            psm.effect_size_plot(save=True)
            plt.close()
        finally:
            os.chdir(original_cwd)

    return psm.df_matched
