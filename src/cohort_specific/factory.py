from .ukbiobank.merge_covariates_ukb import merge_covariates_ukbiobank

# from .trinetx.merge_covariates_tnx import merge_covariates_trinetx

COHORT_SPECIFIC_COV_MERGE_FUNCTIONS = {
    "mock_data": lambda df, **kwargs: df,  # don't merge anything for the mock dataset
    "ukbiobank": merge_covariates_ukbiobank,
    # "trinetx": merge_covariates_trinetx,
}
