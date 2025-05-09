from .ukbiobank.merge_covariates_ukb import merge_covariates_ukbiobank

COHORT_SPECIFIC_COV_MERGE_FUNCTIONS = {
    "mock_data": lambda df, **kwargs: df,  # don't merge anything for the mock dataset
    "ukbiobank": merge_covariates_ukbiobank,
}
