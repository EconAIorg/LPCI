from lpci import LPCI, EvaluateLPCI


def test_overall_coverage_bounds(lpci_instance, fitted_interval_df):
    evaluator = EvaluateLPCI(lpci_instance, alpha=0.1, interval_df=fitted_interval_df)
    coverage = evaluator.overall_coverage()
    assert 0.0 <= float(coverage) <= 1.0


def test_coverage_by_unit_index(lpci_instance, fitted_interval_df):
    lpci = lpci_instance
    evaluator = EvaluateLPCI(lpci, alpha=0.1, interval_df=fitted_interval_df)
    result = evaluator.coverage_by_unit()
    assert set(result.index) == set(fitted_interval_df[lpci.unit_col].unique())


def test_coverage_by_time_index(lpci_instance, fitted_interval_df):
    lpci = lpci_instance
    evaluator = EvaluateLPCI(lpci, alpha=0.1, interval_df=fitted_interval_df)
    result = evaluator.coverage_by_time()
    assert set(result.index) == set(fitted_interval_df[lpci.time_col].unique())
