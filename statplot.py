import itertools
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency, binomtest, norm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest


class Chi2Independence:
    """
    Chi-square test of independence with post-hoc analyses
    and visualization for arbitrary R x C contingency tables.

    Parameters
    ----------
    crosstab : pd.DataFrame
        Contingency table containing observed frequencies.

    correction : bool, default=False
        Apply Yates' continuity correction for 2x2 tables
        in the omnibus chi-square test.

    lambda_ : float or str, optional
        Power divergence parameter passed to chi2_contingency.

    alpha : float, default=0.05
        Significance threshold.

    correction_method : str, default='holm'
        Multiple-testing correction method supported by
        statsmodels.stats.multitest.multipletests.

    Notes
    -----
    Within-group comparisons:
        Exact binomial tests comparing pairs of categories
        under an equal-probability null hypothesis.

    Between-group comparisons:
        Two-proportion z-tests comparing category proportions
        across pairs of groups.

    With two hue categories, complementary between-group
    comparisons are statistically equivalent. They are
    corrected only once, but can both be displayed.

    All observations must be independent.
    """

    def __init__(
        self,
        crosstab,
        correction=False,
        lambda_=None,
        alpha=0.05,
        correction_method="holm",
    ):

        if not isinstance(crosstab, pd.DataFrame):
            raise TypeError("crosstab must be a pandas DataFrame.")

        if crosstab.shape[0] < 2 or crosstab.shape[1] < 2:
            raise ValueError(
                "At least two rows and two columns are required."
            )

        if (
            not crosstab.index.is_unique
            or not crosstab.columns.is_unique
        ):
            raise ValueError(
                "Row and column labels must be unique."
            )

        try:
            values = crosstab.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "The contingency table must contain numeric counts."
            ) from exc

        if not np.isfinite(values).all():
            raise ValueError("Counts must be finite and non-missing.")

        if np.any(values < 0):
            raise ValueError("Counts cannot be negative.")

        if not np.all(values == np.floor(values)):
            raise ValueError("Counts must be integers.")

        if np.any(values.sum(axis=0) == 0):
            raise ValueError(
                "Columns with zero totals must be removed."
            )

        if np.any(values.sum(axis=1) == 0):
            raise ValueError(
                "Rows with zero totals must be removed."
            )

        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1.")

        self.crosstab = crosstab.astype(int).copy()
        self.correction = correction
        self.lambda_ = lambda_
        self.alpha = alpha
        self.correction_method = correction_method

        self.results = None
        self.comparison_results = None
        self.residual_results = None

        self.expected = None
        self.residuals = None
        self.adjusted_residuals = None

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _orient_table(self, x=None, hue=None):
        """
        Orient table so rows represent x-axis groups
        and columns represent hue categories.
        """

        row_name = self.crosstab.index.name
        col_name = self.crosstab.columns.name

        if x is None and hue is None:
            return self.crosstab.copy(), 0

        if row_name == col_name:
            raise ValueError(
                "Index and column names must differ when "
                "specifying x and hue."
            )

        if x == row_name and hue == col_name:
            return self.crosstab.copy(), 0

        if x == col_name and hue == row_name:
            return self.crosstab.T.copy(), 1

        raise ValueError(
            "x and hue must match the index and column "
            "names of the contingency table."
        )

    @staticmethod
    def _validate_order(values, requested, name):

        if requested is None:
            return list(values)

        requested = list(requested)

        if (
            len(requested) != len(values)
            or len(set(requested)) != len(requested)
            or set(requested) != set(values)
        ):
            raise ValueError(
                f"{name} must contain every category exactly once."
            )

        return requested

    def _adjust_pvalues(self, df):
        """Apply multiple-testing correction."""

        df = df.copy()
        df["p_adjusted"] = np.nan
        df["significant"] = False

        if df.empty:
            return df

        valid = df["p"].notna()

        if valid.any():
            rejected, adjusted, _, _ = multipletests(
                df.loc[valid, "p"].to_numpy(),
                alpha=self.alpha,
                method=self.correction_method,
            )

            df.loc[valid, "p_adjusted"] = adjusted
            df.loc[valid, "significant"] = rejected

        return df

    # --------------------------------------------------
    # Omnibus chi-square test
    # --------------------------------------------------

    def chi2_ind(self):
        """Perform a chi-square test of independence."""

        result = chi2_contingency(
            self.crosstab,
            correction=self.correction,
            lambda_=self.lambda_,
        )

        observed = self.crosstab.to_numpy(dtype=float)
        expected = result.expected_freq

        n = observed.sum()
        n_rows, n_cols = observed.shape

        pearson_result = chi2_contingency(
            self.crosstab,
            correction=False,
        )

        cramers_v = np.sqrt(
            pearson_result.statistic
            / (n * min(n_rows - 1, n_cols - 1))
        )

        residuals = (
            (observed - expected) / np.sqrt(expected)
        )

        row_prob = observed.sum(axis=1) / n
        col_prob = observed.sum(axis=0) / n

        denominator = np.sqrt(
            expected
            * (1 - row_prob[:, None])
            * (1 - col_prob[None, :])
        )

        adjusted_residuals = (
            (observed - expected) / denominator
        )

        self.expected = pd.DataFrame(
            expected,
            index=self.crosstab.index,
            columns=self.crosstab.columns,
        )

        self.residuals = pd.DataFrame(
            residuals,
            index=self.crosstab.index,
            columns=self.crosstab.columns,
        )

        self.adjusted_residuals = pd.DataFrame(
            adjusted_residuals,
            index=self.crosstab.index,
            columns=self.crosstab.columns,
        )

        if np.any(expected < 5):
            warnings.warn(
                "Some expected frequencies are below 5. "
                "The chi-square approximation may be unreliable.",
                UserWarning,
            )

        self.results = {
            "chi2": result.statistic,
            "p": result.pvalue,
            "dof": result.dof,
            "expected": self.expected,
            "cramers_v": cramers_v,
            "residuals": self.residuals,
            "adjusted_residuals": self.adjusted_residuals,
        }

        return self.results

    # --------------------------------------------------
    # Post-hoc residual tests
    # --------------------------------------------------

    def test_residuals(self, adjust=True):
        """Test adjusted standardized residuals."""

        if self.results is None:
            self.chi2_ind()

        records = []

        for row in self.crosstab.index:
            for col in self.crosstab.columns:

                z = self.adjusted_residuals.loc[row, col]

                records.append({
                    "group": row,
                    "category": col,
                    "observed": self.crosstab.loc[row, col],
                    "expected": self.expected.loc[row, col],
                    "z": z,
                    "p": 2 * norm.sf(abs(z)),
                })

        df = pd.DataFrame(records)

        if adjust:
            df = self._adjust_pvalues(df)
        else:
            df["p_adjusted"] = df["p"]
            df["significant"] = df["p"] < self.alpha

        self.residual_results = df
        return df

    # --------------------------------------------------
    # Pairwise comparisons
    # --------------------------------------------------

    def pairwise_tests(
        self,
        comparisons="both",
        group_axis=0,
        display_complementary=True,
    ):
        """
        Perform pairwise comparisons for an R x C table.

        comparisons : {'within', 'between', 'both', 'none'}
        group_axis : int
            0: original rows are groups.
            1: original columns are groups.
        display_complementary : bool
            For two hue categories, include both equivalent
            between-group comparisons in the results, while
            counting them only once for p-value adjustment.
        """

        valid_options = {
            "within", "between", "both", "none"
        }

        if comparisons not in valid_options:
            raise ValueError(
                f"comparisons must be one of {valid_options}"
            )

        if group_axis not in (0, 1):
            raise ValueError("group_axis must be 0 or 1.")

        table = (
            self.crosstab.copy()
            if group_axis == 0
            else self.crosstab.T.copy()
        )

        groups = list(table.index)
        categories = list(table.columns)

        records = []
        family_ids = []

        # Within-group comparisons
        if comparisons in ("within", "both"):

            for group in groups:
                for cat1, cat2 in itertools.combinations(
                    categories, 2
                ):

                    count1 = int(table.loc[group, cat1])
                    count2 = int(table.loc[group, cat2])
                    total = count1 + count2

                    p = (
                        binomtest(
                            count1,
                            total,
                            p=0.5,
                            alternative="two-sided",
                        ).pvalue
                        if total > 0
                        else np.nan
                    )

                    records.append({
                        "type": "within",
                        "group1": group,
                        "group2": group,
                        "category1": cat1,
                        "category2": cat2,
                        "p": p,
                    })

                    family_ids.append(len(family_ids))

        # Between-group comparisons
        if comparisons in ("between", "both"):

            binary_categories = len(categories) == 2

            categories_to_test = (
                categories[:1]
                if binary_categories
                else categories
            )

            for category in categories_to_test:

                for group1, group2 in itertools.combinations(
                    groups, 2
                ):

                    counts = np.array([
                        table.loc[group1, category],
                        table.loc[group2, category],
                    ], dtype=float)

                    totals = np.array([
                        table.loc[group1].sum(),
                        table.loc[group2].sum(),
                    ], dtype=float)

                    pooled = counts.sum() / totals.sum()

                    if pooled == 0 or pooled == 1:
                        p = 1.0
                    else:
                        _, p = proportions_ztest(
                            count=counts,
                            nobs=totals,
                            alternative="two-sided",
                        )

                    hypothesis_id = len(family_ids)

                    records.append({
                        "type": "between",
                        "group1": group1,
                        "group2": group2,
                        "category1": category,
                        "category2": category,
                        "p": p,
                    })

                    family_ids.append(hypothesis_id)

                    # Complementary binary category:
                    # same hypothesis, same p-value.
                    if (
                        binary_categories
                        and display_complementary
                    ):
                        other_category = categories[1]

                        records.append({
                            "type": "between",
                            "group1": group1,
                            "group2": group2,
                            "category1": other_category,
                            "category2": other_category,
                            "p": p,
                        })

                        family_ids.append(hypothesis_id)

        columns = [
            "type",
            "group1",
            "group2",
            "category1",
            "category2",
            "p",
        ]

        df = pd.DataFrame(records, columns=columns)

        df["p_adjusted"] = np.nan
        df["significant"] = False

        if not df.empty:

            # Adjust only unique hypotheses.
            unique_indices = []
            seen = set()

            for i, hypothesis_id in enumerate(family_ids):
                if hypothesis_id not in seen:
                    seen.add(hypothesis_id)
                    unique_indices.append(i)

            unique_df = self._adjust_pvalues(
                df.iloc[unique_indices]
            )

            adjusted_by_id = {}
            significant_by_id = {}

            for i, hypothesis_id in enumerate(
                [family_ids[j] for j in unique_indices]
            ):
                adjusted_by_id[hypothesis_id] = (
                    unique_df.iloc[i]["p_adjusted"]
                )
                significant_by_id[hypothesis_id] = (
                    unique_df.iloc[i]["significant"]
                )

            df["p_adjusted"] = [
                adjusted_by_id[hypothesis_id]
                for hypothesis_id in family_ids
            ]

            df["significant"] = [
                significant_by_id[hypothesis_id]
                for hypothesis_id in family_ids
            ]

        self.comparison_results = df
        return df

    # --------------------------------------------------
    # Significance labels
    # --------------------------------------------------

    @staticmethod
    def _p_label(p, style="stars"):

        if pd.isna(p):
            return "NA"

        if style == "pvalue":
            if p < 0.001:
                return "p < .001"
            return f"p = {p:.3f}"

        if style != "stars":
            raise ValueError(
                "style must be 'stars' or 'pvalue'."
            )

        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"

        return "ns"

    # --------------------------------------------------
    # Significance brackets
    # --------------------------------------------------

    @staticmethod
    def _add_bracket(ax, x1, x2, y, height, label):
        """Draw a significance bracket."""

        ax.plot(
            [x1, x1, x2, x2],
            [y, y + height, y + height, y],
            color="black",
            linewidth=1,
            clip_on=False,
        )

        ax.text(
            (x1 + x2) / 2,
            y + height,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            clip_on=False,
        )

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    def plot(
        self,
        x=None,
        hue=None,
        order=None,
        hue_order=None,
        title=None,
        dst_dir=None,
        comparisons="both",
        value_type="percentage",
        significance="stars",
        show_ns=False,
        display_complementary=True,
        figsize=(10, 6),
        palette=None,
        ax=None,
        **kwargs,
    ):
        """
        Plot an R x C contingency table.

        Parameters
        ----------
        comparisons : {'within', 'between', 'both', 'none'}
        value_type : {'percentage', 'frequency'}
        significance : {'stars', 'pvalue'}
        show_ns : bool
            Whether to show non-significant comparisons.
        display_complementary : bool
            Display both equivalent between-group brackets
            for binary hue categories.
        palette : str, dict, or None
            None uses Seaborn's active default colors.

        Returns
        -------
        matplotlib.axes.Axes
        """

        table, group_axis = self._orient_table(
            x=x,
            hue=hue,
        )

        order = self._validate_order(
            table.index,
            order,
            "order",
        )

        hue_order = self._validate_order(
            table.columns,
            hue_order,
            "hue_order",
        )

        table = table.loc[order, hue_order]

        if value_type == "percentage":

            data = table.div(
                table.sum(axis=1),
                axis=0,
            ) * 100

            ylabel = "Percentage within group (%)"

        elif value_type == "frequency":

            data = table.copy()
            ylabel = "Frequency"

        else:
            raise ValueError(
                "value_type must be 'percentage' or 'frequency'."
            )

        df_plot = (
            data.rename_axis(
                index="_group",
                columns="_category",
            )
            .stack()
            .reset_index(name="_value")
        )

        created_fig = ax is None

        if created_fig:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        sns.barplot(
            data=df_plot,
            x="_group",
            y="_value",
            hue="_category",
            order=order,
            hue_order=hue_order,
            palette=palette,
            errorbar=None,
            dodge=True,
            ax=ax,
            **kwargs,
        )

        # Determine bar positions.
        positions = {}

        for j, container in enumerate(
            ax.containers[:len(hue_order)]
        ):
            for i, bar in enumerate(container.patches):
                positions[(order[i], hue_order[j])] = (
                    bar.get_x() + bar.get_width() / 2
                )

        tests = self.pairwise_tests(
            comparisons=comparisons,
            group_axis=group_axis,
            display_complementary=display_complementary,
        )

        if show_ns:
            tests_to_draw = tests[
                tests["p_adjusted"].notna()
            ].copy()
        else:
            tests_to_draw = tests[
                tests["significant"]
            ].copy()

        # Build significance brackets
        brackets = []

        for _, test in tests_to_draw.iterrows():

            key1 = (test["group1"], test["category1"])
            key2 = (test["group2"], test["category2"])

            x1 = positions[key1]
            x2 = positions[key2]

            brackets.append({
                "x1": x1,
                "x2": x2,
                "p": test["p_adjusted"],
                "type": test["type"],
                "span": abs(x2 - x1),
            })

        # Short brackets first
        brackets.sort(key=lambda b: b["span"])

        max_height = float(data.to_numpy().max())
        scale = max(max_height, 1.0)

        step = scale * 0.12
        bracket_height = scale * 0.025

        # Give each comparison its own vertical level.
        for i, bracket in enumerate(brackets):

            y = max_height + step * (i + 1)

            label = self._p_label(
                bracket["p"],
                style=significance,
            )

            self._add_bracket(
                ax=ax,
                x1=bracket["x1"],
                x2=bracket["x2"],
                y=y,
                height=bracket_height,
                label=label,
            )

        # Ensure every bracket is visible
        upper = max_height + step * (len(brackets) + 2)

        if value_type == "percentage":
            ax.set_ylim(0, max(105, upper))
        else:
            ax.set_ylim(0, upper)

        ax.set_xlabel(
            x or table.index.name or "Group"
        )

        ax.set_ylabel(ylabel)

        if title:
            ax.set_title(title)

        ax.legend(
            title=hue or table.columns.name,
            frameon=False,
        )

        sns.despine(ax=ax)

        if created_fig:
            fig.tight_layout()

        if dst_dir is not None:
            fig.savefig(
                dst_dir,
                dpi=600,
                bbox_inches="tight",
            )

        return ax
