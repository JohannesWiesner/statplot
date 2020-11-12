import numpy as np
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt

class Chi2Independence():
    """Class for performing a Chi2 test of independence, additional post-hoc tests 
    and for plotting the results.
        
    Parameters
    ----------
    crosstab: pd.DataFrame
        A dataframe in the format of a crosstable. Row indices are the names
        of the categories of the first variable and column names are the names
        of the categories of the second variable. 
        
        See: https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.crosstab.html
    
    correction: bool, optional
        If True, and the degrees of freedom is 1, apply Yates’ correction 
        for continuity. The effect of the correction is to adjust each 
        observed value by 0.5 towards the corresponding expected value.
        See documentation of :func:`scipy.stats.chi2_contigency`
        
    lambda_: float or str, optional
        By default, the statistic computed in this test is Pearson’s 
        chi-squared statistic. lambda_ allows a statistic from the 
        Cressie-Read power divergence family to be used instead. 
        See power_divergence for details. See documentation of 
        :func:`scipy.stats.chi2_contigency`
    
    shift_zeros: bool, optional
        This value is passed over to :func:`statsmodels.stats.contingency_tables.Table`
        If True and any cell count is zero, add 0.5 to all values in the table.
        See documentation of `statsmodels.stats.contingency_tables.Table`
    
    """
    def __init__(self,crosstab,correction=True,lambda_=None,shift_zeros=False):
        self.crosstab = crosstab
        self.correction = correction
        self.lambda_ = lambda_
        self.shift_zeros = shift_zeros
        
    def chi2_ind(self):
        """Perform a Chi2 test of independence. This is a wrapper function around
        :func:`scipy.stats.chi2_contingency`. `Standardized` and `adjusted 
        standardized` residuals are added to the output of the Chi2 test.
                
        Note
        ----
        For more information see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
        """

        self.results_chi2_ = chi2_contingency(observed=self.crosstab,
                                              correction=self.correction,
                                              lambda_=self.lambda_)
        self._get_residuals()
        self.results = self.results_chi2_ + (self._residuals,) + (self._stdres,)
        
        return self.results
        
    def _get_residuals(self):
        self._table = sm.stats.Table(self.crosstab,shift_zeros=self.shift_zeros)
        self._residuals = self._table.resid_pearson
        self._stdres = self._table.standardized_resids
        
    def _test_single_residual(self,residual,upper_z,lower_z):
          if residual < lower_z or residual > upper_z:
            return True
          else:
            return False
        
    # TO-DO: When testing residuals for significance there should be an option to correct
    # for multiple comparisons.
    def test_residuals(self):
        """Test adjusted residuals for significance.
        
        This method performs post-hoc tests on the adjusted residuals.
        `p-value` correction is not applied.
        
        Returns
        -------
        df_freq: pd.DataFrame
            A DataFrame containing frequencies for each category combination
            and a boolean column telling if this combination got significant.
        
        Note
        ----
        This method automatically calls :func:`~statplot.Chi2Independence.chi2_ind()` 
        if this method has not been called by the user.
        
        """
        if not hasattr(self,'results'):
            self.chi2_ind()
        
        # convert stdres crosstable to one-dimensional array
        stdres_array = self._stdres.values.flatten()
        
        # check for significant local differences
        alpha = 0.05
        
        # get critical upper and lower z-value
        upper_z =  norm.ppf(1-alpha/2)
        lower_z = norm.ppf(alpha/2)
        
        # test every residual for significance (return boolean)
        stdres_sig = np.array([self._test_single_residual(residual,upper_z,lower_z) for residual in np.nditer(stdres_array)])
        
        # convert crosstab to dataframe (append significance vector)
        self.df_freq = self.crosstab.stack().reset_index().rename(columns={0:'Frequency'})
        self.df_freq['sig'] = stdres_sig
        
        return self.df_freq
    
    def plot(self,x,hue,title=None,dst_dir=None,**kwargs):
        """Plot results of Chi2 test of independence as a barplot.
        
        Parameters
        ----------
        x: str
            Name of the variable which should be put on the x-axis.
            
        hue: str
            Name of the variable which is responsible for hueing.
        
        dst_dir: str (default=None)
            A string providing the path to the destination directory
            where the barplot should be saved. If `None` plot will not be saved.
        
        kwargs: key, value mappings
            Other keyword arguments are passed through to seaborn.barplot
        
        Note
        ----
        This method automatically calls :func:`~statplot.Chi2Independence.test_residuals()` 
        if this method has not been called by the user.
            
        """
        if not hasattr(self,'df_freq'):
            self.test_residuals()
        
        barplot = sns.barplot(data=self.df_freq,x=x,y='Frequency',hue=hue,**kwargs)
        
        if title:
            barplot.set_title(title)
        
        # add significance asterisks
        stdres_sig_sorted = self.df_freq.sort_values(hue)['sig']
        
        for p,sig in zip(barplot.patches,stdres_sig_sorted):
            if sig == True:
                barplot.text(p.get_x()+p.get_width()/2.,p.get_height(),'*',ha='center')
                
        if dst_dir is not None:
            plt.savefig(dst_dir,dpi=600)
            
if __name__ == '__main__':
    pass