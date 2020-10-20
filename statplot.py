import numpy as np
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt

# FIXME: Setting a title should not be mandatory.
# FIXME: When testing residuals for significance there should be an option to correct
# for multiple comparisons.
class Chi2Independence():
    """Class to perform a Chi2 test of independence.
    
    Perform a Chi2 test of indepence. Additionally add `standardized` and
    `adjusted standardized` residuals to the output of the Chi2 test.
    
    Parameters
    ----------
    crosstab: pd.DataFrame
        A dataframe in the format of a crosstable. Row indices are the names
        of the categories of the first variable and column names are the names
        of the categories of the second variable. 
        
        See: https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.crosstab.html
    
    """
    def __init__(self,crosstab):
        self.crosstab = crosstab
        
    def chi2_ind(self,correction=False,lambda_=None):
        """Perform a chi2_contingency test. 
        
        Also calculate standardized and adjusted standardized residuals. 
        These are added to the output of chi2_contingency.
        
        Parameters
        ----------
        correction: bool, optional
            See documentation of `scipy.stats.chi2_contigency`
        
        lambda_: float or str, optional
            See documentation of `scipy.stats.chi2_contigency`
        
        Note
        ----
        This function is an addition to chi2 test of indepence provided by `scipy.stats.chi2_contingency`
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
        """
        self.results_chi2_ = chi2_contingency(self.crosstab,
                                              correction=correction,
                                              lambda_=lambda_)
        self._residuals()
        self.results = self.results_chi2_ + (self._residuals,) + (self._stdres,)
        return self.results
        
    def _residuals(self,shift_zeros=False):
        self._table = sm.stats.Table(self.crosstab,shift_zeros=shift_zeros)
        self._residuals = self._table.resid_pearson
        self._stdres = self._table.standardized_resids
        
    def _test_single_residual(self,residual,upper_z,lower_z):
          
          if residual < lower_z:
            return True
          elif residual > upper_z:
            return True
          else:
            return False
    
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
        This method automatically calls chi2_ind() first if this
        method has not yet been called by user.
        
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
    
    def plot(self,x_var,hue_var,title,dst_dir=None):
        """Plot results of Chi2 test of independence as barplot.
        
        Parameters
        ----------
        x_var: str
            Name of the variable which should be put on the x-axis.
            
        hue_var: str
            Name of the variable which responsible for hueing.
        
        dst_dir: str (default=None)
            A string providing the path to the destination directory
            where the barplot should be saved. If `None` plot will not be saved.
        
        Note
        ----
        This method automatically calls test_residuals() if this method
        has not been called by the user.
            
        """
        if not hasattr(self,'df_freq'):
            self.test_residuals()
        
        stdres_sig_sorted = self.df_freq.sort_values(hue_var)['sig']
        
        plt.figure()
        barplot = sns.barplot(x=x_var, y='Frequency', hue=hue_var,data=self.df_freq)
        barplot.set_xticklabels(barplot.get_xticklabels(), rotation=45)
        barplot.get_xticklabels()
        barplot.set_title(title)
        
        for p,sig in zip(barplot.patches,stdres_sig_sorted):
            
            if sig == True:
                barplot.text(p.get_x() + p.get_width() / 2., p.get_height(),'*', ha='center')
        
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1)
        
        plt.tight_layout()
        
        if dst_dir is not None:
            plt.savefig(dst_dir,dpi=600)