import pandas as pd
import matplotlib.pyplot as plt
# NB: pandas requires scipy


def stat_cmp():
    """
    Compare statistics in output of TreeStat
    """

    fn1 = 'out/mcmc_stat.txt'
    fn2 = 'out/beast_stat.txt'

    df1 = pd.read_csv(fn1, delimiter='\t', header=0, index_col='state')
    df2 = pd.read_csv(fn2, delimiter='\t', header=0, index_col='state')

    assert df1.columns == df2.columns

    for name in df1.columns:
        df1[name].plot.kde()
        df2[name].plot.kde()
        plt.xlabel(name)
        plt.legend(["Dodonaphy", "Beast"])
        plt.show()


stat_cmp()
