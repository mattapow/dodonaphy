import os

import matplotlib.pyplot as plt
import pandas as pd

# NB: pandas requires scipy
# import dendropy


def stat_cmp():
    """
    Compare Dodonaphy to Beast outputs

    First run Dodonaphy and Beast to obtain output nexus files containing Newick trees
    Prints the consensus tree and the Maximum Clade Credibility Tree
    """

    dir = "./data/T17"

    # TODO: burnin
    # plot from outputs of TreeStat
    experiments = ("mrbayes", "simple_mst_c5_d5", "simple_nj_c5_d5", "simple_nj_c5_d10")
    fn1 = os.path.join(dir, experiments[0], "treeStats.txt")
    fn2 = os.path.join(dir, 'mcmc', experiments[1], "treeStats.txt")
    fn3 = os.path.join(dir, 'mcmc', experiments[2], "treeStats.txt")
    fn4 = os.path.join(dir, 'mcmc', experiments[3], "treeStats.txt")

    df1 = pd.read_csv(fn1, delimiter='\t', header=0, index_col='state')
    df2 = pd.read_csv(fn2, delimiter='\t', header=0, index_col='state')
    df3 = pd.read_csv(fn3, delimiter='\t', header=0, index_col='state')
    df4 = pd.read_csv(fn4, delimiter='\t', header=0, index_col='state')

    print(df1.columns)
    for name in df1.columns:
        df1[name].plot.kde()
        df2[name].plot.kde()
        df3[name].plot.kde()
        df4[name].plot.kde()
        plt.xlabel(name)
        plt.legend(experiments)
        plt.show()


stat_cmp()
