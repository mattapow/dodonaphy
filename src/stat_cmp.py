import pandas as pd
import matplotlib.pyplot as plt
# NB: pandas requires scipy
# import dendropy


def stat_cmp():
    """
    Compare Dodonaphy to Beast outputs

    First run Dodonaphy and Beast to obtain output nexus files containing Newick trees
    Prints the consensus tree and the Maximum Clade Credibility Tree
    """

    dir = "./data/T6_2"

    fn1 = dir + "/dodo_mcmc.trees"
    fn2 = dir + "/beast.trees"

    # TODO: burnin

    # bst_trees = dendropy.TreeList.get(path=fn1, schema="nexus")
    # mcmc_trees = dendropy.TreeList.get(path=fn2, schema="nexus")
    # wrap_trees = dendropy.TreeList.get(path=fn2, schema="nexus")
    # logit_trees = dendropy.TreeList.get(path=fn2, schema="nexus")

    # bst_splits = bst_trees.split_distribution()
    # mcmc_splits = mcmc_trees.split_distribution()
    # wrap_splits = wrap_trees.split_distribution()
    # logit_splits = logit_trees.split_distribution()

    # print consensus trees
    # print("Dodonaphy Consensus Tree:")
    # dodo_splits.consensus_tree().print_plot()
    # print("BEAST Consensus Tree:")
    # bst_splits.consensus_tree().print_plot()

    # # print maxmim clade credibility tree
    # print("Dodonaphy Maximim Clade Credibility Set:")
    # dodo_trees.maximum_product_of_split_support_tree().print_plot()
    # print("BEAST Maximim Clade Credibility Set:")
    # bst_trees.maximum_product_of_split_support_tree().print_plot()

    # plot from outputs of TreeStat
    fn1 = dir + "/beast/treeStats.txt"
    fn2 = dir + "/mcmc_mst_hot5_jitter_1/treeStats.txt"
    fn3 = dir + "/mcmc_incentre_hot5_1/treeStats.txt"
    fn4 = dir + "/mcmc_geodesics_hot5_jitter_1/treeStats.txt"
    # fn5 = dir + "/logit_k10/treeStats.txt"

    df1 = pd.read_csv(fn1, delimiter='\t', header=0, index_col='state')
    df2 = pd.read_csv(fn2, delimiter='\t', header=0, index_col='state')
    df3 = pd.read_csv(fn3, delimiter='\t', header=0, index_col='state')
    df4 = pd.read_csv(fn4, delimiter='\t', header=0, index_col='state')
    # df5 = pd.read_csv(fn5, delimiter='\t', header=0, index_col='state')

    print(df1.columns)
    for name in df1.columns:
        df1[name].plot.kde()
        df2[name].plot.kde()
        df3[name].plot.kde()
        df4[name].plot.kde()
        # df5[name].plot.kde()
        plt.xlabel(name)
        plt.legend(["Beast", "mcmc_mst_hot5_jitter_1", "mcmc_incentre_hot5_1", "mcmc_geodesics_hot5_jitter_1"])
        plt.show()


stat_cmp()
