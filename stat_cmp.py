import pandas as pd
import matplotlib.pyplot as plt
# NB: pandas requires scipy
import dendropy


def stat_cmp():
    """
    Compare Dodonaphy to Beast outputs

    First run Dodonaphy and Beast to obtain output nexus files containing Newick trees
    Prints the consensus tree and the Maximum Clade Credibility Tree
    """

    dir = "./out"
    fn1 = dir + "/mcmc.trees"
    fn2 = dir + "/beast.trees"

    # TODO: burnin

    dodo_trees = dendropy.TreeList.get(path=fn1, schema="nexus")
    bst_trees = dendropy.TreeList.get(path=fn2, schema="nexus")

    dodo_splits = dodo_trees.split_distribution()
    bst_splits = bst_trees.split_distribution()

    # print consensus trees
    print("Dodonaphy Consensus Tree:")
    dodo_splits.consensus_tree().print_plot()
    print("BEAST Consensus Tree:")
    bst_splits.consensus_tree().print_plot()

    # print maxmim clade credibility tree
    print("Dodonaphy Maximim Clade Credibility Set:")
    dodo_trees.maximum_product_of_split_support_tree().print_plot()
    print("BEAST Maximim Clade Credibility Set:")
    bst_trees.maximum_product_of_split_support_tree().print_plot()

    # plot from outputs of TreeStat
    fn1 = dir + "/mcmc_stat.txt"
    fn2 = dir + "/beast_stat.txt"

    df1 = pd.read_csv(fn1, delimiter='\t', header=0, index_col='state')
    df2 = pd.read_csv(fn2, delimiter='\t', header=0, index_col='state')

    for name in df1.columns:
        df1[name].plot.kde()
        df2[name].plot.kde()
        plt.xlabel(name)
        plt.legend(["Dodonaphy", "Beast"])
        plt.show()


stat_cmp()
