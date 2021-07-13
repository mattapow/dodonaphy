import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


def post_order_traversal(mst, currentNode, peel, visited):
    """Post-order traversal of a constrained-MST

    Args:
        mst ([type]): [description]
        currentNode ([type]): [description]
        peel ([type]): [description]
        visited ([type]): [description]

    Returns:
        [type]: [description]
    """
    visited[currentNode] = True
    if mst[currentNode].__len__() < 2:  # leaf nodes
        return currentNode
    else:  # internal nodes
        childs = []
        for child in mst[currentNode]:
            if (not visited[child]):
                childs.append(post_order_traversal(
                    mst, child, peel, visited))
                # childs.append(child)
        childs.append(currentNode)
        peel.append(childs)
        return currentNode


def dendrophy_to_pb(tree):
    """ Convert Dendrophy tree to peels and blens.
    Parameters
    ----------
    tree : A Dendrophy tree
    Returns
    -------
    peel : adjacencies of internal nodes (left child, right child, node)
    blens : branch lengths
    """
    # Get branch lengths
    S = len(tree)
    n_edges = 2 * S - 2
    blens = torch.zeros(n_edges)
    for i in range(n_edges):
        blens[i] = tree.bipartition_edge_map[tree.bipartition_encoding[i]].length

    # Get peel
    nds = [nd for nd in tree.postorder_internal_node_iter()]
    n_int_nds = len(nds)
    peel = np.zeros((n_int_nds, 3), dtype=int)
    for i in range(n_int_nds):
        peel[i, 0] = tree.bipartition_encoding.index(nds[i].child_edges()[0].bipartition)
        peel[i, 1] = tree.bipartition_encoding.index(nds[i].child_edges()[1].bipartition)
        peel[i, 2] = tree.bipartition_encoding.index(nds[i].bipartition)
    return peel, blens.double()


def plot_tree(ax, peel, X, color=(0, 0, 0), labels=True, root=0, radius=1):
    """ Plot a tree in the Poincare disk

    Parameters
    ----------
    ax (axes): Axes for plotting
    peel (2D Tensor): edges
    X (2D Tensor): node positions [x, y]
    color (tuple): rgb colour

    Returns
    -------

    """
    circ = Circle((0, 0), radius=radius, fill=False, edgecolor='k')
    ax.add_patch(circ)

    # nodes
    ax.plot(X[:, 0], X[:, 1], '.', color=color)

    # edges
    n_parents = peel.shape[0]
    for i in range(n_parents):
        left = peel[i, 0]
        right = peel[i, 1]
        parent = peel[i, 2]
        # TODO: correctly curved lines, not straight lines
        line = Line2D([X[left, 0], X[parent, 0]],
                      [X[left, 1], X[parent, 1]],
                      linewidth=1,
                      color=color)
        ax.add_line(line)
        line = Line2D([X[right, 0], X[parent, 0]],
                      [X[right, 1], X[parent, 1]],
                      linewidth=1,
                      color=color)
        ax.add_line(line)

    if labels:
        n_points = X.shape[0] - 1
        for p in range(n_points):
            msg = str(p)
            ax.annotate(msg,
                        xy=(float(X[p, 0]) + .002, float(X[p, 1])),
                        xycoords='data')


def save_tree(dir, filename, peel, blens, iteration, LL):
    if dir is None:
        return
    S = len(peel)+1
    tipnames = ['T' + str(x+1) for x in range(S)]
    tree = tree_to_newick(tipnames, peel, blens)

    fn = dir + '/' + filename + '.trees'
    with open(fn, 'a+') as file:
        file.write("tree STATE_" + str(iteration))
        file.write(" [&lnP={}] = [&R] ".format(LL))
        file.write(tree + '\n')


def tree_to_newick(tipnames, peel_row, blen_row):
    """This function returns a Tree in newick format from given peel and branch lengths

    Args:
        tipnames (List): List of Taxa labels
        peel_row (List of List): Peel in post-order indexing
        blen_row (List): Branch lengths

    Returns:
        Newick String: A Tree
    """
    chunks = {}
    plen = tipnames.__len__()-1
    for p in range(plen):
        n1 = peel_row[p][0]
        n2 = peel_row[p][1]
        n3 = peel_row[p][2]
        if n1 <= plen:
            chunks[n1] = tipnames[n1] + ":" + str(blen_row[n1].item())
        if n2 <= plen:
            chunks[n2] = tipnames[n2] + ":" + str(blen_row[n2].item())

        if p == (plen-1):
            chunks[n3] = "(" + chunks[n1] + "," + chunks[n2] + ")" + ";"
        else:
            chunks[n3] = "(" + chunks[n1] + "," + chunks[n2] + ")" + ":" + str(blen_row[n3].item())
    return str(chunks[peel_row[-1][2]])


def save_tree_head(path_write, filename, S):
    if path_write is None:
        return
    fn = path_write + '/' + filename + '.trees'
    with open(fn, 'w') as file:
        file.write("#NEXUS\n\n")
        file.write("Begin taxa;\n\tDimensions ntax=" + str(S) + ";\n")
        file.write("\tTaxlabels\n")
        for i in range(S):
            file.write("\t\t" + "T" + str(i+1) + "\n")
        file.write("\t\t;\nEnd;\n\n")
        file.write("Begin trees;\n")
