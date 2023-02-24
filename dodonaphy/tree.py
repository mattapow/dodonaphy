import os

import numpy as np
import torch
from matplotlib.patches import Circle

from dodonaphy.poincare import geodesic_fn


def post_order_traversal(adjacency, currentNode, peel, visited):
    """Post-order traversal of tree in adjacency matrix

    Args:
        adjacency ([type]): [description]
        currentNode ([type]): [description]
        peel ([type]): [description]
        visited ([type]): [description]

    Returns:
        [type]: [description]
    """
    visited[currentNode] = True
    if isinstance(adjacency, dict):
        n_current = adjacency[currentNode].__len__()
    elif isinstance(adjacency, np.ndarray):
        n_current = np.sum(adjacency[currentNode])

    if n_current < 2:  # leaf nodes
        return currentNode
    else:  # internal nodes
        childs = []
        if isinstance(adjacency, dict):
            cur_adj = adjacency[currentNode]
        elif isinstance(adjacency, np.ndarray):
            cur_adj = np.where(adjacency[currentNode])[0]
        for child in cur_adj:
            if not visited[child]:
                childs.append(post_order_traversal(adjacency, child, peel, visited))
                # childs.append(child)
        childs.append(currentNode)
        peel.append(childs)
        return currentNode

class taxon():
    def __init__(self, label):
        self.label = label


def rename_labels(tree, offset=0, get_label_map=False):
    """Given a DendroPy tree, relabel its taxa to integers, starting from the value of offset.

    Args:
    tree (Tree): A DendroPy tree

    Returns:
    label_map (dict): A dictionary with labels as values and their indexes as keys.
    """
    labels = tree.taxon_namespace.labels()
    label_map = dict()
    for (i, label) in enumerate(labels):
        node_to_change = tree.find_node_with_taxon_label(label)
        node_to_change.taxon.label = str(i+offset)
        label_map[label] = str(i+offset)
    if get_label_map:
        return label_map


def dendrophy_to_pb(tree, offset=0):
    """Convert Dendropy tree to peels and blens.
    Taxon names must be indexes, starting from offset (will get converted to 0 indexing).

    Parameters
    ----------
    tree : A Dendrophy tree

    Returns
    -------
    peel : adjacencies of internal nodes (left child, right child, node)
    blens : branch lengths
    """
    n_taxa = len(tree)
    n_edges = 2 * n_taxa - 2
    blens = torch.zeros(n_edges)

    # Get peel
    nds = [nd for nd in tree.postorder_internal_node_iter()]
    for i, nd in enumerate(tree.postorder_internal_node_iter()):
        nd.taxon = taxon(i+n_taxa+1)
    n_int_nds = len(nds)
    peel = np.zeros((n_int_nds+1, 3), dtype=int)
    for i in range(n_int_nds):
        try:
            c0, c1 = nds[i].child_nodes()
        except ValueError:
            c0, c1, c2 = nds[i].child_nodes()
        
        chld0 = int(c0.taxon.label) - offset
        chld1 = int(c1.taxon.label) - offset
        parent = int(nds[i].taxon.label) - offset
        peel[i, 0] = chld0
        peel[i, 1] = chld1
        peel[i, 2] = parent

        blens[chld0] = c0.edge_length
        blens[chld1] = c1.edge_length

    # add fake root above last parent and remaining taxon
    chld2 = int(c2.taxon.label) - 1
    peel[i+1, 0] = chld2
    peel[i+1, 1] = parent
    peel[i+1, 2] = n_int_nds + n_taxa
    blens[chld2] = c2.edge_length
    return peel, blens.double()


def plot_tree(ax, peel, X_torch, color=(0, 0, 0), labels=True, radius=1):
    """Plot a tree in the Poincare disk

    Parameters
    ----------
    ax (axes): Axes for plotting
    peel (2D Tensor): edges
    X (2D Tensor): node positions [x, y]
    color (tuple): rgb colour

    Returns
    -------

    """
    circ = Circle(
        (0, 0), radius=radius, fill=False, edgecolor=(235 / 256, 237 / 256, 240 / 256)
    )
    ax.add_patch(circ)

    # nodes
    X = np.array(X_torch)
    ax.plot(X[:, 0], X[:, 1], ".", color=color)

    # edges
    n_parents = peel.shape[0]
    for i in range(n_parents):
        left = peel[i, 0]
        right = peel[i, 1]
        parent = peel[i, 2]

        points = geodesic_fn(X[left], X[parent], nb_points=100)
        ax.plot(points[:, 0], points[:, 1], linewidth=1, color=color)
        points = geodesic_fn(X[right], X[parent], nb_points=100)
        ax.plot(points[:, 0], points[:, 1], linewidth=1, color=color)
    if labels:
        n_points = X.shape[0] - 1
        for p in range(n_points):
            msg = str(p)
            ax.annotate(
                msg, xy=(float(X[p, 0]) + 0.002, float(X[p, 1])), xycoords="data"
            )


def save_tree(
    root_dir,
    filename,
    peel,
    blens,
    iteration,
    lnL,
    lnPr,
    tip_labels=None,
):
    if root_dir is None:
        return
    if tip_labels == None:
        S = len(peel) + 1
        tip_labels = [str(x + 1) for x in range(S)]
    tree = tree_to_newick(tip_labels, peel, blens)
    fn = os.path.join(root_dir, filename + ".t")
    with open(fn, "a+", encoding="UTF-8") as file:
        file.write("\ttree STATE_" + str(iteration))
        file.write(" [&lnL=%f, &lnPr=%f] = [&U] " % (lnL, lnPr))
        file.write(tree + "\n")


def end_tree_file(path_write):
    fn = os.path.join(path_write, "samples.t")
    with open(fn, "a+", encoding="UTF-8") as file:
        file.write("end;\n")


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
    plen = tipnames.__len__() - 1
    for p in range(plen):
        n1 = peel_row[p][0]
        n2 = peel_row[p][1]
        n3 = peel_row[p][2]
        if n1 <= plen:
            chunks[n1] = tipnames[n1] + ":" + str(blen_row[n1].item())
        if n2 <= plen:
            chunks[n2] = tipnames[n2] + ":" + str(blen_row[n2].item())

        if p == (plen - 1):
            chunks[n3] = "(" + chunks[n1] + "," + chunks[n2] + ")" + ";"
        else:
            chunks[n3] = (
                "("
                + chunks[n1]
                + ","
                + chunks[n2]
                + ")"
                + ":"
                + str(blen_row[n3].item())
            )
    return str(chunks[peel_row[-1][2]])


def save_tree_head(path_write, filename, tip_labels, formatter="MrBayes"):
    if path_write is None:
        return
    fn = path_write + "/" + filename + ".t"
    S = len(tip_labels)
    assert formatter in ("Beast", "MrBayes"), "Invalid save format specified."
    if formatter == "Beast":
        with open(fn, "w", encoding="UTF-8") as file:
            file.write("#NEXUS\n\n")
            file.write("Begin taxa;\n\tDimensions ntax=" + str(S) + ";\n")
            file.write("\tTaxlabels\n")
            for i in range(S):
                file.write("\t\t" + "T" + str(i + 1) + "\n")
            file.write("\t\t;\nEnd;\n\n")
            file.write("Begin trees;\n")
            file.write("\t translate\n")
            for i in range(1, S):
                file.write("\t\tT%d T%d,\n" % (i, i))
            file.write("\t\tT%d T%d;\n" % (S, S))
    else:
        with open(fn, "w", encoding="UTF-8") as file:
            file.write("#NEXUS\n[Param: tree]\nbegin trees;\n\ttranslate\n")
            idx = 1
            for taxon in tip_labels[:-1]:
                file.write(f"\t\t{idx} {taxon},\n")
                idx += 1
            file.write(f"\t\t{idx} {tip_labels[-1]};\n")
