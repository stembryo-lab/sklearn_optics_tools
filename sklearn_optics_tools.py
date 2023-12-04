import numpy as np
from anytree import Node, RenderTree, PostOrderIter, findall
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

def optics_find_clusters_(model):
    """
    Function to find the automatically the clusters and its hierarchy as described by the original (OPTICS paper)[https://dl.acm.org/doi/abs/10.1145/304181.304187].

    Parameters
    ----------
    model
        The OPTICS model fitted as returned by `sklearn.cluster.OPTICS`.

    Returns
    -------
    List of containing the clusters in the form of [`cluster_init`, `cluster_end`], where the bigger clusters are above the other clusters.
    """

    reachabilities = model.reachability_[model.ordering_]

    N = len(reachabilities)

    clusters = []
    open_clusters = [{"pos":0,"res":reachabilities[0]}]

    old_pos = 0
    old_res = reachabilities[0]
    downhill = True
    for new_pos, new_res in enumerate(reachabilities):

        if old_res < new_res: #upphill

            l_pop = []
            for i,clust in enumerate(open_clusters):
                i = len(open_clusters)-i-1
                clust = open_clusters[i]
                
                if  clust["res"] < new_res or new_pos == len(reachabilities)-1: #cluster step by left boundary

                    clusters.append((clust["pos"],old_pos))

                    l_pop.append(i)
            
            open_clusters = list(np.array(open_clusters)[[i not in l_pop for i in range(len(open_clusters))]])

            downhill = False

        elif old_res > new_res: #downhill

            if downhill == False:
                
                clust = open_clusters[-1]
                pos = clust["pos"]
                for j,res in zip(range(pos,new_pos),reachabilities[pos:new_pos]):
                    if res < old_res: #cluster step by right boundary
                        
                        clusters.append((j,old_pos))   
                        # open_clusters.pop(len(open_clusters)-1)        
                        break

                open_clusters.append({"pos":old_pos,"res":old_res})

            downhill = True

        old_pos = new_pos
        old_res = new_res

    #Cluster containing all
    clusters.append((0,len(reachabilities)))

    return clusters

def optics_create_tree_(model, clusters):
    """
    Function to create an `anytree` tree structure capturing the hierarchy of the clustering.

    Parameters
    ----------
    model
        The OPTICS model fitted as returned by `sklearn.cluster.OPTICS`.
    clusters
        The clustering of the data as returned by `optics_find_clusters_`, in this project.

    Returns
    -------
    `anytree` ancestor `Node` object containing the whole structure of the hierarchic clustering. Appart from the properties of the `Node` class, each node contains the following information:

     - name: Name of the cluster
     - id: Numeric value of the cluster
     - color: Color of the cluster as assigned by the function `optics_assign_colors` inside this project.
     - min: Position of the beginning of the cluster in the ordered reachabilities.
     - max: Position of the ending of the cluster in the ordered reachabilities.
     - cluster_size: Size of the cluster.
     - reachability: Reachability of the cluster
     - reachability_relative: Relative reachability as counted from the lowest reachability of the cluster to the highest.
    """

    reachabilities = model.reachability_[model.ordering_]
    reachabilities_ = deepcopy(reachabilities)
    reachabilities_[reachabilities_==np.inf] = np.max(reachabilities_[reachabilities_!=np.inf])

    clusters.reverse()
    nodes = dict()
    for i,c in enumerate(clusters):
        if i == 0:
            nodes[f"cluster_{i}"] = Node(f'cluster_{i}',order=model.ordering_,id=i,color=0,min=c[0],max=c[1],cluster_size=c[1]-c[0],reachability=np.max(reachabilities_[c[0]:c[1]])*0.999,reachability_relative=np.max(reachabilities_[c[0]:c[1]])-np.min(reachabilities_[c[0]:c[1]]))
        else:
            clusters2 = deepcopy(clusters[:i])
            clusters2.reverse()
            for j,c2 in zip(range(i-1,-1,-1),clusters2):

                if c[0] >= c2[0] and c[1] <= c2[1]: #inside other cluster

                    nodes[f"cluster_{i}"] = Node(f'cluster_{i}',id=i,color=0,parent=nodes[f"cluster_{j}"],min=c[0],max=c[1],cluster_size=c[1]-c[0],reachability=np.max(reachabilities_[c[0]:c[1]])*0.999,reachability_relative=np.max(reachabilities_[c[0]:c[1]])-np.min(reachabilities_[c[0]:c[1]]))

                    break

    tree = optics_assign_colors(nodes["cluster_0"])

    return tree

def optics_assign_colors(tree):
    """
    Function that assigns unique colors to each cluster in the hierarchy level in a way that helps the trazability of the clusters along the hierarchy.

    Parameters
    ----------
     tree
        Anytree object returned by the `anytree` structure returned by the `optics_tree` function.

    Returns
    -------
    Tree object with the nodes with the `node.color` assigned.    
    """

    for i,node in enumerate(tree.leaves):

        node.color = sns.color_palette("hls",len(tree.leaves))[i]

    for depth in range(tree.height,-1,-1):

        nodes = findall(tree, filter_=lambda node: node.depth == depth and not node.is_leaf)

        for node in nodes:

            color = [0,0,0]
            norm = 0
            for son in node.children:
                norm += (node.max-node.min)
            for son in node.children:
                color[0] += son.color[0]*(node.max-node.min)/norm
                color[1] += son.color[1]*(node.max-node.min)/norm
                color[2] += son.color[2]*(node.max-node.min)/norm

            node.color = color

    return tree

def optics_tree(model):
    """
    Function to create an `anytree` tree structure capturing the hierarchy of the OPTICS clustering as described by the original (OPTICS paper)[https://dl.acm.org/doi/abs/10.1145/304181.304187].

    Parameters
    ----------
    model
        The OPTICS model fitted as returned by `sklearn.cluster.OPTICS`.

    Returns
    -------
    `anytree` ancestor `Node` object containing the whole structure of the hierarchic clustering. Appart from the properties of the `Node` class, each node contains the following information:

     - name: Name of the cluster
     - id: Numeric value of the cluster
     - color: Color of the cluster as assigned by the function `optics_assign_colors` inside this project.
     - min: Position of the beginning of the cluster in the ordered reachabilities.
     - max: Position of the ending of the cluster in the ordered reachabilities.
     - cluster_size: Size of the cluster.
     - reachability: Reachability of the cluster
     - reachability_relative: Relative reachability as counted from the lowest reachability of the cluster to the highest.
    """

    clusters = optics_find_clusters_(model)
    tree = optics_create_tree_(model, clusters)

    return tree

def optics_tree_to_list(tree):
    """
    Function that saves the tree cluster structure in a list format that is JSON compatible to be saved.

    Parameters
    ----------
     tree
        Anytree object returned by the `anytree` structure returned by the `optics_tree` function.

    Returns
    -------
    `list` object with the tree structure.
    """

    d = {
        "name":[],
        "id":[],
        "color":[],
        "min":[],
        "max":[],
        "cluster_size":[],
        "reachability":[],
        "reachability_relative":[],
        "parent":[]
    }
    for pre, fill, node in RenderTree(tree):
        d["name"].append(node.name)
        d["id"].append(node.id)
        d["color"].append(node.color)
        d["min"].append(node.min)
        d["max"].append(node.max)
        d["cluster_size"].append(node.cluster_size)
        d["reachability"].append(node.reachability)
        d["reachability_relative"].append(node.reachability_relative)

        if node.parent != None:
            d["parent"].append(node.parent.name)
        else:
            d["parent"].append("-1")

    return d

def optics_list_to_tree(d):

    origin = None
    tree = {}
    for name, id_, color, min_, max_, cluster_size, reachability, reachability_relative, parent in zip(d["name"], d["id"], d["color"], d["min"], d["max"], d["cluster_size"], d["reachability"], d["reachability_relative"], d["parent"]):
        if parent == "-1":
            parent = None
        else:
            parent = tree[parent] 

        tree[name] = Node(
            name,
            id = id_,
            color = color,
            min = min_,
            max = max_,
            cluster_size = cluster_size,
            reachability = reachability,
            reachability_relative = reachability_relative,
            parent = parent
        )

    return origin

def optics_prune_tree(tree, reachability_relative_min=0, size_min=0, reachability_max=np.inf, cluster_hierarchy_max=np.inf):
    """
    Function to remove clusters in the tree based on some criteria and return a prunned tree. Possible criteria:

     - Min Relative Reachability: Will remove nodes in the tree which relative reachability (Distance between maximum and minimum reachability of the cluster members) is smaller that a certain threshold.
     - Min Size: Will remove nodes in the tree which cluster size is smaller that a certain threshold.
     - Max Reachability: Will remove nodes in the tree which the absolute reachability is bigger that a certain threshold.
     - Max hierarchy: Will remove nodes in the tree hierarchy is bigger than a certain threshold.
    
    Parameters
    ----------
     tree
        Anytree object returned by the `anytree` structure returned by the `optics_tree` function.
     reachability_relative_min (default `0`)
        Minimum reachability threshold.
     size_min (default `0`)
        Minimum size threshold.
     reachability_max (default `np.inf`)
        Maximum reachability.
     cluster_hierarchy_max (default `np.inf`)
        Maximum hierarchy of the node threshold.

    Returns
    -------
    Prunned tree.
    """

    tree_ = deepcopy(tree)
    #reachability_relative_min
    for pre, fill, node in RenderTree(tree_):

        if node.reachability_relative < reachability_relative_min:
            node.parent = None
    #reachability_max
    for pre, fill, node in RenderTree(tree_):

        if node.reachability > reachability_max:
            node.parent = None
    #size_min
    for pre, fill, node in RenderTree(tree_):

        if node.max-node.min < size_min:
            node.parent = None
    #cluster_hierarchy_max
    for pre, fill, node in RenderTree(tree_):

        if node.depth >= cluster_hierarchy_max:
            node.parent = None

    #Remove leaves from branches with just one leave after prunning (e.c. keep biggest cluster)
    pruned = True
    while pruned:
        pruned = False
        for pre, fill, node in RenderTree(tree_):

            if len(node.children) == 1 and node.children[0].is_leaf:
                pruned = True
                node.children[0].parent = None
            elif len(node.children) == 1: #assign children of children
                pruned = True
                for child in node.children[0].children:
                    child.parent = node
                node.children[0].parent = None
                        
    # tree_ = optics_assign_colors(tree_)

    return tree_

def optics_plot_tree(tree,ax,flavor="reachability_relative",log_scale=False,bar_width_scale=0.02):
    """
    Plot the clustering tree in a provided matplotlib `axis` object. 

    Parameters
    ----------
     tree
        Anytree object returned by the `anytree` structure returned by the `optics_tree` function.
     axis
        `matplotlib.pyplot.axis` object.
     flavor (dafault `"reachability_relative"`)
        Style of plot of the clusters in the y axis between ["reachability_relative","reachability","cluster_size","hierarchy"]
     log_scale (default `False`)
        If to plot in the log scale the y axis.
     bar_width_scale (default `0.02`)
        Scale of the bar plots representing the clusters.

    Returns
    -------
    Nothing 
    """

    if flavor == "hierarchy":
        g = optics_plot_tree_by_hierarchy_(tree,ax)
    else:
        g = optics_plot_tree_by_measure_(tree,ax,flavor=flavor,log_scale=log_scale,bar_width_scale=bar_width_scale)    

    ax.set_ylabel(flavor)
    return

def optics_plot_tree_by_measure_(tree,ax,flavor="reachability_relative",log_scale=False,bar_width_scale=0.02):
    """
    Plot the clustering tree in a provided matplotlib `axis` object for the ["reachability_relative","reachability","cluster_size"] flavors.

    Parameters
    ----------
     tree
        Anytree object returned by the `anytree` structure returned by the `optics_tree` function.
     axis
        `matplotlib.pyplot.axis` object.
     flavor (dafault `"reachability_relative"`)
        Style of plot of the clusters in the y axis between ["reachability_relative","reachability","cluster_size","hierarchy"]
     log_scale (default `False`)
        If to plot in the log scale the y axis.
     bar_width_scale (default `0.02`)
        Scale of the bar plots representing the clusters.

    Returns
    -------
    Nothing 
    """

    posx = []
    posy = []
    color = []
    color_order = [] 
    size = []
    for pre, fill, node in RenderTree(tree):
        
        posx.append((node.max+node.min)/2)

        if flavor=="reachability":
            y = node.reachability
        elif flavor=="cluster_size":
            y = node.cluster_size
        else:
            y = node.reachability_relative

        posy.append(y)

        size.append(node.cluster_size)

        color_order.append(node.id)
        color.append(node.color)

    scale = np.round(np.max(posy))
    posy = np.array(posy)/scale
    height = (np.max(posy)-np.min(posy))*bar_width_scale
    if log_scale:
        height = posy*np.exp(height)-posy

    ax.bar(x=posx,height=height*scale,width=size,bottom=posy*scale,color=color)
    if log_scale:
        ax.set_yscale("log")

    return

def optics_plot_tree_by_hierarchy_(tree,ax):
    """
    Plot the clustering tree in a provided matplotlib `axis` object for the ["hierarchy"] flavor.

    Parameters
    ----------
     tree
        Anytree object returned by the `anytree` structure returned by the `optics_tree` function.
     axis
        `matplotlib.pyplot.axis` object.

    Returns
    -------
    Nothing 
    """

    posx = []
    posy = []
    color = []
    color_order = [] 
    size = []

    height = tree.height
    for pre, fill, node in RenderTree(tree):

        if node.is_leaf: #Leave
            for i in range(height+1-node.depth):        
                posx.append((node.max+node.min)/2)
                posy.append(node.depth+i)
                color.append(node.color)
                size.append(node.cluster_size)
        else:
                posx.append((node.max+node.min)/2)
                posy.append(node.depth)
                color.append(node.color)
                size.append(node.cluster_size)

        color_order.append(node.id)

    posy = np.array(posy)

    ax.bar(x=posx,height=1,width=size,bottom=-posy-.5,color=color)
    ax.set_yticks([-i for i in range(tree.height,-1,-1)])
    ax.set_yticklabels([str(i) for i in range(tree.height,-1,-1)])

    return

def optics_plot_reachability(model,tree,ax,**kwargs):
    """
    Plot the reachability plot of the OPTICS model in a provided matplotlib `axis` with the hierarchy colored in it.

    Parameters
    ----------
    model
        The OPTICS model fitted as returned by `sklearn.cluster.OPTICS`.
    tree 
        Anytree object returned by the `anytree` structure returned by the `optics_tree` function.
    axis 
        `matplotlib.pyplot.axis` object.
    **kwargs
        Arguments to be sent to `matplotlib.pyplot.fill_between` function.

    Returns
    -------
    Nothing 
    """

    reachability = model.reachability_[model.ordering_]
    reachability[reachability == np.inf] = reachability[reachability == np.inf].max()

    posx = []
    posy = []
    color = []
    color_order = [] 
    arrowsx = [] 
    arrowsy = [] 
    size = []
    for pre, fill, node in RenderTree(tree):
        
        posx.append((node.max+node.min)/2)

        y = node.reachability

        posy.append(y)

        size.append(node.cluster_size)

        color_order.append(int(node.name.split("_")[-1]))
        color.append(node.name.split("_")[-1])

    for i in range(tree.height+1):
        for pre, fill, node in RenderTree(tree):

            if node.depth == i:
                order = int(node.name.split("_")[-1])
                x_total = np.arange(0,len(reachability))
                x = x_total[node.min:node.max]
                f = reachability[node.min:node.max]
                ax.fill_between(x,f,node.reachability,color=node.color,**kwargs)

    return

def optics_get_labels(tree):
    """
    Function that returns a list of labels of the leaves for each element of the dataset.

    Parameters
    ----------
    tree 
        Anytree object returned by the `anytree` structure returned by the `optics_tree` function.

    Returns
    -------
    Array of labels.     
    """

    labels = -np.ones(tree.max,int)

    for pre, fill, node in RenderTree(tree):

        if node.is_leaf:
            labels[node.min:node.max] = node.id

    labels2 = -np.ones(tree.max,int)
    labels2[tree.order] = labels

    return labels2

def optics_get_colors(tree, labels = None):
    """
    Function that returns a list of colors assigned to the labels of the leaves for each element of the dataset.

    Parameters
    ----------
    tree 
        Anytree object returned by the `anytree` structure returned by the `optics_tree` function.
    labels (default `None`)
        If `None`, assigns lables by the leaves of the tree object, uses the provided labels otherwise.
    Returns
    -------
    Array of colors (in RGBS format).     
    """

    if np.all(labels == None):
        labels = optics_get_labels(tree)

    m = {-1:(0,0,0,0)}
    for pre, fill, node in RenderTree(tree):
        m[node.id] = (node.color[0],node.color[1],node.color[2],1)

    return list(map(lambda x : m[x], labels))

def optics_get_all_labels(tree):
    """
    Function that returns a list of lists with the labels at each hierarchical level assigned to the labels of the leaves for each element of the dataset.

    Parameters
    ----------
    tree 
        Anytree object returned by the `anytree` structure returned by the `optics_tree` function.

    Returns
    -------
    List array of labels.     
    """

    reachlist = []
    for pre, fill, node in RenderTree(tree):

        reachlist.append(node.depth+1)

    reachlist = np.sort(np.unique(reachlist))

    labels_list = []
    for reach in reachlist:

        new_tree = deepcopy(tree)
        for pre, fill, node in RenderTree(new_tree):

            if node.depth >= reach:
                node.parent = None

        labels = optics_get_labels(new_tree)

        labels_list.append(deepcopy(labels))

    return labels_list

def optics_label_outliers(X, labels, **kwargs):
    """
    Function that takes the original matrix used to fit OPTICS, a set of labels from the optics tree, and uses a `sklearn.cluster.KNeighborsClassifier` to assign a cluster label

    Parameters
    ----------
    X
        Matrix of observations x parameters used to fit the model.
    labels
        Array with labels. 
    **kwargs
        Args to be sent to `KNeighborsClassifier` class.
        
    Returns
    -------
    Array with labels.
    """

    labels = KNeighborsClassifier(**kwargs).fit(X[labels != -1,:],labels[labels != -1]).predict(X)

    return labels
