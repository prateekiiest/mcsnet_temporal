#####################################################################################
#   Taken from 
#   https://github.com/yunshengb/GraphSim/blob/master/src/nx_to_mivia.py
#   With minor edits 
#####################################################################################


import struct

def convert_to_mivia(graph, labeled, label_key='label', label_dict={}):
    """Returns a binary representation of the graph that is consistent with:
    http://mivia.unisa.it/datasets/graph-database/arg-database/documentation/.

    BIG NOTE:
    McCreesh 2016 paper code only accepts labeled graphs. It can use a
    unlabeled setting, but the input format must be in mivia labeled. So to
    ignore the graph's labels, just use labeled=False.

    The edge numbering format is based double sorted. First by source node id,
    then for each node, by target node id. Be careful changing this because
    the answers from the cpp solver uses absolute edge numbering, so we have
    to keep the ordering consistent.

    Can handle labeled or unlabeled graphs.
    Does not handle directed graphs.
    Does not handle edge labels.

    Assumes that the gexf graph file starts with node id 0. This is because
    the mivia format uses 0 based indexing for everything without
    allowing graph ids.

    Args:
        graph: networkx graph, undirected, with all node ids starting from 0.
        labeled: Boolean whether or not the graph is labeled
                 (edge labels not supported yet).
        label_key: Default 'label'. Only for labeled graphs. The key that
                   should be used to access a networkx node to get its label.
        label_dict: A map from a node label to its unique id.
    Returns:
        List of results of struct.pack, to be iterated through and written.
        idx_to_node: A map of mivia index to the corresponding node from graph.nodes()
    """
    # 1. Number of nodes.
    data_bytes = []
    data_bytes.append(len(graph.nodes()))

    # 2. Node label for each node.
    idx_to_node = {}
    node_to_idx = {}
    node_iter = sorted(graph.nodes(data=True), key=lambda x: int(x[0]))
    for idx, (node, attr) in enumerate(node_iter):
        idx_to_node[idx] = int(node)
        node_to_idx[int(node)] = idx
        if labeled:
            current_label = attr[label_key]
            data_bytes.append(label_dict[current_label])
        else:
            # Just use any default label for all nodes.
            data_bytes.append(0)

    # 3. Adj list for each node, sorted by the source node id.
    adj_iter = sorted(graph.adj.items(), key=lambda x: int(x[0]))
    for source_id, adj_list in adj_iter:
        # 4. Add the length of the current node's adj list.
        data_bytes.append(len(adj_list))

        # 5. Add the indices of the connected nodes for each edge.
        for target_id, attr in sorted(adj_list.items(), key=lambda x: int(x[0])):
            # Edge labels are unsupported.
            edge_label = 0
            data_bytes.append(node_to_idx[int(target_id)])
            data_bytes.append(edge_label)

    return int_list_to_bytes_list(data_bytes), idx_to_node


def int_list_to_bytes_list(int_list):
    format_string = '<H'  # 16 bit little endian.
    return [struct.pack(format_string, i) for i in int_list]

def write_mivia_input_file(graph, filepath, labeled=False, label_key=None, label_map=None):
    bytes, idx_to_node = convert_to_mivia(graph, labeled, label_key, label_map)
    with open(filepath, 'wb') as writefile:
        for byte in bytes:
            writefile.write(byte)
    return idx_to_node
