from pebble import ProcessPool
import torch
from concurrent.futures import TimeoutError
import numpy as np
import networkx as nx
import scipy
from mcs.nx_to_mivia import * 

def dump_to_mivia(av,list_nx_graphs,graph_type):
    """
        list_nx_graphs : list of networkx graphs
        graph_type : str query/corpus
        Since in current setup, corpus graphs are shared
        across tr-test-val, no distinction is necessary for now
    """
    for idx in range(len(list_nx_graphs)):
        fp1 = av.DIR_PATH+"/mcs_mccreesh/data/" + graph_type+"/" +av.DATASET_NAME+\
         "_" + graph_type+"_graphs_" + str(idx)+".mivia"
        node_id_map = write_mivia_input_file(list_nx_graphs[idx],fp1)
        #our nx graphs should have node ids 0..n .
        #Assert here and speak of this node_id_map no more
        assert all([node_id_map[k]==k for k in node_id_map.keys()])

def mydraw(g):
    nx.draw(g, pos=nx.kamada_kawai_layout(g),\
        with_labels=True,node_size=1000,node_color='white')

def random_graph_generator(seed_graph, num_new_nodes, max_edges_per_node, a, b): 
    """
      seed_graph: nx graph - may be nx.empty_graph()
      num_new_nodes: int denoting o of new vertices to add 
      a*degree + b : choose nodes of new connection 
      TODO: how to factor in edge density ? 
    """
    if seed_graph is None:
        seed_graph = nx.empty_graph(1)
        num_new_nodes = num_new_nodes-1

    new_graph = seed_graph.copy()
    new_node_list = np.arange(num_new_nodes) + seed_graph.number_of_nodes()

    for node in new_node_list: 
        degree_list = np.array([new_graph.degree[x] for x in new_graph.nodes])
        weighted_degree_list = degree_list*a+b
        probs =weighted_degree_list / np.sum(weighted_degree_list)
        dist = scipy.stats.rv_discrete(values=(np.arange(len(probs)), probs))
        num_new_edges = min(randrange(1,max_edges_per_node),len(probs))
        nbr_set = set()
        while(len(nbr_set)!=num_new_edges):
            nbr_set.add(dist.rvs())
        for nbr in nbr_set:
            new_graph.add_edge(nbr,node)
        
    return new_graph 

def print_mcs_stats(datalist):
    a,b = zip(*datalist)
    node_data = np.array(a)
    edge_data = np.array(b)
    total = len(a)
    print("total pairs: ", total)
    print("node mcs stats")
    for v in set(node_data):
        print("No. of {}: {}".format(v,(node_data == v).sum()))
    print("edge mcs stats")
    for v in set(edge_data):
        print("No. of {}: {}".format(v,(edge_data == v).sum()))    
     
def check_ratios(datalist,qgr_size):
    a,b = zip(*datalist)
    node_data = np.array(a)
    edge_data = np.array(b)
    total = len(a)
    #Assuming threshold 10 to deem pos examples
    pos2neg = (node_data>=10).sum()/800
    #We want few exact subgraph iso matches
    exact_iso = (node_data==qgr_size).sum()
    #print(pos2neg,exact_iso)
    return pos2neg<0.3 and pos2neg>0.1 and exact_iso<0.01

def return_ratio(datalist):
    a,b = zip(*datalist)
    node_data = np.array(a)
    edge_data = np.array(b)
    total = len(a)
    #Assuming threshold 10 to deem pos examples
    pos2neg = (node_data>=10).sum()/len(datalist)
    return pos2neg
 

def parallel_mcs_mccreesh(mcs_func,pair_id_list,qnx_list):
    all_result = []
    len_list = len(qnx_list)
    with ProcessPool(max_workers=100) as pool:
        future = pool.map(mcs_func, zip(pair_id_list,qnx_list), timeout=30)
        iterator = future.result()

        for c_i in range(len_list): 
              try:
                  result = next(iterator)
                  all_result.append(result)

              except StopIteration:
                  break
              except TimeoutError as error:  
                  all_result.append((-1,-1)) 
              except AssertionError: 
                  all_result.append((-2,-2))
    return all_result

def iterative_gossip(A,x_0):
    """
        A: adjacency matrix (should be square)
        x_0 : initial node values
    """
    #Note: don't use below inplace command since backprop hindered
    #A.fill_diagonal_(1)
    #Use this version instead
    A =  torch.eye(A.shape[0]) + A
    x_k = x_0 
    for i in range(A.shape[0]):
        x_k = torch.max(A.double()*x_k.double().expand(x_k.shape[0],x_k.shape[0]),axis=0)[0].unsqueeze(1)
    return x_k

def gossip_score(A):
    V = A.shape[0]
    x_0 = (torch.arange(V)+1)*V
    x_k = iterative_gossip(A,x_0) 
    err = torch.abs(x_k.expand(x_k.shape[0],x_k.shape[0]) \
                    -x_k.t().expand(x_k.shape[0],x_k.shape[0]) ) 
    res = torch.max(torch.sum((err==0).long(), dim=1))
    return res.item()

def QAPcost(P, adj_G, adj_G1):
    return np.linalg.norm(adj_G - P@adj_G1@P.T)**2

def  MCSHingeScore(P, Q, C):
    v = Q - P@C@P.T
    return (Q-np.maximum(0,v)).sum()

def  MCSHingeScoreGossip(P, Q, C):
    v = Q - P@C@P.T
    a = Q-np.maximum(0,v)
    return gossip_score(a)

def compute_qap_obj(data):
    q_gr,c_gr = data
    #adjacency mat of query graph
    adj_q = np.array(nx.adjacency_matrix(q_gr,nodelist=list(range(q_gr.number_of_nodes()))).todense())
        
    #pad appropriately   
    max_set_size = max(q_gr.number_of_nodes(), c_gr.number_of_nodes())
    adj_q_pad =  np.pad(adj_q,(0,max_set_size-adj_q.shape[0]))

    all_scores = []
    all_mcs_hinge_scores = []
    
    savedDataList = []

    for it in range(100):
      savedData = {}
      #generate random permutation of corpus ids
      nodeperm = {y:x for x,y in list(enumerate(np.random.permutation(c_gr.nodes)))}
      savedData['nodeperm'] = nodeperm
      #relabel corpus nodes to obtain permuted corpus graph
      c_gr_perm = nx.relabel_nodes(c_gr,mapping=nodeperm, copy=True)
      
      adj_c = np.array(nx.adjacency_matrix(c_gr_perm,nodelist=list(range(c_gr_perm.number_of_nodes()))).todense())
      savedData['adj_c'] = adj_c
      #print(adj_q_pad.shape, adj_c.shape)
      out = scipy.optimize.quadratic_assignment(adj_q_pad, adj_c,method='faq',options={'maximize':True})
      Pqap = np.eye(adj_q_pad.shape[0])[out['col_ind']]
      savedData['col_ind'] = out['col_ind']
      qapObj = QAPcost(Pqap,adj_q_pad,adj_c) 
      savedData['qapObj'] = qapObj
      all_scores.append(qapObj)
      savedDataList.append(savedData)  
      
      P = np.eye(adj_c.shape[0])[out['col_ind']]
      Q = adj_q_pad
      C = adj_c
      mcs_hinge_score = MCSHingeScore(P, Q, C)
      all_mcs_hinge_scores.append(mcs_hinge_score)
      

    maxId = np.argmax(all_mcs_hinge_scores)
    rdata = savedDataList[maxId]
    rdata['mcs_hinge_score'] =all_mcs_hinge_scores[maxId]/2
    return rdata

def compute_gossip_qap_obj(data):
    q_gr,c_gr = data
    #adjacency mat of query graph
    adj_q = np.array(nx.adjacency_matrix(q_gr,nodelist=list(range(q_gr.number_of_nodes()))).todense())
        
    #pad appropriately   
    max_set_size = max(q_gr.number_of_nodes(), c_gr.number_of_nodes())
    adj_q_pad =  np.pad(adj_q,(0,max_set_size-adj_q.shape[0]))

    all_scores = []
    all_mcs_hinge_scores = []
    
    savedDataList = []

    for it in range(100):
      savedData = {}
      #generate random permutation of corpus ids
      nodeperm = {y:x for x,y in list(enumerate(np.random.permutation(c_gr.nodes)))}
      savedData['nodeperm'] = nodeperm
      #relabel corpus nodes to obtain permuted corpus graph
      c_gr_perm = nx.relabel_nodes(c_gr,mapping=nodeperm, copy=True)
      
      adj_c = np.array(nx.adjacency_matrix(c_gr_perm,nodelist=list(range(c_gr_perm.number_of_nodes()))).todense())
      savedData['adj_c'] = adj_c
      #print(adj_q_pad.shape, adj_c.shape)
      out = scipy.optimize.quadratic_assignment(adj_q_pad, adj_c,method='faq',options={'maximize':True})
      Pqap = np.eye(adj_q_pad.shape[0])[out['col_ind']]
      savedData['col_ind'] = out['col_ind']
      qapObj = QAPcost(Pqap,adj_q_pad,adj_c) 
      savedData['qapObj'] = qapObj
      all_scores.append(qapObj)
      savedDataList.append(savedData)  
      
      P = np.eye(adj_c.shape[0])[out['col_ind']]
      Q = adj_q_pad
      C = adj_c
      mcs_hinge_score = MCSHingeScoreGossip(P, Q, C)
      all_mcs_hinge_scores.append(mcs_hinge_score)

    maxId = np.argmax(all_mcs_hinge_scores)
    rdata = savedDataList[maxId]
    rdata['mcs_hinge_score'] =all_mcs_hinge_scores[maxId]
    return rdata

def run_parallel_pool(func,input_list):
    all_result = []
    len_list = len(input_list)
    with ProcessPool(max_workers=100) as pool:
        future = pool.map(func, input_list, timeout=3000)
        iterator = future.result()

        for c_i in range(len_list): 
              try:
                  result = next(iterator)
                  all_result.append(result)

              except StopIteration:
                  break
              except TimeoutError as error:  
                  all_result.append(None) 
              except AssertionError: 
                  all_result.append(None)
    return all_result
