import torch
import pickle
import tqdm
import scipy
import networkx as nx
import numpy as np
#import subgraph.iso_matching_models as iso
#import torch_geometric.utils as pyg_utils
#from subgraph.graphs import TUDatasetGraph
from random import randrange
from pebble import ProcessPool
from concurrent.futures import TimeoutError
import subprocess
import re
from collections import defaultdict
from scipy.stats import kendalltau
from mcs.nx_to_mivia import *
from mcs.utils import * 
import argparse

np.random.seed(12)

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', choices=['msrc_21', 'cox2', 'dd', 'ptc_fm', 'ptc_fr', 'ptc_mm', 'ptc_mr'], default='ptc_mm', help='dataset name')
args = parser.parse_args()

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

av = Namespace(   want_cuda                    = True,
                  has_cuda                   = torch.cuda.is_available(),
                  MIN_QUERY_SUBGRAPH_SIZE    = 9,
                  MAX_QUERY_SUBGRAPH_SIZE    = 10,
                  MIN_CORPUS_SUBGRAPH_SIZE   = 11,
                  MAX_CORPUS_SUBGRAPH_SIZE   = 15,
                  DIR_PATH                   =".",
                  DATASET_NAME               = "ptc_fr",
                  FEAT_TYPE                  = "One",#"Adjrow",#"Onehot",
                  filters_1                  = 10,
                  filters_2                  = 10,
                  filters_3                  = 10,
                  transform_dim              = 16,
                  bottle_neck_neurons        = 10,
                  tensor_neurons             = 10,               
                  dropout                    = 0.5,
                  BATCH_SIZE                 =128,
                  CONV                       = "GCN",
                  MARGIN                     = 0.1,
                  NOISE_FACTOR               = 0,
                  GMN_NPROPLAYERS = 5, 
                  TASK = "",
                #   bins                       = 16,
                  histogram                  = False
                #   WEIGHT_DECAY               =5*10**-4,
                #   TASK                       ="SimGnnIsoSeparateData"
              )

def run_mccreesh_mcs(data):
    (qid,cid),nxq = data
    fpq = av.DIR_PATH+"/mcs_mccreesh/data/query/" +av.DATASET_NAME+\
     "_query_graphs_" + str(qid)+".mivia"
    fpc = av.DIR_PATH+"/mcs_mccreesh/data/corpus/" +av.DATASET_NAME+\
     "_corpus_graphs_" + str(cid)+".mivia"
    cpp_binary = av.DIR_PATH + "/mcs_mccreesh/model/mcsp"
    cmds = "--connected --quiet min_product" 
    full_cmd = "{binary} {commands} {g1} {g2}".format(binary=cpp_binary,commands=cmds, g1=fpq, g2=fpc)
    proc = subprocess.Popen([full_cmd], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    mcs_node = int(str(out).split('\\n')[0].split(' ')[-1])
    temp = list(map(int, re.findall(r'\d+', str(out).split('\\n')[1])))
    mcs_edge = nxq.subgraph(temp[0::2]).number_of_edges()
    return mcs_node,mcs_edge



av.DATASET_NAME = args.dataset
no_of_query_subgraphs = 50
no_of_corpus_subgraphs = 800


fp = av.DIR_PATH + "/Datasets/preprocessed/" + av.DATASET_NAME +"80k_corpus_subgraphs_" + \
              str(no_of_corpus_subgraphs)+"_min_"+str(av.MIN_CORPUS_SUBGRAPH_SIZE) + "_max_" + \
              str(av.MAX_CORPUS_SUBGRAPH_SIZE)+".pkl"
print("Loading corpus subgraphs from", fp)

with open(fp, 'rb') as f:
    #pickle.dump((corpus_subgraph_list,corpus_anchor_list,corpus_subgraph_id_list),f)
    corpus_subgraph_list,corpus_anchor_list,corpus_subgraph_id_list = pickle.load(f)
#c_gr_all = corpus_subgraph_list

fp = av.DIR_PATH + "/Datasets/preprocessed/" + av.DATASET_NAME +"40k_query_subgraphs_" + \
              str(no_of_query_subgraphs)+"_min_"+str(av.MIN_QUERY_SUBGRAPH_SIZE) + "_max_" + \
              str(av.MAX_QUERY_SUBGRAPH_SIZE)+".pkl"
print("Loading query subgraphs from ", fp)
with open(fp, 'rb') as f:
    query_subgraph_list,query_anchor_list,query_subgraph_id_list = pickle.load(f)
#all_query_graphs = query_subgraph_list

### label all graphs to 0,...,n-1 ###
c_gr_all = [nx.convert_node_labels_to_integers(o) for o in corpus_subgraph_list]
all_query_graphs = [nx.convert_node_labels_to_integers(o) for o in query_subgraph_list]


fp = av.DIR_PATH + "/Datasets/mcs/splits/" + av.DATASET_NAME +"80k_corpus_subgraphs.pkl"
pickle.dump(c_gr_all,open(fp,"wb"))
print("saving corpus graphs with sorted node idx to %s", fp)


#train_data = iso.OurMatchingModelSubgraphIsoData(av,mode="train")
#val_data = iso.OurMatchingModelSubgraphIsoData(av,mode="val")
#test_data = iso.OurMatchingModelSubgraphIsoData(av,mode="test")

#c_gr_all = train_data.corpus_graphs
c_sz_all_node = np.array([x.number_of_nodes() for x in c_gr_all])
c_sz_all_edge = np.array([x.number_of_edges() for x in c_gr_all])
#dump_to_mivia(av,test_data.corpus_graphs, "corpus")
dump_to_mivia(av,c_gr_all, "corpus")

#all_query_graphs = train_data.query_graphs + val_data.query_graphs  + test_data.query_graphs 
print(len(all_query_graphs))
qgraphs_10 = list(filter(lambda x: x.number_of_nodes()==10, all_query_graphs))
print('number of 10 node graphs: %s' % len(qgraphs_10))

print('generating graphs')
successes = defaultdict(int)
new_qgrs = []
fff = "./Datasets/new_qgrs_mcs_"+ av.DATASET_NAME +"80k_enhanced_query_subgraphs.pkl"
for nn in [1,2]:
  for ne in [3,4,5]:
    for idx in tqdm.tqdm(range(len(qgraphs_10))):
      for i in range(10):
        #for i in range(400//len(qgraphs_10)):
        new_qgraph = random_graph_generator(qgraphs_10[idx],num_new_nodes=nn, max_edges_per_node=ne,a=0,b=2)
        dump_to_mivia(av,[new_qgraph], "query")

        call = list(zip([0]*800,range(800)))
        mcs_all = parallel_mcs_mccreesh(run_mccreesh_mcs,call,[new_qgraph]*len(call))
        key = str(nn)+str(ne)+str(idx)
        if check_ratios(mcs_all,new_qgraph.number_of_nodes()):
            new_qgrs.append(new_qgraph)
            successes[key]+=1
            pickle.dump(new_qgrs,open(fff,"wb"))



unique_qgrs = []
for g in new_qgrs:
    dup  = False
    for g1 in unique_qgrs: 
        if nx.is_isomorphic(g,g1):
            dup = True
            break
    if not dup: 
        unique_qgrs.append(g)
print('%s graphs generated, %s unique' % (len(new_qgrs), len(unique_qgrs)))

print('computing qap mcs')
mcs_qap = []
for idx in tqdm.tqdm(range(len(unique_qgrs))):
    all_data = run_parallel_pool(compute_qap_obj,list(zip([unique_qgrs[idx]]*len(c_gr_all),c_gr_all)))
    mcs_hinge_score_all = [x['mcs_hinge_score'] for x in all_data]
    mcs_qap.append(mcs_hinge_score_all)

mcs_data = {}
mcs_data['all_qgrs'] = new_qgrs
mcs_data['unique_qgrs'] = unique_qgrs
mcs_data['mcs_qap'] = mcs_qap
fp = "%s/Datasets/%s_generated_qgrs_mcs_qap_unique.pkl" % (av.DIR_PATH, av.DATASET_NAME)
pickle.dump(mcs_data,open(fp,"wb"))

all_ktaus=[]
for idx in range(len(mcs_qap)):
    qnodes = unique_qgrs[idx].number_of_nodes()
    qedges = unique_qgrs[idx].number_of_edges()
    all_ktaus.append(max(kendalltau(mcs_qap[idx],c_sz_all_edge)[0],\
        kendalltau(mcs_qap[idx],c_sz_all_node)[0],\
              kendalltau(mcs_qap[idx],c_sz_all_edge-qnodes)[0],\
              kendalltau(mcs_qap[idx],c_sz_all_edge+qnodes)[0],\
              kendalltau(mcs_qap[idx],np.minimum(c_sz_all_edge,qnodes))[0],\
              kendalltau(mcs_qap[idx],c_sz_all_node-qnodes)[0],\
              kendalltau(mcs_qap[idx],c_sz_all_node+qnodes)[0],\
              kendalltau(mcs_qap[idx],np.minimum(c_sz_all_node,qnodes))[0],\
                  kendalltau(mcs_qap[idx],c_sz_all_edge-qedges)[0],\
              kendalltau(mcs_qap[idx],c_sz_all_edge+qedges)[0],\
              kendalltau(mcs_qap[idx],np.minimum(c_sz_all_edge,qedges))[0],\
              kendalltau(mcs_qap[idx],c_sz_all_node-qedges)[0],\
              kendalltau(mcs_qap[idx],c_sz_all_node+qedges)[0],\
              kendalltau(mcs_qap[idx],np.minimum(c_sz_all_node,qedges))[0]))

min_ktau_idx500 = np.argsort(all_ktaus)[:500]
min_ktau_idx500_qgrs = [unique_qgrs[x] for x in min_ktau_idx500]
min_ktau_idx500_mcs_qap = [mcs_qap[x] for x in min_ktau_idx500]

min_ktau_idx500_mcs_mccreesh = []
call = list(zip([0]*800,range(800)))

print('computing mccreesh mcs')
for new_qgraph in tqdm.tqdm(min_ktau_idx500_qgrs): 
    dump_to_mivia(av,[new_qgraph], "query")    
    mcs_all = parallel_mcs_mccreesh(run_mccreesh_mcs,call,[new_qgraph]*len(call))
    min_ktau_idx500_mcs_mccreesh.append(mcs_all)

random_idx = np.random.permutation(500)
train_idx = random_idx[:300]
val_idx = random_idx[300:400]
test_idx = random_idx[400:500]

tr_qgr = [min_ktau_idx500_qgrs[i] for i in train_idx]
val_qgr = [min_ktau_idx500_qgrs[i] for i in val_idx]
test_qgr = [min_ktau_idx500_qgrs[i] for i in test_idx]

tr_qap = [min_ktau_idx500_mcs_qap[i] for i in train_idx]
val_qap = [min_ktau_idx500_mcs_qap[i] for i in val_idx]
test_qap = [min_ktau_idx500_mcs_qap[i] for i in test_idx]

tr_mcs = [min_ktau_idx500_mcs_mccreesh[i] for i in train_idx]
val_mcs = [min_ktau_idx500_mcs_mccreesh[i] for i in val_idx]
test_mcs = [min_ktau_idx500_mcs_mccreesh[i] for i in test_idx]

av.DATASET_NAME +="_500qgrlarge"

unique_qgrs = tr_qgr + val_qgr + test_qgr

print('computing gossip qap mcs')
gossip_mcs_qap = []
for idx in tqdm.tqdm(range(len(unique_qgrs))):
    all_data = run_parallel_pool(compute_gossip_qap_obj,list(zip([unique_qgrs[idx]]*len(c_gr_all),c_gr_all)))
    mcs_hinge_score_all = [x['mcs_hinge_score'] for x in all_data]
    gossip_mcs_qap.append(mcs_hinge_score_all)
tr_gossip_qap_mcs = gossip_mcs_qap[:len(tr_qgr)]
val_gossip_qap_mcs = gossip_mcs_qap[len(tr_qgr):-len(test_qgr)]
test_gossip_qap_mcs = gossip_mcs_qap[-len(test_qgr):]




mode = "test" 
fp = av.DIR_PATH + "/Datasets/mcs/splits/" + mode + "/" + mode + "_" +\
                av.DATASET_NAME +"80k_enhanced_query_subgraphs.pkl"
pickle.dump(test_qgr,open(fp,"wb"))
fp = av.DIR_PATH + "/Datasets/mcs/splits/" + mode + "/" + mode + "_" +\
            av.DATASET_NAME + "80k_rel_gossip_qap_mcs.pkl"
pickle.dump(test_gossip_qap_mcs,open(fp,"wb"))
fp = av.DIR_PATH + "/Datasets/mcs/splits/" + mode + "/" + mode + "_" +\
            av.DATASET_NAME + "80k_rel_mccreesh_mcs.pkl"
pickle.dump(test_mcs,open(fp,"wb"))
fp = av.DIR_PATH + "/Datasets/mcs/splits/" + mode + "/" + mode + "_" +\
            av.DATASET_NAME + "80k_rel_qap_mcs.pkl"
pickle.dump(test_qap,open(fp,"wb"))

mode = "train" 
fp = av.DIR_PATH + "/Datasets/mcs/splits/" + mode + "/" + mode + "_" +\
                av.DATASET_NAME +"80k_enhanced_query_subgraphs.pkl"
pickle.dump(tr_qgr,open(fp,"wb"))
fp = av.DIR_PATH + "/Datasets/mcs/splits/" + mode + "/" + mode + "_" +\
            av.DATASET_NAME + "80k_rel_gossip_qap_mcs.pkl"
pickle.dump(tr_gossip_qap_mcs,open(fp,"wb"))
fp = av.DIR_PATH + "/Datasets/mcs/splits/" + mode + "/" + mode + "_" +\
            av.DATASET_NAME + "80k_rel_mccreesh_mcs.pkl"
pickle.dump(tr_mcs,open(fp,"wb"))
fp = av.DIR_PATH + "/Datasets/mcs/splits/" + mode + "/" + mode + "_" +\
            av.DATASET_NAME + "80k_rel_qap_mcs.pkl"
pickle.dump(tr_qap,open(fp,"wb"))

mode = "val" 
fp = av.DIR_PATH + "/Datasets/mcs/splits/" + mode + "/" + mode + "_" +\
                av.DATASET_NAME +"80k_enhanced_query_subgraphs.pkl"
pickle.dump(val_qgr,open(fp,"wb"))
fp = av.DIR_PATH + "/Datasets/mcs/splits/" + mode + "/" + mode + "_" +\
            av.DATASET_NAME + "80k_rel_gossip_qap_mcs.pkl"
pickle.dump(val_gossip_qap_mcs,open(fp,"wb"))
fp = av.DIR_PATH + "/Datasets/mcs/splits/" + mode + "/" + mode + "_" +\
            av.DATASET_NAME + "80k_rel_mccreesh_mcs.pkl"
pickle.dump(val_mcs,open(fp,"wb"))
fp = av.DIR_PATH + "/Datasets/mcs/splits/" + mode + "/" + mode + "_" +\
            av.DATASET_NAME + "80k_rel_qap_mcs.pkl"
pickle.dump(val_qap,open(fp,"wb"))

print('gossip qap stats')
datalist = sum(gossip_mcs_qap,[])
keys = set(datalist)
datalistnp = np.array(datalist)

for k in keys: 
    print(k,(datalistnp==k).sum())


print('qap stats')
datalist = sum(min_ktau_idx500_mcs_qap,[])
keys = set(datalist)
datalistnp = np.array(datalist)

for k in keys: 
    print(k,(datalistnp==k).sum())

print('mccreesh stats')
datalist_pair = sum(min_ktau_idx500_mcs_mccreesh,[])
a,b = zip(*datalist_pair)
datalist = list(b)
keys = set(datalist)
datalistnp = np.array(datalist)

for k in keys:
    print(k,(datalistnp==k).sum())


#for dataset in ["msrc_21_500qgrlarge","cox2_500qgrlarge","dd_500qgrlarge"]:
#    av.DATASET_NAME = dataset
#    av.MAX_SET_SIZE = 55
#    fp = 'Datasets/mcs/splits/stats/%s_dataset_stats.pkl' % av.DATASET_NAME
#    dstats = {}
#    val_data = McsData(av,mode="val")
#    val_data.data_type = "gmn"
#    test_data = McsData(av,mode="test")
#    test_data.data_type = "gmn"
#    train_data = McsData(av,mode="train")
#    train_data.data_type = "gmn"
#    
#    a = max([x.number_of_nodes() for x in test_data.query_graphs+train_data.query_graphs+val_data.query_graphs])
#    b = max([x.number_of_nodes() for x in test_data.corpus_graphs])
#    c = max([x.number_of_edges() for x in test_data.query_graphs+train_data.query_graphs+val_data.query_graphs])
#    d = max([x.number_of_edges() for x in test_data.corpus_graphs])
#    dstats['max_num_edges'] = max(c,d)
#    dstats['max_num_nodes'] = max(a,b)
#    pickle.dump(dstats,open(fp,"wb"))
