import collections
import pickle
import sys
from GMN.configure import *
from common import logger, set_log
import numpy as np
import time
import random
import pandas as pd
import networkx as nx
import torch
import GMN.graphembeddingnetwork as gmngen
from torch_geometric.data import Data, Batch
import torch.nn.functional as F
from mcs.evaluate import evaluate
from subgraph.earlystopping import EarlyStoppingModule

from subgraph.utils import cudavar, save_initial_model
from mcs.models import pytorch_sinkhorn_iters
import argparse


df = pd.read_csv("mcs/dummy.csv", names=["src", "dest", "time"])
print(df)


def trainingGraphList(trainDf):

    timeDict = trainDf.groupby("time").groups
    graphList = []
    for time in timeDict.keys():
        graphList.append(nx.from_pandas_edgelist(
            trainDf.loc[timeDict[time], :], "src", "dest"))
    return graphList


def queryPreviousTimeGraphs(t: int, df: pd.DataFrame):
    """
    This function returns a list of graphs from time 1 to t-1
    """
    time_df = df[df["time"] <= t]
    timeDict = (time_df.groupby("time").groups)
    graphList = []
    for time in timeDict.keys():
        graphList.append(nx.from_pandas_edgelist(
            time_df.loc[timeDict[time], :], "src", "dest"))
    return graphList


def queryGraphatTBatch(t: int, batch: pd.DataFrame) -> nx.Graph:
    """
    This function returns a graph at a given time t for a particular batch
    """
    time_df = batch[batch["time"] == t]
    G: nx.Graph = nx.from_pandas_edgelist(time_df, "src", "dest")
    return G


def queryGraphatT(t: int) -> nx.Graph:
    """
    This function returns a graph at a given time t
    """
    time_df = df[df["time"] == t]
    G: nx.Graph = nx.from_pandas_edgelist(time_df, "src", "dest")
    return G

# current timeinstance G and current timeinstance LP


def nonExistentEdges(G: nx.Graph, LP):
    """
    This function adds all non-existent edges to the LP dictionary with a value of 0    
    """
    nodeList = list(G.nodes)
    for n1 in nodeList:
        for n2 in nodeList:
            if n1 != n2 and not G.has_edge(n1, n2):
                LP[(n1, n2)] = 0


LP = {}
G1 = queryGraphatT(1)
nonExistentEdges(G1, LP)

print(LP)


def convertToPyTorchFormat(LP):
    """
    This function converts the LP dictionary to a pytorch tensor
    """
    df = pd.json_normalize(LP, sep='_')
    ls = df.to_dict(orient='records')[0]
    return torch.Tensor(list(ls.values()))


def checkTarget(t, u, v):
    """
    check if edge (u,v) exists at time t
    """
    G = queryGraphatT(t+1)
    if G.has_edge(u, v):
        return 1
    else:
        return 0


class DataRetriever(object):
    """
    """

    def __init__(self, av, mode="train") -> None:
        self.av = av
        self.data_type = "pyg"
        self.training_mode = self.av.training_mode
        self.load_graphs()

    def load_graphs(self):
        """
        """

        df = pd.read_csv("mcs/dummy.csv", names=["src", "dest", "time"])
        trainThreshold: int = self.av.trainThreshold
        valThreshold: int = self.av.valThreshold
        self.train_data: pd.DataFrame = df[df["time"] <= trainThreshold]
        self.val_data: pd.DataFrame = df[(
            df["time"] > trainThreshold) & (df["time"] <= valThreshold)]
        self.test_data: pd.DataFrame = df[df["time"] > valThreshold]
        self.train_data = self.train_data.reset_index(drop=True)
        self.val_data = self.val_data.reset_index(drop=True)
        self.test_data = self.test_data.reset_index(drop=True)

        self.trainGraphs = trainingGraphList(self.train_data) # list of graphs from time 1 to t-1   where t is the last time index of train file.

    def fetch_batched_data_by_id(self, i: int):
        """
        """
        assert(i < self.num_batches)
        batch: pd.DataFrame = self.batches[i]
        timeDict = batch.groupby("time").groups
        timeWiseLP = {}
        timeWiseTarget = {}
        for time in timeDict.keys():
            LPscore = {}
            G = queryGraphatTBatch(time, batch)
            TargetScore = {}
            # add all non-existent edges (u-v) pairs to LPscore with a value of 0
            nonExistentEdges(G, LPscore)

            for u, v in LPscore:
                TargetScore[(u, v)] = checkTarget(time, u, v)
            timeWiseLP[time] = LPscore
            timeWiseTarget[time] = TargetScore

        return batch, timeWiseLP, timeWiseTarget

    def create_batches(self, dataDF, batch_size):
        """
        """
        self.batches = []
        for i in range(0, len(dataDF), batch_size):
            self.batches.append(dataDF[i:i+batch_size])

        self.num_batches = len(self.batches)

        return self.num_batches

    def create_pyG_data_object(self, g):
        if self.av.FEAT_TYPE == "One":
            # This sets node features to one aka [1]
            x1 = cudavar(self.av, torch.FloatTensor(
                torch.ones(g.number_of_nodes(), 1)))
        #x1 = cudavar(self.av, torch.ones(g.number_of_nodes(),1).double())
        else:
            raise NotImplementedError()

        l = list(g.edges)
        edges_1 = [[x, y] for (x, y) in l] + [[y, x] for (x, y) in l]
        edge_index = cudavar(self.av, torch.from_numpy(
            np.array(edges_1, dtype=np.int64).T).type(torch.long))
        # TODO: save sizes and whatnot as per mode - node/edge
        return Data(x=x1, edge_index=edge_index), g.number_of_nodes()



    def _pack_batch(self, graphs):
        """Pack a batch of graphs into a single `GraphData` instance.
        Args:
        graphs: a list of generated networkx graphs.
        Returns:
            graph_data: a `GraphData` instance, with node and edge indices properly
            shifted.
        """
        Graphs = []
        for graph in graphs:
            for inergraph in graph:
                Graphs.append(inergraph)
        graphs = Graphs
        from_idx = []
        to_idx = []
        graph_idx = []

        n_total_nodes = 0
        n_total_edges = 0
        for i, g in enumerate(graphs):
            n_nodes = g.number_of_nodes()
            n_edges = g.number_of_edges()
            edges = np.array(g.edges(), dtype=np.int32)
            # shift the node indices for the edges
            from_idx.append(edges[:, 0] + n_total_nodes)
            to_idx.append(edges[:, 1] + n_total_nodes)
            graph_idx.append(np.ones(n_nodes, dtype=np.int32) * i)

            n_total_nodes += n_nodes
            n_total_edges += n_edges

        GraphData = collections.namedtuple('GraphData', [
            'from_idx',
            'to_idx',
            'node_features',
            'edge_features',
            'graph_idx',
            'n_graphs'])

        return GraphData(
            from_idx=np.concatenate(from_idx, axis=0),
            to_idx=np.concatenate(to_idx, axis=0),
            # this task only cares about the structures, the graphs have no features
            node_features=np.ones((n_total_nodes, 1), dtype=np.float32),
            edge_features=np.ones((n_total_edges, 1), dtype=np.float32),
            graph_idx=np.concatenate(graph_idx, axis=0),
            n_graphs=len(graphs),
        )


class MCSTemporal(torch.nn.Module):
    """
        MCS Temporal mcs hinge
    """

    def __init__(self, av, config, input_dim):
        """
        """
        super(MCSTemporal, self).__init__()
        self.av = av
        self.config = config
        self.input_dim = input_dim
        self.build_masking_utility()
        self.build_layers()
        self.diagnostic_mode = False

    def build_masking_utility(self):
        self.max_set_size = self.av.MAX_SET_SIZE
        self.graph_size_to_mask_map = [torch.cat((torch.tensor([1]).repeat(x, 1).repeat(1, self.av.transform_dim),
                                                  torch.tensor([0]).repeat(self.max_set_size-x, 1).repeat(1, self.av.transform_dim))) for x in range(0, self.max_set_size+1)]

    def build_layers(self):

        self.encoder = gmngen.GraphEncoder(**self.config['encoder'])
        prop_config = self.config['graph_embedding_net'].copy()
        prop_config.pop('n_prop_layers', None)
        prop_config.pop('share_prop_params', None)
        self.prop_layer = gmngen.GraphPropLayer(**prop_config)

        # NOTE:FILTERS_3 is 10 for now - hardcoded into config
        self.fc_transform1 = torch.nn.Linear(
            self.av.filters_3, self.av.transform_dim)
        self.relu1 = torch.nn.ReLU()
        self.fc_transform2 = torch.nn.Linear(
            self.av.transform_dim, self.av.transform_dim)

        self.fc_scores = torch.nn.Linear(
            self.config['graph_embedding_net']['n_prop_layers'], 1, bias=False)

    def get_graph(self, batchGraph):
        graph = batchGraph
        node_features = cudavar(self.av, torch.from_numpy(graph.node_features))
        edge_features = cudavar(self.av, torch.from_numpy(graph.edge_features))
        from_idx = cudavar(self.av, torch.from_numpy(graph.from_idx).long())
        to_idx = cudavar(self.av, torch.from_numpy(graph.to_idx).long())
        graph_idx = cudavar(self.av, torch.from_numpy(graph.graph_idx).long())
        return node_features, edge_features, from_idx, to_idx, graph_idx

    def forward(self, batch_data: pd.DataFrame):
        """
        """


        """
        The following three lines may be required to be changed
        """

        timeWiseLP = {}
        timeDict = batch_data.groupby("time").groups

        for time in timeDict.keys():
            # batchGraph is a graph object
            batchGraph = queryGraphatTBatch(time, batch_data) # G(t) returns a nx Graph
            # previousGraphList is a list of graph objects
            previousGraphList = queryPreviousTimeGraphs(time, df) # G(tau)s returns a list of nx Graphs

            currTime_graph_size_list= []
            currTime_graph_data_list = []
            
            data, size = self.create_pyG_data_object(batchGraph)
            currTime_graph_data_list.append(data)
            currTime_graph_size_list.append(size) #


            prevTime_graph_data_list = []
            prevTime_graph_size_list = []
            n_prevGraphs = len(previousGraphList)
            for i in range(n_prevGraphs):
            # trainGraphs or validationGraphs or testGraphs notation will change accordingly which mode we are selecting
                data, size = self.create_pyG_data_object(self.previousGraphList[i])
                prevTime_graph_data_list.append(data)
                prevTime_graph_size_list.append(size) # g1_size


            batch_data_sizes = list(zip(currTime_graph_size_list, prevTime_graph_size_list))
            a, b = zip(*batch_data_sizes)
            # query => graph (G(t)) in cuda memory
            qgraph_sizes = cudavar(self.av, torch.tensor(a))
            # corpus => G(tau) in cuda memory
            cgraph_sizes = cudavar(self.av, torch.tensor(b))


            batchGraphData = self._pack_batch(
                zip([batchGraph], previousGraphList))
            node_features, edge_features, from_idx, to_idx, graph_idx = self.get_graph(
                batchGraphData)
            node_features_enc, edge_features_enc = self.encoder(
                node_features, edge_features)

            list_nf_enc = []
            num_propagation_layers = self.config['graph_embedding_net']['n_prop_layers'] # R propagation layers
            for i in range(num_propagation_layers):
                node_features_enc = self.prop_layer(
                    node_features_enc, from_idx, to_idx, edge_features_enc)
                list_nf_enc.append(node_features_enc)

            # [(8, 12), (10, 13), (10, 14)] -> [8, 12, 10, 13, 10, 14]
            batch_data_sizes_flat = [
                item for sublist in batch_data_sizes for item in sublist]

            all_node_features_enc = torch.cat(list_nf_enc)
            node_feature_enc_split = torch.split(all_node_features_enc,
                                                 batch_data_sizes_flat*num_propagation_layers,
                                                 dim=0)
            node_feature_enc_query = node_feature_enc_split[0::2]
            node_feature_enc_corpus = node_feature_enc_split[1::2]
            assert(list(zip([x.shape[0] for x in node_feature_enc_query],
                            [x.shape[0] for x in node_feature_enc_corpus]))
                   == batch_data_sizes*num_propagation_layers)
   # TODO: need to do for corresponding non-existent edge node pairs.

        # TODO :Add for Loop here for time steps
            stacked_currTime_node_emb = torch.stack([F.pad(x, pad=(0, 0, 0, self.max_set_size-x.shape[0]))
                                                     for x in node_feature_enc_query])
            stacked_prevTime_node_emb = torch.stack([F.pad(x, pad=(0, 0, 0, self.max_set_size-x.shape[0]))
                                                     for x in node_feature_enc_corpus])

            transformed_currTime_node_emb = self.fc_transform2(
                self.relu1(self.fc_transform1(stacked_currTime_node_emb)))
            transformed_prevTime_node_emb = self.fc_transform2(
                self.relu1(self.fc_transform1(stacked_prevTime_node_emb)))
            currTimeGraph_mask = cudavar(self.av, torch.stack(
                [self.graph_size_to_mask_map[i] for i in qgraph_sizes]))
            prevTimeGraph_mask = cudavar(self.av, torch.stack(
                [self.graph_size_to_mask_map[i] for i in cgraph_sizes]))
            masked_currTime_node_emb = torch.mul(currTimeGraph_mask.repeat(
                num_propagation_layers, 1, 1), transformed_currTime_node_emb)
            masked_prevTime_node_emb = torch.mul(prevTimeGraph_mask.repeat(
                num_propagation_layers, 1, 1), transformed_prevTime_node_emb)
            sinkhorn_input = torch.matmul(
                masked_currTime_node_emb, masked_prevTime_node_emb.permute(0, 2, 1))
            transport_plan = pytorch_sinkhorn_iters(self.av, sinkhorn_input)
            if self.diagnostic_mode:
                # return transport_plan, stacked_qnode_emb, stacked_cnode_emb
                return transport_plan

            scores: torch.Tensor = torch.sum(stacked_currTime_node_emb - torch.maximum(stacked_currTime_node_emb - transport_plan@stacked_prevTime_node_emb,
                                                                                       cudavar(self.av, torch.tensor([0]))),
                                             dim=(1, 2))
            scores_reshaped = scores.view(-1, self.av.BATCH_SIZE).T
            #final_scores = self.fc_scores(scores_reshaped).squeeze()
            final_scores = scores_reshaped@(torch.nn.ReLU()
                                            (self.fc_scores.weight.T)).squeeze()

            timeWiseLP[time] = final_scores

        return timeWiseLP


def train(av, config):
    device = "cuda" if av.has_cuda and av.want_cuda else "cpu"
    print("============ DEVICE ==================== : ", device)
    # TODO: remove this hardcoded abomination
    av.MAX_SET_SIZE = av.dataset_stats['max_num_edges']
    train_data = DataRetriever(av, mode="train")

    #print("Training Data ===>>> ", train_data.query_graphs)
    val_data = DataRetriever(av, mode="val")
    es = EarlyStoppingModule(av, av.ES)

    logger.info("Loading model MCSTemporal")
    logger.info(
        "This uses basic ISONET node alignment model with hinge MCS loss. We apply sinkhorn on embedding from every layer")
    av.MAX_SET_SIZE = av.dataset_stats['max_num_nodes']

    model = MCSTemporal(av, config, 1).to(device)
    train_data.data_type = "gmn"
    val_data.data_type = "gmn"

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=av.LEARNING_RATE,
                                 weight_decay=av.WEIGHT_DECAY)
    cnt = 0
    for param in model.parameters():
        cnt = cnt+torch.numel(param)
    logger.info("no. of params in model: %s", cnt)

    # If this model has been trained before, then load latest trained model
    # Check status of last model, and continue/abort run accordingly
    checkpoint = es.load_latest_model()
    if not checkpoint:
        save_initial_model(av, model)
        run = 0
    else:
        if es.should_stop_now:
            logger.info(
                "Training has been completed. This logfile can be deleted.")
            return
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optim_state_dict'])
            run = checkpoint['epoch'] + 1

    while av.RUN_TILL_ES or run < av.NUM_RUNS:
        model.train()
        start_time = time.time()
        n_batches = train_data.create_batches()
        epoch_loss = 0
        start_time = time.time()
        for i in range(n_batches):
            batch_data, prediction, target = train_data.fetch_batched_data_by_id(
                i)
            optimizer.zero_grad()
            prediction = model(batch_data) # this is calling the forward pass
            losses = torch.nn.functional.mse_loss(
                convertToPyTorchFormat(target), prediction, reduction="mean")
            losses.backward()
            optimizer.step()
            epoch_loss = epoch_loss + losses.item()

        logger.info("Run: %d train loss: %f Time: %.2f",
                    run, epoch_loss, time.time()-start_time)
        start_time = time.time()
        ndcg, mse, rankcorr, mae = evaluate(av, model, val_data)
        logger.info("Run: %d VAL ndcg_score: %.6f mse_loss: %.6f rankcorr: %.6f Time: %.2f",
                    run, ndcg, mse, rankcorr, time.time()-start_time)

        if av.RUN_TILL_ES:
            es_score = -mse
            if es.check([es_score], model, run, optimizer):
                break
        run += 1
































if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--logpath",                        type=str,
                    default="logDir/logfile", help="/path/to/log")
    ap.add_argument("--want_cuda",
                    type=bool,  default=True)
    ap.add_argument("--RUN_TILL_ES",
                    type=bool,  default=True)
    ap.add_argument("--has_cuda",                       type=bool,
                    default=torch.cuda.is_available())
    #ap.add_argument("--is_sig",                         type=bool,  default=False)
    ap.add_argument("--ES",                             type=int,   default=50)
    #ap.add_argument("--MIN_QUERY_SUBGRAPH_SIZE",        type=int,   default=5)
    #ap.add_argument("--MAX_QUERY_SUBGRAPH_SIZE",        type=int,   default=10)
    #ap.add_argument("--MIN_CORPUS_SUBGRAPH_SIZE",       type=int,   default=11)
    #ap.add_argument("--MAX_CORPUS_SUBGRAPH_SIZE",       type=int,   default=15)
    #ap.add_argument("--MAX_GRAPH_SIZE",                 type=int,   default=0)
    ap.add_argument("--n_layers",                       type=int,   default=3)
    ap.add_argument("--conv_type",
                    type=str,   default='SAGE')
    ap.add_argument("--gt_mode",                         type=str,
                    default='qap', help="qap/glasgow")
    ap.add_argument("--mcs_mode",                        type=str,
                    default='edge', help="edge/node")
    ap.add_argument("--training_mode",
                    type=str,   default='mse', help="mse/rank")
    ap.add_argument("--method_type",
                    type=str,   default='order')
    ap.add_argument("--skip",
                    type=str,   default='learnable')
    ap.add_argument("--neuromatch_hidden_dim",          type=int,   default=10)
    ap.add_argument("--post_mp_dim",                    type=int,   default=64)
    ap.add_argument("--filters_1",
                    type=int,   default=10)
    ap.add_argument("--filters_2",
                    type=int,   default=10)
    ap.add_argument("--filters_3",
                    type=int,   default=10)
    ap.add_argument("--dropout",                        type=float, default=0)
    ap.add_argument("--COMBO",                          type=float, default=0)
    ap.add_argument("--tensor_neurons",                 type=int,   default=10)
    ap.add_argument("--transform_dim",                 type=int,   default=10)
    ap.add_argument("--bottle_neck_neurons",            type=int,   default=10)
    ap.add_argument("--bins",                           type=int,   default=16)
    ap.add_argument("--histogram",
                    type=bool,  default=False)
    ap.add_argument("--GMN_NPROPLAYERS",                type=int,   default=5)
    ap.add_argument("--MARGIN",
                    type=float, default=0.1)
    ap.add_argument("--KRON_LAMBDA",                    type=float, default=0)
    ap.add_argument("--CONVEX_KRON_LAMBDA",
                    type=float, default=1.0)
    ap.add_argument("--NOISE_FACTOR",                   type=float, default=0)
    ap.add_argument("--LP_LOSS_REG",
                    type=float, default=1.0)
    ap.add_argument("--TEMP",
                    type=float, default=0.1)
    ap.add_argument("--GOSSIP_TEMP",
                    type=float, default=1.0)
    ap.add_argument("--NITER",                          type=int,   default=20)
    ap.add_argument("--NUM_GOSSIP_ITER",                type=int,   default=15)
    ap.add_argument("--NUM_RUNS",                       type=int,   default=2)
    ap.add_argument("--BATCH_SIZE",
                    type=int,   default=128)
    ap.add_argument("--LEARNING_RATE",
                    type=float, default=0.001)
    ap.add_argument("--WEIGHT_DECAY",
                    type=float, default=5*10**-4)
    ap.add_argument("--FEAT_TYPE",                      type=str,
                    default="One", help="One/Onehot/Onehot1/Adjrow/Adjrow1/AdjOnehot")
    ap.add_argument("--CONV",                           type=str,
                    default="GCN", help="GCN/GAT/GIN/SAGE")
    ap.add_argument("--DIR_PATH",                       type=str,
                    default=".", help="path/to/datasets")
    ap.add_argument("--DATASET_NAME",                   type=str,
                    default="ptc_mm", help="TODO")
    ap.add_argument("--TASK",                           type=str,
                    default="OurMatchingSimilarity", help="TODO")

    av = ap.parse_args()

    # if "qap" in av.gt_mode:
    av.TASK = av.TASK + "_gt_mode_" + av.gt_mode
    # if av.training_mode == "rank":
    av.TASK = av.TASK + "_trMode_" + av.training_mode
    if av.FEAT_TYPE == "Adjrow" or av.FEAT_TYPE == "Adjrow1" or av.FEAT_TYPE == "AdjOnehot":
        av.TASK = av.TASK + "_" + av.FEAT_TYPE
    if av.CONV != "GCN":
        av.TASK = av.TASK + "_" + av.CONV
    av.logpath = av.logpath+"_"+av.TASK+"_"+av.DATASET_NAME+str(time.time())
    set_log(av)
    logger.info("Command line")
    logger.info('\n'.join(sys.argv[:]))

    # Print configure
    config = get_default_config()
    config['encoder']['node_hidden_sizes'] = [av.filters_3]  # [10]
    config['encoder']['node_feature_dim'] = 1
    config['encoder']['edge_feature_dim'] = 1
    config['aggregator']['node_hidden_sizes'] = [av.filters_3]  # [10]
    config['aggregator']['graph_transform_sizes'] = [av.filters_3]  # [10]
    config['aggregator']['input_size'] = [av.filters_3]  # [10]
    config['graph_matching_net']['node_state_dim'] = av.filters_3  # 10
    #config['graph_matching_net'] ['n_prop_layers'] = av.GMN_NPROPLAYERS
    config['graph_matching_net']['edge_hidden_sizes'] = [
        2*av.filters_3]  # [20]
    config['graph_matching_net']['node_hidden_sizes'] = [av.filters_3]  # [10]
    config['graph_matching_net']['n_prop_layers'] = 5
    config['graph_embedding_net']['node_state_dim'] = av.filters_3  # 10
    #config['graph_embedding_net'] ['n_prop_layers'] = av.GMN_NPROPLAYERS
    config['graph_embedding_net']['edge_hidden_sizes'] = [
        2*av.filters_3]  # [20]
    config['graph_embedding_net']['node_hidden_sizes'] = [av.filters_3]  # [10]
    config['graph_embedding_net']['n_prop_layers'] = 5

    #logger.info("av gmn_prop_param")
    # logger.info(av.GMN_NPROPLAYERS)
    #logger.info("config param")
    #logger.info(config['graph_embedding_net'] ['n_prop_layers'] )
    config['graph_embedding_net']['n_prop_layers'] = av.GMN_NPROPLAYERS
    config['graph_matching_net']['n_prop_layers'] = av.GMN_NPROPLAYERS
    #logger.info("config param")
    #logger.info(config['graph_embedding_net'] ['n_prop_layers'] )

    config['training']['batch_size'] = av.BATCH_SIZE
    #config['training']['margin']  = av.MARGIN
    config['evaluation']['batch_size'] = av.BATCH_SIZE
    config['model_type'] = "embedding"
    config['graphsim'] = {}
    config['graphsim']['conv_kernel_size'] = [10, 4, 2]
    config['graphsim']['linear_size'] = [24, 16]
    config['graphsim']['gcn_size'] = [10, 10, 10]
    config['graphsim']['conv_pool_size'] = [3, 3, 2]
    config['graphsim']['conv_out_channels'] = [2, 4, 8]
    config['graphsim']['dropout'] = av.dropout

    for (k, v) in config.items():
        logger.info("%s= %s" % (k, v))

    # Set random seeds
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.backends.cudnn.deterministic = False
    #  torch.backends.cudnn.benchmark = True

    av.dataset_stats = pickle.load(
        open('Datasets/mcs/splits/stats/%s_dataset_stats.pkl' % av.DATASET_NAME, "rb"))

    av.dataset = av.DATASET_NAME
    train(av, config)




