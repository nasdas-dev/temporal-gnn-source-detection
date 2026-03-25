# --------------------------------------------------------------------------
# Functions to import empirical networks or to generate synthetic networks
# --------------------------------------------------------------------------

import torch
import networkx as nx
from random import getrandbits
from subprocess import Popen, PIPE, STDOUT

def import_nwk (fname):
    raise NotImplementedError

def create_nwk (args):
    # Extract graph model from config
    model = args["graph"]
    # Match case statement to generate graph
    match model:
        # ============================================
        # ERDÖS-RÉNYI GRAPH
        case "ER":
            # Initialize condition for while condition
            last_node_is_iso = True
            # This is necessary because the simulation code expects the node
            # with the largest node ID NOT to be an isolate.
            while (last_node_is_iso):
                # Create graph
                G = nx.erdos_renyi_graph(args["n"], args["p"])
                # Check whether node with largest ID is in list of isolates or not.
                if (G.number_of_nodes() - 1 not in list(nx.isolates(G))):
                    # If it is not, then set condition to False so that the while loop breaks.
                    last_node_is_iso = False
            # Return the graph
            return G
        # ============================================
        # BARABASI-ALBERT GRAPH
        case "BA":
            # Initialize condition for while condition
            last_node_is_iso = True
            # This is necessary because the simulation code expects the node
            # with the largest node ID NOT to be an isolate.
            while (last_node_is_iso):
                # Create graph
                G = nx.barabasi_albert_graph(args["n"], args["m"])
                # Check whether node with largest ID is in list of isolates or not.
                if (G.number_of_nodes() - 1 not in list(nx.isolates(G))):
                    # If it is not, then set condition to False so that the while loop breaks.
                    last_node_is_iso = False
            # Return the graph
            return G
        # ============================================
        # EMPIRICAL GRAPHS
        case "CONFERENCE" | "DOLPHIN" | "FRATERNITY" | "WORKPLACE" | "HIGHSCHOOL" | "HIGHSCHOOL2010" | "HIGHSCHOOL2013" | "ICELAND" | "KARATE" | "POWERGRID" | "AIRTRAFFIC" | "SMALL_ER":
            # Open the file with edge list
            with open('nwk/' + model.lower() + '.csv') as f:
                lines = f.readlines()
            # Create the edge list
            elist = [tuple(map(int, line.strip().split())) for line in lines]
            # Find the number of nodes based on edgelist
            n_nodes = max(elist, key = lambda i : i)[0] + 1
            # Load weights if there are any
            if (args["weights"]):
                with open('nwk/' + model.lower() + '_weights.csv') as f:
                    lines = f.readlines()
                # Create the tensor
                weights = [int(line.strip()) for line in lines]
                # Add weights to edge list
                elist = [t + (n,) for t, n in zip(elist, weights)]
            # Empty graph
            G = nx.Graph()
            # Add nodes
            G.add_nodes_from([v for v in range(n_nodes)])
            # Add edges to graph based on whether there are weights or not
            if (args["weights"]):
                G.add_weighted_edges_from(elist)
            else:
                G.add_edges_from(elist)
            # Terminate if last node is isolate
            if (G.number_of_nodes() - 1 in list(nx.isolates(G))):
                raise Exception("Node with highest ID is isolate.")
            # Return the graph
            return G
        # ============================================
        case _:
            raise ValueError('No matching graph model.')
