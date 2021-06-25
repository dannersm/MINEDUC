"""The functions in this module calculate several CM from an edge file

functions:
    * get_page_rank_from_edge_file - Calculates the PageRank of the keywords
    * get_diffusion_centrality_from_edge_file - Calculates the
                        Diffusion Centrality of the keywords
    * get_katz_centrality_from_edge_file - Calculates the
                        Katz-Bonacich Centrality of the keywords
    * export_to_file - exports a CM dict to a .txt file
    * cm_file_to_dict - reads a CM .txt into a dict

"""
import networkx as nx
import numpy as np
from nltk import SnowballStemmer
from sklearn.preprocessing import normalize


def get_page_rank_from_edge_file(edges_file, sep=' ', stochastic=False):
    """Calculates the PageRank from the weighted edgelist

    :type edges_file: str
    :param edges_file: path for the weighted edge list
    :rtype: dict
    :return: words as keys and pagerank as value
    """
    G = nx.read_weighted_edgelist(edges_file, create_using=nx.DiGraph(), delimiter=sep)
    G = nx.stochastic_graph(G) if stochastic else G
    pr = nx.pagerank(G)
    return pr


def get_diffusion_centrality_from_edge_file(edges_file, L, delta, sep=' ', stochastic=False):
    """Calculates the Diffusion Centrality from the weighted edgelist

    :type edges_file: str
    :param edges_file: path for the weighted edge list
    :type L: int
    :param L: number of layers for calculating the diffusion centrality
    :type delta: float
    :param delta: attenuation coefficient for the diffusion centrality
    :rtype: dict
    :return: words as keys and diffusion centrality as value
    """
    G = nx.read_weighted_edgelist(edges_file, create_using=nx.DiGraph(), delimiter=sep)
    G = nx.stochastic_graph(G) if stochastic else G

    V = list(G.nodes)
    A = nx.adjacency_matrix(G).todense()

    centrality = np.zeros(len(V))
    while i <= L:
        A_i = (delta ** i) * np.linalg.matrix_power(A, i)
        centrality = centrality + A_i.sum(axis=0)
        i = i + 1

    centrality = centrality / centrality.sum(axis=1)

    res = {}
    for i in range(len(V)):
        res[V[i]] = centrality[0, i]

    return res


def get_katz_centrality_from_edge_file(edges_file, delta, sep=' ', stochastic=False):
    """Calculates the Katz-Bonacich Centrality from the weighted edgelist

    :type edges_file: str
    :param edges_file: path for the weighted edge list
    :type delta: float
    :param delta: attenuation coefficient for the katz centrality
    :rtype: dict
    :return: words as keys and katz centrality as value
    """
    G = nx.read_weighted_edgelist(edges_file, create_using=nx.DiGraph(), delimiter=sep)
    G = nx.stochastic_graph(G) if stochastic else G

    V = list(G.nodes)

    A = nx.adjacency_matrix(G).todense()

    I = np.identity(len(V))

    aux = I - delta * A.transpose()
    aux = np.linalg.inv(aux) - I

    centrality = aux.sum(axis=1)
    centrality = centrality / centrality.sum(axis=0)
    res = {}
    for i in range(len(V)):
        res[V[i]] = centrality[i, 0]

    return res


def get_closeness_centrality_from_edge_file(edges_file, sep=' ', stochastic=False):
    """calculates current flow closeness centrality from edge file

    :type edges_file: str
    :param edges_file: path for the weighted edge list
    :rtype: dict
    :return: words as keys and current flow closeness centrality as value
    """
    G = nx.read_weighted_edgelist(edges_file, create_using=nx.DiGraph(), delimiter=sep)
    G = nx.stochastic_graph(G) if stochastic else G
    return nx.current_flow_closeness_centrality(G, weight='weight')


def get_betweenness_centrality_from_edge_file(edges_file, sep=' ', stochastic=False):
    """calculates current flow betweenness centrality from edge file

    :type edges_file: str
    :param edges_file: path for the weighted edge list
    :rtype: dict
    :return: words as keys and current flow betweenness centrality as value
    """
    G = nx.read_weighted_edgelist(edges_file, create_using=nx.DiGraph(), delimiter=sep)
    G = nx.stochastic_graph(G) if stochastic else G
    return nx.current_flow_betweenness_centrality(G, weight='weight')


def export_to_file(cm_dict, keywords, file_name):
    """exports a centrality measure dictionary to a txt file

    :type cm_dict: dict
    :param cm_dict: dictionary with words as keys and their centrality as value
    :type keywords: list
    :param keywords: keywords to be exported
    :type file_name: str
    :param file_name: path for the output
    """
    # exporting to file
    stemmer = SnowballStemmer("spanish")
    file = open(file_name, 'w', encoding='utf-8')

    for word in keywords:
        key = stemmer.stem(word)
        file.write(word + " " + str(cm_dict.get(key, 0)) + "\n")

    file.close()


def cm_file_to_dict(cm_file, sep=' '):
    """Reads a CM file and turns it into a dictionary

    :type cm_file: str
    :param cm_file: path for the CM file
    :rtype: dict
    :return: dictionary with words as keys and their centrality as value
    """
    file = open(cm_file, 'r', encoding='utf-8')
    key_dict = {}

    for line in file:
        aux = line.split(sep)
        key = ''
        for i in range(len(aux) - 1):
            key = key + ' ' + aux[i]
        key = key[1:]
        key_dict[key] = float(aux[len(aux) - 1])

    file.close()

    return key_dict

