import gudhi
import numpy as np
import matplotlib.pyplot as plt
import json
import gensim
import networkx as nx
#from gensim.models import Word2Vec
from node2vec import Node2Vec
from watsonx import ontology_generator_model
import re

#import open3d as o3d
from sklearn.manifold import TSNE

def build_ontology(actions: list[str]):
    policy = f'''
        [INST] Extract connected subject, object, predicate triples from the action provided. 
        
        You should reply in the following format:

        Concept1: <concept>
        Concept2: <concept>

        Concept1: <concept>
        Concept2: <concept>
        [/INST]

        [EXAMPLE]
        For example:

        Concept1: climate_change
        Concept2: paris_agreement

        Concept1: deforestation
        Concept2: carbon_capture
        
        [CONDITION]
        If the name of the concept is two words use the snake case format, for example: climate_awareness or financial_planning.
        Consider how pairs are related and show these as concept pairs.
        Try to generate at least 50 pairs.

        The actions are: {actions}
        '''

    results = ontology_generator_model.generate(prompt=policy)
    generated_text = results['results'][0]['generated_text']

    # Split the text into lines
    lines = generated_text.strip().split('\n')

    # Initialize an empty list to store the pairs
    concept_pairs = []

    # Iterate through the lines and extract adjacent Concept1 and Concept2 pairs
    i = 0
    while i < len(lines) - 1:  # We check up to the second-to-last line
        line1 = lines[i].strip()
        line2 = lines[i + 1].strip()

        if line1.startswith("Concept1:") and line2.startswith("Concept2:"):
            concept1 = line1.split(":", 1)[1].strip()
            concept2 = line2.split(":", 1)[1].strip()
            concept_pairs.append([concept1, concept2])
            i += 2  # Skip the next line as we've already processed it
        else:
            i += 1  # Move to the next line

    return concept_pairs

def construct_graph(ontology):

    # reconstruct the graph
    G = nx.Graph()
    #concepts = [[item['concept1'], item['concept2']] for item in ontology]
    for pair in ontology:
        for node in pair:
            if node not in G.nodes:
                G.add_node(node)

        G.add_edge(pair[0], pair[1])
    #pos = nx.spring_layout(G)  # You can choose different layout algorithms based on your preference
    #nx.draw(G, pos, with_labels=True, font_size=8, font_color='black', node_size=500, node_color='lightblue', edge_color='gray', linewidths=0.5)
    #plt.show()
    return G
    
def embed_graph(G:nx.Graph):
    # walk the graph and embed
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    embeddings = model.wv.vectors
    #print("embeddings",embeddings)
    # Apply t-SNE for dimensionality reduction to 3D
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_reduced_dim = tsne.fit_transform(embeddings)
    #print("embeddings reduced",embeddings_reduced_dim)
    return embeddings_reduced_dim

#def create_point_cloud(embeddings_3d):
    # Create a point cloud from the 3D embeddings
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(embeddings_3d)
    # o3d.visualization.draw_geometries([pcd])

def create_simplex(embeddings):
    # Create a Rips complex from the point cloud
    point_cloud = embeddings
    rips_complex = gudhi.RipsComplex(points=point_cloud)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    return simplex_tree,point_cloud

def compute_homology(simplex_tree):
    # Compute persistent homology
    #persistence = simplex_tree.persistence()
    simplex_tree.compute_persistence()
    #persistence = simplex_tree.persistence_intervals_in_dimension(1)
    #print("persistence",persistence)

    persistence_pairs = simplex_tree.persistence_pairs()
    #print("persistence_pairs",persistence_pairs)

    simplices_list = list(simplex_tree.get_simplices())
    #print("simplicies_list",simplices_list)

    # get the birth simplicies with the top_n largest filtraction values
    birth_simplices_with_largest_filtration = []
    for birth_simplex_index,death_simplicies_indices in persistence_pairs:
        birth_simplex = simplices_list[birth_simplex_index[0]]
        birth_simplices_with_largest_filtration.append([birth_simplex_index[0],birth_simplex[0],birth_simplex[1]])
    #print(birth_simplices_with_largest_filtration)
    sorted_data = sorted(birth_simplices_with_largest_filtration, key=lambda x: x[2], reverse=True)[10:]
    #print(sorted_data)

    # I want to order the entries in the persistent pairs by the order of the birth simplicies 
    birth_index_to_filtration = {entry[0]: entry[2] for entry in birth_simplices_with_largest_filtration}
    #print("index",birth_index_to_filtration)

    # Sort the persistent pairs based on the highest filtration value of the birth simplex
    sorted_persistence_pairs = sorted(persistence_pairs, key=lambda pair: max(birth_index_to_filtration[entry] for entry in pair[0]), reverse=True)
    #print("sorted list",sorted_persistence_pairs)

    # get the top_n most persistent pairs based on the highest filtration value of the birth simplex
    top_n_sorted_persistence_pairs = sorted_persistence_pairs[:10]
    #print("top_n sorted persistent pairs", top_n_sorted_persistence_pairs)

    # Extract the death simplices associated with filling of the hole (the boundary simplicies)
    holes = []
    for birth_simplex_index,death_simplicies_indices in top_n_sorted_persistence_pairs:
        boundary_simplicies = []
        for death_simplex in death_simplicies_indices:
            death_simplices = simplices_list[death_simplex]
            #print("index:",death_simplex,"verticies",death_simplices[0],"filtration",death_simplices[1])
            #print("death_simplicies",death_simplices[0])
            boundary_simplicies.append(death_simplices[0])
        holes.append(boundary_simplicies)
    #print(holes)

    #gudhi.plot_persistence_diagram(persistence)
    #plt.show()
    #print(persistence)
    return holes

def get_boundary_words(boundary_simplicies,point_cloud,low_dim_node_embedding_mapping):
    # Map back to the words associated in the point cloud
    boundary_words = []
    for hole in boundary_simplicies:
        words = []
        for simplicies in hole:
            for index in simplicies:
                node_name=next((key for key, value in low_dim_node_embedding_mapping.items() if np.array_equal(value, point_cloud[index])), None)
                if node_name not in words:
                    words.append(node_name)
                #print(index,pointcloud,node_name)
        boundary_words.append(words)

    boundary_words_json = json.dumps(boundary_words)
    return boundary_words_json