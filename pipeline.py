import cohere
import pandas as pd
from multiprocessing import Pool
from generate_examples import generate_examples
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
import umap


co = cohere.Client('QFTD2O4Y78t2iGFuy1bBYuqS4d3NrNi84KUww6wW')

with open('./prompt.txt') as f:
    prompt = f.read()


def get_examples_multiprocess(descriptions, num_generations, p=prompt):
    """
    This function runs an async generate request for each description that is passed in
    The goal is to run until num_generations is met or a timeout
    descriptions = list
    num_generations = int
    """
    func_inputs = [(p, num_generations, d) for d in descriptions]
    pool = Pool(len(descriptions))
    des_e = pool.starmap_async(generate_examples, func_inputs)
    des_e.wait()
    output = des_e.get()  
    return output


def prune(description_examples, route_embeds):
    """
    this will attempt to remove outliers from each generated cluster
    """
    sims = cosine_similarity(route_embeds)

    examples_processed = 0
    outliers = []
    pruned_embeds = []
    pruned_routes = []
    pruned_examples = []

    # this is looping through each route "cluster"
    for route_num, route in enumerate(description_examples):
        e_count = len(route['examples'])
        route_start = examples_processed
        route_end = examples_processed + e_count
        group_embed = sims[route_start:route_end,route_start:route_end]
        group_mean = group_embed.mean()
        group_std = group_embed.std()

        for ex_num, ex in enumerate(route['examples']):
            line_embeds = sims[examples_processed+ex_num][route_start:route_end]
            average_sim = line_embeds.mean()
            if average_sim < group_mean - group_std:
                outliers.append((description_examples[route_num]['description'], description_examples[route_num]['examples'][ex_num]))
            else:
                pruned_embeds.append(route_embeds[examples_processed+ex_num])
                pruned_routes.append(route['route_name'])
                pruned_examples.append(ex)

        examples_processed += e_count

    return pruned_routes, pruned_embeds, pruned_examples


def search(search_phrase, embeddings, route_names, n_comparisons=10):
    """
    this is a basic search function over an embedding space.
    TODO check if pinecone will work
    """
    search_embed = co.embed(texts=[search_phrase] , model='small').embeddings
    sims = cosine_similarity(X=search_embed, Y=embeddings)

    
    # get the top n results
    top_idx = np.argsort(sims)[0][-n_comparisons:]
    top_values = [route_names[i] for i in top_idx]

    # get the most common value
    occurence_count = Counter(top_values)
    most_common_prediction = occurence_count.most_common(1)[0][0]

    return most_common_prediction, search_embed

def umap_reduce(embeds, n_neighbors=10):
    print('#############\nreducing\n#############')
    a = np.array(embeds)
    reducer = umap.UMAP(n_neighbors=n_neighbors)
    umap_embeds = reducer.fit_transform(a)

    df = pd.DataFrame()
    df['x'] = umap_embeds[:, 0]
    df['y'] = umap_embeds[:, 1]
    return df


def generate_example_embeddings(route_names, descriptions):
    """
    This will run the main pipeline that is triggered by the front end
    It will run the pipeline for any descriptions not currently in session storage
    
    Returns a list of route names and embeddings
    """
    print('generating examples....')
    res = get_examples_multiprocess(descriptions, p=prompt, num_generations=15)
    description_examples = [{'route_name': route_names[i], 'description': descriptions[i], 'examples': r['results']} for i, r in enumerate(res)]
    generated_examples = [[i['route_name'], i['description'], e] for i in description_examples for e in i['examples']]
    route_embeds = co.embed(texts=[i[2] for i in generated_examples] , model='small').embeddings
    pruned_routes, pruned_embeds, pruned_examples = prune(description_examples, route_embeds)

    return pruned_routes, pruned_embeds, pruned_examples

