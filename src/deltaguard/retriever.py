import numpy as np

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if a.shape != b.shape or a.ndim != 1 or b.ndim != 1 or a.size == 0 or b.size == 0:
        raise ValueError("Input vectors must be 1D, non-empty, and of the same length.")
 

    magitude_a = np.linalg.norm(a)
    magitude_b = np.linalg.norm(b)
    normalized_a = a / magitude_a if magitude_a != 0 else a
    normalized_b = b / magitude_b if magitude_b != 0 else b

    return np.dot(normalized_a, normalized_b)

def retrieve_top_k(query_embedding, document_embeddings, k):
    if not isinstance(k, int):
        raise ValueError("k must be an integer.")
    elif k == 0:
        return np.asarray([]), np.asarray([])
    elif k < 0:
        raise ValueError("k must be a non-negative integer.")
    elif k > len(document_embeddings):
        raise ValueError("k cannot be greater than the number of document embeddings.")

    
    cosine_similarities = [cosine_similarity(query_embedding, doc_embedding) for doc_embedding in document_embeddings]
    top_k_indices = np.argpartition(cosine_similarities, -k)[-k:]
    top_k_indices_sorted = top_k_indices[np.argsort(np.array(cosine_similarities)[top_k_indices])[::-1]]

    return top_k_indices_sorted, [cosine_similarities[i] for i in top_k_indices_sorted]

 