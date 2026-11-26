import pytest  
from deltaguard.retriever import cosine_similarity, retrieve_top_k


def test_cosine_similarity():
    a = [1, 0]
    b = [0, 1]
    assert cosine_similarity(a, b) == 0.0

    a = [1, 0]
    b = [1, 0]
    assert cosine_similarity(a, b) == 1.0

    a = [1, 1]
    b = [1, 1]
    assert cosine_similarity(a, b) == pytest.approx(1.0)

    a = [1, 2]
    b = [2, 4]
    assert cosine_similarity(a, b) == pytest.approx(1.0)

def test_cosine_similarity_with_zero_vector():
    a = [0, 0]
    b = [1, 1]
    assert cosine_similarity(a, b) == 0.0

    a = [1, 1]
    b = [0, 0]
    assert cosine_similarity(a, b) == 0.0

    a = [0, 0]
    b = [0, 0]
    assert cosine_similarity(a, b) == 0.0

def test_cosine_similarity_with_negative_values():
    a = [-1, -1]
    b = [1, 1]
    assert cosine_similarity(a, b) == pytest.approx(-1.0)

    a = [-1, 0]
    b = [0, 1]
    assert cosine_similarity(a, b) == 0.0

    a = [-1, -1]
    b = [-1, -1]
    assert cosine_similarity(a, b) == pytest.approx(1.0)

def test_cosine_similarity_with_different_lengths():
    a = [1, 0]
    b = [1, 0, 0]
    with pytest.raises(ValueError):
        cosine_similarity(a, b)

    a = [1, 0, 0]
    b = [1, 0]
    with pytest.raises(ValueError):
        cosine_similarity(a, b)

def test_cosine_similarity_with_two_dimensional_arrays():
    a = [[1, 0], [0, 1]]
    b = [[1, 0], [0, 1]]
    with pytest.raises(ValueError):
        cosine_similarity(a, b)

def test_cosine_similarity_with_empty_arrays():
    a = []
    b = []
    with pytest.raises(ValueError):
        cosine_similarity(a, b)
 
def test_retrieve_top_k():
    query_embedding = [1, 0]
    document_embeddings = [[1, 0], [0, 1], [1, 1]]
    k = 2
    top_k_indices, top_k_similarities = retrieve_top_k(query_embedding, document_embeddings, k)
    assert list(top_k_indices) == [0, 2]
    assert list(top_k_similarities) == [1.0, pytest.approx(0.7071067811865475)]

def test_retrieve_top_k_with_k_zero():
    query_embedding = [1, 0]
    document_embeddings = [[1, 0], [0, 1], [1, 1]]
    k = 0
    top_k_indices, top_k_similarities = retrieve_top_k(query_embedding, document_embeddings, k)
    assert list(top_k_indices) == []
    assert list(top_k_similarities) == []

def test_retrieve_top_k_with_k_negative():
    query_embedding = [1, 0]
    document_embeddings = [[1, 0], [0, 1], [1, 1]]
    k = -1
    with pytest.raises(ValueError):
        retrieve_top_k(query_embedding, document_embeddings, k)

def test_retrieve_top_k_with_k_greater_than_document_embeddings():
    query_embedding = [1, 0]
    document_embeddings = [[1, 0], [0, 1], [1, 1]]
    k = 4
    with pytest.raises(ValueError):
        retrieve_top_k(query_embedding, document_embeddings, k)

def test_retrieve_top_k_with_k_equal_to_document_embeddings():
    query_embedding = [1, 0]
    document_embeddings = [[1, 0], [0, 1], [1, 1]]
    k = 3
    top_k_indices, top_k_similarities = retrieve_top_k(query_embedding, document_embeddings, k)
    assert list(top_k_indices) == [0, 2, 1]
    assert list(top_k_similarities) == [1.0, pytest.approx(0.7071067811865475), 0.0]

def test_retrieve_top_k_with_k_as_float():
    query_embedding = [1, 0]
    document_embeddings = [[1, 0], [0, 1], [1, 1]]
    k = 2.7
    with pytest.raises(ValueError):
        retrieve_top_k(query_embedding, document_embeddings, k)