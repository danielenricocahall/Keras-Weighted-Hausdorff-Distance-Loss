import tensorflow as tf
from pytest import fixture

from hausdorff.hausdorff import cdist


@fixture
def tf_session():
    session = tf.Session()
    yield session
    session.close()


def test_cdist(tf_session):
    # GIVEN we have 2 identical matrices A and B with two pairs, (1, 2) and (3, 4)
    data = [[1, 2], [3, 4]]
    A = tf.convert_to_tensor(data, dtype=tf.float32)
    B = tf.convert_to_tensor(data, dtype=tf.float32)
    # WHEN we run compute the pairwise distance
    d = tf_session.run(cdist(A, B))
    # THEN we expect the result to be a diagonal matrix with 0s along the right diagonal and sqrt(8) along the left
    assert d.tolist() == [[0.0, 2.8284270763397217],
                          [2.8284270763397217, 0.0]]


