import io
import numpy as np
import sagemaker.amazon.common as smac
import pickle, gzip


def transform_input():
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

    vectors = np.array([t.tolist() for t in train_set[0]]).astype('float32')
    labels = np.where(np.array([t.tolist() for t in train_set[1]]) == 0, 1, 0).astype('float32')

    # The Amazon SageMaker implementation of Linear Learner takes recordIO-wrapped
    # protobuf, where the data we have today is a pickle-ized numpy array on disk.
    buf = io.BytesIO()
    smac.write_numpy_to_dense_tensor(buf, vectors, labels)
    buf.seek(0)

    with open('inputs.protobuf', 'wb') as f:
        f.write(buf.getbuffer())
