import numpy as np


def create_sequences(features, targets,sequence_length=20):
    X=[]
    y=[]

    for i in range(sequence_length,len(features)):
        X.append(features[i-sequence_length:i])
        y.append(targets[i])

    return np.asarray(X,dtype=np.float32),np.asarray(y,dtype=np.float32)


def load_and_sequence(path,sequence_length=20):
    data=np.load(path)

    features=data["features"]
    soh=data["soh"]

    return create_sequences(
        features,
        soh,
        sequence_length
    )