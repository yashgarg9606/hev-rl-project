import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def battery_aware_split(X,y,batteries,test_size=0.25,random_state=42):
    batteries=np.asarray(batteries)

    splitter=GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    train_idx,test_idx=next(
        splitter.split(
            X,
            y,
            groups=batteries
        )
    )

    return (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx],
        batteries[train_idx],
        batteries[test_idx]
    )