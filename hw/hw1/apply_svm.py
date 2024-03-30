import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm  # type: ignore
from sklearn.inspection import DecisionBoundaryDisplay  # type: ignore


def main() -> None:
    test_set = pd.read_csv("testSet.txt", sep="\t", names=["x", "y", "label"])
    coordinates = np.asarray((test_set["x"], test_set["y"])).T
    labels = test_set["label"]

    model = svm.LinearSVC(loss="hinge", dual="auto").fit(coordinates, labels)

    one_set = test_set[test_set['label'] ==1]
    minus_one_set = test_set[test_set['label'] ==-1]

    plt.scatter(
        one_set["x"],
        one_set["y"],
        c='red',
        marker="x",
        label="Sample Points with Label 1",
    )
    plt.scatter(
        minus_one_set["x"],
        minus_one_set["y"],
        c='blue',
        marker="x",
        label="Sample Points with Label -1",
    )

    DecisionBoundaryDisplay.from_estimator(
        model,
        coordinates,
        plot_method="contour",
        colors="k",
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=["--", "-", "--"],
        ax=plt.gca(),
    )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Sample Points and SVM Decision Line")
    plt.legend()
    plt.show(block=True)


main() if __name__ == "__main__" else None