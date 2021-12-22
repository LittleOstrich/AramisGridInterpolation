import os

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from helpers.general_tools import getColours


def createScatterPlot(data, title="", dst=None, save=False, show=True, dpi=500, colours=None, labels=None, alpha=0.5):
    numFeatures = 1
    if len(data.shape) > 1:
        numFeatures = data.shape[1]

    if colours is None:
        colours = 'b'
    else:
        pass

    if numFeatures == 3:
        scatterPlot3D(data=data, title=title, dst=dst, save=save, show=show, dpi=dpi, colours=colours, labels=labels,
                      alpha=alpha)
    elif numFeatures == 2:
        scatterPlot2D(data=data, title=title, dst=dst, save=save, show=show, dpi=dpi, colours=colours, labels=labels,
                      alpha=alpha)
    elif numFeatures == 1:
        scatterPlot1D(data=data, title=title, dst=dst, save=save, show=show, dpi=dpi, colours=colours, labels=labels,
                      alpha=alpha)


def nearestNeighbourscatterPlot(data, dst=None, title="", save=True, show=False,
                                dpi=500, colours=None, labels=None,
                                alpha=0.5):
    numFeatures = 1
    if len(data.shape) > 1:
        numFeatures = data.shape[1]

    if colours is None:
        colours = 'b'
    else:
        pass

    if numFeatures == 3:
        nearestNeighbourscatterPlot3D(data=data, dst=dst, title=title, save=save, show=show,
                                      dpi=dpi, colours=colours,
                                      labels=labels,
                                      alpha=alpha)


def scatterPlot2D(data, title="", dst=None, save=True, show=False, dpi=500, colours='b', labels=None, alpha=0.03):
    fig = plt.figure()
    ax = fig.subplots()
    plt.title(title)

    # Generate the values
    x_vals = data[:, 0:1]
    y_vals = data[:, 1:2]

    # Plot the values
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    ax.scatter(x_vals, y_vals, c=colours, marker='o', label=labels, alpha=alpha)
    ax.legend()

    if save:
        plt.title(title)
        if dst is not None:
            os.makedirs(dst, exist_ok=True)
            dst = dst + os.sep + title
        plt.savefig(dst, dpi=dpi)
    if show:
        plt.show(block=True)
    plt.close(fig)


def scatterPlot3D(data, save=True, dst=None, show=False, title="", dpi=500, colours='b', alpha=0.03, labels=None):
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    plt.title(title)
    # Generate the values
    x_vals = data[:, 0:1]
    y_vals = data[:, 1:2]
    z_vals = data[:, 2:3]

    # Plot the values
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    ax.scatter(x_vals, y_vals, z_vals, c=colours, marker='o', alpha=alpha)

    if save:
        plt.title(title)
        if dst is not None:
            os.makedirs(dst, exist_ok=True)
            dst = dst + os.sep + title
        plt.savefig(dst, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show(block=True)
    plt.close('all')


def nearestNeighbourscatterPlot3D(data, dst=None, save=False, show=True, title="", dpi=500,
                                  colours='b', alpha=0.03, labels=None):
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    plt.title(title)
    # Generate the values
    x_vals = data[:, 0:1]
    y_vals = data[:, 1:2]
    z_vals = data[:, 2:3]

    # Plot the values
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    ax.scatter(x_vals, y_vals, z_vals, c=colours, marker='o', alpha=alpha)

    uniques, counts = np.unique(labels, return_counts=True)
    numC = len(np.unique(uniques))
    cSpace = getColours(numColours=numC)
    ax2 = fig.add_subplot(122)
    ax2.axis("off")
    labels = ["isNN of " + str(_ + 1) for _ in range(numC)]
    for _ in range(numC):
        ax2.plot([], [], 'o', c=cSpace[_], label=labels[_])
    ax2.legend()

    if save:
        if dst is not None:
            os.makedirs(dst, exist_ok=True)
            title = dst + os.sep + title
        plt.savefig(title, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show(block=True)
    plt.close('all')


def scatterPlot1D(data, save=False, show=True, title="", dst=None, dpi=500, colours='b', labels="", alpha=0.03):
    fig = plt.figure()
    ax = plt.axes()
    plt.title(title)
    plt.plot(data, np.zeros_like(data), ".", alpha=0.03)

    if save:
        if dst is not None:
            os.makedirs(dst, exist_ok=True)
            title = dst + os.sep + title
        plt.savefig(title, dpi=dpi)
    if show:
        plt.show()
    plt.close('all')


def plotHistogram(data, title="", dst=None, save=True, show=False, dpi=500, bins=10):
    fig = plt.figure()
    plt.hist(data, bins=bins)
    if save:
        if dst is not None:
            os.makedirs(dst, exist_ok=True)
            title = dst + os.sep + title
        plt.savefig(title, dpi=dpi)
    if show:
        plt.show(block=True)
    plt.close(fig)
