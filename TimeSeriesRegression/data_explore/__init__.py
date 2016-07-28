import matplotlib.pyplot as plt
from matplotlib import colors



def create_fig():
    fig_number = create_fig.fig_number + 1
    plt.figure(fig_number, figsize=(20,10))

create_fig.fig_number = 0


def draw_scatterplot(data_sets, x_title, subplotnum, y_title=None, c=None):
    plt.subplot(subplotnum)
    plt.xlabel(x_title, fontsize=18)
    plt.ylabel(y_title, fontsize=18)

    #if c is None:
    #    c = range(len(data_sets))
    for i, d in enumerate(data_sets):
        point_size = 30 if len(d[0]) < 100 else 8
        if c is None:
            plt.scatter(d[0], d[1], alpha=0.5, s=point_size)
        else:
            plt.scatter(d[0], d[1], alpha=0.5, s=point_size, c=c[i])
