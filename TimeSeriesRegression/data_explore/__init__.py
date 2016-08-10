import matplotlib.pyplot as plt
from matplotlib import colors



def create_fig():
    fig_number = create_fig.fig_number + 1
    plt.figure(fig_number, figsize=(20,10))


create_fig.fig_number = 0


def draw_simple_scatterplot(x,y , x_title, subplotnum):
    print x.shape,y.shape , x_title, subplotnum
    plt.subplot(subplotnum)
    plt.xlabel(x_title, fontsize=18)
    plt.scatter(x, y, alpha=0.3)


def draw_scatterplot_one(x,y , x_title, subplotnum, y_title=None, c=None):
    draw_scatterplot([(x,y)], x_title, subplotnum, y_title=y_title, c=c)


def draw_scatterplot(data_sets, x_title, subplotnum, y_title=None, c=None):
    plt.subplot(subplotnum)
    plt.xlabel(x_title, fontsize=18)
    plt.ylabel(y_title, fontsize=18)

    #if c is None:
    #    c = range(len(data_sets))
    for i, d in enumerate(data_sets):
        point_size = 30 if len(d[0]) < 100 else 8
        if c is None:
            plt.scatter(d[0], d[1], alpha=0.3, s=point_size)
        else:
            plt.scatter(d[0], d[1], alpha=0.3, s=point_size, c=c[i])


def draw_error_bars(pltnum, x, mean, ulimit, llimit, x_label, y_label=None, do_log=True):
    ax = plt.subplot(pltnum)
    plt.xlabel(x_label, fontsize=18)
    if y_label is not None:
        plt.ylabel(y_label, fontsize=18)
    if do_log:
        ax.set_yscale('log')
    plt.errorbar(x, mean, yerr=[llimit, ulimit],
                 ms=5, mew=2, fmt='--o')
