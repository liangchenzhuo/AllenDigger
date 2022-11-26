import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

from Demo.accessdata import data_slice, anno_slice,  get_data
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.pyplot import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scanpy as sc
import anndata as ad


def plot_anno(time: str,
              section: str,
              section_id: int,
              path=None,
              annotation=True):

    '''
    Plot a slice of annotation data and show.
    :param path: None or a path string of a directory to save the image
    :param time: the developing period of mouse data ['E11pt5','E13pt5','E15pt5','E18pt5','P4','P14','P28','P56']
    :param section: the direction of slicing 'sag' for sagittal,'hor' for horizontal,'cor' for coronal
    :param section_id: No. section number
    :param annotation: bool, True: label the annotation by different color defined by ABA
    :return: None
    example: plot_anno(time='P56', section='Coronal', section_id=20, path='data/figure')
    '''

    # slice the annotation data
    maxx, maxy, anno = anno_slice(time=time, section=section, section_id=section_id)

    # get the pixel coordinates and labels
    x = anno.iloc[:, 0].tolist()
    y = anno.iloc[:, 1].tolist()
    # label = anno['label'].tolist()
    color = anno['color'].tolist()
    l = []
    for c in color:
        l.append('#'+c)

    # plot
    fig = plt.figure(figsize=(15,15))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    x_major_locator = MultipleLocator(1)
    y_major_locator = MultipleLocator(1)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.axis('equal')
    if annotation:
        # ax.set_title(time + ' ' + section + ' No.' + str(section_id))
        ax.scatter(x, y, s=35, c=l, marker='s')
    else:
        ax.scatter(x, y, s=35, c='grey', marker='s')
    frame = plt.gca()
    frame.invert_yaxis()
    frame.axes.get_yaxis().set_visible(False)
    frame.axes.get_xaxis().set_visible(False)
    if path:
        plt.savefig(path + '/' + time + '_' + section + '_No' + str(section_id) + '.png', dpi=300, transparent=True,
                    bbox_inches='tight')
    plt.show()
    return None


def plot2D_anno(time, section, section_id, anno_level, legend=True):
    x, y, adata = data_slice(time=time, section=section, section_id=section_id)
    x = adata.obsm['spatial'].iloc[:, 0]
    y = adata.obsm['spatial'].iloc[:, 1]
    label = adata.obs[anno_level]
    labely = label.astype('category')
    labelx = labely.cat.codes
    num = len(set(labelx))
    cmap = plt.get_cmap("hsv", num)
    ax = plt.subplot()
    ax.scatter(x, y, s=100, c=cmap(labelx), marker='s')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.axis('equal')
    ax.set_title(time + ' ' + section + ' No.' + str(section_id))
    frame = plt.gca()
    frame.invert_yaxis()
    frame.axes.get_yaxis().set_visible(False)
    frame.axes.get_xaxis().set_visible(False)
    if legend:
        patchlist = []
        for m in range(num):
            label_ = labely.cat.categories[m]
            patch = mpatches.Patch(color=cmap(m), label=label_)
            patchlist.append(patch)
        ax.legend(handles=patchlist, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()


def plot3D_anno(time, anno_level):
    adata = get_data(time=time)
    size = {'E11pt5': [70, 75, 40],
            'E13pt5': [89, 109, 69],
            'E15pt5': [94, 132, 65],
            'E18pt8': [67, 43, 40],
            'P4': [77, 43, 50],
            'P14': [68, 40, 50],
            'P28': [73, 41, 53],
            'P56': [67, 41, 58],
            'Adult': [67, 41, 58]
    }
    x = adata.obs['x'].tolist()
    y = adata.obs['y'].tolist()
    z = adata.obs['z'].tolist()
    array = np.zeros(size[time])
    size_c = size[time]
    size_c.append(4)
    color = np.zeros(size_c)
    label = adata.obs[anno_level]
    labely = label.astype('category')
    labelx = labely.cat.codes
    num = len(set(labelx))
    cmap = plt.get_cmap("coolwarm", num)
    for (i, j, k, l) in zip(x, y, z, labelx):
        array[i][j][k] = 1
        color[i][j][k] = np.asarray(cmap(l))
    fig = plt.figure()
    ax = plt.subplot(projection='3d')
    ax.voxels(array, facecolors=color, edgecolor=None, alpha=0.3)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(-90, -90)
    ax.set_title(time + ' Mouse ' + anno_level)
    ax.axis('equal')
    patchlist = []
    for m in range(num):
        label_ = labely.cat.categories[m]
        patch = mpatches.Patch(color=cmap(m), label=label_)
        patchlist.append(patch)
    ax.legend(handles=patchlist, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('./Adult_3D.png', dpi=300, bbox_inches='tight', transparent=True)

    plt.show()
    return None


def plot_anno_legend(time, section, section_id, path=None):

    '''
    Plot the legend of annotation slice and show.
    :param time: the developing period of mouse data ['E11pt5','E13pt5','E15pt5','E18pt5','P4','P14','P28','P56']
    :param section: the direction of slicing 'sag' for sagittal,'hor' for horizontal,'cor' for coronal
    :param section_id: No. section number
    :param path: None or a path string of a directory to save the image
    :return: None
    example: plot_anno_legend(time='P56', section='Coronal', section_id=20, path='data/figure')
    '''

    maxx, maxy, anno = anno_slice(time=time, section=section, section_id=section_id)
    color = anno['color'].tolist()
    patchlist = []
    for color_ in list(set(color)):
        anno_ = anno[anno['color'] == color_]
        label_ = list(set(anno_['label'].tolist()))
        patch = mpatches.Patch(color='#' + color_, label=label_)
        patchlist.append(patch)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.legend(handles=patchlist, loc='center')
    frame = plt.gca()
    frame.axes.get_yaxis().set_visible(False)
    frame.axes.get_xaxis().set_visible(False)
    plt.show()
    if path:
        plt.savefig(path + '/' + time + '_' + section + '_No' + str(section_id) + '_legend.png',
                    dpi=300, transparent=True)
    return None


def plot_expression(gene_name, time, section, section_id, annotation=False, anno_level='custom_4'):

    '''
    Plot a gene expression on one section
    :param gene_name:str, the gene symbol of target gene
    :param time: str, the developing period of mouse data ['E11pt5','E13pt5','E15pt5','E18pt5','P4','P14','P28','P56','Adult']
    :param section: the direction of slicing 'sag' for sagittal,'hor' for horizontal,'cor' for coronal
    :param section_id: No. section number
    :param annotation: bool, whether plot the structure annotation
    :param anno_level: str, the annotation level from level_1 to level_10
    :return:
    example: plot_expression(gene_name'PITX1',time='P56',section='Coronal',section_id=20)
    '''

    maxx, maxy, adata = data_slice(time=time, section=section, section_id=section_id)
    x = adata.obsm['spatial'].iloc[:, 0]
    y = adata.obsm['spatial'].iloc[:, 1]
    X = adata[:, adata.var['gene'] == gene_name].X
    fig = plt.figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot()
    ax.set_title(time + ' ' + section + ' No.' + str(section_id) + ' gene:' + gene_name)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    x_major_locator = MultipleLocator(1)
    y_major_locator = MultipleLocator(1)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.axis('equal')
    ax.set_xlim(xmin=-1, xmax=maxx)
    ax.set_ylim(ymin=-1, ymax=maxy)
    frame = plt.gca()
    frame.invert_yaxis()
    frame.axes.get_yaxis().set_visible(False)
    frame.axes.get_xaxis().set_visible(False)
    if annotation:
        labels = adata.obs[anno_level]
        label = labels.astype('category')
        labelx = label.cat.codes
        ax.scatter(x, y, c=labelx, marker='s')
    ax.scatter(x, y, c='grey', marker='s')
    scatter = ax.scatter(x, y, c=X, alpha=0.5)
    plt.colorbar(scatter, location='bottom', pad=0.01)
    plt.show()
    return None


def plot2D_expression(gene_name, time, section, section_id, anno_level, label=None, legend=True):
    maxx, maxy, adata = data_slice(time=time, section=section, section_id=section_id)
    x = adata.obsm['spatial'].iloc[:, 0]
    y = adata.obsm['spatial'].iloc[:, 1]
    X = adata[:, adata.var['gene'] == gene_name]
    # sc.pp.scale(X)
    if label:
        labels = adata.obs[anno_level]
        labels = labels.astype('str')
        for i in labels.index.tolist():
            if labels.loc[i, ] in label:
                continue
            else:
                labels.loc[i, ] = 'other'
        labely = labels.astype('category')
    else:
        label = adata.obs[anno_level]
        labely = label.astype('category')
    labelx = labely.cat.codes
    num = len(set(labelx))
    cmap = plt.get_cmap("coolwarm", num)
    ax = plt.subplot()
    ax.scatter(x, y, s=40, c=cmap(labelx), marker='s')
    scatter = ax.scatter(x, y, s=X.X, c='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.axis('equal')
    ax.set_title(time + ' ' + section + ' ' + ' No.' + str(section_id) + ' gene:' + gene_name )
    frame = plt.gca()
    frame.invert_yaxis()
    frame.axes.get_yaxis().set_visible(False)
    frame.axes.get_xaxis().set_visible(False)
    if legend:
        patchlist = []
        for m in range(num):
            label_ = labely.cat.categories[m]
            patch = mpatches.Patch(color=cmap(m), label=label_)
            patchlist.append(patch)
        legend1 = ax.legend(handles=patchlist, loc='center left', bbox_to_anchor=(1, 0.5),title='Region')
        ax.add_artist(legend1)
        handles, labels = scatter.legend_elements(prop="sizes", color='black', alpha=0.6)
        legend2 = ax.legend(handles, labels, loc="center right", bbox_to_anchor=(0, 0.5), title="Expression")
    # plt.savefig('./'+time + ' ' + section + ' ' + ' No.' + str(section_id) + ' gene:' + gene_name +'.png',dpi=300,bbox_inches='tight', transparent=True)
    plt.show()


def plot3D_expression(gene_name, time, anno_level, legend=True):
    '''
    plot the gene expression distribution on 3D voxel space
    :param gene_name: str, the gene symbol of targeted gene
    :param time: str, the developing period of mouse data ['E11pt5','E13pt5','E15pt5','E18pt5','P4','P14','P28','P56','Adult']
    :param anno_level: str, the annotation level from level_1 to level_10
    :param legend: bool, whether to plot the legend of structure label
    :return:
    '''
    adata = get_data(time=time)
    size = {'E11pt5': [70, 75, 40],
            'E13pt5': [89, 109, 69],
            'E15pt5': [94, 132, 65],
            'E18pt8': [67, 43, 40],
            'P4': [77, 43, 50],
            'P14': [68, 40, 50],
            'P28': [73, 41, 53],
            'P56': [67, 41, 58],
            'Adult': [67, 41, 58]
            }
    x = adata.obs['x'].tolist()
    y = adata.obs['y'].tolist()
    z = adata.obs['z'].tolist()
    array = np.zeros(size[time])
    size_c = size[time]
    size_c.append(4)
    color = np.zeros(size_c)
    label = adata.obs[anno_level]
    labely = label.astype('category')
    labelx = labely.cat.codes
    num = len(set(labelx))
    cmap = plt.get_cmap("coolwarm", num)
    for (i, j, k, l) in zip(x, y, z, labelx):
        array[i][j][k] = 1
        color[i][j][k] = np.asarray(cmap(l))
    sc.pp.scale(adata)
    data = adata[:, adata.var['gene'] == gene_name]
    s = data.X
    fig = plt.figure()
    ax = plt.subplot(projection='3d')
    ax.voxels(array, facecolors=color, edgecolor=None, alpha=0.3)
    scatter = ax.scatter(x, y, z, s=s, c='orange', alpha=0.1)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(-90, -90)
    ax.set_title(time + ' Mouse ' + gene_name +' ' + anno_level)
    ax.axis('equal')
    if legend:
        patchlist = []
        for m in range(num):
            label_ = labely.cat.categories[m]
            patch = mpatches.Patch(color=cmap(m), label=label_)
            patchlist.append(patch)
        legend1 = ax.legend(handles=patchlist, loc='center left', bbox_to_anchor=(1, 0.5), title='Region')
        ax.add_artist(legend1)
        handles, labels = scatter.legend_elements(prop="sizes", color='orange', num=5, alpha=0.6)
        legend2 = ax.legend(handles, labels, loc="center right", bbox_to_anchor=(0, 0.5), title="Expression")
    plt.savefig('./Adult_3d_Etv1.png',dpi=300,bbox_inches='tight',transparent=True)
    plt.show()


