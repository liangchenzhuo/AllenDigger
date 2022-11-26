import scanpy as sc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from typing import Optional
from Demo.accessdata import data_slice, get_data, find_overlap_gene, get_section_structure
from Demo.preprocess import update_label, accuracy, macro_precision, macro_f1, macro_recall, roc_auc_score_multiclass
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
from scipy.stats import mannwhitneyu
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def get_spatial_clusters(
        section: str,
        section_id: int,
        time: str,
        lr_rate: Optional[float] = 1e-3,
        num_of_cluster: Optional[int] = 5,
        epochs: Optional[int] = 10,
        seed: Optional[int] = 1,
        weight_decay: Optional[int] = 1e-5,
        use_variable_genes: Optional[bool] = False,
        plot_cluster_result: Optional[bool] = False):
    '''
    Return
    ------
    cluster_label: str

    '''
    if not type(section_id) == int:
        raise TypeError('section id should be an integer')
    time_list = ['E11pt5', 'E13pt5', 'E15pt5', 'E18pt5', 'P4', 'P14', 'P28', 'P56', 'Adult']
    section_list = ['sagittal', 'horizontal', 'coronal']
    if not (time in time_list):
        raise ValueError("the input time is not available, available time points are" + '\n' +
                         "'E11pt5','E13pt5','E15pt5','E18pt5','P4','P14','P28','P56','Adult'")

    if not (section in section_list):
        raise ValueError("the input section_is not available, available sections are" + '\n' +
                         "'sagittal','horizontal','coronal'")
    section_dict = {'sagittal': 'z', 'horizontal': 'y', 'coronal': 'x'}

    # read in data
    adata = sc.read_h5ad('data/' + time + '_adata.h5ad')
    # extract data of inputted section and section_id
    adata = adata[adata.obs[section_dict[section]] == section_id, :]
    mat = adata.X.astype(np.float32)
    if use_variable_genes:
        sc.pp.highly_variable_genes(adata, min_mean=0.01, max_mean=2, min_disp=0.25)
        mat = adata.X[:, adata.var['highly_variable']]
    else:
        mat = adata.X

    # scale expression data into range [0,1]
    min_max_scaler = preprocessing.MinMaxScaler()
    mat = min_max_scaler.fit_transform(mat)
    # generator tensor
    mat = torch.tensor(mat)

    # build model
    input_shape = np.shape(mat)[1]

    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(1 * input_shape, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 15),
            )

            self.decoder = nn.Sequential(
                nn.Linear(15, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, input_shape),
                nn.Sigmoid()
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return encoded, decoded

    model = Autoencoder()
    # define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate, weight_decay=weight_decay)

    # train model
    num_epochs = epochs
    outputs = []
    for epoch in range(num_epochs):
        encoded, decoded = model(mat)
        loss = criterion(decoded, mat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch:{epoch + 1},Loss:{loss.item():.4f}')
        outputs.append((epoch, mat, decoded))

    # clustering analysis
    kmeans = KMeans(n_clusters=6, random_state=0)
    kmeans.fit(encoded.detach().numpy())
    kmeans.pred = kmeans.predict(encoded.detach().numpy())
    cluster_result = pd.DataFrame({'cluster': kmeans.pred}, index=adata.obs['x'].index)
    adata.obs['cluster'] = cluster_result

    # plot cluster result
    if plot_cluster_result:
        if section == 'sagittal':
            frame = {'X': adata.obs['x'], 'Y': adata.obs['y'], 'annot_L2': adata.obs['level_2'],
                     'cluster': adata.obs['cluster']}
            df = pd.DataFrame(frame)
            sns.lmplot(x='X', y='Y', data=df, hue='cluster', fit_reg=False)
            plt.axis('equal')
            #plt.show()
        elif section == 'horizontal':
            frame = {'X': adata.obs['x'], 'Z': adata.obs['z'], 'annot_L2': adata.obs['level_2'],
                     'cluster': adata.obs['cluster']}
            df = pd.DataFrame(frame)
            sns.lmplot(x='X', y='Z', data=df, hue='cluster', fit_reg=False)
            plt.axis('equal')
            #plt.show()
        else:
            frame = {'Y': adata.obs['y'], 'Z': adata.obs['z'], 'annot_L2': adata.obs['level_2'],
                     'cluster': adata.obs['cluster']}
            df = pd.DataFrame(frame)
            sns.lmplot(x='Y', y='Z', data=df, hue='cluster', fit_reg=False)
            plt.axis('equal')
            #plt.show()
        plt.savefig('./spatial_cluster.png',dpi=300,bbox_inches='tight',transparent=True)

    return adata


# adata = get_spatial_clusters('sagittal', 10, 'E18pt5', plot_cluster_result=True, use_variable_genes=True)


def find_differential_expression_marker(time: str,
                                        labels: list,
                                        anno_level: str,
                                        plot_result=True,
                                        method='ranksum',
                                        n=30):
    '''
    find the differentiL expression gene from different brain regions using wilcoxon rank sum test
    :param time: str, the developing period of mouse data ['E11pt5','E13pt5','E15pt5','E18pt5','P4','P14','P28','P56','Adult']
    :param labels: list, the target structure acronyms
    :param anno_level: str, the annotation level from level_1 to level_10
    :param plot_result: bool, whether to plot the heatmap containing auc value of top gene markers
    :param method: str, the method used
    :param n: int, the cutoff of top genes
    :return: dict, the keys represent target structure acronyms, the values are a list of top genes and a DataFrame of auc value of all genes
    '''
    adata = get_data(time=time)
    result = {}
    for label in labels:
        df = pd.DataFrame(index=adata.var['gene'].tolist(), columns=['auc', 'pvalue'])
        x = adata[adata.obs[anno_level] == label]
        y = adata[adata.obs[anno_level] != label]
        U1, pvalue = mannwhitneyu(x.X, y.X, alternative='two-sided')
        n1 = x.shape[0]
        n2 = y.shape[0]
        df['auc'] = U1/(n1*n2)
        df['pvalue'] = pvalue
        # df['js'] = js
        df_s = df.sort_values(by=['auc'], ascending=False)
        s_max = df_s.iloc[:n, :].index.tolist()
        df_p = df[df['pvalue'] < 0.05]
        df_p = df_p.sort_values(by=['pvalue'])
        p_min = df_p.iloc[:n, :].index.tolist()
        # df_js = df.sort_values('js', ascending=False)
        # js_max = df_js.iloc[20, :].index.tolist()
        top = list(set(s_max) & set(p_min))
        result[label] = [top, df]
    if plot_result:
        columns = []
        for label_ in labels:
            gene = result[label_][0]
            for g in gene:
                columns.append(g)
        # columns = list(set(columns))
        result_df = pd.DataFrame(index=labels, columns=columns)
        for label in labels:
            for column in columns:
                result_df.loc[label, column] = result[label][1].loc[column, 'auc']
        array = result_df.values.astype('float64')
        result_df = pd.DataFrame(array, index=labels, columns=columns)
        map = sns.heatmap(result_df, xticklabels=True, yticklabels=True)
        map.fig.set_size_inches(20, 20)
        plt.show()

    return result


'''adata = get_data(time='Adult')
labels = list(set(adata.obs['level_8'].tolist()))
result = find_differential_expression_marker(time='Adult', labels=labels, anno_level='level_8')
'''


def section_find_differential_expression_marker(time,
                                                section,
                                                section_id,
                                                labels,
                                                anno_level,
                                                n=5,
                                                plot_result=True,
                                                method='ranksum',
                                                ):
    # find the differential expression genes among structures in one section
    i, j, adata = data_slice(time=time, section=section, section_id=section_id)
    # sc.pp.filter_cells(adata, min_genes=200)
    # sc.pp.filter_genes(adata, min_cells=3)
    result = {}
    genes = adata.var['gene'].tolist()
    for label in labels:
        df = pd.DataFrame(index=adata.var['gene'].tolist(), columns=['auc', 'pvalue', 'log2FC'])
        x = adata[adata.obs[anno_level] == label]
        y = adata[adata.obs[anno_level] != label]
        FC=[]
        for gene in genes:
            xx = x[:, gene].X
            yy = y[:, gene].X
            mean_x = xx.mean()
            mean_y = yy.mean()
            fc = mean_x/mean_y
            FC.append(fc)
        log2FC = np.log2(FC)
        df['log2FC'] = log2FC
        U1, pvalue = mannwhitneyu(x.X, y.X)
        n1 = x.shape[0]
        n2 = y.shape[0]
        df['auc'] = U1/(n1*n2)
        df['pvalue'] = pvalue
        df_l = df[df['log2FC'] > 1.1]
        df_p = df_l[df_l['pvalue'] < 0.05]
        df_p = df_p.sort_values(by=['pvalue'])
        df_s = df_p.sort_values(by=['auc'], ascending=False)
        s_max = df_s.iloc[:n-1, :].index.tolist()
        p_min = df_p.iloc[:n-1, :].index.tolist()
        top = list(set(s_max) & set(p_min))
        result[label] = [top, df]
    if plot_result:
        columns = []
        for label_ in labels:
            gene = result[label_][0]
            for g in gene:
                columns.append(g)
        result_df = pd.DataFrame(index=labels, columns=columns)
        for label in labels:
            for column in columns:
                result_df.loc[label, column] = result[label][1].loc[column, 'auc']
        array = result_df.values.astype('float64')
        result_df = pd.DataFrame(array, index=labels, columns=columns)
        sns.heatmap(result_df, xticklabels=True, yticklabels=True, cmap='OrRd')
        # plt.xlabel('x_label', fontsize=8, color='k')
        # plt.ylabel('y_label', fontsize=8, color='k')
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.savefig('./Adult_sag_20_DEG_level_6.png',dpi=300,bbox_inches='tight',transparent=True)
        plt.show()
    return result


'''time = 'Adult'
section = 'sagittal'
section_id = 20
anno_level = 'level_6'
n = 10
d = get_section_structure(time=time, section=section, section_id=section_id, anno_level=anno_level)
labels = d.keys()
# labels=['SCs', 'SCm']
result = section_find_differential_expression_marker(time=time, section=section, section_id=section_id, labels=labels, anno_level=anno_level)
'''


def structure_mapping_rf(time,
                         anno_level,
                         scrna,
                         label,
                         gene_use=None,
                         plot_predict_result=True,
                         method='random forest'):
    '''
    register cells into structures using randomforestclassifier
    :param time: str, the developing period of mouse data ['E11pt5','E13pt5','E15pt5','E18pt5','P4','P14','P28','P56','Adult']
    :param anno_level:str, the annotation level from level_1 to level_10, consistent with the label of scrna data
    :param scrna: AnnData object, cell-gene expression data of scRNA-seq
    :param label: str, the obs_name of structure annotation label from scrna data
    :param gene_use: list, the genes used for classifier's training and predicting
    :param plot_predict_result: bool, whether to show the prediction result using confusionmatrixdisplay
    :param method: str, method used for registration
    :return: the classifier, dataframe of importance score of genes/features, anndata of scrna data with prediction labels
    '''
    adata_spatial = get_data(time=time)
    level = eval(anno_level[-1])
    scrna = update_label(scrna=scrna, label=label, time=time, level=level)
    if gene_use:
        gene_use = gene_use
    else:
        gene_use = find_overlap_gene(adata_spatial, scrna, time=time)
    # random forest training
    # putting feature variable to X and target variable to y
    df = pd.DataFrame(adata_spatial.X, columns=adata_spatial.var['gene'])
    df = df[gene_use]
    x = df
    y = adata_spatial.obs[anno_level]
    label = y.astype('category')
    labely = label.cat.codes
    # split the data into train and test
    # x_train, x_test, y_train, y_test = train_test_split(x, labely, train_size=0.8, random_state=42)
    x_train = x
    y_train = labely
    st_x = preprocessing.StandardScaler()
    x_train = st_x.fit_transform(x_train)
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=20, min_samples_leaf=5, n_estimators=50)
    rf.fit(x_train, y_train)
    '''params = {
        'max_depth': [2, 3, 5, 10, 20],
        'min_samples_leaf': [5, 10, 20, 50, 100, 200],
        'n_estimators': [10, 25, 30, 50, 100, 200]
    }
    grid_search = GridSearchCV(estimator=rf,
                               param_grid=params,
                               cv=4,
                               n_jobs=-1, verbose=1, scoring="accuracy")
    grid_search.fit(x_train, y_train)
    rf_best = grid_search.best_estimator_
    # sort the data with feature importance
    # rf_best.feature_importances_'''
    imp_df = pd.DataFrame({
        "Gene name": x.columns,
        "Imp": rf.feature_importances_
    })
    imp_df.sort_values(by="Imp", ascending=False, inplace=True)
    if time == 'Adult':
        gene_use_ = gene_use
    else:
        gene_use_ = []
        for gene in scrna.var['gene'].tolist():
            if gene.upper() in gene_use:
                gene_use_.append(gene)
    scrna_ = scrna[:, gene_use_]
    scrna_X = st_x.fit_transform(scrna_.X)
    result = rf.predict(scrna_X)
    probablity = rf.predict_proba(scrna_X)
    l = []
    for i in result:
        label_ = label.cat.categories[i]
        l.append(label_)
    label_name = 'pred_rf_' + anno_level
    scrna.obs[label_name] = l

    # evaluate the prediction
    y_true = scrna.obs[anno_level]
    y_pred = scrna.obs[label_name]
    cm = confusion_matrix(y_true, y_pred)
    cm_display = ConfusionMatrixDisplay(cm).plot()
    plt.show()
    print(f"Accuracy: {accuracy(y_true, y_pred)}")
    print(f"Macro-averaged Precision score : {macro_precision(y_true, y_pred)}")
    # print(f"Micro-averaged Precision score : {micro_precision(y_true, y_pred)}")
    print(f"Macro-averaged recall score : {macro_recall(y_true, y_pred)}")
    # print(f"Micro-averaged recall score : {micro_recall(y_true, y_pred)}")
    print(f"Macro-averaged f1 score : {macro_f1(y_true, y_pred)}")
    # print(f"Micro-averaged recall score : {micro_f1(y_true, y_pred)}")
    roc_auc_dict = roc_auc_score_multiclass(y_true, y_pred)

    # plot the predicted result
    if plot_predict_result:
        sc.pp.neighbors(scrna)
        sc.tl.umap(scrna)
        sc.pl.umap(scrna, color=label_name)
    return rf, imp_df, scrna


# adata_spatial = get_data(time='P56')
# adata_scrna = ad.read_h5ad('data/scRNA/scRNA_aca_hip_ex_neuron_adata.h5ad')
# rf, imp_df, adata_scrna = structure_mapping_rf(time='Adult', anno_level='level_6', scrna=adata_scrna, label='region')


def section_structure_mapping_rf(time,
                                 section,
                                 section_id,
                                 scrna,
                                 anno_level,
                                 gene_use=None,
                                 plot_predict_result=True,
                                 method='random forest'):
    '''
    register cells into structures in one section using randomforestclassifier
    :param time: str, the developing period of mouse data ['E11pt5','E13pt5','E15pt5','E18pt5','P4','P14','P28','P56','Adult']
    :param section: the direction of slicing 'sag' for sagittal,'hor' for horizontal,'cor' for coronal
    :param section_id:No. section number
    :param scrna:anndata object, the cell-gene expression data of scRNA-seq
    :param anno_level: str, the annotation level from level_1 to level_10
    :param gene_use: list, the genes used for classifier training and predicting
    :param plot_predict_result: bool, whether to show the prediction result by confusionmatrixdisplay
    :param method: str, the method used for registration
    :return:the classifier, dataframe of importance score of genes/features, anndata of scrna data with prediction labels
    '''
    corx, cory, adata_spatial = data_slice(time=time, section=section, section_id=section_id)
    if gene_use:
        gene_use = gene_use
    else:
        gene_use = find_overlap_gene(adata_spatial, scrna, time=time)
    df = pd.DataFrame(adata_spatial.X, columns=adata_spatial.var['gene'])
    df = df[gene_use]
    x = df
    y = adata_spatial.obs[anno_level]
    label = y.astype('category')
    labely = label.cat.codes
    # x_train, x_test, y_train, y_test = train_test_split(x, labely, train_size=0.8, random_state=42)
    x_train = x
    y_train = labely
    st_x = preprocessing.StandardScaler()
    x_train = st_x.fit_transform(x_train)
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=20, min_samples_leaf=5, n_estimators=50)
    rf.fit(x_train, y_train)
    # calculate accuracy
    # score = rf.score(x_test, y_test)
    # print("Accuracy: ", round(score, ndigits=4))
    # find beat parameters for classifier
    '''params = {
        'max_depth': [2, 3, 5, 10, 20],
        'min_samples_leaf': [5, 10, 20, 50, 100, 200],
        'n_estimators': [10, 25, 30, 50, 100, 200]
    }
    grid_search = GridSearchCV(estimator=rf,
                               param_grid=params,
                               cv=4,
                               n_jobs=-1, verbose=1, scoring="accuracy")
    grid_search.fit(x_train, y_train)
    # rf_best = grid_search.best_estimator_
    # sort the data with feature importance
    # rf_best.feature_importances_
    print(grid_search.best_params_)'''

    imp_df = pd.DataFrame({
        "Gene name": x.columns,
        "Imp": rf.feature_importances_
    })
    imp_df.sort_values(by="Imp", ascending=False, inplace=True)
    if time == 'Adult':
        gene_use_ = gene_use
    else:
        gene_use_ = []
        for gene in scrna.var['gene'].tolist():
            if gene.upper() in gene_use:
                gene_use_.append(gene)

    scrna_ = scrna[:, gene_use_]
    scrna_X = st_x.fit_transform(scrna_.X)
    result = rf.predict(scrna_X)
    l = []
    for i in result:
        label_ = label.cat.categories[i]
        l.append(label_)
    label_name = 'pred_rf_' + section + '_' + str(section_id)
    scrna.obs[label_name] = l
    # evaluate the prediction
    y_true = scrna.obs[anno_level]
    y_pred = scrna.obs[label_name]
    cm = confusion_matrix(y_true, y_pred)
    cm_display = ConfusionMatrixDisplay(cm).plot()
    plt.show()
    print(f"Accuracy: {accuracy(y_true, y_pred)}")
    print(f"Macro-averaged Precision score : {macro_precision(y_true, y_pred)}")
    # print(f"Micro-averaged Precision score : {micro_precision(y_true, y_pred)}")
    print(f"Macro-averaged recall score : {macro_recall(y_true, y_pred)}")
    # print(f"Micro-averaged recall score : {micro_recall(y_true, y_pred)}")
    print(f"Macro-averaged f1 score : {macro_f1(y_true, y_pred)}")
    # print(f"Micro-averaged recall score : {micro_f1(y_true, y_pred)}")
    roc_auc_dict = roc_auc_score_multiclass(y_true, y_pred)

    # plot the result
    if plot_predict_result:
        sc.pp.pca(scrna)
        sc.pp.neighbors(scrna)
        sc.tl.umap(scrna)
        sc.pl.umap(scrna, color=label_name)
    return rf, imp_df, scrna










