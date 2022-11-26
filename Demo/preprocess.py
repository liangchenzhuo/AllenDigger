import anndata as ad
import scanpy as sc
import pandas as pd
import SpatialDE
import numpy as np
from sklearn.metrics import roc_auc_score, auc, roc_curve
import matplotlib.pyplot as plt
from sklearn import preprocessing
from Demo.accessdata import get_data
from numpy import array
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier


def find_highly_variable_gene(
        adata: ad.AnnData,
        section: str,
        section_id: int):
    '''
    Return
    ------
    spatial variable genes

    '''

    if not type(section_id) == int:
        raise TypeError('section id should be an integer')
    time_list = ['E11pt5', 'E13pt5', 'E15pt5', 'E18pt5', 'P4', 'P14', 'P28', 'P56']
    section_list = ['sagittal', 'horizontal', 'coronal']

    if not (section in section_list):
        raise ValueError("inputed section_is not avalible, avalible sections are" + '\n' +
                         "'sagittal','horizontal','coronal'")
    section_dict = {'sagittal': 'z', 'horizontal': 'y', 'coronal': 'x'}
    mat = adata.X.astype(np.float32)
    if section == 'sagittal':
        frame = {'X': adata.obs['x'], 'Y': adata.obs['y']}
        df = pd.DataFrame(frame)
        mat_df = pd.DataFrame(mat)
        mat_df.index = adata.obs_names
        mat_df.columns = adata.var
    if section == 'horizontal':
        frame = {'X': adata.obs['x'], 'Z': adata.obs['z']}
        df = pd.DataFrame(frame)
        mat_df = pd.DataFrame(mat)
        mat_df.index = adata.obs_names
        mat_df.columns = adata.var
    if section == 'coronal':
        frame = {'Y': adata.obs['y'], 'Z': adata.obs['z']}
        df = pd.DataFrame(frame)
        mat_df = pd.DataFrame(mat)
        mat_df.index = adata.obs_names
        mat_df.columns = adata.var

    spatial_var_gene = SpatialDE.run(df, pd.DataFrame(mat_df))
    # spatial_var_gene = spatial_var_gene.sort_values('qval')['g'][0:200]

    return spatial_var_gene


def update_label(scrna, label, time, level):
    '''
    update the annotation of scRNA data to make label consistent with spatial data
    :param scrna: the anndata object, scrna cell-gene data
    :param label: str, the obs_name of annotation label from scrna data
    :param time: str, the developing time period of mouse brain data
    :param level: str, the annotation level
    :return: the anndata object of scrna data with updated labels
    '''
    if time == 'Adult':
        anno = pd.read_csv('./data/ABA_id_structure_annotation.csv')
    labels = scrna.obs[label].tolist()
    acronym = anno['acronym'].tolist()
    name = anno['name'].tolist()
    l = []
    for label_ in labels:
        if label_ in acronym:
            st_level = int(anno.loc[anno['acronym'] == label_, 'st_level'])
            if st_level <= level:
                l.append(label_)
            elif st_level > level:
                while st_level > level:
                    id = int(anno.loc[anno['acronym'] == label_, 'parent_structure_id'])
                    st_level = int(anno.loc[anno['id'] == id, 'st_level'])
                    label__ = anno.loc[anno['id'] == id, 'acronym']
                    label__ = label__.tolist()
                    label__ = label__[0]
                l.append(label__)
        elif label_ in name:
            st_level = int(anno.loc[anno['name'] == label_, 'st_level'])
            if st_level <= level:
                label__ = anno.loc[anno['name'] == label_, 'acronym']
                l.append(label__)
            elif st_level > level:
                while st_level > level:
                    id = int(anno.loc[anno['name'] == label_, 'parent_structure_id'])
                    st_level = int(anno.loc[anno['id'] == id, 'st_level'])
                    label__ = anno.loc[anno['id'] == id, 'name']
                    label__ = label__.tolist()
                    label__ = label__[0]
                l.append(label__)
        else:
            l.append('NA')
    scrna.obs['level_'+str(level)] = l
    scrna = scrna[scrna.obs['level_'+str(level)] != 'NA']
    return scrna


# scrna = ad.read_h5ad('./data/scRNA/scRNA_aca_hip_ex_neuron_adata.h5ad')
# scrna = update_label(scrna, label='region', time='Adult', level=7)


def rf_cross_validation(time, anno_level, n):
    adata_spatial = get_data(time=time)
    df = pd.DataFrame(adata_spatial.X, columns=adata_spatial.var['gene'])
    X = df
    y = adata_spatial.obs[anno_level]
    classes = y.unique()
    n_classes = classes.shape[0]
    label = y.astype('category')
    labely = label.cat.codes
    y = labely
    values = array(label)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    y_ = onehot_encoded
    st_x = preprocessing.StandardScaler()
    X = st_x.fit_transform(X)
    cv = StratifiedKFold(n_splits=n, shuffle=True)
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=20, min_samples_leaf=5, n_estimators=50)
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i, (train, test) in enumerate(cv.split(X, y)):
        rf.fit(X[train, :], y[train])
        y_score = rf.predict_proba(X[test, :])
        fpri = dict()
        tpri = dict()
        roc_auci = dict()
        for j in range(n_classes):
            fpri[j], tpri[j], _ = roc_curve(y_[test][:, j], y_score[:, j])
            roc_auci[j] = auc(fpri[j], tpri[j])
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpri[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for k in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpri[k], tpri[k])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro_"+str(i)] = all_fpr
        tpr["macro_"+str(i)] = mean_tpr
        roc_auc["macro_"+str(i)] = auc(fpr["macro_"+str(i)], tpr["macro_"+str(i)])

    color_list = plt.cm.tab10(np.linspace(0, 1, n))
    for i, color in zip(range(n_classes), color_list):
        plt.plot(
            fpr["macro_"+str(i)],
            tpr["macro_"+str(i)],
            color=color,
            lw=2,
            label="macro-average ROC curve of Fold {0} (area = {1:0.2f})".format(i, roc_auc["macro_"+str(i)]),
        )
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.legend(loc="best")
    plt.title("Random Forest ROC curve")
    plt.show()


# rf_cross_validation(time='Adult', anno_level='level_2', n=5)


def sgd_cross_validation(time, anno_level, n):
    adata_spatial = get_data(time=time)
    df = pd.DataFrame(adata_spatial.X, columns=adata_spatial.var['gene'])
    X = df
    y = adata_spatial.obs[anno_level]
    classes = y.unique()
    n_classes = classes.shape[0]
    label = y.astype('category')
    labely = label.cat.codes
    y = labely
    values = array(label)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    y_ = onehot_encoded
    st_x = preprocessing.StandardScaler()
    X = st_x.fit_transform(X)
    cv = StratifiedKFold(n_splits=n, shuffle=True)
    sgdc = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i, (train, test) in enumerate(cv.split(X, y)):
        sgdc.fit(X[train, :], y[train])
        y_score = sgdc.predict_proba(X[test, :])
        fpri = dict()
        tpri = dict()
        roc_auci = dict()
        for j in range(n_classes):
            fpri[j], tpri[j], _ = roc_curve(y_[test][:, j], y_score[:, j])
            roc_auci[j] = auc(fpri[j], tpri[j])
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpri[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for k in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpri[k], tpri[k])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro_"+str(i)] = all_fpr
        tpr["macro_"+str(i)] = mean_tpr
        roc_auc["macro_"+str(i)] = auc(fpr["macro_"+str(i)], tpr["macro_"+str(i)])

    color_list = plt.cm.tab10(np.linspace(0, 1, n))
    for i, color in zip(range(n_classes), color_list):
        plt.plot(
            fpr["macro_"+str(i)],
            tpr["macro_"+str(i)],
            color=color,
            lw=2,
            label="macro-average ROC curve of Fold {0} (area = {1:0.2f})".format(i, roc_auc["macro_"+str(i)]),
        )
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.legend(loc="best")
    plt.title("SGD Classifier ROC curve")
    plt.show()


# sgd_cross_validation('Adult', 'level_2', 3)


def accuracy(y_true, y_pred):
    # Intitializing variable to store count of correctly predicted classes
    correct_predictions = 0

    for yt, yp in zip(y_true, y_pred):

        if yt == yp:
            correct_predictions += 1

    # returns accuracy
    return correct_predictions / len(y_true)


def true_positive(y_true, y_pred):
    tp = 0

    for yt, yp in zip(y_true, y_pred):

        if yt == 1 and yp == 1:
            tp += 1

    return tp


def true_negative(y_true, y_pred):
    tn = 0

    for yt, yp in zip(y_true, y_pred):

        if yt == 0 and yp == 0:
            tn += 1

    return tn


def false_positive(y_true, y_pred):
    fp = 0

    for yt, yp in zip(y_true, y_pred):

        if yt == 0 and yp == 1:
            fp += 1

    return fp


def false_negative(y_true, y_pred):
    fn = 0

    for yt, yp in zip(y_true, y_pred):

        if yt == 1 and yp == 0:
            fn += 1

    return fn


def macro_precision(y_true, y_pred):
    # find the number of classes
    num_classes = len(np.unique(y_true))

    # initialize precision to 0
    precision = 0

    # loop over all classes
    for class_ in list(y_true.unique()):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # compute true positive for current class
        tp = true_positive(temp_true, temp_pred)

        # compute false positive for current class
        fp = false_positive(temp_true, temp_pred)

        # compute precision for current class
        temp_precision = tp / (tp + fp + 1e-6)
        # keep adding precision for all classes
        precision += temp_precision

    # calculate and return average precision over all classes
    precision /= num_classes

    return precision


def micro_precision(y_true, y_pred):
    # find the number of classes
    num_classes = len(np.unique(y_true))

    # initialize tp and fp to 0
    tp = 0
    fp = 0

    # loop over all classes
    for class_ in y_true.unique():
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # calculate true positive for current class
        # and update overall tp
        tp += true_positive(temp_true, temp_pred)

        # calculate false positive for current class
        # and update overall tp
        fp += false_positive(temp_true, temp_pred)

    # calculate and return overall precision
    precision = tp / (tp + fp)
    return precision


def macro_recall(y_true, y_pred):
    # find the number of classes
    num_classes = len(np.unique(y_true))

    # initialize recall to 0
    recall = 0

    # loop over all classes
    for class_ in list(y_true.unique()):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # compute true positive for current class
        tp = true_positive(temp_true, temp_pred)

        # compute false negative for current class
        fn = false_negative(temp_true, temp_pred)

        # compute recall for current class
        temp_recall = tp / (tp + fn + 1e-6)

        # keep adding recall for all classes
        recall += temp_recall

    # calculate and return average recall over all classes
    recall /= num_classes

    return recall


def micro_recall(y_true, y_pred):
    # find the number of classes
    num_classes = len(np.unique(y_true))

    # initialize tp and fp to 0
    tp = 0
    fn = 0

    # loop over all classes
    for class_ in y_true.unique():
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # calculate true positive for current class
        # and update overall tp
        tp += true_positive(temp_true, temp_pred)

        # calculate false negative for current class
        # and update overall tp
        fn += false_negative(temp_true, temp_pred)

    # calculate and return overall recall
    recall = tp / (tp + fn)
    return recall


def macro_f1(y_true, y_pred):
    # find the number of classes
    num_classes = len(np.unique(y_true))

    # initialize f1 to 0
    f1 = 0

    # loop over all classes
    for class_ in list(y_true.unique()):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # compute true positive for current class
        tp = true_positive(temp_true, temp_pred)

        # compute false negative for current class
        fn = false_negative(temp_true, temp_pred)

        # compute false positive for current class
        fp = false_positive(temp_true, temp_pred)

        # compute recall for current class
        temp_recall = tp / (tp + fn + 1e-6)

        # compute precision for current class
        temp_precision = tp / (tp + fp + 1e-6)

        temp_f1 = 2 * temp_precision * temp_recall / (temp_precision + temp_recall + 1e-6)

        # keep adding f1 score for all classes
        f1 += temp_f1

    # calculate and return average f1 score over all classes
    f1 /= num_classes

    return f1


def micro_f1(y_true, y_pred):


    #micro-averaged precision score
    P = micro_precision(y_true, y_pred)

    #micro-averaged recall score
    R = micro_recall(y_true, y_pred)

    #micro averaged f1 score
    f1 = 2*P*R / (P + R)

    return f1


def roc_auc_score_multiclass(actual_class, pred_class, average="macro"):
    # creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        # creating a list of all the classes except the current class
        other_class = [x for x in unique_class if x != per_class]

        # marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        # using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average=average)
        roc_auc_dict[per_class] = roc_auc

    return roc_auc_dict

