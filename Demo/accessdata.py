import anndata as ad
import pandas as pd
import scanpy as sc


def csv_to_spatial_adata(counts, spatial):
    adata = ad.read_csv(counts)
    coor = pd.read_csv(spatial)
    adata.obs['spatial'] = coor
    return adata


def csv_to_scRNA_adata(counts, annotation):
    adata = ad.read_csv(counts)
    adata.var['gene'] = adata.var.index.tolist()
    anno = pd.read_csv(annotation)
    for column in anno.columns:
        adata.obs[column] = anno[column].tolist()
    return adata


def txt_to_scRNA_adata(counts, annotation):
    adata = sc.read_text(counts)
    anno = pd.read_table(annotation)
    for column in anno.columns:
        adata.obs[column] = anno[column]
    return adata


def get_data(time):
    '''
    Load the Developing Mouse Brain ISH data, form: .h5ad file and anndata.
    :param time: the developing period of mouse data ['E11pt5','E13pt5','E15pt5','E18pt5','P4','P14','P28','P56','Adult']
    :return: AnnDate object
    example: adata = get_data('P56')
    '''

    adata = ad.read_h5ad('./data/'+time+'_adata.h5ad')
    return adata


def get_data_section_id(time, section):
    '''
    get the range of section_id from voxel-gene expression data
    :param time: the developing period of mouse data ['E11pt5','E13pt5','E15pt5','E18pt5','P4','P14','P28','P56','Adult']
    :param section: the direction of slicing 'sag' for sagittal,'hor' for horizontal,'cor' for coronal
    :return: the min and the max id of the section
    '''
    adata = ad.read_h5ad('data/' + time + '_adata.h5ad')
    maxx = max(adata.obs['x'])
    minx = min(adata.obs['x'])
    maxy = max(adata.obs['y'])
    miny = min(adata.obs['y'])
    maxz = max(adata.obs['z'])
    minz = min(adata.obs['z'])
    if ('sag' in section) or ('Sag' in section):
        return minz, maxz
    elif ('hor' in section) or ('Hor' in section):
        return miny, maxy
    elif ('cor' in section) or ('Cor' in section):
        return minx, maxx


def get_anno_section_id(time, section):
    '''
    get the range of section_id from voxel-annotation data
    :param time: the developing period of mouse data ['E11pt5','E13pt5','E15pt5','E18pt5','P4','P14','P28','P56','Adult']
    :param section: the direction of slicing 'sag' for sagittal,'hor' for horizontal,'cor' for coronal
    :return: the min and the max id of the section
    '''
    anno = pd.read_csv('data/' + time + '_DevMouse2012_gridAnnotation/' + time + '_voxel_anno.csv')
    maxx = max(anno['x'])
    minx = min(anno['x'])
    maxy = max(anno['y'])
    miny = min(anno['y'])
    maxz = max(anno['z'])
    minz = min(anno['z'])
    if ('cor' in section) or ('Cor' in section):
        return minz, maxz
    elif ('hor' in section) or ('Hor' in section):
        return miny, maxy
    elif ('sag' in section) or ('Sag' in section):
        return minx, maxx


def get_structure(time, anno_level):
    '''
    get the structure acronym and its coordinates of one annotation level from one developing mouse brain data
    :param time: the developing period of mouse data ['E11pt5','E13pt5','E15pt5','E18pt5','P4','P14','P28','P56','Adult']
    :param anno_level: the level of annotation from level_1 to level_10
    :return: a dict, the key represents structure acronym while the value is a DataFrame containing x,y,z coordinates
    '''
    adata = ad.read_h5ad('data/' + time + '_adata.h5ad')
    structures = list(set(adata.obs[anno_level].tolist()))
    d = {}
    for structure in structures:
        data = adata[adata.obs[anno_level] == structure]
        x = data.obs['x'].tolist()
        y = data.obs['y'].tolist()
        z = data.obs['z'].tolist()
        df = pd.DataFrame({'x': x, 'y': y, 'z': z})
        d[structure] = df
    return d


# d = get_structure(time='Adult', anno_level='level_7')


def get_section_structure(time, section, section_id, anno_level):
    '''
    get the structure acronym and its coordinates of one annotation level from one section of data
    :param time: the developing period of mouse data ['E11pt5','E13pt5','E15pt5','E18pt5','P4','P14','P28','P56','Adult']
    :param section: the direction of slicing 'sag' for sagittal,'hor' for horizontal,'cor' for coronal
    :param section_id: the id of the section
    :param anno_level: the level of annotation from level_1 to level_10
    :return: a dict, the key represents structure acronym while the value is a DataFrame containing x,y,z coordinates
    '''
    adata = ad.read_h5ad('data/' + time + '_adata.h5ad')
    maxx = max(adata.obs['x'])
    minx = min(adata.obs['x'])
    maxy = max(adata.obs['y'])
    miny = min(adata.obs['y'])
    maxz = max(adata.obs['z'])
    minz = min(adata.obs['z'])
    if ('sag' in section) or ('Sag' in section):
        if section_id > maxz:
            raise ValueError('please input the right section_id, <='+str(maxz))
        elif section_id < minz:
            raise ValueError('please input the right section_id, >='+str(minz))
        filter = adata.obs['z'] == section_id
        adata = adata[filter]
        structures = list(set(adata.obs[anno_level].tolist()))
        d = {}
        for structure in structures:
            data = adata[adata.obs[anno_level] == structure]
            d[structure] = data.obs[['x', 'y', 'z']]
        return d
    elif ('hor' in section) or ('Hor' in section):
        if section_id > maxy:
            raise ValueError('please input the right section_id, <='+str(maxy))
        elif section_id < miny:
            raise ValueError('please input the right section_id, >='+str(miny))
        filter = adata.obs['y'] == section_id
        adata = adata[filter]
        structures = list(set(adata.obs[anno_level].tolist()))
        d = {}
        for structure in structures:
            data = adata[adata.obs[anno_level] == structure]
            d[structure] = data.obs[['x', 'y', 'z']]
        return d
    elif ('cor' in section) or ('Cor' in section):
        if section_id > maxx:
            raise ValueError('please input the right section_id, <='+str(maxx))
        elif section_id < minx:
            raise ValueError('please input the right section_id, >='+str(minx))
        filter = adata.obs['x'] == section_id
        adata = adata[filter]
        structures = list(set(adata.obs[anno_level].tolist()))
        d = {}
        for structure in structures:
            data = adata[adata.obs[anno_level] == structure]
            d[structure] = data.obs[['x', 'y', 'z']]
        return d


def data_slice(time, section, section_id):

    '''
    'Cut' a slice from the 3D data
    :param time: the developing period of mouse data ['E11pt5','E13pt5','E15pt5','E18pt5','P4','P14','P28','P56', 'Adult']
    :param section: the direction of slicing 'sag' for sagittal,'hor' for horizontal,'cor' for coronal
    :param section_id: No. section number
    :return: max x,y and a slice of anndata
    example: slice_cor = data_slice(time='P56', section='coronal', section_id=20)
    '''

    # load data
    adata = ad.read_h5ad('data/'+time+'_adata.h5ad')
    # get the range of the slice
    maxx = max(adata.obs['x'])
    minx = min(adata.obs['x'])
    maxy = max(adata.obs['y'])
    miny = min(adata.obs['y'])
    maxz = max(adata.obs['z'])
    minz = min(adata.obs['z'])
    # slicing
    if ('sag' in section) or ('Sag' in section):
        if section_id > maxz:
            raise ValueError('please input the right section_id, <='+str(maxz))
        elif section_id < minz:
            raise ValueError('please input the right section_id, >='+str(minz))
        filter = adata.obs['z'] == section_id
        adata = adata[filter]
        spatial = pd.DataFrame()
        spatial['x'] = adata.obs['x']
        spatial['y'] = adata.obs['y']
        adata.obsm['spatial'] = spatial
        return maxx, maxy, adata

    elif ('hor' in section) or ('Hor' in section):
        if section_id > maxy:
            raise ValueError('please input the right section_id, <='+str(maxy))
        elif section_id < miny:
            raise ValueError('please input the right section_id, >='+str(miny))
        filter = adata.obs['y'] == section_id
        adata = adata[filter]
        spatial = pd.DataFrame()
        spatial['x'] = adata.obs['x']
        spatial['z'] = adata.obs['z']
        adata.obsm['spatial'] = spatial
        return maxx, maxz, adata

    elif ('cor' in section) or ('Cor' in section):
        if section_id > maxx:
            raise ValueError('please input the right section_id, <='+str(maxx))
        elif section_id < minx:
            raise ValueError('please input the right section_id, >='+str(minx))
        filter = adata.obs['x'] == section_id
        adata = adata[filter]
        spatial = pd.DataFrame()
        spatial['z'] = adata.obs['z']
        spatial['y'] = adata.obs['y']
        adata.obsm['spatial'] = spatial
        return maxz, maxy, adata
    else:
        raise ValueError("please select the direction of the section:'sag', 'hor' or 'cor'")

# slice_cor = data_slice(time='P56', section='coronal', section_id=20)


def anno_slice(time, section, section_id):

    '''
    'Cut' a slice from 3D developing mouse brain annotation data
    :param time: the developing period of mouse data ['E11pt5','E13pt5','E15pt5','E18pt5','P4','P14','P28','P56','Adult']
    :param section: the direction of slicing 'sag' for sagittal,'hor' for horizontal,'cor' for coronal
    :param section_id: No. section number
    :return: max x, y and a dataframe of slice pixel coordinates, label, and ABA_defined_color
    example: spatial = anno_slice(time='P56', section='coronal', section_id=20)
    '''

    # load the data
    anno = pd.read_csv('data/' + time + '_DevMouse2012_gridAnnotation/' + time + '_voxel_anno.csv')
    # get the range of the slice
    maxx = max(anno['x'])
    minx = min(anno['x'])
    maxy = max(anno['y'])
    miny = min(anno['y'])
    maxz = max(anno['z'])
    minz = min(anno['z'])
    # slicing
    if ('cor' in section) or ('Cor' in section):
        if section_id > maxz:
            raise ValueError('please input the right section_id, <=' + str(maxz))
        elif section_id < minz:
            raise ValueError('please input the right section_id,>=' + str(minz))
        anno = anno[anno['z'] == section_id]
        coordinates = anno[['x', 'y', 'acronym', 'color_hex_triplet']]
        coordinates.columns = ['x', 'y', 'label', 'color']
        return maxx, maxy, coordinates

    elif ('hor' in section) or ('Hor' in section):
        if section_id > maxy:
            raise ValueError('please input the right section_id, <=' + str(maxy))
        elif section_id < miny:
            raise ValueError('please input the right section_id,>=' + str(miny))
        anno = anno[anno['y'] == section_id]
        coordinates = anno[['x', 'z', 'acronym', 'color_hex_triplet']]
        coordinates.columns = ['x', 'z', 'label', 'color']
        return maxx, maxz, coordinates

    elif ('sag' in section) or ('Sag' in section):
        if section_id > maxx:
            raise ValueError('please input the right section_id, <=' + str(maxx))
        elif section_id < minx:
            raise ValueError('please input the right section_id,>=' + str(minx))
        anno = anno[anno['x'] == section_id]
        coordinates = anno[['z', 'y', 'acronym', 'color_hex_triplet']]
        coordinates.columns = ['z', 'y', 'label', 'color']
        return maxz, maxy, coordinates
    else:
        raise ValueError("please select the direction of the section:'sag', 'hor' or 'cor'")


# data = anno_slice(time='P56',section='Coronal',section_id=20)

def find_overlap_gene(spatial, scrna, time):
    '''
    find the overlap gene between spatial and scrna data
    :param spatial: spatial adata
    :param scrna: scrna adata
    :param time:the developing period of mouse data ['E11pt5','E13pt5','E15pt5','E18pt5','P4','P14','P28','P56','Adult']
    :return:a list of overlap gene
    '''
    if time == 'Adult':
        overlap_gene = []
        for gene in scrna.var['gene'].tolist():
            if gene in spatial.var['gene'].tolist():
                overlap_gene.append(gene)
    else:
        overlap_gene = []
        for gene in scrna.var['gene'].tolist():
            gene = gene.upper()
            if gene in spatial.var['gene'].tolist():
                overlap_gene.append(gene)
    return overlap_gene
