import pdb
import glob
import copy
import os
import pickle
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import sklearn.feature_selection

class FeatureColumn:
    def __init__(self, category, field, preprocessor, args=None, cost=None):
        self.category = category
        self.field = field
        self.preprocessor = preprocessor
        self.args = args
        self.data = None
        self.cost = cost

class NHANES:
    def __init__(self, db_path=None, columns=None):
        self.db_path = db_path
        self.columns = columns # Depricated
        self.dataset = None # Depricated
        self.column_data = None
        self.column_info = None
        self.df_features = None
        self.df_targets = None
        self.costs = None

    def process(self):
        df = None
        cache = {}
        # collect relevant data
        df = []
        for fe_col in self.columns:
            sheet = fe_col.category
            field = fe_col.field
            data_files = glob.glob(self.db_path+sheet+'/*.XPT')
            df_col = []
            for dfile in data_files:
                print(80*' ', end='\r')
                print('\rProcessing: ' + dfile.split('/')[-1], end='')
                # read the file
                if dfile in cache:
                    df_tmp = cache[dfile]
                else:
                    df_tmp = pd.read_sas(dfile)
                    cache[dfile] = df_tmp
                # skip of there is no SEQN
                if 'SEQN' not in df_tmp.columns:
                    continue
                #df_tmp.set_index('SEQN')
                # skip if there is nothing interseting there
                sel_cols = set(df_tmp.columns).intersection([field])
                if not sel_cols:
                    continue
                else:
                    df_tmp = df_tmp[['SEQN'] + list(sel_cols)]
                    df_tmp.set_index('SEQN', inplace=True)
                    df_col.append(df_tmp)

            try:
                df_col = pd.concat(df_col)
            except:
                #raise Error('Failed to process' + field)
                raise Exception('Failed to process' + field)
            df.append(df_col)
        df = pd.concat(df, axis=1)
        #df = pd.merge(df, df_sel, how='outer')
        # do preprocessing steps
        df_proc = []#[df['SEQN']]
        for fe_col in self.columns:
            field = fe_col.field
            fe_col.data = df[field].copy()
            # do preprocessing
            if fe_col.preprocessor is not None:
                prepr_col = fe_col.preprocessor(df[field], fe_col.args)
            else:
                prepr_col = df[field]
            # handle the 1 to many
            if (len(prepr_col.shape) > 1):
                fe_col.cost = [fe_col.cost] * prepr_col.shape[1]
            else:
                fe_col.cost = [fe_col.cost]
            df_proc.append(prepr_col)
        self.dataset = pd.concat(df_proc, axis=1)
        return self.dataset
    
    
# Preprocessing functions
def preproc_onehot(df_col, args=None):
    return pd.get_dummies(df_col, prefix=df_col.name, prefix_sep='#')

def preproc_real(df_col, args=None):
    if args is None:
        args={'cutoff':np.inf}
    # other answers as nan
    df_col[df_col > args['cutoff']] = np.nan
    # nan replaced by mean
    df_col[pd.isna(df_col)] = df_col.mean()
    # statistical normalization
    df_col = (df_col-df_col.mean()) / df_col.std()
    return df_col

def preproc_impute(df_col, args=None):
    # nan replaced by mean
    df_col[pd.isna(df_col)] = df_col.mean()
    return df_col

def preproc_cut(df_col, bins):
    # limit values to the bins range
    df_col = df_col[df_col >= bins[0]]
    df_col = df_col[df_col <= bins[-1]]
    return pd.cut(df_col.iloc[:,0], bins, labels=False)

def preproc_dropna(df_col, args=None):
    df_col.dropna(axis=0, how='any', inplace=True)
    return df_col

#### Add your own preprocessing functions ####

# Dataset loader
class Dataset():
    """ 
    Dataset manager class
    """
    def  __init__(self, data_path=None):
        """
        Class intitializer.
        """
        # set database path
        if data_path == None:
            self.data_path = './run_data/'
        else:
            self.data_path = data_path
        # feature and target vecotrs
        self.features = None
        self.targets = None
        self.costs = None
        
    def load_arthritis(self, opts=None):
        columns = [
            # TARGET: Ever told you had cancer or malignancy
            FeatureColumn('Questionnaire', 'MCQ220', 
                                    None, {'cutoff':2}),

            ######################## DEMOGRAPHIC ########################
            # Country of birth
            FeatureColumn('Demographics', 'DMDBORN4', 
                                 preproc_real, {'cutoff':2}),

            # Age in years at time of screening - till 80
            FeatureColumn('Demographics', 'RIDAGEYR', 
                                 preproc_real, None),

            # Gender
            FeatureColumn('Demographics', 'RIAGENDR', 
                                 preproc_onehot, {'cutoff':2}),

            # Ratio of family income to poverty
            FeatureColumn('Demographics', 'INDFMPIR', 
                                 preproc_onehot, {'cutoff':5}),
            # # Annual household income
            # FeatureColumn('Demographics', 'INDHHINC', 
            #                      preproc_real, {'cutoff':11}),

            # Race/Hispanic origin w/ NH Asian
            FeatureColumn('Demographics', 'RIDRETH3', 
                                 preproc_real, {'cutoff':7}),

            # Census 2000 FIPS State Code - 2 digit
            FeatureColumn('Demographics', 'STATE2K', 
                                 preproc_real, None),

            # Census 2000 FIPS County Code - 3 digit
            FeatureColumn('Demographics', 'CNTY2K', 
                                 preproc_real, None),
            ######################## DEMOGRAPHIC ########################


            ########################## DIETARY ##########################
            # How often add salt to food at table
            FeatureColumn('Dietary', 'DBD100', 
                                 preproc_real, {'cutoff':3}),

            # Alpha-carotene (mcg) - 0 to 35057
            FeatureColumn('Dietary', 'DR1TACAR', 
                                 preproc_real, None),

            # Beta-carotene (mcg) - 0 to 78752
            FeatureColumn('Dietary', 'DR1TBCAR', 
                                 preproc_real, None),
            # Alcohol (gm) - 0 to 591.4
            FeatureColumn('Dietary', 'DR1TALCO', 
                                 preproc_real, None),

            # Vitamin E as alpha-tocopherol (mg) - 0 to 172.32
            FeatureColumn('Dietary', 'DR1TATOC', 
                                 preproc_real, None),

            # Total choline (mg) - 0 to 2909.1
            FeatureColumn('Dietary', 'DR1TCHL', 
                                 preproc_real, None),

            # Cholesterol (mg) - 0 to 3515
            FeatureColumn('Dietary', 'DR1TCHOL', 
                                 preproc_real, None),

            # Beta-cryptoxanthin (mcg) - 0 to 24328
            FeatureColumn('Dietary', 'DR1TCRYP', 
                                 preproc_real, None),

            # Protein (gm) - 0 to 869.49
            FeatureColumn('Dietary', 'DR1TPROT', 
                                 preproc_real, None),

            # Vitamin A, RAE (mcg) - 0 to 10597
            FeatureColumn('Dietary', 'DR1TVARA', 
                                 preproc_real, None),

            # Total sugars (gm) - 0.13 to 1115.5
            FeatureColumn('Dietary', 'DR1TSUGR', 
                                 preproc_real, None),

            # Total fat (gm) - 0 to 548.38
            FeatureColumn('Dietary', 'DR1TTFAT', 
                                 preproc_real, None),

            # Sodium (mg) - 17 to 21399	
            FeatureColumn('Dietary', 'DR1TSODI', 
                                 preproc_real, None),

            # Vitamin C (mg) - 0 to 2008
            FeatureColumn('Dietary', 'DR1TVC', 
                                 preproc_real, None),

            # Vitamin K (mcg) - 0 to 4080.5	
            FeatureColumn('Dietary', 'DR1TVK', 
                                 preproc_real, None),
            ########################## DIETARY ##########################


            ######################## EXAMINATION ########################
            # Body Mass Index (kg/m**2) - 12.4 to 82.1
            FeatureColumn('Examination', 'BMXBMI', 
                                 preproc_real, None),
            ######################## EXAMINATION ########################


            ######################## LABORATORY ########################
            # Urinary Chlamydia
            FeatureColumn('Laboratory', 'URXUCL', 
                                 preproc_real, None),

            # Trichomonas, Urine
            FeatureColumn('Laboratory', 'URXUTRI', 
                                 preproc_real, None),

            # Basophils number (1000 cells/uL) - 0 to 0.8
            FeatureColumn('Laboratory', 'LBDBANO', 
                                 preproc_real, None),

            # Eosinophils number (1000 cells/uL) - 0 to 4.3
            FeatureColumn('Laboratory', 'LBDEONO', 
                                 preproc_real, None),

            # Lymphocyte number (1000 cells/uL) - 0.2 to 49
            FeatureColumn('Laboratory', 'LBDLYMNO', 
                                 preproc_real, None),

            # Monocyte number (1000 cells/uL) - 0.1 to 3.4	
            FeatureColumn('Laboratory', 'LBDMONO', 
                                 preproc_real, None),

            # Segmented neutrophils num (1000 cell/uL) - 0.4 to 25.6
            FeatureColumn('Laboratory', 'LBDNENO', 
                                 preproc_real, None),

            # Basophils percent (%) - 0 to 5.8
            FeatureColumn('Laboratory', 'LBXBAPCT', 
                                 preproc_real, None),

            #  Eosinophils percent (%) - 0 to 36.6
            FeatureColumn('Laboratory', 'LBXEOPCT', 
                                 preproc_real, None),

            # Hematocrit (%) - 17.9 to 56.5	
            FeatureColumn('Laboratory', 'LBXHCT', 
                                 preproc_real, None),

            # Hemoglobin (g/dL) - 6.4 to 19.5
            FeatureColumn('Laboratory', 'LBXHGB', 
                                 preproc_real, None),

            #  Lymphocyte percent (%) - 2.6 to 88
            FeatureColumn('Laboratory', 'LBXLYPCT', 
                                 preproc_real, None),

            # Mean Cell Hgb Conc. (g/dL) - 13.8 to 38.5	
            FeatureColumn('Laboratory', 'LBXMCHSI', 
                                 preproc_real, None),

            # Mean cell volume (fL) - 50.8 to 115.6
            FeatureColumn('Laboratory', 'LBXMCVSI', 
                                 preproc_real, None),

            # Monocyte percent (%) - 1.4 to 29.3
            FeatureColumn('Laboratory', 'LBXMOPCT', 
                                 preproc_real, None),

            # Mean platelet volume (fL) - 5.6 to 15.1	
            FeatureColumn('Laboratory', 'LBXMPSI', 
                                 preproc_real, None),

            # Segmented neutrophils percent (%) - 3.6 to 93.2
            FeatureColumn('Laboratory', 'LBXNEPCT', 
                                 preproc_real, None),

            # Platelet count (1000 cells/uL) - 14 to 777
            FeatureColumn('Laboratory', 'LBXPLTSI', 
                                 preproc_real, None),

            # Red blood cell count (million cells/uL) - 2.52 to 7.9
            FeatureColumn('Laboratory', 'LBXRBCSI', 
                                 preproc_real, None),

            # Red cell distribution width (%) - 11.5 to 26.4
            FeatureColumn('Laboratory', 'LBXRDW', 
                                 preproc_real, None),

            # White blood cell count (1000 cells/uL) - 1.4 to 117.2
            FeatureColumn('Laboratory', 'LBXWBCSI', 
                                 preproc_real, None),

            # Insulin (pmol/L) - 0.84 to 4094.88
            FeatureColumn('Laboratory', 'LBDINSI', 
                                 preproc_real, None),

            # HPV Type 16
            FeatureColumn('Laboratory', 'ORXH16', 
                                 preproc_real, {'cutoff':2}),

            # HPV Type 18
            FeatureColumn('Laboratory', 'ORXH18', 
                                 preproc_real, {'cutoff':2}),

            # Blood Benzene (ng/mL) - 0.017 to 1.97	
            FeatureColumn('Laboratory', 'LBXVBZ', 
                                 preproc_real, None),

            # Blood Tetrachloroethene (ng/mL) - 0.0339 to 57.5
            FeatureColumn('Laboratory', 'LBXV4C', 
                                 preproc_real, None),

            # Blood Bromoform (ng/mL) - 0.0057 to 0.254
            FeatureColumn('Laboratory', 'LBXVBF', 
                                 preproc_real, None),

            # Estradiol (pg/mL) - 2.117 to 1220
            FeatureColumn('Laboratory', 'LBXEST', 
                                 preproc_real, None),

            # SHBG (nmol/L) - 5.18 to 1067
            FeatureColumn('Laboratory', 'LBXSHBG', 
                                 preproc_real, None),

            # Testosterone total (ng/dL) - 0.25 to 2543.99
            FeatureColumn('Laboratory', 'LBXTST', 
                                 preproc_real, None),

            # Urinary Bisphenol A (ng/mL) - 0.14 to 792
            FeatureColumn('Laboratory', 'URXBPH', 
                                 preproc_real, None),

            # 2,5-dichlorophenol (ug/L) - 0.07 to 30426.3
            FeatureColumn('Laboratory', 'URX14D', 
                                 preproc_real, None)
            ######################## LABORATORY ########################
        ]
        nhanes_dataset = NHANES(self.data_path, columns)
        df = nhanes_dataset.process()
        fe_cols = df.drop(['MCQ220'], axis=1)
        features = fe_cols.values
        target = df['MCQ220'].values
        # remove nan labeled samples
        inds_valid = ~ np.isnan(target)
        features = features[inds_valid]
        target = target[inds_valid]

        # Put each person in the corresponding bin
        targets = np.zeros(target.shape[0])
        targets[target == 1] = 0 # yes arthritis
        targets[target == 2] = 1 # no arthritis

       # random permutation
        perm = np.random.permutation(targets.shape[0])
        self.features = features[perm]
        self.targets = targets[perm]
        self.costs = [c.cost for c in columns[1:]]
        self.costs = np.array(
            [item for sublist in self.costs for item in sublist])
        
        
    #### Add your own dataset loader ####
