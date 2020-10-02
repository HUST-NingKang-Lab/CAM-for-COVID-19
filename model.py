import pandas as pd, numpy as np, tensorflow as tf
from tensorflow import keras
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    #...
    args = parser.parse_args()
    if args.mode == 'predict':
        model = CAM(restore_from=args.model)
        data = pd.read_excel(args.input, index_col=0, sheet_name='data').set_index('SampleID')
        missing_cols = {'lymphocyte(%)', 'Neutrophil', 'LDH', 'CRP'} - data.columns.to_set()
        assertEqual(len(missing_cols), 0, msg='Missing columns: {}'.format(missing_cols))

    if args.mode == 'adapt':
        data = pd.read_excel(args.input, index_col=0, sheet_name='data').set_index('SampleID')
        missing_cols = {'lymphocyte(%)', 'Neutrophil', 'LDH', 'CRP', 'Status'} - data.columns.to_set()
        assertEqual(len(missing_cols), 0, msg='Missing columns: {}'.format(missing_cols))

        model = CAM(restore_from=args.model)
        X = model.normalize(data[['lymphocyte(%)', 'Neutrophil', 'LDH', 'CRP']])
        Y = data['Status']
        model.update_adaptor()
        model.freeze_predictor()
        model.update_nn()
        model.nn.compile(optimizer='', loss='', metrics='')
        callbacks = []
        model.nn.fit(X, Y, batchsize=16, epochs=1000, callbacks=callbacks, validation_split=0.1)
        model.unfreeze_predictor()
        model.update_nn()
        model.nn.compile(optimizer='', loss='', metrics='')
        callbacks = []
        model.nn.fit(X, Y, batchsize=16, epochs=1000, callbacks=callbacks, validation_split=0.1)
        model.save(args.output)

    elif args.mode == 'fit':
        data = pd.read_excel(args.input, index_col=0, sheet_name='data').set_index('SampleID')
        missing_cols = {'lymphocyte(%)', 'Neutrophil', 'LDH', 'CRP', 'Status'} - data.columns.to_set()
        assertEqual(len(missing_cols), 0, msg='Missing columns: {}'.format(missing_cols))

        model = CAM()
        X = model.normalize(data[['lymphocyte(%)', 'Neutrophil', 'LDH', 'CRP']])
        Y = data['Status']
        model.nn.compile(optimizer='', loss='', metrics='')
        callbacks = []
        model.nn.fit(X, Y, batchsize=16, epochs=1000, callbacks=callbacks, validation_split=0.1)
        model.save(args.output)

    elif args.mode == 'evaluate':
        data = pd.read_excel(args.input, index_col=0, sheet_name='data').set_index('SampleID')
        missing_cols = {'lymphocyte(%)', 'Neutrophil', 'LDH', 'CRP', 'Status'} - data.columns.to_set()
        assertEqual(len(missing_cols), 0, msg='Missing columns: {}'.format(missing_cols))

        model = CAM(restore_from=args.model)
        X = model.normalize(data[['lymphocyte(%)', 'Neutrophil', 'LDH', 'CRP']])
        Y = data['Status']
        model.nn.compile(optimizer='', loss='', metrics='')
        callbacks = []
        model.nn.evaluate(X, Y)

class CAM(object):

    def __init__(self, restore_from=None):
        if restore_from:
            self.restore(restore_from)
        else:
            self.stats = {}
            self.initializer = keras.initializers.HeNormal(seed=2)
            self.update_adaptor()
            self.predictor = self.init_predictor()
            self.update_nn()

    def normalize(self, X):
        if len(X) == 0:
            self.stats['variances'] = X.var()
            self.stats['mean'] = X.mean()
        return (X - self.stats['mean']).fillna(0) / self.stats['variances']

    def init_adaptor(self):
        block = keras.Sequential([], name='Adaptor')
        block.add(keras.layers.Dense(4))
        return block

    def init_predictor(self):
        block = keras.Sequential([], name='Predictor')
        block.add(keras.layers.Dense(64, activation='relu', kernel_initializer=self.initializer))
        block.add(keras.layers.Dropout(0.75))
        block.add(keras.layers.Dense(16, activation='relu', kernel_initializer=self.initializer))
        block.add(keras.layers.Dropout(0.75))
        block.add(keras.layers.Dense(4, activation='relu', kernel_initializer=self.initializer))
        block.add(keras.layers.Dropout(0.75))
        block.add(keras.layers.Dense(1, activation='sigmoid'))
        return block

    def build_graph():
        X = keras.layers.Input(shape(4,))
        X_tmp = self.adaptor(X)
        Y = self.predictor(X_tmp) # training?
        return Y

    def update_adaptor(self):
        self.adaptor = self.init_adaptor()

    def update_nn(self):
        self.nn = self.build_graph()

    def freeze_predictor():
        self.predictor.trainable = False

    def unfreeze_predictor():
        self.predictor.trainable = True

    def save_blocks(path):
        pass

