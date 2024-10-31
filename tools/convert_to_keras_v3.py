# This file is part of the Extra-P Adaptive Modeler software (https://github.com/extra-p/extrap-adaptive-modeler)
#
# Copyright (c) 2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import argparse


def convert(input_file="../src/normalizedTerms_newPoints.h5", output_file="../src/normalizedTerms_newPoints.keras"):
    import h5py
    import keras
    import numpy

    assert (keras.__version__.startswith('3.'))
    model = keras.Sequential([
        keras.Input(shape=(11,)),
        keras.layers.Dense(1500, activation='tanh', name='dense_1089'),
        keras.layers.Dense(1500, activation='tanh', name='dense_1090'),
        keras.layers.Dense(750, activation='tanh', name='dense_1091'),
        keras.layers.Dense(250, activation='tanh', name='dense_1092'),
        keras.layers.Dense(250, activation='tanh', name='dense_1093'),
        keras.layers.Dense(43, activation='softmax', name='dense_1094')
    ])
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        optimizer=keras.optimizers.Adamax(learning_rate=1e-3, name="Adamax"),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.TopKCategoricalAccuracy(name="top2_ca", k=2),
            keras.metrics.TopKCategoricalAccuracy(name="top3_ca", k=3),
        ],
    )
    model.summary()
    x = numpy.array([numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])])
    y = numpy.array([numpy.array([0, 0, 1] + [0] * 40)])
    model.fit(x, y)
    model.load_weights(input_file)
    with h5py.File(input_file) as data:
        var_iter = iter(model.optimizer.variables)
        iteration_var = next(var_iter)
        optimizer_name, var_name = iteration_var.path.split('/')
        assert (var_name == "iteration")
        iteration_var.assign(data['optimizer_weights'][optimizer_name]['iter:0'])
        assert (next(var_iter).path.endswith("/learning_rate"))
        for var in var_iter:
            optimizer_name, var_name = var.path.split('/')
            model_name, layertype, layer_id, wtype, ctype = var_name.split('_')
            if ctype == 'norm':
                ds_name = 'v:0'
            elif ctype == 'momentum':
                ds_name = 'm:0'
            else:
                raise RuntimeError("Unknown ctype.")
            weights = data['optimizer_weights'][optimizer_name][layertype + '_' + layer_id][wtype][ds_name]
            var.assign(weights)
    model.save(output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", required=True)
    parser.add_argument("output_file", required=True)
    args = parser.parse_args()

    convert(input_file=args.input_file, output_file=args.output_file)


if __name__ == "__main__":
    main()
