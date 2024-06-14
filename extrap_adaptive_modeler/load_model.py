# This file is part of the Extra-P Adaptive Modeler software (https://github.com/extra-p/extrap-adaptive-modeler)
#
# Copyright (c) 2024, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.


import os
from pathlib import Path

import importlib_resources
from extrap.entities.terms import CompoundTerm

from extrap_adaptive_modeler.data_generator import TrainingDataGenerator

_ML_MODEL_NAME = 'normalizedTerms_newPoints'

_TERMS = [CompoundTerm.create(*expo) for expo in
          [(0, 1, 0), (0, 1, 1), (0, 1, 2), (1, 4, 0), (1, 3, 0),
           (1, 4, 1), (1, 3, 1), (1, 4, 2), (1, 3, 2), (1, 2, 0),
           (1, 2, 1), (1, 2, 2), (2, 3, 0), (3, 4, 0), (2, 3, 1),
           (3, 4, 1), (4, 5, 0), (2, 3, 2), (3, 4, 2), (1, 1, 0),
           (1, 1, 1), (1, 1, 2), (5, 4, 0), (5, 4, 1), (4, 3, 0),
           (4, 3, 1), (3, 2, 0), (3, 2, 1), (3, 2, 2), (5, 3, 0),
           (7, 4, 0), (2, 1, 0), (2, 1, 1), (2, 1, 2), (9, 4, 0),
           (7, 3, 0), (5, 2, 0), (5, 2, 1), (5, 2, 2), (8, 3, 0),
           (11, 4, 0), (3, 1, 0), (3, 1, 1)]]


def get_model(bucket_indices, noise_category, positions, tf, retrain_epochs, retrain_examples_per_class,
              _cached_mlmodels):
    ml_model_key = (noise_category, tuple(positions))
    if ml_model_key not in _cached_mlmodels:
        ml_model_path = os.path.join(os.path.dirname(__file__), _ML_MODEL_NAME + '.h5')
        with importlib_resources.as_file(Path(ml_model_path)) \
                as model_file:
            ml_model = tf.keras.models.load_model(model_file)
        data_gen = TrainingDataGenerator(_TERMS, positions, bucket_indices)
        data_gen.noise = noise_category
        train_data = data_gen.create_data(retrain_examples_per_class)
        # batch size is tensorflow default value (32)
        ml_model.fit(train_data[0], train_data[1], batch_size=32, epochs=retrain_epochs)
        _cached_mlmodels[ml_model_key] = ml_model
    else:
        ml_model = _cached_mlmodels[ml_model_key]
    return ml_model, _cached_mlmodels
