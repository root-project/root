#!/usr/bin/env python3

import os
import tempfile
import unittest

try:
    from keras import layers, models
    import numpy as np
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False

try:
    import ROOT
    from ROOT.TMVA.Experimental import SOFIE
    HAS_ROOT = True
except (ImportError, AttributeError):
    HAS_ROOT = False


@unittest.skipIf(not HAS_KERAS, "Keras not available")
@unittest.skipIf(not HAS_ROOT, "ROOT with PyROOT not available")
class TestKerasParserConv2DChannelsFirstSamePadding(unittest.TestCase):

    def test_conv2d_channels_first_same_padding_matches_keras(self):
        model = models.Sequential([
            layers.Input(shape=(2, 5, 5)),
            layers.Conv2D(4, (3, 3), padding='same', strides=(2, 2),
                         data_format='channels_first', activation='relu')
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        x = np.random.rand(2, 2, 5, 5).astype('float32')
        y = np.random.rand(2, 4, 3, 3).astype('float32')
        model.fit(x, y, epochs=1, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "conv2d_channels_first_same.keras")
            model.save(model_path)

            rmodel = ROOT.TMVA.Experimental.SOFIE.PyKeras.Parse(model_path, batch_size=1)
            hxx_path = os.path.join(tmpdir, "conv2d_channels_first_same.hxx")
            rmodel.Generate()
            rmodel.OutputGenerated(hxx_path)

            compile_status = ROOT.gInterpreter.Declare(f'#include "{hxx_path}"')
            if not compile_status:
                self.fail(f"Failed to compile generated header {hxx_path}")

            SessionClass = getattr(ROOT, "TMVA_SOFIE_conv2d_channels_first_same").Session
            dat_path = hxx_path.replace(".hxx", ".dat")
            session = SessionClass(dat_path)

            keras_model = models.load_model(model_path)
            input_arr = np.ones((1, 2, 5, 5), dtype='float32')
            sofie_result = np.asarray(session.infer(input_arr)).flatten()
            keras_result = np.asarray(keras_model(input_arr)).flatten()

            self.assertEqual(
                sofie_result.size,
                keras_result.size,
                "Output tensor dimensions from SOFIE and Keras do not match"
            )
            np.testing.assert_allclose(
                sofie_result,
                keras_result,
                rtol=1e-2,
                err_msg="Inference results from SOFIE and Keras do not match"
            )


if __name__ == '__main__':
    unittest.main()
