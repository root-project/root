import ROOT

'''
The test file contains two types of functions:

    generate_and_test_inference:
        - This function accepts the following inputs:
            - Model file path: Path to the input model.
            - Destination directory for the generated header file: If set to None, the header file will be generated in
              the model's directory.
            - Batch size.
        - After generating the inference code, we instantiate the session for inference. To validate the results from
          SOFIE, we compare the outputs from both SOFIE and Keras.
            - Load the Keras model.
            - Extract the input dimensions of the Keras model to avoid hardcoding.
            - For Sequential models or functional models with a single input:
                - Extract the model's input specification and create a NumPy array of ones with the same shape as the
                  model's input specification, replacing None with the batch size. This becomes the input tensor.
            - For functional models with multiple inputs:
            - Extract the dimensions for each input, set the batch size, create a NumPy array of ones for each input,
              and append each tensor to the list of input tensors.
            - These input tensors are then fed to both the instantiated session object and the Keras model.
        - Verify the output tensor dimensions:
          Since SOFIE always flattens the output tensors before returning them, we need to extract the output tensor
          shape from the model object.
        - Convert the inference results to NumPy arrays:
          The SOFIE result is of type vector<float>, and the Keras result is a TensorFlow tensor. Both are converted to
          NumPy arrays before being passed to the np.testing.assert_allclose function for comparison.

'''
def is_channels_first_supported() :
      #channel first is not supported on tensorflow CPU versions
      from keras import backend
      if backend.backend() == "tensorflow" :
         import os
         if os.environ.get("ROOT_TMVA_SOFIE_KERAS_CPU_CHANNELS_FIRST", "") == "1":
            return True
         import tensorflow as tf
         if len(tf.config.list_physical_devices("GPU")) == 0:
            return False

      return True

def generate_and_test_inference(model_file_path: str, generated_header_file_dir: str = None, batch_size=1):

    import keras
    import numpy as np
    import tensorflow as tf

    print("Tensorflow version: ", tf.__version__)
    print("Keras version: ", keras.__version__)
    print("Numpy version:", np.__version__)

    model_name = model_file_path[model_file_path.rfind('/')+1:].removesuffix(".keras")
    rmodel = ROOT.TMVA.Experimental.SOFIE.PyKeras.Parse(model_file_path, batch_size)
    if generated_header_file_dir is None:
        last_idx = model_file_path.rfind("/")
        if last_idx == -1:
            generated_header_file_dir = "./"
        else:
            generated_header_file_dir = model_file_path[:last_idx]
    generated_header_file_path = generated_header_file_dir + "/" + model_name + ".hxx"
    print(f"Generating inference code for the Keras model from {model_file_path} in the header {generated_header_file_path}")
    rmodel.Generate()
    rmodel.OutputGenerated(generated_header_file_path)
    print(f"Compiling SOFIE model {model_name}")
    compile_status = ROOT.gInterpreter.Declare(f'#include "{generated_header_file_path}"')
    if not compile_status:
        raise AssertionError(f"Error compiling header file {generated_header_file_path}")
    sofie_model_namespace = getattr(ROOT, "TMVA_SOFIE_" + model_name)
    inference_session = sofie_model_namespace.Session(generated_header_file_path.removesuffix(".hxx") + ".dat")
    keras_model = keras.models.load_model(model_file_path)

    input_tensors = []
    for model_input in keras_model.inputs:
      input_shape = list(model_input.shape)
      input_shape[0] = batch_size
      input_tensors.append(np.ones(input_shape, dtype='float32'))
    sofie_inference_result = inference_session.infer(*input_tensors)
    sofie_output_tensor_shape = list(rmodel.GetTensorShape(rmodel.GetOutputTensorNames()[0]))   # get output shape
                                                                                                # from SOFIE
    # Keras explicitly forbids input tensor lists of size 1
    if len(keras_model.inputs) == 1:
        keras_inference_result = keras_model(input_tensors[0])
    else:
        keras_inference_result = keras_model(input_tensors)
    if sofie_output_tensor_shape != list(keras_inference_result.shape):
        raise AssertionError("Output tensor dimensions from SOFIE and Keras do not match")

    np.testing.assert_allclose(
        np.asarray(sofie_inference_result).flatten(),
        np.asarray(keras_inference_result).flatten(),
        atol=1e-2,
        rtol=0.  # explicitly disable relative tolerance (NumPy uses |a - b| <= atol + rtol * |b|)
    )
