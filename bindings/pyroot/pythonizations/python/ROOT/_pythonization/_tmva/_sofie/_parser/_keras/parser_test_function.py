import ROOT
import numpy as np
import keras

'''
The test file contains two types of functions:
    is_accurate:
        - This function checks whether the inference results from SOFIE and Keras are accurate within a specified 
          tolerance. Since the inference result from Keras is not flattened, the function flattens both tensors before 
          performing the comparison.
          
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
          NumPy arrays before being passed to the is_accurate function for comparison.

'''

def is_accurate(tensor_a, tensor_b, tolerance=1e-3):
    tensor_a = tensor_a.flatten()
    tensor_b = tensor_b.flatten()
    for i in range(len(tensor_a)):
        difference = abs(tensor_a[i] - tensor_b[i])
        if difference > tolerance:
            print(tensor_a[i], tensor_b[i])
            return False
    return True

def generate_and_test_inference(model_file_path: str, generated_header_file_dir: str = None, batch_size=1):
    model_name = model_file_path[model_file_path.rfind('/')+1:].removesuffix(".h5")
    rmodel = ROOT.TMVA.Experimental.SOFIE.RModelParser_Keras.Parse(model_file_path, batch_size)
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
    inference_session = sofie_model_namespace.Session(generated_header_file_path[:-4] + ".dat")
    keras_model = keras.models.load_model(model_file_path)
    keras_model.load_weights(model_file_path)
    if len(keras_model.inputs) == 1:
        input_shape = list(keras_model.inputs[0].shape)
        input_shape[0] = batch_size
        input_tensors = np.ones(input_shape, dtype='float32')
    else:
        input_tensors = []
        for model_input in keras_model.inputs:
            input_shape = list(model_input.shape)
            input_shape[0] = batch_size
            input_tensors.append(np.ones(input_shape, dtype='float32'))
    sofie_inference_result = inference_session.infer(*input_tensors)
    sofie_output_tensor_shape = list(rmodel.GetTensorShape(rmodel.GetOutputTensorNames()[0]))   # get output shape
                                                                                                # from SOFIE
    keras_inference_result = keras_model(input_tensors)
    if sofie_output_tensor_shape != list(keras_inference_result.shape):
        raise AssertionError("Output tensor dimensions from SOFIE and Keras do not match")
    sofie_inference_result = np.asarray(sofie_inference_result)
    keras_inference_result = np.asarray(keras_inference_result)
    is_inference_accurate = is_accurate(sofie_inference_result, keras_inference_result)
    if not is_inference_accurate:
        raise AssertionError("Inference results from SOFIE and Keras do not match")    