/// \file
/// \ingroup tutorial_ml
/// \notebook -nodraw
/// This macro shows how to use the SOFIE profiler.
/// It parses a model, generates profiled C++ code, runs inference,
/// and prints the timing results for each operation.
///
/// \macro_code
/// \macro_output
/// \author Olha Sirikova, Lorenzo Moneta, Sanjiban Sengupta

using namespace TMVA::Experimental;

void TMVA_SOFIE_Profiler(const std::string& modelName = "Linear_16") {
   // Use a standard ONNX model from the ROOT tutorials directory.
   std::string inputFile = std::string(gROOT->GetTutorialsDir()) + "/machine_learning/" + modelName + ".onnx";
   if (gSystem->AccessPathName(inputFile.c_str())) {
      std::cout << "Error: Could not find input file: " << inputFile << std::endl;
      return;
   }

   // Parse the ONNX file into a SOFIE RModel object.
   SOFIE::RModelParser_ONNX parser;
   SOFIE::RModel model = parser.Parse(inputFile);

   // Generate inference code with profiling enabled using the kProfile option.
   std::cout << "Generating profiled inference code..." << std::endl;
   model.Generate(SOFIE::Options::kProfile);

   // Write the generated code to .hxx and .dat files.
   model.OutputGenerated();
   std::cout << "Generated files: " << modelName << ".hxx and " << modelName << ".dat" << std::endl;

   // Load and compile the generated model's header file.
   std::cout << "\nCompiling the generated code..." << std::endl;
   gROOT->ProcessLine(TString::Format(".L %s.hxx+", modelName.c_str()));

   // Construct the name of the generated Session class.
   TString sessionTypeName = TString::Format("TMVA_SOFIE_%s::Session", modelName.c_str());

   // Create a new Session object via the interpreter and get its address.
   Long_t sessionAddr = gROOT->ProcessLine(TString::Format("new %s();", sessionTypeName.Data()));

   // Prepare input and output data vectors for the model.
   std::vector<float> input_tensor(1, 1.0f);
   std::vector<float> output_tensor(16, 0.0f);

   // Run inference many times to collect timing statistics.
   int n_inferences = 1000;
   std::cout << "\nRunning inference " << n_inferences << " times..." << std::endl;
   for (int i = 0; i < n_inferences; ++i) {
      // Call the doInfer method via the interpreter, passing pointers to the data.
      gROOT->ProcessLine(TString::Format("((%s*)%ld)->doInfer((float*)%p, *(std::vector<float>*)%p);",
                                         sessionTypeName.Data(), sessionAddr, input_tensor.data(), &output_tensor));
   }
   std::cout << "Inference complete." << std::endl;

   // Display the profiling results.
   // Print results ordered by time (slowest first).
   gROOT->ProcessLine(TString::Format("((%s*)%ld)->PrintProfilingResults(true);", sessionTypeName.Data(), sessionAddr));

   // Print results in their original execution order.
   gROOT->ProcessLine(TString::Format("((%s*)%ld)->PrintProfilingResults(false);", sessionTypeName.Data(), sessionAddr));

   // Reset the profiler data.
   std::cout << "Resetting profiling data..." << std::endl;
   gROOT->ProcessLine(TString::Format("((%s*)%ld)->ResetProfilingResults();", sessionTypeName.Data(), sessionAddr));
   gROOT->ProcessLine(TString::Format("((%s*)%ld)->PrintProfilingResults(true);", sessionTypeName.Data(), sessionAddr));

   // Clean up the Session object to avoid memory leaks.
   gROOT->ProcessLine(TString::Format("delete ((%s*)%ld);", sessionTypeName.Data(), sessionAddr));
}
