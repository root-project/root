// Author: Federico Sossai
// Last modified: 2021/07/30
// Description:
//    Emits the code for all ONNX models in a directory into the current
//    working directory, for the SofieCompileModels_ONNX test.
//    Usage: ./emitFromONNX <models_dir>

#include "SOFIE/RModel.hxx"
#include "SOFIE/RModelParser_ONNX.hxx"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <vector>

int EmitModel(std::string filename, std::string outname)
{
   using namespace SOFIE;

   std::cout << "parsing file ..." << filename << std::endl;
   {
      // The generated code with all optimizations. Used for most SOFIE tests.
      RModelParser_ONNX parser;
      RModel model = parser.Parse(filename);
      model.Generate();
      model.OutputGenerated(outname + "_FromONNX.hxx");
   }
   {
      // Generate code without memory re-use for intermediate tensors.
      //
      // IMPORTANT:
      // When memory re-use is enabled, SOFIE may assign multiple intermediate
      // tensors to the same memory buffer. This means that values produced earlier
      // in the forward pass can be overwritten by later operations.
      //
      // This is safe for inference, but it breaks source-transformation automatic
      // differentiation (e.g. with Clad). In reverse-mode AD, the backward pass
      // needs access to the original intermediate values from the forward pass
      // (e.g. inputs to activations like ReLU). If those values have been
      // overwritten, the generated gradient code will read incorrect data and
      // produce wrong results.
      //
      // Since Clad operates on the generated source code and is not aware of these
      // aliasing/reuse optimizations, it cannot reconstruct or recompute the lost
      // values. Therefore we disable memory re-use here to ensure correctness of
      // the differentiated code.
      //
      // Note: this increases memory usage but is required for AD correctness.
      RModelParser_ONNX parser;
      RModel model = parser.Parse(filename);
      model.SetOptimizationLevel(OptimizationLevel::kBasic);
      model.Generate();
      model.OutputGenerated(outname + "_FromONNX_unoptimized.hxx");
   }

   return 0;
}

int main(int argc, char *argv[])
{
   if (argc != 2) {
      std::cerr << "Usage: " << argv[0] << " <models_dir>" << std::endl;
      return 1;
   }

   std::vector<std::filesystem::path> files;
   for (auto const &entry : std::filesystem::directory_iterator(argv[1])) {
      if (entry.path().extension() == ".onnx")
         files.push_back(entry.path());
   }
   if (files.empty()) {
      std::cerr << "No .onnx files found in " << argv[1] << std::endl;
      return 1;
   }
   std::sort(files.begin(), files.end());

   for (auto const &file : files)
      EmitModel(file.string(), file.stem().string());

   return 0;
}
