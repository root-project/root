#include "TMVA/RModelProfiler.hxx"
#include "TMVA/SOFIE_common.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

// The constructor now just registers the necessary C++ libraries.
RModelProfiler::RModelProfiler(RModel &model) : fModel(model)
{
   fModel.AddNeededStdLib("chrono");      // for timing operators
   fModel.AddNeededStdLib("vector");      // for storing profiling results
   fModel.AddNeededStdLib("string");      // for operator names
   fModel.AddNeededStdLib("map");         // for the results map
   fModel.AddNeededStdLib("iostream");    // for printing results
   fModel.AddNeededStdLib("iomanip");     // for printing results
}

// This function generates the helper functions inside the Session struct.
void RModelProfiler::GenerateUtilityFunctions()
{
   auto &gc = fModel.fProfilerGC;

   // Generate PrintProfilingResults function
   gc += "   void PrintProfilingResults() const {\n";
   gc += "      if (fProfilingResults.empty()) {\n";
   gc += "         std::cout << \"No profiling results to display.\" << std::endl;\n";
   gc += "         return;\n";
   gc += "      }\n";
   gc += "\n";
   gc += "      std::cout << \"\\n\" << std::string(50, '=') << std::endl;\n";
   gc += "      std::cout << \"         AVERAGE PROFILING RESULTS\" << std::endl;\n";
   gc += "      std::cout << std::string(50, '=') << std::endl;\n";
   gc += "      for (const auto& op : fProfilingResults) {\n";
   gc += "         double sum = 0.0;\n";
   gc += "         for (double time : op.second) {\n";
   gc += "            sum += time;\n";
   gc += "         }\n";
   gc += "         double average = sum / op.second.size();\n";
   gc += "         std::cout << \"  \" << std::left << std::setw(20) << op.first\n";
   gc += "                   << \": \" << std::fixed << std::setprecision(6) << average << \" us\"\n";
   gc += "                   << \"  (over \" << op.second.size() << \" runs)\" << std::endl;\n";
   gc += "      }\n";
   gc += "      std::cout << std::string(50, '=') << \"\\n\" << std::endl;\n";
   gc += "   }\n";
   gc += "\n";

   // Generate ResetProfilingResults function
   gc += "   void ResetProfilingResults() {\n";
   gc += "      fProfilingResults.clear();\n";
   gc += "   }\n";
   gc += "\n";

   // Generate GetOpAvgTime function
   gc += "   std::map<std::string, double> GetOpAvgTime() const {\n";
   gc += "      if (fProfilingResults.empty()) {\n";
   gc += "         return {};\n";
   gc += "      }\n";
   gc += "\n";
   gc += "      std::map<std::string, double> avg;\n";
   gc += "      for (const auto& op : fProfilingResults) {\n";
   gc += "         double mean = 0.0;\n";
   gc += "         for (double time : op.second) {\n";
   gc += "            mean += time;\n";
   gc += "         }\n";
   gc += "         mean /= op.second.size();\n";
   gc += "         avg[op.first] = mean;\n";
   gc += "      }\n";
   gc += "\n";
   gc += "      return avg;\n";
   gc += "   }\n";
   gc += "\n";

   // Generate GetOpVariance function 
   gc += "   std::map<std::string, double> GetOpVariance() const {\n";
   gc += "      if (fProfilingResults.empty()) {\n";
   gc += "         return {};\n";
   gc += "      }\n";
   gc += "\n";
   gc += "      std::map<std::string, double> variance;\n";
   gc += "      for (const auto& op : fProfilingResults) {\n";
   gc += "         // Var[X] = E[X^2] - E[X]^2\n";
   gc += "         double mean = 0.0, mean2 = 0.0;\n";
   gc += "         for (double time : op.second) {\n";
   gc += "            mean += time;\n";
   gc += "            mean2 += time * time;\n";
   gc += "         }\n";
   gc += "         mean /= op.second.size();\n";
   gc += "         mean2 /= op.second.size();\n";
   gc += "         variance[op.first] = mean2 - mean * mean;\n";
   gc += "      }\n";
   gc += "\n";
   gc += "      return variance;\n";
   gc += "   }\n";
}

// Main generation function for the profiler.
void RModelProfiler::Generate()
{
   // Clear the profiler's code string to start fresh.
   fModel.fProfilerGC.clear();
   auto &gc = fModel.fProfilerGC;

   // 1. Add the data member to the Session struct to store results.
   gc += "public:\n";
   gc += "   // Maps an operator name to a vector of its execution times (in microseconds).\n";
   gc += "   std::map<std::string, std::vector<double>> fProfilingResults;\n\n";

   // 2. Generate and add the utility functions like PrintProfilingResults.
   GenerateUtilityFunctions();

   // 3. Generate the signature for the profiled doInfer method.
   std::string doInferSignature = fModel.GenerateInferSignature();
   if (!doInferSignature.empty()) doInferSignature += ", ";
   for (auto const &name : fModel.GetOutputTensorNames()) {
      doInferSignature += " std::vector<" + ConvertTypeToString(fModel.GetTensorType(name)) + "> &output_tensor_" + name + ",";
   }
   if (!fModel.GetOutputTensorNames().empty()) {
      doInferSignature.back() = ' ';
   }
   gc += "void doInfer(" + doInferSignature + ") {\n";

   // 4. Generate the body of the doInfer method with timing instrumentation.
   gc += "   // Timer variable for profiling\n";
   gc += "   std::chrono::steady_clock::time_point tp_start, tp_overall_start;\n\n";
   gc += "   tp_overall_start = std::chrono::steady_clock::now();\n\n";

   for (size_t op_idx = 0; op_idx < fModel.fOperators.size(); ++op_idx) {
      const auto& op = fModel.fOperators[op_idx];
      gc += "   // -- Profiling for operator " + op->name + " --\n";
      gc += "   tp_start = std::chrono::steady_clock::now();\n\n";
      
      // Add the actual operator inference code
      gc += op->Generate(std::to_string(op_idx));
      
      // Add the code to stop the timer and store the result
      gc += "\n   fProfilingResults[\"" + op->name + "\"].push_back(\n";
      gc += "      std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(\n";
      gc += "         std::chrono::steady_clock::now() - tp_start).count());\n\n";
   }

   // 5. Generate the code to fill the output tensors.
   gc += "   using TMVA::Experimental::SOFIE::UTILITY::FillOutput;\n\n";
   for (std::string const &name : fModel.GetOutputTensorNames()) {
      bool isIntermediate = fModel.fIntermediateTensorInfos.count(name) > 0;
      std::string n = isIntermediate ? std::to_string(ConvertShapeToLength(fModel.GetTensorShape(name)))
                                     : ConvertDynamicShapeToLength(fModel.GetDynamicTensorShape(name));
      gc += "   FillOutput(tensor_" + name + ", output_tensor_" + name + ", " + n + ");\n";
   }

   gc += "\n   // -- Record overall inference time --\n";
   gc += "   fProfilingResults[\"Overall_Time\"].push_back(\n";
   gc += "      std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(\n";
   gc += "         std::chrono::steady_clock::now() - tp_overall_start).count());\n";

   gc += "}\n\n"; // End of doInfer function
}

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
