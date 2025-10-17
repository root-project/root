#include "TMVA/RModelProfiler.hxx"
#include "TMVA/SOFIE_common.hxx"
#include <sstream>

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
   // Additions for supervisor's requests
   fModel.AddNeededStdLib("utility");
   fModel.AddNeededStdLib("algorithm");
   fModel.AddNeededStdLib("cmath");
   fModel.AddNeededStdLib("sstream");
}

// This function generates the helper functions inside the Session struct.
void RModelProfiler::GenerateUtilityFunctions()
{
   auto &gc = fModel.fProfilerGC;

   // Generate PrintProfilingResults function
   gc += "   void PrintProfilingResults(bool ordered = true) const {\n";
   gc += "      if (fProfilingResults.empty()) {\n";
   gc += "         std::cout << \"No profiling results to display.\" << std::endl;\n";
   gc += "         return;\n";
   gc += "      }\n";
   gc += "\n";
   gc += "      // Helper struct to store full profiling info\n";
   gc += "      struct ProfileData { std::string name; double average; double error; size_t runs; };\n";
   gc += "      std::vector<ProfileData> results;\n\n";

   gc += "      if (ordered) {\n";
   gc += "         // For ordered view, iterate through the map (which is alphabetical)\n";
   gc += "         for (const auto& op : fProfilingResults) {\n";
   gc += "            double sum = 0.0, sum2 = 0.0;\n";
   gc += "            for (double time : op.second) { sum += time; sum2 += time * time; }\n";
   gc += "            const size_t n_runs = op.second.size();\n";
   gc += "            const double average = (n_runs > 0) ? sum / n_runs : 0.0;\n";
   gc += "            const double variance = (n_runs > 1) ? (sum2 / n_runs - average * average) : 0.0;\n";
   gc += "            const double error_on_mean = (n_runs > 0) ? std::sqrt(variance / n_runs) : 0.0;\n";
   gc += "            results.push_back({op.first, average, error_on_mean, n_runs});\n";
   gc += "         }\n";
   gc += "         // Then sort the results by average time\n";
   gc += "         std::sort(results.begin(), results.end(), [](const ProfileData& a, const ProfileData& b) { return a.average > b.average; });\n";
   gc += "      } else {\n";
   gc += "         // For execution order view, iterate through our explicitly stored execution order vector\n";
   gc += "         results.reserve(fExecutionOrder.size()); // Pre-allocate memory for efficiency\n";
   gc += "         for (const auto& op_name : fExecutionOrder) {\n";
   gc += "            const auto& op_timings = fProfilingResults.at(op_name);\n";
   gc += "            double sum = 0.0, sum2 = 0.0;\n";
   gc += "            for (double time : op_timings) { sum += time; sum2 += time * time; }\n";
   gc += "            const size_t n_runs = op_timings.size();\n";
   gc += "            const double average = (n_runs > 0) ? sum / n_runs : 0.0;\n";
   gc += "            const double variance = (n_runs > 1) ? (sum2 / n_runs - average * average) : 0.0;\n";
   gc += "            const double error_on_mean = (n_runs > 0) ? std::sqrt(variance / n_runs) : 0.0;\n";
   gc += "            results.push_back({op_name, average, error_on_mean, n_runs});\n";
   gc += "         }\n";
   gc += "      }\n";
   gc += "\n";

   // --- PRINTING LOGIC (remains the same, but now uses the correctly ordered 'results' vector) ---\n";
   gc += "      if (ordered) {\n";
   gc += "         std::cout << \"\\n\" << std::string(80, '=') << std::endl;\n";
   gc += "         std::cout << \"                  PROFILING RESULTS (ORDERED BY TIME)\" << std::endl;\n";
   gc += "      } else {\n";
   gc += "         std::cout << \"\\n\" << std::string(80, '=') << std::endl;\n";
   gc += "         std::cout << \"                PROFILING RESULTS (EXECUTION ORDER)\" << std::endl;\n";
   gc += "      }\n";
   gc += "      std::cout << std::string(80, '=') << std::endl;\n";
   gc += "      for (const auto & op : results) {\n";
   gc += "         std::stringstream ss;\n";
   gc += "         ss << std::fixed << std::setprecision(4) << op.average << \" +/- \" << op.error;\n";
   gc += "         std::cout << \"  \" << std::left << std::setw(25) << op.name\n";
   gc += "                   << \": \" << std::left << std::setw(25) << ss.str()\n";
   gc += "                   << \"(over \" << op.runs << \" runs)\" << std::endl;\n";
   gc += "      }\n";
   gc += "      std::cout << std::string(80, '=') << \"\\n\" << std::endl;\n";
   gc += "   }\n";
   gc += "\n";

   // Generate ResetProfilingResults function
   gc += "   void ResetProfilingResults() {\n";
   gc += "      fProfilingResults.clear();\n";
   gc += "      fExecutionOrder.clear(); // Also clear the execution order vector\n";
   gc += "   }\n";
   gc += "\n";

   // Generate GetOpAvgTime function
   gc += "   std::map<std::string, double> GetOpAvgTime() const {\n";
   gc += "      if (fProfilingResults.empty()) {\n";
   gc += "         return {};\n";
   gc += "      }\n";
   gc += "      std::map<std::string, double> avg;\n";
   gc += "      for (const auto& op : fProfilingResults) {\n";
   gc += "         double mean = 0.0; for (double time : op.second) { mean += time; } mean /= op.second.size();\n";
   gc += "         avg[op.first] = mean;\n";
   gc += "      }\n";
   gc += "      return avg;\n";
   gc += "   }\n";
   gc += "\n";

   // Generate GetOpVariance function
   gc += "   std::map<std::string, double> GetOpVariance() const {\n";
   gc += "      if (fProfilingResults.empty()) { return {}; }\n";
   gc += "      std::map<std::string, double> variance;\n";
   gc += "      for (const auto& op : fProfilingResults) {\n";
   gc += "         double mean = 0.0, mean2 = 0.0; for (double time : op.second) { mean += time; mean2 += time * time; }\n";
   gc += "         mean /= op.second.size(); mean2 /= op.second.size();\n";
   gc += "         variance[op.first] = mean2 - mean * mean;\n";
   gc += "      }\n";
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
   gc += "   std::map<std::string, std::vector<double>> fProfilingResults;\n";
   gc += "   // Stores operator names to preserve the original execution order.\n";
   gc += "   std::vector<std::string> fExecutionOrder;\n\n";

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
      gc += op->Generate(std::to_string(op_idx));
      gc += "\n   fProfilingResults[\"" + op->name + "\"].push_back(\n";
      gc += "      std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(\n";
      gc += "         std::chrono::steady_clock::now() - tp_start).count());\n";
      gc += "   if (fProfilingResults.at(\"" + op->name + "\").size() == 1) {\n";
      gc += "      fExecutionOrder.push_back(\"" + op->name + "\");\n";
      gc += "   }\n\n";
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
   gc += "   if (fProfilingResults.at(\"Overall_Time\").size() == 1) {\n";
   gc += "      fExecutionOrder.push_back(\"Overall_Time\");\n";
   gc += "   }\n";


   gc += "}\n\n"; // End of doInfer function
}

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
