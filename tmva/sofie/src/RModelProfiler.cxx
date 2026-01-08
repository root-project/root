#include "TMVA/RModelProfiler.hxx"
#include "TMVA/SOFIE_common.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

// The constructor now just registers the necessary C++ libraries.
void RModelProfiler::AddNeededStdLibs(RModel &model)
{
   model.AddNeededStdLib("chrono");      // for timing operators
   model.AddNeededStdLib("vector");      // for storing profiling results
   model.AddNeededStdLib("string");      // for operator names
   model.AddNeededStdLib("map");         // for the results map
   model.AddNeededStdLib("iostream");    // for printing results
   model.AddNeededStdLib("iomanip");     // for printing results
}

// This function generates the helper functions inside the Session struct.
std::string RModelProfiler::GenerateUtilityFunctions()
{
   std::string gc;

   // Generate PrintProfilingResults function
   gc += "   // generate code for printing operator results. By default order according to time (from higher to lower)\n";
   gc += "   void PrintProfilingResults(bool order = true) const {\n";
   gc += "      if (fProfilingResults.empty()) {\n";
   gc += "         std::cout << \"No profiling results to display.\" << std::endl;\n";
   gc += "         return;\n";
   gc += "      }\n";
   gc += "\n";
   gc += "      // compute summary statistics of profiling results and sort them in decreasing time\n";
   gc += "      std::vector<std::tuple<std::string, double, double, int>> averageResults;\n";
   gc += "      std::cout << \"\\n\" << std::string(50, '=') << std::endl;\n";
   gc += "      std::cout << \"         AVERAGE PROFILING RESULTS\" << std::endl;\n";
   gc += "      std::cout << std::string(50, '=') << std::endl;\n";
   gc += "      for (const auto& op : fProfilingResults) {\n";
   gc += "         double sum = 0.0;\n";
   gc += "         double sum2 = 0.0;\n";
   gc += "         for (double time : op.second) {\n";
   gc += "            sum += time;\n";
   gc += "            sum2 += time*time;\n";
   gc += "         }\n";
   gc += "         double average = sum / op.second.size();\n";
   gc += "         double stddev = std::sqrt(( sum2 - sum *average)/ (op.second.size()-1));\n";
   gc += "         averageResults.push_back({op.first, average, stddev, op.second.size()});\n";
   gc += "      }\n";
   gc += "\n";
   gc += "      // sort average results in decreasing time\n";
   gc += "      std::sort(averageResults.begin(), averageResults.end(),\n";
   gc += "          []( std::tuple<std::string,double,double,int> a, std::tuple<std::string,double,double,int> b) {return std::get<1>(a) > std::get<1>(b); });\n";
   gc += "\n";
   gc += "      for (const auto & r : averageResults) {\n";
   gc += "         std::cout << \"  \" << std::left << std::setw(20) << std::get<0>(r)\n";
   gc += "                   << \": \" << std::fixed << std::setprecision(6) << std::get<1>(r) << \" +/- \" \n";
   gc += "                   << std::get<2>(r)/std::sqrt(std::get<3>(r)) << \" us\"\n";
   gc += "                   << \"  (over \" << std::get<3>(r) << \" runs)\" << std::endl;\n";
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

   return gc;
}

// Generate code for adding session member

// Main generation function for the profiler.
std::string RModelProfiler::GenerateSessionMembers()
{
   std::string gc;
   gc += "// Maps an operator name to a vector of its execution times (in microseconds).\n";
   // need to use mutable because we pass a const Session in the doInfer function
   gc += "mutable std::map<std::string, std::vector<double>> fProfilingResults;\n\n";
   return gc;
}
std::string RModelProfiler::GenerateBeginInferCode() {
   std::string gc;
   // 1. Add necessary code for instrumenting with timers
   gc += "   // Timer variable for profiling\n";
   gc += "   std::chrono::steady_clock::time_point tp_start, tp_overall_start;\n\n";
   gc += "   tp_overall_start = std::chrono::steady_clock::now();\n\n";
   gc += "   auto & fProfilingResults = session.fProfilingResults;\n\n";
   return gc;
}
std::string RModelProfiler::GenerateOperatorCode(ROperator &op, size_t op_idx) {
   std::string gc;
   gc += "   // -- Profiling for operator " + op.Name() + " --\n";
   gc += "   tp_start = std::chrono::steady_clock::now();\n\n";

   // Add the actual operator inference code
   gc += op.Generate(std::to_string(op_idx));

   // Add the code to stop the timer and store the result
   gc += "\n   fProfilingResults[\"" + op.Name() + "\"].push_back(\n";
   gc += "      std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(\n";
   gc += "         std::chrono::steady_clock::now() - tp_start).count());\n\n";
   return gc;
}

std::string RModelProfiler::GenerateEndInferCode() {
   std::string gc;
   gc += "\n   // -- Record overall inference time --\n";
   gc += "   fProfilingResults[\"Overall_Time\"].push_back(\n";
   gc += "      std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(\n";
   gc += "         std::chrono::steady_clock::now() - tp_overall_start).count());\n";
   return gc;
}

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
