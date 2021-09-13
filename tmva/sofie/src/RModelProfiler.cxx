#include "TMVA/RModelProfiler.hxx"

namespace TMVA{
namespace Experimental{
namespace SOFIE{

   RModelProfiler::RModelProfiler(RModel& model) : fModel(model) {
      fModel.fNeededStdLib.insert("chrono");           // for timing operators
      fModel.fNeededStdLib.insert("unordered_map");    // for storing profiling results
      fModel.fNeededStdLib.insert("string");           // operator names
   }

   void RModelProfiler::GenerateUtilityFunctions() {
      // Generation of 'GetOpAvgTime()'
      fModel.fGC +=
         "ProfilerResult GetOpAvgTime() {\n"
         "\tif (profiler_results.size() == 0) {\n"
         "\t\treturn {};\n"
         "\t}\n"
         "\t\n"
         "\tProfilerResult avg;\n"
         "\tfor (auto&& op : profiler_results) {\n"
         "\t\tdouble mean = 0;\n"
         "\t\tfor (double d : op.second) {\n"
         "\t\t\tmean += d;\n"
         "\t\t}\n"
         "\t\tmean /= op.second.size();\n"
         "\t\tavg[op.first] = mean;\n"
         "\t}\n"
         "\t\n"
         "\treturn avg;\n"
         "}\n";

      // Generation of 'GetOpVariance()'
      // To exploit locality of reference the variance is calculated according
      // to the formula:
      // Var[X] = E[X^2] - E[X]^2
      fModel.fGC +=
         "ProfilerResult GetOpVariance() {\n"
         "\tif (profiler_results.size() == 0) {\n"
         "\t\treturn {};\n"
         "\t}\n"
         "\t\n"
         "\tProfilerResult var;\n"
         "\tfor (auto&& op : profiler_results) {\n"
         "\t\t// Var[X] = E[X^2] - E[X]^2\n"
         "\t\tdouble mean = 0, mean2 = 0;\n"
         "\t\tfor (double d : op.second) {\n"
         "\t\t\tmean += d;\n"
         "\t\t\tmean2 += d * d;\n"
         "\t\t}\n"
         "\t\tmean /= op.second.size();\n"
         "\t\tmean2 /= op.second.size();\n"
         "\t\tvar[op.first] = mean2 - mean * mean;\n"
         "\t}\n"
         "\t\n"
         "\treturn var;\n"
         "}\n";
   }

   void RModelProfiler::Generate(){
      fModel.Initialize();
      auto& fGC = fModel.fGC;
      fGC += "//Code for profiling and benchmarking purposes.\n";
      fGC += ("//Code generated automatically by TMVA for Inference of Model file [" + fModel.fFileName + "] at [" + fModel.fParseTime.substr(0, fModel.fParseTime.length()-1) +"] \n");
      for (auto& i: fModel.fNeededStdLib) {
         fGC += "#include<" + i + ">\n";
      }
      fGC += ("namespace TMVA_SOFIE_" + fModel.fName + "{\n");
      if (!fModel.fNeededBlasRoutines.empty()) {
         fGC += ("namespace BLAS{\n");
         for (auto &routine : fModel.fNeededBlasRoutines) {
            if (routine == "Gemm") {
               fGC += ("\textern \"C\" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,\n"
                       "\t                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,\n"
                       "\t                       const float * beta, float * C, const int * ldc);\n");
            } else if (routine == "Gemv") {
               fGC += ("\textern \"C\" void sgemv_(const char * trans, const int * m, const int * n, const float * alpha, const float * A,\n"
                       "\t                       const int * lda, const float * X, const int * incx, const float * beta, const float * Y, const int * incy);\n");
            } else if (routine == "Axpy") {
               fGC += ("\textern \"C\" void saxpy_(const int * n, const float * alpha, const float * x,\n"
                       "\t                         const int * incx, float * y, const int * incy);\n");
            }
         }
         fGC += ("}//BLAS\n");
      }

      // Every time 'infer' is called every operator gets timed in this variable
      fGC += "// Maps an operator name to its execution time in a run.\n";
      fGC += "using ProfilerResult = std::unordered_map<std::string,double>;\n";
      fGC += "std::unordered_map<std::string,std::vector<double>> profiler_results;\n";

      if (UtilityFunctionsGeneration) {
         GenerateUtilityFunctions();
      }
      fGC += "\n";

      for (auto& i: fModel.fInitializedTensors){
         if (i.second.fType == ETensorType::FLOAT){
            size_t length = 1;
            for (auto & dim: i.second.fShape){
               length *= dim;
            }
            fGC += "float tensor_" + i.first + "[" + std::to_string(length) + "] = {";
            std::shared_ptr<float> data = std::static_pointer_cast<float>(i.second.fData);
            std::stringstream floats;
            for (size_t idx = 0; idx < length-1; idx++){
               floats << std::setprecision(std::numeric_limits<float>::max_digits10) << data.get()[idx] << ", ";
            }
            floats << std::setprecision(std::numeric_limits<float>::max_digits10) << data.get()[length-1];
            fGC += floats.str() +"};\n";
         }
      }
      for (auto&i: fModel.fIntermediateTensorInfos){
         if (i.second.type == ETensorType::FLOAT){
            size_t length = 1;
            for (auto & dim: i.second.shape){
               length *= dim;
            }
            fGC += "float tensor_" + i.first + "[" + std::to_string(length) + "];\n";
         }
      }

      if (fModel.fOutputTensorNames.size() == 1){
         auto f = fModel.fIntermediateTensorInfos.find(fModel.fOutputTensorNames[0]);
         if (f == fModel.fIntermediateTensorInfos.end()){
            throw std::runtime_error("TMVA-SOFIE: output tensor " + fModel.fOutputTensorNames[0] + " not found when trying to get its info");
         }else{
            if (f->second.type == ETensorType::FLOAT){
               fGC += "std::vector<float> ";
            }
         }
      }else{
         std::cout << fModel.fOutputTensorNames.size() << std::endl;
         throw std::runtime_error("TMVA-SOFIE: More than 1 output tensor is not yet supported");
      }

      fGC += "infer(";
      for (auto& i: fModel.fReadyInputTensorInfos){
         size_t length = 1;
         for (auto& dim: i.second.shape){
            length *= dim;
         }
         if (i.second.type == ETensorType::FLOAT){
         fGC += "float* tensor_" + i.first + ",";
         }
      }
      fGC.pop_back(); //remove last ","
      fGC += "){\n";

      // Creating variables for timing
      fGC += "\tstd::chrono::steady_clock::time_point tp_start;\n";
      //fGC += "\tProfilerResult current_execution;\n";

      for (size_t id = 0; id < fModel.fOperators.size(); id++){
         // Starting timer
         fGC += "\ttp_start = std::chrono::steady_clock::now();\n";
         fGC += (fModel.fOperators[id]->Generate(std::to_string(id)));
         // Stopping timer
         fGC += "\profiler_results[\"" + fModel.fOperators[id]->name + "\"].push_back(std::chrono::duration_cast<std::chrono::microseconds>(\n";
         fGC += "\t\tstd::chrono::steady_clock::now() - tp_start).count() / 1e0);\n";
      }
      //fGC += "\tprofiler_results.push_back(std::move(current_execution));\n";
      if (fModel.fOutputTensorNames.size() == 1){
         fGC += "\tstd::vector<float> ret (tensor_" + fModel.fOutputTensorNames[0] + ", tensor_" + fModel.fOutputTensorNames[0] + " + sizeof(tensor_" +
               fModel.fOutputTensorNames[0] + ") / sizeof(tensor_" + fModel.fOutputTensorNames[0] + "[0]));\n";
         fGC += "\treturn ret;\n";
      }
      fGC += "}\n";
      fGC += ("} //TMVA_SOFIE_" + fModel.fName + "\n");
   }

}//SOFIE
}//Experimental
}//TMVA
