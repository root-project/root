#ifndef TMVA_SOFIE_RMODELPROFILER
#define TMVA_SOFIE_RMODELPROFILER

#include <vector>
#include <unordered_map>
#include <iostream>
#include <memory>
#include <ctime>
#include <set>
#include <iomanip>
#include <fstream>
#include <sstream>

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

namespace TMVA{
namespace Experimental{
namespace SOFIE{

class RModelProfiler : public RModel {

private:
   void GenerateUtilityFunctions();

public:

   RModelProfiler() = delete;
   ~RModelProfiler() { }
   RModelProfiler(RModel&& model);

   // default move ctor and move assn
   RModelProfiler(RModelProfiler&& other) = default;
   RModelProfiler& operator=(RModelProfiler&& other) = default;

   // disallow copy
   RModelProfiler(const RModelProfiler& other) = delete;
   RModelProfiler& operator=(const RModelProfiler& other) = delete;

   void Generate();

   // Set whether to generate utility functions in the output code.
   // Utility functions include:
   //   - GetOpAvgTime()
   bool UtilityFunctionsGeneration = true;

};

}//SOFIE
}//Experimental
}//TMVA

#endif //TMVA_SOFIE_RMODELPROFILER
