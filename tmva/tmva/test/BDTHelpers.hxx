#ifndef TMVA_TEST_BDTHELPERS
#define TMVA_TEST_BDTHELPERS

#include "TFile.h"
#include "TDirectory.h"
#include "TInterpreter.h"

#include <vector>
#include <string>
#include <sstream>
#include <functional> // std::function

void WriteModel(std::string key, std::string filename, std::string objective, std::vector<int> inputs,
                std::vector<int> outputs, std::vector<float> thresholds, std::vector<int> max_depth,
                std::vector<int> num_trees, std::vector<int> num_inputs, std::vector<int> num_outputs)
{
   auto f = TFile::Open(filename.c_str(), "RECREATE");
   f->mkdir(key.c_str());
   auto d = (TDirectory *)f->Get(key.c_str());
   d->WriteObjectAny(&inputs, "std::vector<int>", "inputs");
   d->WriteObjectAny(&outputs, "std::vector<int>", "outputs");
   d->WriteObjectAny(&thresholds, "std::vector<float>", "thresholds");
   d->WriteObjectAny(&objective, "std::string", "objective");
   d->WriteObjectAny(&max_depth, "std::vector<int>", "max_depth");
   d->WriteObjectAny(&num_trees, "std::vector<int>", "num_trees");
   d->WriteObjectAny(&num_inputs, "std::vector<int>", "num_inputs");
   d->WriteObjectAny(&num_outputs, "std::vector<int>", "num_outputs");
   f->Write();
   f->Close();
}

template <typename T>
T JittedTreeInference(const std::string& nameSpace, const std::string& funcName, const std::string& code, const T* inputs, const int stride)
{
   // JIT the function in the given namespace
   std::stringstream jitss;
   jitss << "#pragma cling optimize(3)\nnamespace " << nameSpace << "{\n" << code << "\n}";
   const std::string jit = jitss.str();
   gInterpreter->Declare(jit.c_str());

   // Get the function pointer and make a function
   std::stringstream refss;
   refss << "#pragma cling optimize(3)\n" << nameSpace << "::" << funcName;
   const std::string refname = refss.str();
   auto ptr = gInterpreter->Calc(refname.c_str());
   T (*f)(const T *, const int) = reinterpret_cast<T (*)(const T *, const int)>(ptr);
   std::function<T(const T *, const int)> func{f};

   // Make computation and return
   return func(inputs, stride);
}

#endif // TMVA_TEST_BDTHELPERS
