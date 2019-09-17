#ifndef TMVA_TEST_BDTHELPERS
#define TMVA_TEST_BDTHELPERS

#include "TFile.h"
#include "TDirectory.h"

#include <vector>
#include <string>

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

#endif // TMVA_TEST_BDTHELPERS
