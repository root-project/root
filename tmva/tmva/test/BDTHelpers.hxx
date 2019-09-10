#ifndef TMVA_TEST_BDTHELPERS
#define TMVA_TEST_BDTHELPERS

#include "TFile.h"
#include "TDirectory.h"

#include <vector>
#include <string>

void WriteModel(std::string key, std::string filename, std::string objective, std::vector<int> features,
                std::vector<float> thresholds, std::vector<int> max_depth, std::vector<int> num_trees,
                std::vector<int> num_features, std::vector<int> num_classes)
{
   auto f = TFile::Open(filename.c_str(), "RECREATE");
   f->mkdir(key.c_str());
   auto d = (TDirectory *)f->Get(key.c_str());
   d->WriteObjectAny(&features, "std::vector<int>", "features");
   d->WriteObjectAny(&thresholds, "std::vector<float>", "thresholds");
   d->WriteObjectAny(&objective, "std::string", "objective");
   d->WriteObjectAny(&max_depth, "std::vector<int>", "max_depth");
   d->WriteObjectAny(&num_trees, "std::vector<int>", "num_trees");
   d->WriteObjectAny(&num_features, "std::vector<int>", "num_features");
   d->WriteObjectAny(&num_classes, "std::vector<int>", "num_classes");
   f->Write();
   f->Close();
}

#endif // TMVA_TEST_BDTHELPERS
