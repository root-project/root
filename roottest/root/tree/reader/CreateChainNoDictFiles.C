#ifndef __CreateChainNoDictFiles__
#define __CreateChainNoDictFiles__

#include "TFile.h"
#include "TTree.h"

#include <memory>
#include <string>
#include <vector>

struct ChainNoDictHit {
   double fX = 0;
   int fId = 0;
};

#ifdef __ROOTCLING__
#pragma link C++ class ChainNoDictHit+;
#pragma link C++ class std::vector<ChainNoDictHit>+;
#endif

// Write two files with a split std::vector of a struct. They are read in
// ChainNoDict.C without this dictionary, so that the collection proxy is emulated.
int CreateChainNoDictFiles()
{
   for (int file = 0; file < 2; ++file) {
      auto f = std::make_unique<TFile>(("chainNoDict" + std::to_string(file) + ".root").c_str(), "RECREATE");
      auto t = std::make_unique<TTree>("t", "test tree");
      std::vector<ChainNoDictHit> hits;
      t->Branch("hits", &hits, 32000, 99);
      for (int entry = 0; entry < 10; ++entry) {
         hits.clear();
         for (int i = 0; i < entry; ++i)
            hits.push_back({100. * file + entry + 0.1 * i, i});
         t->Fill();
      }
      t->Write();
   }
   return 0;
}

#endif
