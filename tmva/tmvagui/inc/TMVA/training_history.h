#ifndef training_history__HH
#define training_history__HH
#include "tmvaglob.h"
#include "TH2F.h"
#include "TFile.h"
#include "TIterator.h"
#include "TKey.h"
namespace TMVA{

   void plot_training_history(TString dataset, TFile* file, TDirectory* BinDir=0);
   void training_history(TString dataset, TString fin = "TMVA.root", Bool_t useTMVAStyle = kTRUE );
}
#endif
