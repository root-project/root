#ifndef efficiencies__HH
#define efficiencies__HH
#include "tmvaglob.h"
#include "TH2F.h"
#include "TFile.h"
#include "TIterator.h"
#include "TKey.h"
namespace TMVA{

   void plot_efficiencies( TFile* file, Int_t type = 2, TDirectory* BinDir=0);
   void efficiencies( TString fin = "TMVA.root", Int_t type = 2, Bool_t useTMVAStyle = kTRUE );
}
#endif
