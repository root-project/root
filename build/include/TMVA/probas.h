#ifndef probas__HH
#define probas__HH

#include "TString.h"

namespace TMVA{

   // this macro plots the MVA probability distributions (Signal and
   // Background overlayed); of different MVA methods run in TMVA
   // (e.g. running TMVAnalysis.C).

   // input: - Input file (result from TMVA);
   //        - use of TMVA plotting TStyle
   void probas(TString dataset, TString fin = "TMVA.root", Bool_t useTMVAStyle = kTRUE );
}
#endif
