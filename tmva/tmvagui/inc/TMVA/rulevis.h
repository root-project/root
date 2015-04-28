#ifndef rulevis__HH
#define rulevis__HH
#include "tmvaglob.h"
#include "rulevisHists.h"
#include "rulevisCorr.h"
namespace TMVA{

   // This macro plots the distributions of the different input variables overlaid on
   // the sum of importance per bin.
   // The scale goes from violett (no importance); to red (high importance).
   // Areas where many important rules are active, will thus be very red.
   //
   // input: - Input file (result from TMVA),
   //        - normal/decorrelated/PCA
   //        - use of TMVA plotting TStyle
   void rulevis( TString fin = "TMVA.root", TMVAGlob::TypeOfPlot type = TMVAGlob::kNorm, bool useTMVAStyle=kTRUE );
}
#endif
