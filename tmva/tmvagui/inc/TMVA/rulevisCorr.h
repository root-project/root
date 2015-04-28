#ifndef rulevisCorr__HH
#define rulevisCorr__HH
#include "tmvaglob.h"
namespace TMVA{

   // This macro plots the distributions of the different input variables overlaid on
   // the sum of importance per bin.
   // The scale goes from violett (no importance); to red (high importance).
   // Areas where many important rules are active, will thus be very red.
   //
   // input: - Input file (result from TMVA),
   //        - normal/decorrelated/PCA
   //        - use of TMVA plotting TStyle
   void rulevisCorr( TString fin = "TMVA.root", TMVAGlob::TypeOfPlot type = TMVAGlob::kNorm, bool useTMVAStyle=kTRUE );
   void rulevisCorr( TDirectory *rfdir, TDirectory *vardir, TDirectory *corrdir, TMVAGlob::TypeOfPlot type); 
}
#endif
