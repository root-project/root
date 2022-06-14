#ifndef correlationsMultiClass__HH
#define correlationsMultiClass__HH
#include "tmvaglob.h"
namespace TMVA{

   // this macro plots the correlation matrix of the various input
   // variables used in TMVA (e.g. running TMVAnalysis.C).  Signal and
   // Background are plotted separately

   // input: - Input file (result from TMVA),
   //        - use of colors or grey scale
   //        - use of TMVA plotting TStyle
   void correlationsMultiClass(TString dataset, TString fin = "TMVA.root", Bool_t isRegression = kFALSE, 
                               Bool_t greyScale = kFALSE, Bool_t useTMVAStyle = kTRUE );
}
#endif
