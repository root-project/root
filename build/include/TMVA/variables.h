#ifndef variables__HH
#define variables__HH
#include "tmvaglob.h"
namespace TMVA{

   // this macro plots the distributions of the different input variables
   // used in TMVA (e.g. running TMVAnalysis.C).  Signal and Background are overlayed.

   // input: - Input file (result from TMVA),
   //        - normal/decorrelated/PCA
   //        - use of TMVA plotting TStyle
   void variables(TString dataset, TString fin = "TMVA.root", TString dirName = "InputVariables_Id", TString title = "TMVA Input Variables",Bool_t isRegression = kFALSE, Bool_t useTMVAStyle = kTRUE );
}
#endif
