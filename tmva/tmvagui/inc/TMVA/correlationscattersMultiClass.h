#ifndef correlationscattersMultiClass__HH
#define correlationscattersMultiClass__HH
#include "tmvaglob.h"
namespace TMVA{

   // this macro plots the correlations (as scatter plots); of
   // the various input variable combinations used in TMVA (e.g. running
   // TMVAnalysis.C).  Signal and Background are plotted separately

   // input: - Input file (result from TMVA),
   //        - normal/decorrelated/PCA
   //        - use of TMVA plotting TStyle
   void correlationscattersMultiClass( TString fin = "TMVA.root", TString var= "var3", 
                                       TString dirName_ = "InputVariables_Id", TString title = "TMVA Input Variable",
                                       Bool_t isRegression = kFALSE,
                                       Bool_t useTMVAStyle = kTRUE );
}
#endif
