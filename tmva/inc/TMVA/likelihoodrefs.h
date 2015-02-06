#ifndef likelihoodrefs__HH
#define likelihoodrefs__HH
#include <vector>
#include <string>
#include "tmvaglob.h"
namespace TMVA{


   // this macro plots the reference distribuions for the Likelihood
   // methods for the various input variables used in TMVA (e.g. running
   // TMVAnalysis.C).  Signal and Background are plotted separately

   // input: - Input file (result from TMVA),
   //        - use of TMVA plotting TStyle


   void likelihoodrefs( TDirectory *lhdir ); 

   void likelihoodrefs( TString fin = "TMVA.root", Bool_t useTMVAStyle = kTRUE );
}
#endif
