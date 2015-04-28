#ifndef BoostControlPlots__HH
#define BoostControlPlots__HH
#include <vector>
#include <string>
#include "TLegend.h"
#include "TText.h"
#include "tmvaglob.h"
namespace TMVA{


   // input: - Input file (result from TMVA),
   //        - use of TMVA plotting TStyle
   // this macro is based on BDTControlPlots.C
   void BoostControlPlots( TString fin = "TMVA.root", Bool_t useTMVAStyle = kTRUE );

   void boostcontrolplots( TDirectory *boostdir ); 

}
#endif
