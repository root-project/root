#ifndef BDTControlPlots__HH
#define BDTControlPlots__HH
#include <vector>
#include <string>
#include "tmvaglob.h"

#include "TH1.h"
#include "TGraph.h"
namespace TMVA{

   // input: - Input file (result from TMVA),
   //        - use of TMVA plotting TStyle
   void bdtcontrolplots(TDirectory *);

   void BDTControlPlots( TString fin = "TMVA.root", Bool_t useTMVAStyle = kTRUE );
   void bdtcontrolplots( TDirectory *bdtdir ); 
  
  
}
#endif
