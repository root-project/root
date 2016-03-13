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
   void bdtcontrolplots(TString dataset,TDirectory *);

   void BDTControlPlots(TString dataset, TString fin = "TMVA.root", Bool_t useTMVAStyle = kTRUE );
   void bdtcontrolplots(TString dataset, TDirectory *bdtdir ); 
  
  
}
#endif
