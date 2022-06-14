#ifndef mvaweights__HH
#define mvaweights__HH
#include "tmvaglob.h"
namespace TMVA{

   // input: - Input file (result from TMVA);
   //        - use of TMVA plotting TStyle
   void mvaweights( TString fin = "TMVA.root", Bool_t useTMVAStyle = kTRUE );
}
#endif
