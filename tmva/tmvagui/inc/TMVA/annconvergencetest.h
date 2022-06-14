#ifndef annconvergencetest__HH
#define annconvergencetest__HH
#include "TMVA/tmvaglob.h"
namespace TMVA{

   // this macro serves to assess the convergence of the MLP ANN. 
   // It compares the error estimator for the training and testing samples.
   // If overtraining occurred, the estimator for the training sample should 
   // monotoneously decrease, while the estimator of the testing sample should 
   // show a minimum after which it increases.

   // input: - Input file (result from TMVA),
   //        - use of TMVA plotting TStyle

   void annconvergencetest(TString dataset, TDirectory *lhdir );

   void annconvergencetest(TString dataset, TString fin = "TMVA.root", Bool_t useTMVAStyle = kTRUE );
}
#endif
