#ifndef regression_averagedevs__HH
#define regression_averagedevs__HH
#include "tmvaglob.h"
namespace TMVA{

   /*
     this macro plots the quadratic deviation of the estimated from the target value, averaged over the first nevt events in test sample (all if Nevt=-1);
     a); normal average
     b); truncated average, using best 90%
     created January 2009, Eckhard von Toerne, University of Bonn, Germany
   */

   void regression_averagedevs(TString dataset,TString fin, Int_t Nevt=-1, Bool_t useTMVAStyle = kTRUE );

}
#endif
