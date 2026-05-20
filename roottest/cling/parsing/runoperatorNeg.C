#include "TString.h"
int runoperatorNeg() {
 int N = 2;
 int M = 2;
TString**table = 0;
   if (!(table = new TString*[N])) return -1;

   for (Int_t i=0; i<N; i++) {
      //some code
      table[i] = 0;
      if (!(table[i] = new TString[M])) return -1;
      //some code
   }
   return 0; 
}
