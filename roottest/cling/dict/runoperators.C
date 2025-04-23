#if defined(__CINT__) && !defined(__MAKECINT__)
#include "operators_dict.h+"
#else 
#include "operators.h"
#endif
#include "cstdio"

void runoperators() {
   myiterator i,j,k;
   ++j;
   ++k;
   k = k + 1;
   i = j+k;
   if ( i == j ) {
      printf("problem with equal operator\n");
   }
   if ( i != k ) {
      // We are fine
   } else {
      printf("problem with not equal operator\n");
   }
   
   std::vector<myiterator> v;
   std::vector<myiterator>::iterator iter;
   for( iter = v.begin(); iter != v.end(); ++iter) {
      static_cast<void>(iter + 1);
   }
}
