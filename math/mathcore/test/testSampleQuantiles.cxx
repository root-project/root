// test sample quantiles
// function TMath::Quantiles and indirectly also TMath::kOrdStat
// compare with results from R (hardcoded in the test)
// L.M 9/11/2011

#include "TMath.h"

#include <iostream>
#include <algorithm>
using std::cout;
using std::endl;

bool debug = false;

// restults obtained runnnig r for all types from 1 to 9
// there are 13 values for each ( size is 13*9 = 121
double result[121] =  { 
      0.1 , 0.1 , 0.1 , 0.3 , 0.7 , 1 , 1.2 , 1.5 , 1.8 , 2 , 10 , 10 , 10,         // type = 1
      0.1 , 0.1 , 0.2 , 0.5 , 0.85 , 1.1 , 1.35 , 1.65 , 1.9 , 6 , 10 , 10 , 10,    // type = 2
      0.1 , 0.1 , 0.1 , 0.3 , 0.7 , 1 , 1.2 , 1.5 , 1.8 , 2 , 10 , 10 , 10,         // type = 3
      0.1 , 0.1 , 0.1 , 0.3 , 0.7 , 1 , 1.2 , 1.5 , 1.8 , 2 , 10 , 10 , 10,                   // type = 4 
      0.1 , 0.1 , 0.2 , 0.5 , 0.85 , 1.1 , 1.35 , 1.65 , 1.9 , 6 , 10 , 10 , 10,       
      0.1 , 0.1 , 0.12 , 0.38 , 0.79 , 1.08 , 1.35 , 1.68 , 1.94 , 8.4 , 10 , 10 , 10, 
      0.1 , 0.118 , 0.28 , 0.62 , 0.91 , 1.12 , 1.35 , 1.62 , 1.86 , 3.6 , 10 , 10 , 10,
      0.1 , 0.1 , 0.1733333 , 0.46 , 0.83 , 1.093333 , 1.35 , 1.66 , 1.913333 , 6.8 , 10 , 10 , 10,
      0.1 , 0.1 , 0.18 , 0.47 , 0.835 , 1.095 , 1.35 , 1.6575 , 1.91 , 6.6 , 10 , 10 , 10 };


bool testQuantiles(int type = 0, bool sorted = true) { 

   const int n = 10;
   double x[] = {0.1,0.3,0.7,1.,1.2,1.5,1.8,2.,10,10};

   const int np = 13;
   double p[] = { 0.,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,1.0 };
   double quant[np];

   if (!sorted) { 
      // shuffle the data
      std::random_shuffle(x, x+10);
      if (debug) { 
         std::cout << "shuffle data " << std::endl;
         cout << " data = { ";
         for (int i = 0; i < n; ++i) 
            cout << x[i] << "  ";
         cout << " }\n";      
      }
   }
    

   if (type >0 && type < 10) 
      TMath::Quantiles(n,np,x,quant,p,sorted,0,type);
   else
      TMath::Quantiles(n,np, x,quant,p,sorted);

   if (debug) { 
      for (int i = 0; i < np; ++i) { 
         printf("  %5.2f ",p[i]);
      }
      cout << endl;
      for (int i = 0; i < np; ++i) { 
         printf("  %5.3f ",quant[i]);
      }
      cout << endl;
   }

   // test if result is OK
   if (type == 0) type = 7; 
   if (type < 0) type = - type;
   bool ok = true; 
   cout << "Testing for type " << type << " :\t\t"; 
   for (int i = 0; i < np; ++i) {       
      double r_result = result[ (type-1)*np + i];
      if (TMath::AreEqualAbs(quant[i], r_result, 1.E-6) )
         cout << ".";
      else { 
         cout << "  Failed for prob = " << p[i] << " -  R gives " << r_result << " TMath gives " << quant[i] << std::endl;
         ok = false; 
      }
   }
   if (ok) 
      cout << "\t OK !\n";
   else
      cout << "\nTest Failed for type " << type << std::endl;

   return ok;
}

int main(int argc,  char *argv[]) {  
   
   if (argc > 1) {
      int type =  atoi(argv[1]);
      debug = true;
      bool ret = testQuantiles(type,true);
      return (ret) ? 0 : -1;
   }

   bool ok = true; 
   cout << "Test ordered data ....\n";
   //itype == 0 is considered the defafult
   for (int itype = 0; itype < 10; ++itype) { 
      ok &= testQuantiles(itype,true);
   }
   cout << "\nTest  data in random order....\n";
   for (int itype = 0; itype < 10; ++itype) { 
      ok &= testQuantiles(itype,false);
   }

   if (!ok) { 
      cout << "Test sample quantiles FAILED " << endl;
      return -1;
   } 
   return 0;

}
