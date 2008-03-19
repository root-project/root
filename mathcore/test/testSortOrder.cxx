#include <iostream>
#include <algorithm>
#include <ctime>
#include <vector>

#include "TMath.h"
#include "TRandom2.h"

using namespace std;

const int npass = 100000;
const int maxint = 20;
const int minsize = 20;
const int maxsize = 500;
const int increment = 10;
const int arraysize = (maxsize-minsize)/10 + 1;

#ifndef ROOT_TMath

template<typename T> 
struct CompareDesc { 

   CompareDesc(const T *  d) : fData(d) {}

   bool operator()(int i1, int i2) { 
      return fData[i1] > fData[i2];
   }

   const T * fData; 
};

template<typename T> 
struct CompareAsc { 

   CompareAsc(const T *  d) : fData(d) {}

   bool operator()(int i1, int i2) { 
      return fData[i1] < fData[i2];
   }

   const T * fData; 
};

#endif

template <typename T> void testSort(const int n)
{
   vector<T> k(n);
   vector<T> index(n);

   TRandom2 r( time( 0 ) );

   cout << "k: ";
   for ( Int_t i = 0; i < n; i++) {
      k[i] = (T) r.Integer( maxint ); 
      cout << k[i] << ' ';
   }
   cout << endl;



   for(Int_t i = 0; i < n; i++) { index[i] = i; }
   TMath::Sort(n,&k[0],&index[0],kTRUE);  

   cout << "TMath[kTRUE]\n\tindex = ";
   for ( Int_t i = 0; i < n; ++i )
      cout << k[index[i]] << ' ';
   cout << endl;

  

   for(Int_t i = 0; i < n; i++) { index[i] = i; }
   std::sort(&index[0],&index[n], CompareDesc<T>(&k[0]) );

   cout << "std::sort[CompareDesc]\n\tindex = ";
   for ( Int_t i = 0; i < n; ++i )
      cout << k[index[i]] << ' ';
   cout << endl;



   for(Int_t i = 0; i < n; i++) { index[i] = i; }
   TMath::Sort(n,&k[0],&index[0],kFALSE);  

   cout << "TMath[kFALSE]\n\tindex = ";
   for ( Int_t i = 0; i < n; ++i )
      cout << k[index[i]] << ' ';
   cout << endl;



   for(Int_t i = 0; i < n; i++) { index[i] = i; }
   std::sort(&index[0],&index[n], CompareAsc<T>(&k[0]) );

   cout << "std::sort[CompareAsc]\n\tindex = ";
   for ( Int_t i = 0; i < n; ++i )
      cout << k[index[i]] << ' ';
   cout << endl;
}

void stdsort() 
{
   testSort<Int_t>(20);
}

int main(int argc, char **argv)
{
   stdsort();

   return 0;
}
