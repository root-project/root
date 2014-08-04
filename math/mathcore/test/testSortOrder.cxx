#include <iostream>
#include <algorithm>
#include <ctime>
#include <vector>

#include "TMath.h"
#include "TRandom2.h"

using namespace std;

const int maxint = 20;

#ifndef ROOT_TMath

template<typename T>
struct CompareDesc {

   CompareDesc(T d) : fData(d) {}

   bool operator()(int i1, int i2) {
      return *(fData + i1) > *(fData + i2);
   }

   T fData;
};

template<typename T>
struct CompareAsc {

   CompareAsc(T d) : fData(d) {}

   bool operator()(int i1, int i2) {
      return *(fData + i1) < *(fData + i2);
   }

   T fData;
};

#endif

template <typename T> bool testSort(const int n)
{
   vector<T> k(n);
   vector<T> indexM(n);
   vector<T> indexS(n);

   bool equals = true;

   TRandom2 r( time( 0 ) );

   cout << "k: ";
   for ( Int_t i = 0; i < n; i++) {
      k[i] = (T) r.Integer( maxint );
      cout << k[i] << ' ';
   }
   cout << endl;

   for(Int_t i = 0; i < n; i++) { indexM[i] = i; }
   TMath::Sort(n,&k[0],&indexM[0],kTRUE);

   cout << "TMath[kTRUE]\n\tindex = ";
   for ( Int_t i = 0; i < n; ++i )
      cout << k[indexM[i]] << ' ';
   cout << endl;

   for(Int_t i = 0; i < n; i++) { indexS[i] = i; }
   std::sort(&indexS[0],&indexS[0]+n, CompareDesc<const T*>(&k[0]) );

   cout << "std::sort[CompareDesc]\n\tindex = ";
   for ( Int_t i = 0; i < n; ++i )
      cout << k[indexS[i]] << ' ';
   cout << endl;


   equals &= std::equal(indexM.begin(), indexM.end(), indexS.begin());
   cout << "Equals? " << (char*) (equals?"OK":"FAILED") << endl;


   for(Int_t i = 0; i < n; i++) { indexM[i] = i; }
   TMath::Sort(n,&k[0],&indexM[0],kFALSE);

   cout << "TMath[kFALSE]\n\tindex = ";
   for ( Int_t i = 0; i < n; ++i )
      cout << k[indexM[i]] << ' ';
   cout << endl;

   for(Int_t i = 0; i < n; i++) { indexS[i] = i; }
   std::sort(&indexS[0],&indexS[0]+n, CompareAsc<const T*>(&k[0]) );

   cout << "std::sort[CompareAsc]\n\tindex = ";
   for ( Int_t i = 0; i < n; ++i )
      cout << k[indexS[i]] << ' ';
   cout << endl;


   equals &= std::equal(indexM.begin(), indexM.end(), indexS.begin());
   cout << "Equals? " << (char*) (equals?"OK":"FAILED") << endl;

   return equals;
}

bool stdsort()
{
   return testSort<Int_t>(20);
}

int main(int /* argc */ , char ** /* argv */ )
{
   bool equals = stdsort();

   if ( !equals )
      return 1;
   else
      return 0;
}
