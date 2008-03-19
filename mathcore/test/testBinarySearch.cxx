#include <iostream>
#include <ctime>

#include <TRandom2.h>
#include <TMath.h>

using namespace std;

const int n = 20;
const int maxint = 10;
const int except = 8;

template <typename T> void testBinarySearch()
{
   T k[n];

   TRandom2 r( time( 0 ) );
   for ( Int_t i = 0; i < n; i++) {
      T number = (T) r.Integer( maxint );
      while ( number == except )
         number = (T) r.Integer( maxint );
      k[i] = number;
   }

   std::sort(k, k+n);

   for ( Int_t i = 0; i < n; i++) {
      cout << k[i] << ' ';
   }
   cout << endl;


   for ( T elem = -1; elem <= maxint; ++elem ) {
      Long_t index = TMath::BinarySearch((Long_t) 20, k, elem);
      
      T* pind;
      pind = std::lower_bound(k, k+20, elem);
      Long_t index2 = ((*pind == elem)? (pind - k): ( pind - k - 1));

      pind = std::upper_bound(k, k+20, elem);
      Long_t index3 = ((*pind == elem)? (pind - k): ( pind - k - 1));

      cout << " ELEM = " << elem
           << " [TMATH] [i:" << index  << " k[i]:" << k[index] << ']'
           << " [LOWER] [i:" << index2 << " k[i]:" << k[index2] << ']'
           << " [UPPER] [i:" << index3 << " k[i]:" << k[index3] << ']'
           << endl;

   }
}

void testBinarySearch()
{
   testBinarySearch<Double_t>();

   cout << "Test done!" << endl;
}

int main()
{
   testBinarySearch();

   return 0;
}
