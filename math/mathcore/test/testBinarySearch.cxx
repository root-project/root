#include <iostream>
#include <ctime>

#include <TRandom2.h>
#include <TMath.h>

using namespace std;

const int nn = 20;
const int maxint = 10;
const int except = 8;

template <typename T> void testBinarySearch()
{
   T k[nn];

   TRandom2 r( time( 0 ) );
   for ( Int_t i = 0; i < nn; i++) {
      T number = (T) r.Integer( maxint );
      while ( number == except )
         number = (T) r.Integer( maxint );
      k[i] = number;
   }

   std::sort(k, k+nn);

   for ( Int_t i = 0; i < nn; i++) {
      cout << k[i] << ' ';
   }
   cout << endl;


   auto begin = k;
   auto end = k + nn;

   for ( T elem = -1; elem <= maxint; ++elem ) {
      Long_t index = TMath::BinarySearch((Long_t) nn, k, elem);

      T* pind;
      pind = std::lower_bound(begin, end, elem);
      Long_t index2 = ((pind!=end && (*pind == elem)) ? (pind - k): ( pind - k - 1));

      pind = std::upper_bound(begin, end, elem);
      Long_t index3 = ((pind!=end && (*pind == elem)) ? (pind - k): ( pind - k - 1));

      cout << " ELEM = " << elem;
      cout << " [TMATH] [i:" << index  << " k[i]:"; if (index>=0 && index<nn) cout << k[index]; else cout << "n/a"; cout << ']';
      cout << " [LOWER] [i:" << index2 << " k[i]:"; if (index2>=0 && index<nn) cout << k[index2]; else cout << "n/a"; cout << ']';
      cout << " [UPPER] [i:" << index3 << " k[i]:"; if (index3>=0 && index<nn) cout << k[index3]; else cout << "n/a"; cout << ']';
      cout << endl;

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
