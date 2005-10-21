#include "TBits.h"
#include "Riostream.h"

bool runtbits() {
   TBits a;  a.SetBitNumber(2,1);  a.SetBitNumber(12,1); 
   TBits b;  b.SetBitNumber(2,1);  b.SetBitNumber(15,1); 
   // Operation to test:
   cout << a << endl;
   cout << b << endl;
   TBits c = a & b;
   cout << c << endl;
   c = a ^ b;
   cout << c << endl;
   c = a | b;
   cout << c << endl;
   c <<= 2;
   cout << c << endl;
   c >>= 2;
   cout << c << endl;
   Bool_t bb = a[2] & b[15];
   cout << bb << endl;
   bb = a[2] & b[3];
   cout << bb << endl;
   return 0;
}
