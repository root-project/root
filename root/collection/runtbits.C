#include "TBits.h"
#include "Riostream.h"

#ifndef __CINT__
#include <bitset>
using namespace std;
#endif

bool testEqual() {
   bool result = true;
   
   TBits bits1, bits2;
   
   bits1.SetBitNumber(10);
   bits1.ResetAllBits();
   
   bits1.SetBitNumber(1);
   bits1.SetBitNumber(2);
   
   bits2.SetBitNumber(1);
   
   TBits bits_and = bits1 & bits2;
   
   bits_and.Print();
   bits2.Print();
   
   if ( bits_and == bits2) {
      printf("ok\n");
   } else {
      result = false;
      printf("bug\n");
   }
   if ( bits_and != bits2) {
      result = false;
      printf("bug\n");
   } else {
      printf("ok\n");
   }
   
   if ( bits2 == bits_and) {
      printf("ok\n");
   } else {
      result = false;
      printf("bug\n");
   }
   if ( bits2 != bits_and) {
      result = false;
      printf("bug\n");
   } else {
      printf("ok\n");
   }
   
   bits_and.SetBitNumber(10);
   if ( bits_and == bits2) {
      result = false;
      printf("bug\n");
   } else {
      printf("ok\n");
   }
   if ( bits_and != bits2) {
      printf("ok\n");
   } else {
      result = false;
      printf("bug\n");
   }
   if ( bits2 == bits_and) {
      result = false;
      printf("bug\n");
   } else {
      printf("ok\n");
   }
   if ( bits2 != bits_and) {
      printf("ok\n");
   } else {
      result = false;
      printf("bug\n");
   }
   return result;
}


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

#ifndef __CINT__
   bitset<16> bs;
   bs[2] = 1;
   //bs[12]= 1;
   cout << bs << endl;
#endif
   testEqual();
   return 0;
}
