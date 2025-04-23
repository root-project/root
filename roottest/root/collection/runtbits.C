#include "TBits.h"
#include "Riostream.h"
#include "TRandom.h"

#ifndef __CINT__
#include <bitset>
using namespace std;
#endif

void testFirstAndLast()
{
   const UInt_t size = 100000;
   TBits b(size);
   Bool_t success = kTRUE;
   UInt_t i, j;
   printf("Testing TBits::LastSetBit() ...\n");
   // Set a single bit then search the set bit both from the end and from 1 bit close
   for (i=0; i<size; i++) {
      b.SetBitNumber(i);
      if ((b.LastSetBit() != i) || (b.LastSetBit(i+1) != i)) {
         printf("Error for bit %d: lastset=%d lastset(i+1)=%d\n", i, b.LastSetBit(), b.LastSetBit(i+1));
         success = kFALSE;
      }   
      b.SetBitNumber(i, kFALSE);
   }
   if (success) printf("Test 1: Set/Find SUCCESS\n");
   else         printf("Test 1: Set/Find FAILED\n");
   
   // Set random bits, search sequentially and checksum their index, both forward 
   // and backward
   for (i=0; i<size; i++) {
      j = size*gRandom->Rndm();
      b.SetBitNumber(j);
   }
   Long64_t checksum1 = 0;
   j = b.FirstSetBit();
   while (j<size) {
      checksum1 += j;
      j = b.FirstSetBit(j+1);
   }
   Long64_t checksum2 = 0;
   j = b.LastSetBit();
   while (j<size) {
      checksum2 += j;
      if (j==0) break;
      j = b.LastSetBit(j-1);
   }
   printf("Test 2: checksum1=%lld  checksum2=%lld ... ", checksum1, checksum2);
   if (checksum1==checksum2) printf("SUCCESS\n");
   else         printf("FAILED\n");   

   printf("Testing TBits::LastNullBit() ...\n");
   for (i=0; i<size; i++) b.SetBitNumber(i);
   // Reset a single bit then search the reset bit both from the end and from 1 bit close
   for (i=0; i<size; i++) {
      b.SetBitNumber(i, false);
      if ((b.LastNullBit() != i) || (b.LastNullBit(i+1) != i)) {
         printf("Error for bit %d: lastnull=%d lastnull(i+1)=%d\n", i, b.LastNullBit(),b.LastNullBit(i+1));
         success = kFALSE;
      }   
      b.SetBitNumber(i);
   }
   if (success) printf("Test 3: Reset/Find SUCCESS\n");
   else         printf("Test 3: Reset/Find FAILED\n");
   
   // Set random bits, search sequentially and checksum the index of reset bits, both forward 
   // and backward
   for (i=0; i<size; i++) {
      j = size*gRandom->Rndm();
      b.SetBitNumber(j, false);
   }
   Long64_t checksum3 = 0;
   j = b.FirstNullBit();
   while (j<size) {
      checksum3 += j;
      j = b.FirstNullBit(j+1);
   }
   Long64_t checksum4 = 0;
   j = b.LastNullBit();
   while (j<size) {
      checksum4 += j;
      if (j==0) break;
      j = b.LastNullBit(j-1);
   }
   printf("Test 4: checksum3=%lld  checksum4=%lld ... ", checksum3, checksum4);
   if (checksum3==checksum4) printf("SUCCESS\n");
   else         printf("FAILED\n");   
}

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

   bitset<16> bs;
   bs[2] = 1;
   //bs[12]= 1;
   cout << bs << endl;
   testEqual();
   testFirstAndLast();
   return 0;
}
