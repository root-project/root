/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifdef __hpux
#include <iostream.h>
#include <iomanip.h>
#else
#include <iostream>
#include <iomanip>
using namespace std;
#endif
typedef int Int_t;

class Test {
public:
   Int_t v[2];
   Test() { v[0]=1; v[1]=2;}
   void print1() {
      for (Int_t i=0; i<2 ; i++)
         cout << setw(4) << v[i] << endl;
   }
   void print2() {
      cout << setw(4) << v[0] << endl;
      cout << setw(4) << v[1] << endl;
   }
};

Test tt;

//Int_t v[2] = { 1, 2 };

void print() {
   for (Int_t i=0; i<2 ; i++) 
      cout << setw(4) << tt.v[i] << endl;
   //cout << setw(4) << v[i] << endl;
}

int main() {
   tt.print1();
   tt.print2();
   print();
   for(int i=0;i<3;i++) {
      cout << "====================" << endl;
      tt.print1();
      tt.print2();
      print();
   }
   return 0;
}
