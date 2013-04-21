#include <bitset>
#include "TFile.h"
#include <iostream>

class Something {
public:
   std::bitset<5> fBits;

   void Print() {
     for(unsigned int i = 0; i < fBits.size(); ++i) {
       cout << "i: " << i << " val: ";
       if (fBits.test(i)) cout << "true";
       else cout << "false";
       cout << '\n';
     }
   }
   void Fill(int seed) {
     for(unsigned int i = 0; i < fBits.size(); ++i) {
       fBits.set(i,(i+seed)%2);
     }
   }
};

void execBitset() {
   TFile *f = TFile::Open("bitset.root","RECREATE");
   Something s;
   s.Fill(0);
   s.Print();
   cout << "About to write\n";
   f->WriteObject(&s,"bits");
   Something *ptr;
   cout << "About to read\n";
   f->GetObject("bits",ptr);
   cout << "Finished reading\n";
   if(!ptr) cout << "Can't find the bitset object\n";
   else ptr->Print();
}

