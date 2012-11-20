#ifndef __CINT__
#include "TString.h"
#include "Riostream.h"
#include <TLorentzVector.h>
#include "TH1F.h"
#endif

struct SelLambda {
   static const int Stopmin = 0;
   void Loop(Long64_t inEntries)
   {
      
      Long64_t nentries = 1;
      TH1F *invMassPosNeg = new TH1F();
      
      
      for (Long64_t jentry=0; jentry<nentries;jentry++) {
         
         if(Stopmin!=1) continue;
         
         TLorentzVector piMinus(3, 2, 1, 0);         
         TLorentzVector proton(3, 2, 1, 0);          
         TLorentzVector sum = piMinus+proton;
         
         invMassPosNeg->Fill(3);
      }
      delete invMassPosNeg;
      invMassPosNeg = 0;
   }
};

class Top {

public:
   void func(TString &s) {
      cout << s.Data() << endl;
   }
   void Run() {
      func("test");
   }
};

int staticfunc() {
   gROOT->ProcessLine("SelLambda l; l.Loop(0);");

   Top p;
   p.Run();
   Top::Run();

   return 0;
}
