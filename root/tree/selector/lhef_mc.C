#include "lhef_mc_gen.C"
#include <stdio.h>

class lhef_mc : public lhef_mc_gen {
   UInt_t nullValues = 0;
   UInt_t lowValues = 0;
   UInt_t highValues = 0;
   UInt_t allValues = 0;

public:
   using lhef_mc_gen::lhef_mc_gen;


   void Loop()
   {
      if (fChain == 0) return;

      Long64_t nentries = fChain->GetEntriesFast();

      Long64_t nbytes = 0, nb = 0;
      for (Long64_t jentry=0; jentry<nentries;jentry++) {
         Long64_t ientry = LoadTree(jentry);
         if (ientry < 0) break;
         nb = fChain->GetEntry(jentry);   nbytes += nb;

         for (Int_t j=0; j< Particle_ ;j++)
         {
            double k = Particle_M[j];
            if ( k < 0.001 ) nullValues++;
            else if (k < 100) lowValues++;
            else highValues++;
            allValues++;
         }
         // if (Cut(ientry) < 0) continue;
      }
      fprintf(stdout,"all=%u zero=%u low=%u high=%u\n",allValues,nullValues,lowValues,highValues);
   }


};
