#include "lhef_leg_sel_gen.C"
#include <stdio.h>

class lhef_leg_sel : public lhef_leg_sel_gen {
   UInt_t nullValues = 0;
   UInt_t lowValues = 0;
   UInt_t highValues = 0;
   UInt_t allValues = 0;

public:
   using lhef_leg_sel_gen::lhef_leg_sel_gen;

   virtual Bool_t  Process(Long64_t entry) {
      fChain->GetEntry(entry);

      for (Int_t j=0; j< Particle_ ;j++)
      {
         double k = Particle_M[j];
         if ( k < 0.001 ) nullValues++;
         else if (k < 100) lowValues++;
         else highValues++;
         allValues++;
      }
      return true;
   }

   virtual void    Terminate() {
      fprintf(stdout,"all=%u zero=%u low=%u high=%u\n",allValues,nullValues,lowValues,highValues);
   }
};
