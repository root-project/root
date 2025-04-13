#include "lhef_sel_gen.C"
#include <stdio.h>

class lhef_sel : public lhef_sel_gen {
   UInt_t nullValues = 0;
   UInt_t lowValues = 0;
   UInt_t highValues = 0;
   UInt_t allValues = 0;

public:
   using lhef_sel_gen::lhef_sel_gen;

   virtual Bool_t  Process(Long64_t entry) {
      fReader.SetEntry(entry);

      for (size_t j=0; j< Particle_M.GetSize() ;j++)
      {
         Double_t k = Particle_M.At(j);
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
