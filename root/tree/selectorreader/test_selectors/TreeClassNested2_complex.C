#define TreeClassNested2_complex_cxx

#include "../generated_selectors/TreeClassNested2_complex.h"
#include <TH2.h>
#include <TStyle.h>


void TreeClassNested2_complex::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void TreeClassNested2_complex::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();
}

Bool_t TreeClassNested2_complex::Process(Long64_t entry)
{
   fReader.SetEntry(entry);

   fprintf(stderr, "Ev (as leaf): %d Ev (from fC): %d Ev (from ClassB): %d "
                   "Px (as leaf): %.1f Px (from fC): %.1f Px (from ClassB): %.1f\n",
                     *fC_fEv, fC->GetEv(), ClassB_branch->GetC().GetEv(),
                     *fC_fPx, fC->GetPx(), ClassB_branch->GetC().GetPx());
   return kTRUE;
}

void TreeClassNested2_complex::SlaveTerminate() { }

void TreeClassNested2_complex::Terminate() { }
