#include "TChain.h"
#include "TDirectory.h"
#include "TEntryList.h"

// First run: root $ROOTSYS/tutorials/tree/cernbuild.C
// Then run: zip -n root cernstaff cernstaff.root

int execTEntryListZip()
{
   TChain chain("T");
   chain.Add("cernstaff.zip#cernstaff.root");
   chain.Draw(">> elist0", "Age > 40", "entrylist");

   // gDirectory->ls();
   auto el = static_cast<TEntryList *>(gDirectory->Get("elist0"));
   if (!el)
      return 2;
   // el->Print("V");

   chain.SetEntryList(nullptr);
   chain.SetEntryList(el, "");

   return (el == chain.GetEntryList()) ? 0 : 1;
}
