
#include "TFile.h"
#include "TObjString.h"
#include "TString.h"
#include "TList.h"
#include "TNamed.h"
#include "TH1F.h"
#include "TObject.h"

void createFile(const char *fn, const char *nm);
void execMergeMulti()
{

    TString fnm, onm;
    for (Int_t i = 1; i < 5; i++) {
       fnm.Form("mfile%d.root", i);
       onm.Form("%d", i);
       createFile(fnm, onm);
    }
}

void createFile(const char *fn, const char *nm)
{

   // Open the file
   TFile f(fn, "RECREATE");

   // Hist directory
   f.mkdir("hist");
   f.cd("hist");
   TH1F h("Gaus", "Gaus", 100, -5., 5.);
   h.FillRandom("gaus", 100);
   h.Write();
   h.SetDirectory(0);
   f.cd("..");

   // TNamed directory
   f.mkdir("named");
   f.cd("named");
   TNamed n("MyNamed", nm);
   n.Write();
   f.cd();

   // TList
   TList ll;
   ll.Add(new TObjString("uno"));
   ll.Add(new TObjString("due"));
   ll.Add(new TObjString("tre"));
   ll.Write("MyList", TObject::kSingleKey);

   f.Close();

   return;
}
