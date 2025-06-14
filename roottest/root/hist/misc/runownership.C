#include "TH1F.h"
#include "TFile.h"
#include "TCanvas.h"
#include "Riostream.h"

class TH1F_inst : public TH1F {
public:
   static int fgCount;

   TH1F_inst() : TH1F()
   {
      ++fgCount;
   }
   TH1F_inst(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup) : TH1F(name,title,nbinsx,xlow,xup)
   {
      ++fgCount;
   }
   ~TH1F_inst()
   {
      --fgCount;
   }
   ClassDefOverride(TH1F_inst,2);
};

int TH1F_inst::fgCount = 0;

void write(const char *filename = "histo.root")
{
   TFile * f = TFile::Open(filename,"RECREATE");
   TH1F *histo = new TH1F_inst("h1","h1",10,0,10); histo->Fill(3);
   histo = new TH1F_inst("h2","h2",10,0,10); histo->Fill(3);
   TCanvas *c1 = new TCanvas("c1");
   histo->SetBit(kCanDelete);
   histo->Draw();
   c1->Write();
   f->Write();
   delete f;
}

bool read(const char *filename = "histo.root")
{
   TFile * f = TFile::Open(filename,"READ");
   TH1F *histo; f->GetObject("h1",histo);
   if (histo==0) {
      cout << "h1 is not found on the file\n";
      return false;
   }
   TCanvas *c1; f->GetObject("c1",c1);
   c1->Draw();
   delete c1;
   delete f;
   return true;
}

int runownership(const char *filename = "histo.root")
{
   write(filename);
   cout << "So far: " << TH1F_inst::fgCount << '\n';
   read(filename);
   cout << "So far: " << TH1F_inst::fgCount << '\n';
   return 0;
}
