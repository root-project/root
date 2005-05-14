///////////////////////////////////////////////////////////////////////////
//
//  Tutorial illustrating use and precision of the Double32_t data type
//
// You must run this tutorial with ACLIC
//    root > .x double32.C+
//
// The following cases are supported for streaming a Double32_t type
// depending on the range declaration in the comment field of the data member:
//  A-    Double32_t     fNormal;
//  B-    Double32_t     fTemperature; //[0,100]
//  C-    Double32_t     fCharge;      //[-1,1,2]
//  D-    Double32_t     fVertex[3];   //[-30,30,10]
//  E     Int_t          fNsp;
//        Double32_t*    fPointValue;   //[fNsp][0,3]
//
// In case A fNormal is converted from a Double_t to a Float_t
// In case B fTemperature is converted to a 32 bit unsigned integer
// In case C fCharge is converted to a 2 bits unsigned integer
// In case D the array elements of fVertex are converted to an unsigned 10 bits integer
// In case E the fNsp elements of array fPointvalue are converted to an unsigned 32 bit integer
//           Note that the range specifier must follow the dimension specifier.
// the case B has more precision (9 to 10 significative digits than case A (6 to 7 digits).
//
// The range specifier has the general format: [xmin,xmax] or [xmin,xmax,nbits]
//  [0,1]
//  [-10,100];
//  [-pi,pi], [-pi/2,pi/4],[-2pi,2*pi]
//  [-10,100,16]
// if nbits is not specified, or nbits <2 or nbits>32 it is set to 32
//
///////////////////////////////////////////////////////////////////////////

#include "TCanvas.h"
#include "TTree.h"
#include "TH1.h"
#include "TRandom3.h"
#include "TGraph.h"
#include "TText.h"
#include "TFrame.h"
#include "TPaveLabel.h"
   
class DemoDouble32  {
private:
   Double_t    fD64;     //reference member with full double precision
   Double32_t  fF32;     //saved as a 32 bit Float_t
   Double32_t  fI32;     //[0,pi]    saved as a 32 bit unsigned int
   Double32_t  fI30;     //[0,pi,30] saved as a 30 bit unsigned int
   Double32_t  fI28;     //[0,pi,28] saved as a 28 bit unsigned int
   Double32_t  fI26;     //[0,pi,26] saved as a 26 bit unsigned int
   Double32_t  fI24;     //[0,pi,24] saved as a 24 bit unsigned int
   Double32_t  fI22;     //[0,pi,22] saved as a 22 bit unsigned int
   Double32_t  fI20;     //[0,pi,20] saved as a 20 bit unsigned int
   Double32_t  fI18;     //[0,pi,18] saved as a 18 bit unsigned int
   Double32_t  fI16;     //[0,pi,16] saved as a 16 bit unsigned int
   Double32_t  fI14;     //[0,pi,14] saved as a 14 bit unsigned int
   Double32_t  fI12;     //[0,pi,12] saved as a 12 bit unsigned int
   Double32_t  fI10;     //[0,pi,10] saved as a 10 bit unsigned int
   Double32_t  fI8;      //[0,pi, 8] saved as a  8 bit unsigned int
   Double32_t  fI6;      //[0,pi, 6] saved as a  6 bit unsigned int
   Double32_t  fI4;      //[0,pi, 4] saved as a  4 bit unsigned int
   Double32_t  fI2;      //[0,pi, 2] saved as a  2 bit unsigned int
       
public:
   DemoDouble32() {;}
   void Set(Double_t ref);
};

void DemoDouble32::Set(Double_t ref) {
   fD64 = fF32 = fI32 = fI30 = fI28 = fI26 = fI24 = fI22 = fI20 = ref;
   fI18 = fI16 = fI14 = fI12 = fI10 = fI8  = fI6  = fI4  = fI2  = ref;
}
      
void double32() {
   // show the use and precision of the Double32_t data type
   
   DemoDouble32 *d = new DemoDouble32();
   
   //create a Tree with 10000 objects DemoDouble32
   TFile::Open("DemoDouble32.root","recreate");
   TTree *T = new TTree("T","DemoDouble32");
   TBranch *bd = T->Branch("d","DemoDouble32",&d);
   TRandom3 r;
   Double_t pi = TMath::Pi();
   Int_t i, n = 10000;
   for (i=0;i<n;i++) {
      d->Set(r.Uniform(0,pi));
      T->Fill();
   }
   T->Write();
   
   //Create the frame histogram and the graphs
   TObjArray *branches = bd->GetListOfBranches();
   Int_t nb = branches->GetEntries();
   TBranch *br = (TBranch*)branches->At(0);
   Double_t zip64 = br->GetZipBytes();
   Double_t cx = 1;
   Double_t di = 15;
   TCanvas *c1 = new TCanvas("c1","c1",800,600);
   c1->SetGrid();
   c1->SetHighLightColor(0);
   c1->SetFillColor(17);
   c1->SetFrameFillColor(20);
   c1->SetFrameBorderSize(10);
   TH1F *h = new TH1F("h","",nb,0,nb);
   h->SetMaximum(16);
   h->SetStats(0);
   h->Draw();
   c1->GetFrame()->SetFillColor(21);
   c1->GetFrame()->SetBorderSize(12);
   TGraph *gcx = new TGraph(nb);
   TGraph *gdi = new TGraph(nb);
   gcx->SetMarkerStyle(21);
   gcx->SetMarkerColor(kBlue);
   gdi->SetMarkerStyle(20);
   gdi->SetMarkerColor(kRed);
   TText *tdi = new TText(2.5,10.5,"Number of significative digits");
   tdi->SetTextColor(kRed);
   tdi->SetTextSize(0.05);
   tdi->Draw();
   TText *tcx = new TText(1.5,2.6,"Compression factor");
   tcx->SetTextColor(kBlue);
   tcx->SetTextSize(0.05);
   tcx->Draw();
   TPaveLabel *title = new TPaveLabel(.15,.92,.85,.97,"Double32_t compression and precision","brNDC");   
   title->Draw();
   
   //loop on branches to get the precision and compression factors
   for (i=0;i<nb;i++) {
      br = (TBranch*)branches->At(i);
      h->GetXaxis()->SetBinLabel(i+1,br->GetName());
      cx = zip64/br->GetZipBytes();
      gcx->SetPoint(i,i+0.5,cx);
      if (i > 0) {
         T->Draw(Form("fD64-%s",br->GetName()),"","goff");
         Double_t rms = TMath::RMS(n,T->GetV1());
         di = -(1)*TMath::Log10(2*rms/pi);
      }
      gdi->SetPoint(i,i+0.5,di);
   }
   gcx->Draw("lp");
   gdi->Draw("lp");
}
   
   
   
       
