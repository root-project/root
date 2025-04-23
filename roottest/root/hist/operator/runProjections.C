#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TRandom.h"

void PD(TH1* h) {
   TAxis* a = h->GetXaxis();
   printf("   H1<%s>(%d:%g[%g,%g]%g):%g\n",
          h->GetName(), a->GetNbins(),
          a->GetBinLowEdge(0),
          a->GetBinLowEdge(a->GetFirst()),
          a->GetBinUpEdge(a->GetLast()),
          a->GetBinUpEdge(a->GetNbins() + 1),
          h->GetEntries());
}

void PD(TH2* h) {
   TAxis* a = h->GetXaxis();
   TAxis* b = h->GetYaxis();
   printf("   H2<%s>(%d:%g[%g,%g]%g|%d:%g[%g,%g]%g):%g\n",
          h->GetName(),
          a->GetNbins(),
          a->GetBinLowEdge(0),
          a->GetBinLowEdge(a->GetFirst()),
          a->GetBinUpEdge(a->GetLast()),
          a->GetBinUpEdge(a->GetNbins() + 1),
          b->GetNbins(),
          b->GetBinLowEdge(0),
          b->GetBinLowEdge(b->GetFirst()),
          b->GetBinUpEdge(b->GetLast()),
          b->GetBinUpEdge(b->GetNbins() + 1),
          h->GetEntries());
}

bool Compare(TH1* a,  TH1* b) {
   TH1* diff = (TH1*) a->Clone("diff");
   diff->Add(b, -1.);
   Double_t max = diff->GetMaximum();
   Double_t min = diff->GetMinimum();
   if (max < 0.) max = -max;
   if (min < 0.) min = -min;
   if (max < min) max = min;
   if (max > 0.001) {
      printf("%s == %s: FAIL (%g)\n", a->GetName(), b->GetName(), max);
      PD(a);
      PD(b);
      TCanvas* c = new TCanvas(a->GetName(), a->GetName());
      c->Divide(1,3);
      c->cd(1); a->Draw();
      c->cd(2); b->Draw();
      c->cd(3); diff->Draw();
      TFile f("runProjections.root", "UPDATE");
      c->Write();
      delete c;
   } else
      printf("%s == %s: OK\n", a->GetName(), b->GetName());
   delete diff;
   return (max <= 0.001);
}

void Test(Int_t range) {
   printf("\n\nRANGE: %d\n", range);
   TH3F * h3 = new TH3F("h3", "h3", 10, 0.1, 0.9, 20, 0.1, 0.9, 30, 0.1, 0.9);

   for (int i = 0; i < 1000; ++i)
      h3->Fill(gRandom->Rndm(), gRandom->Rndm(), gRandom->Rndm());

   if (range) {
      h3->GetXaxis()->SetRange(range , 10 + 1 - range);
      h3->GetYaxis()->SetRange(range , 20 + 1 - range);
      h3->GetZaxis()->SetRange(range , 30 + 1 - range);
   }

   // Project3D wants them in inverse order :-(
   TH2* xy = (TH2*)h3->Project3D("yx");
   TH2* yz = (TH2*)h3->Project3D("zy");
   TH2* xz = (TH2*)h3->Project3D("zx");

   TH1 *xyx, *xyy, *yzy, *yzz, *xzx, *xzz;
   if (range > 1) {
      xyx = xy->ProjectionX("xyx", 1, xy->GetYaxis()->GetNbins());
      xyy = xy->ProjectionY("xyy", 1, xy->GetXaxis()->GetNbins());
      yzy = yz->ProjectionX("yzy", 1, yz->GetYaxis()->GetNbins());
      yzz = yz->ProjectionY("yzz", 1, yz->GetXaxis()->GetNbins());
      xzx = xz->ProjectionX("xzx", 1, xz->GetYaxis()->GetNbins());
      xzz = xz->ProjectionY("xzz", 1, xz->GetXaxis()->GetNbins());
   } else {
      xyx = xy->ProjectionX("xyx");
      xyy = xy->ProjectionY("xyy");
      yzy = yz->ProjectionX("yzy");
      yzz = yz->ProjectionY("yzz");
      xzx = xz->ProjectionX("xzx");
      xzz = xz->ProjectionY("xzz");
   }

   TH1* x3 = h3->Project3D("x");
   TH1* y3 = h3->Project3D("y");
   TH1* z3 = h3->Project3D("z");

   TH1* x = 0; 
   TH1* y = 0; 
   TH1* z = 0; 
   // need to distinguish cases when use underflow/overlow, when excluding them and 
   // when using a range
   if (range > 1) {       
      x = h3->ProjectionX("x",
                          h3->GetYaxis()->GetFirst(),  
                          h3->GetYaxis()->GetLast(),
                          h3->GetZaxis()->GetFirst(),
                          h3->GetZaxis()->GetLast());   
      y = h3->ProjectionY("y",
                          h3->GetXaxis()->GetFirst(),  
                          h3->GetXaxis()->GetLast(),
                          h3->GetZaxis()->GetFirst(),
                          h3->GetZaxis()->GetLast());   
      z = h3->ProjectionZ("z",
                          h3->GetXaxis()->GetFirst(),  
                          h3->GetXaxis()->GetLast(),
                          h3->GetYaxis()->GetFirst(),
                          h3->GetYaxis()->GetLast());   
   } else if (range == 1) { 
      x = h3->ProjectionX("x",  
                          1,h3->GetYaxis()->GetNbins(),
                          1,h3->GetZaxis()->GetNbins() );   
      y = h3->ProjectionY("y",  
                          1,h3->GetXaxis()->GetNbins(),
                          1,h3->GetZaxis()->GetNbins() );   
      z = h3->ProjectionZ("z",  
                          1,h3->GetXaxis()->GetNbins(),
                          1,h3->GetYaxis()->GetNbins() );   
   }   else {    
      x = h3->ProjectionX("x");   
      y = h3->ProjectionY("y");   
      z = h3->ProjectionZ("z");   
   }
   
   if (!Compare(xyx, xzx)) {
      PD(xy); PD(xz);
   }
   if (!Compare(xyy, yzy)) {
      PD(xy); PD(yz);
   }
   if (!Compare(yzz, xzz)) {
      PD(yz); PD(xz);
   }

   if (!Compare(x3, xyx)) {
      PD(x3); PD(xy);
   }
   if (!Compare(y3, xyy)) {
      PD(y3); PD(xy);
   }
   if (!Compare(z3, xzz)) {
      PD(z3); PD(xz);
   }

   if (!Compare(xyx, xzx)) {
      PD(xy); PD(xz);
   }
   if (!Compare(xyy, yzy)) {
      PD(xy); PD(yz);
   }
   if (!Compare(xzz, yzz)) {
      PD(xz); PD(yz);
   }

   if (!Compare(x, xzx)) {
      PD(z); PD(yz);
   }
   if (!Compare(y, yzy)) {
      PD(z); PD(yz);
   }
   if (!Compare(z, yzz)) {
      PD(z); PD(yz);
   }

   delete h3;
   delete xy;
   delete yz;
   delete xz;
   delete x3;
   delete y3;
   delete z3;
   delete xyx;
   delete xyy;
   delete yzy;
   delete yzz;
   delete xzx;
   delete xzz;
   delete x;
   delete y;
   delete z;
};

void runProjections() {
   for (Int_t range = 0; range < 5; ++range)
      Test(range);
}
