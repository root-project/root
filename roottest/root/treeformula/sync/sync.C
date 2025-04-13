#include "TCanvas.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1.h"
#include "TClassTable.h"
#ifndef __CINT__
#include "Event.h"
#endif
#include "TMath.h"
#include "Riostream.h"

bool Compare(TH1F *draw, TH1F *loop, const char *title) {

  if (draw->GetEntries()!=loop->GetEntries()) {
    cout << title << ": incorrect number of entries (" << draw->GetEntries()
         << " vs " << loop->GetEntries() << ")" << endl;
    return false;
  }

  double dMean = draw->GetMean();
  double lMean = loop->GetMean();
  // Assume precision of 1E-6
  if (fabs(dMean - lMean) * 1E6 > fabs(dMean + lMean)) {
    cout <<  title << ": incorrect mean (" << draw->GetMean()
         << " vs " << loop->GetMean() << ")" << endl;
    return false;
  }
  return true;

}

Int_t sync(bool skipKnownFail) {
  if (!TClassTable::GetDict("Event")) {
    gSystem->Load("libEvent");
  }

  TFile * file = new TFile("Event.root");
  TTree * tree = (TTree*)file->Get("T");

  new TCanvas("c1");
  tree->Draw("fPx>>h3","fMatrix || fMatrix==0");

  tree->Draw("fMatrix>>h1","fVertex>=2");
  tree->Draw("fMatrix>>h5","fPx");
  tree->Draw("fTracks.fVertex - fTracks.fVertex[][fTracks.fNpoint%3]>>h7");


  TH1F * h1 = (TH1F*)gROOT->FindObject("h1");
  TH1F * h3 = (TH1F*)gROOT->FindObject("h3");
  TH1F * h5 = (TH1F*)gROOT->FindObject("h5");
  TH1F * h7 = (TH1F*)gROOT->FindObject("h7");

  TH1F * h2 = new TH1F("h2","h2",9,-3,+7);
  //h2->Reset();
  //h2->SetBit(TH1::kCanRebin);
  TH1F * h4 = new TH1F("h4","h4",h3->GetNbinsX(),
                       h3->GetXaxis()->GetXmin(),
                       h3->GetXaxis()->GetXmax());
  TH1F * h6 = new TH1F("h6","h6",h5->GetNbinsX(),
                       h5->GetXaxis()->GetXmin(),
                       h5->GetXaxis()->GetXmax());
  TH1F * h8 = new TH1F("h8","h8",h7->GetNbinsX(),
                       h7->GetXaxis()->GetXmin(),
                       h7->GetXaxis()->GetXmax());

  Event *e = 0;
  Track *t;
  tree->SetBranchAddress("event",&e);
  double nentries = tree->GetEntries();
  for (int i = 0; i< nentries; i++ ) {
    tree->GetEntry(i);
    int ntracks = e->GetNtrack();
    for (int i1=0; i1<ntracks; i1++) {
       t = (Track*) e->GetTracks()->At(i1);
       int point = t->GetNpoint() % 3;
       for (int i2=0; i2<3; i2++) {
          if (  t->GetVertex(i2) - t->GetVertex(point) < -40 ) {
             fprintf(stderr,"at %d, %d (%d) found %f",
                     i1, i2, point,  t->GetVertex(i2) - t->GetVertex(point));
          }
          h8->Fill( t->GetVertex(i2) - t->GetVertex(point) );
       }
    }


    ntracks = TMath::Min(ntracks,4);
    for (int j=0; j<ntracks; j++) {
      t = (Track*) e->GetTracks()->At(j);

      for (int i0=0; i0<4; i0++) {
         h4->Fill(t->GetPx());
         if (t->GetPx()) h6->Fill(e->GetMatrix(j,i0),t->GetPx());
      }

      for (int k=0; k<3; k++) {
         if (t->GetVertex(k) >=2) {
            h2->Fill(e->GetMatrix(j,k));
         }
      }
    }
  }
  tree->SetBranchAddress("event", 0);
  new TCanvas("c2");
  bool result = true;
  cout << result << endl;
  result &= Compare(h1,h2,h1->GetTitle());
  cout << result << endl;
  result &= Compare(h3,h4,h3->GetTitle());
  cout << result << endl;
  result &= Compare(h5,h6,h5->GetTitle());
  if (!skipKnownFail) {
    cout << result << endl;
    result &= Compare(h7,h8,h7->GetTitle());
  }
  // h7->Dump();
  // h8->Dump();
  h8->Draw();
  cout << result << endl;
  return result;
}
