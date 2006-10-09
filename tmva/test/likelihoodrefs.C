#include <vector>
#include <string>
#include "tmvaglob.C"

void likelihoodrefs( TString fin = "TMVA.root")
{
  // plot likelihood control plots
  gROOT->Reset();
  gROOT->SetStyle("Plain");

  gStyle->SetOptStat(0);
  TList* loc = gROOT->GetListOfCanvases();
  TListIter itc(loc);
  TObject *o(0);
  while( (o = itc()) ) delete o;
  
  //open file
  TFile *file = new TFile( fin );
  file->cd("Likelihood");
  TDirectory *current_sourcedir = gDirectory;
  Int_t color=1;
  TIter next(current_sourcedir->GetListOfKeys());
  TKey *key;
  TLegend *legS = new TLegend(0.14,0.7,0.87,0.87);
  TLegend *legB = new TLegend(0.14,0.7,0.87,0.87);

  Bool_t newCanvas = kTRUE;

  const UInt_t maxCanvas = 200;
  TCanvas** c = new TCanvas*[maxCanvas];
  Int_t width  = 500;
  Int_t height = 500;

  // avoid duplicated printing
  std::vector<std::string> hasBeenUsed;
  
  UInt_t ic = -1;

  while ((key = (TKey*)next())) {
    TClass *cl = gROOT->GetClass(key->GetClassName());
    if (!cl->InheritsFrom("TH1")) continue;
    TH1 *h = (TH1*)key->ReadObj();
    TH1F *b( 0 );
    TString hname( h->GetName() );

    // avoid duplicated plotting
    Bool_t found = kFALSE;
    for (UInt_t j = 0; j < hasBeenUsed.size(); j++) {
      if (hasBeenUsed[j] == hname.Data()) found = kTRUE;
    }
    if (!found) {

      // draw original plots
      if (hname.EndsWith("_sig")) {

	cout << "--- likelihoodrefs:draw variable: " << hname << endl;

	if (newCanvas) {
	  char cn[20];
	  sprintf( cn, "canvas%d", ic+1 );
	  ++ic;
	  TString n = hname;	  
	  c[ic] = new TCanvas( cn, Form( "Likelihood reference for variable: %s", 
					 (n.ReplaceAll("_sig","")).Data() ), 
			       ic*50+200, ic*20, width, height ); 
	  c[ic]->Divide(2,2);
	  newCanvas = kFALSE;
	}      

	// signal
	h->SetMaximum(h->GetMaximum()*1.1);
	color = 4; 
	c[ic]->cd(1);
	TString plotname = hname;
	legS->Clear();
	legS->SetBorderSize(1);
	h->SetMarkerColor(color);
	h->SetMarkerSize( 0.7 );
	h->SetMarkerStyle( 20 );
	h->SetLineWidth(1);
	h->SetLineColor(color);
	color++;
	legS->AddEntry(h,"Input data (signal)","p");
	h->Draw("e1");

	// background
	TString bname( hname );	
	b = (TH1F*)gDirectory->Get( bname.ReplaceAll("_sig","_bgd") );
	c[ic]->cd(3);
	color = 2;
	legB->Clear();
	legB->SetBorderSize(1);
	b->SetMaximum(b->GetMaximum()*1.1);
	b->SetLineWidth(1);
	b->SetLineColor(color);
	b->SetMarkerColor(color);
	b->SetMarkerSize( 0.7 );
	b->SetMarkerStyle( 20 );
	legB->AddEntry(b,"Input data (backgr.)","p");
	b->Draw("e1");       

	// register
	hasBeenUsed.push_back( bname.Data() );

	// the smooth histograms
	TString hsmooth = hname + "_smooth";
	h = (TH1F*)gDirectory->Get( hsmooth );
	if (h == 0) {
	  cout << "ERROR in likelihoodrefs.C: unknown histogram: " << hsmooth << endl;
	  return;
	}
	b = (TH1F*)gDirectory->Get( hsmooth.ReplaceAll("_sig","_bgd") );
     
	color = 1;
	c[ic]->cd(1);
	h->SetLineWidth(2);
	h->SetLineColor(color);
	h->SetMarkerColor(color);
	color++;
	legS->AddEntry(h,"Smoothed histogram (signal)","l");
	h->Draw("histsame");

	color = 1;
	c[ic]->cd(3);
	b->SetLineWidth(2);
	b->Draw("histsame");
	legB->AddEntry(b,"Smoothed histogram (backgr.)","l");

	hasBeenUsed.push_back( hname.Data() );

 	// the splines
	for (int i=0; i<= 5; i++) {
	  TString hspline = hname + Form( "_smooth_hist_from_spline%i", i );
	  h = (TH1F*)gDirectory->Get( hspline );
	  if (h) {
	    b = (TH1F*)gDirectory->Get( hspline.ReplaceAll("_sig","_bgd") );
	    break;
	  }
	}
	if (h == 0 || b == 0) {
	  cout << "--- likelihoodrefs.C: did not find spline for histogram: " << hname.Data() << endl;
	}
	else {

	  h->SetMaximum(h->GetMaximum()*1.5);
	  color = 4;
	  c[ic]->cd(2);
	  h->SetLineWidth(2);
	  h->SetLineColor(color);
	  legS->AddEntry(h,"Splined PDF (norm. signal)","l");
	  h->Draw("hist");
	  legS->Draw();
	  
	  b->SetMaximum(b->GetMaximum()*1.5);
	  color = 2;
	  c[ic]->cd(4);
	  b->SetLineColor(color);
	  b->SetLineWidth(2);
	  legB->AddEntry(b,"Splined PDF (norm. backgr.)","l");
	  b->Draw("hist");

	  // draw the legends
	  legB->Draw();
	  
	  hasBeenUsed.push_back( hname.Data() );
	}	  

	c[ic]->Update();

 	// write to file
 	TString fname = Form( "plots/likelihoodrefs_c%i", ic+1 );
	TMVAGlob::imgconv( c[ic], fname );
	//	c[ic]->Update();

 	newCanvas = kTRUE;
 	hasBeenUsed.push_back( hname.Data() );
      }
    }
  }
}

