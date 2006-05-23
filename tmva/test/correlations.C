#include "tmvaglob.C"

void correlations( TString fin = "TMVA.root", Bool_t greyScale = kFALSE )
{
   gROOT->SetStyle("Plain");
   gStyle->SetOptStat(0);
   TList * loc = gROOT->GetListOfCanvases();
   TListIter itc(loc);
   TObject *o(0);
   while( (o = itc()) ) delete o;

   // signal and background
   const TString hName[2] = { "CorrelationMatrixS", "CorrelationMatrixB" };
   const Int_t width = 600;
   for (Int_t ic=0; ic<2; ic++) {

      TFile *file = new TFile( fin );
      TCanvas* c = new TCanvas( hName[ic], Form("Correlations between MVA input variables (%s)", (ic==0?"signal":"background")), 
                                ic*(width+5)+300, 0, width, width ); 
      Float_t newMargin1 = 0.13;
      Float_t newMargin2 = 0.18;

      gPad->SetGrid();
      gPad->SetTicks();
      gPad->SetLeftMargin  ( newMargin2 );
      gPad->SetBottomMargin( newMargin2 );
      gPad->SetRightMargin ( newMargin1 );
      gPad->SetTopMargin   ( newMargin1 );
      gStyle->SetPalette( 1, 0 );

      TH2* h2 = file->Get( hName[ic] );

      h2->SetMarkerSize( 1.5 );
      h2->SetMarkerColor( 0 );
      Float_t labelSize = 0.050;
      h2->GetXaxis()->SetLabelSize( labelSize );
      h2->GetYaxis()->SetLabelSize( labelSize );
      h2->LabelsOption( "d" );
      h2->SetLabelOffset( 0.011 );// label offset on x axis    

      h2->Draw("colz"); // color pads   
      c->Update();

      // modify properties of paletteAxis
      TPaletteAxis* paletteAxis = (TPaletteAxis*)h2->GetListOfFunctions()->FindObject( "palette" );
      paletteAxis->SetLabelSize( 0.03 );
      paletteAxis->SetX1NDC( paletteAxis->GetX1NDC() + 0.02 );

      h2->Draw("textsame");  // add text

      // add comment    
      TText* t = new TText( 0.31, 0.88, "absolute values for correlation coefficients given in %" );
      t->SetNDC();
      t->SetTextSize( 0.026 );
      t->AppendPad();    

      c->Modified();

      TMVAGlob::plot_logo( 0.85 );

      TString fname = "plots/";
      fname += hName[ic];
      TMVAGlob::imgconv( c, fname );
   }
}
