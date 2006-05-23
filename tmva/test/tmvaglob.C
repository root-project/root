// global TMVA style settings

namespace TMVAGlob {

  //signal
  const Int_t FillColor__S = 38;
  const Int_t FillStyle__S = 1001;
  const Int_t LineColor__S = 1;
  const Int_t LineWidth__S = 1;

  // background
  const Int_t FillColor__B = 2;
  const Int_t FillStyle__B = 3002;
  const Int_t LineColor__B = 2;
  const Int_t LineWidth__B = 1;

  // set the style
  void SetSignalAndBackgroundStyle( TH1* sig, TH1* bgd, TH1* all = 0 ) 
  {
    if (sig != NULL) {
      sig->SetLineColor( LineColor__S );
      sig->SetLineWidth( LineWidth__S );
      sig->SetFillStyle( FillStyle__S );
      sig->SetFillColor( FillColor__S );
    }
    
    if (bgd != NULL) {
      bgd->SetLineColor( LineColor__B );
      bgd->SetLineWidth( LineWidth__B );
      bgd->SetFillStyle( FillStyle__B );
      bgd->SetFillColor( FillColor__B );
    }

    if (all != NULL) {
      all->SetLineColor( LineColor__S );
      all->SetLineWidth( LineWidth__S );
      all->SetFillStyle( FillStyle__S );
      all->SetFillColor( FillColor__S );
    }
  }

  // set frame styles
  SetFrameStyle( TH1* frame, Float_t scale = 1.0 )
  {
    frame->SetLabelOffset( 0.012, "X" );// label offset on x axis
    frame->SetLabelOffset( 0.012, "Y" );// label offset on x axis
    frame->GetXaxis()->SetTitleOffset( 1.25 );
    frame->GetYaxis()->SetTitleOffset( 1.22 );
    frame->GetXaxis()->SetTitleSize( 0.045*scale );
    frame->GetYaxis()->SetTitleSize( 0.045*scale );
    Float_t labelSize = 0.04*scale;
    frame->GetXaxis()->SetLabelSize( labelSize );
    frame->GetYaxis()->SetLabelSize( labelSize );

    // global style settings
    gPad->SetTicks();
    gPad->SetLeftMargin  ( 0.108*scale );
    gPad->SetRightMargin ( 0.050*scale );
    gPad->SetBottomMargin( 0.120*scale  );
  }

  // used to create output file for canvas
  void imgconv( TCanvas* c, TString fname )
  {
    if (NULL == c) {
      cout << "--- Error in TMVAGlob::imgconv: canvas is NULL" << endl;
    }
    else {
      // create directory if not existing
      TString f = fname;
      TString dir = f.Remove( f.Last( '/' ), f.Length() - f.Last( '/' ) );
      gSystem->mkdir( dir );

      TString pngName = fname + ".png";
      TString gifName = fname + ".gif";
      TString epsName = fname + ".eps";
      
      // create png
      TImage *img = TImage::Create();      
      img->FromPad( c );
      img->WriteImage( gifName );
      img->WriteImage( pngName );
      
      // create eps (other option: c->Print( epsName ))
      c->SaveAs(epsName);
      
      cout << "--- Create output files: " << pngName << " (+ .gif and .eps)" << endl;
    }
  }

  void plot_logo( Float_t v_scale = 1.0 )
  {
    TImage *img = TImage::Open("tmva_logo.gif");
    if (!img) {
      printf("Could not create an image... exit\n");
      return;
    }
    img->SetConstRatio(kFALSE);
    UInt_t h_ = img->GetHeight();
    UInt_t w_ = img->GetWidth();

    Float_t d = 0.045;
    // absolute coordinates
    Float_t x1L = 0.7803;
    Float_t y1L = 0.91;
    TPad *p1 = new TPad("img", "img", x1L, y1L, x1L + d*w_/h_, y1L + d*1.5*v_scale );

    p1->Draw();
    p1->cd();
    img->Draw();
  } 

}

