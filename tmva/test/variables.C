#include "tmvaglob.C"

void variables( TString fin = "TMVA.root" )
{
  gROOT->Reset();
  gROOT->SetStyle("Plain");
  gStyle->SetOptStat(0);
  TList * loc = gROOT->GetListOfCanvases();
  TListIter itc(loc);
  TObject *o(0);
  while( (o = itc()) ) delete o;

  TFile *file = new TFile( fin );

  input_variables->cd();

  // how many plots are in the "input_variables" directory?
  Int_t noPlots = ((gDirectory->GetListOfKeys())->GetEntries()) / 2;

  // define Canvas layout here!
  // default setting
  Int_t xPad;  // no of plots in x
  Int_t yPad;  // no of plots in y
  Int_t width; // size of canvas
  Int_t height;
  switch (noPlots) {
  case 1:
    xPad = 1; yPad = 1; width = 500; height = width; break;
  case 2:
    xPad = 2; yPad = 1; width = 600; height = 0.7*width; break;
  case 3:
    xPad = 3; yPad = 1; width = 800; height = 0.5*width; break;
  case 4:
    xPad = 2; yPad = 2; width = 600; height = width; break;
  default:
    xPad = 3; yPad = 2; width = 800; height = 0.7*width; break;
  }
  Int_t noPad = xPad * yPad ;   

  // this defines how many canvases we need
  const Int_t noCanvas = 1 + (Int_t)(noPlots/noPad);
  TCanvas **c = new TCanvas*[noCanvas];
  for (Int_t ic=0; ic<noCanvas; ic++) c[ic] = 0;

  cout << "--- Found: " << noPlots << " plots; "
       << "will produce: " << noCanvas << " canvas" << endl;

  // counter variables
  Int_t countCanvas = 0;
  Int_t countPad    = 1;

  // loop over all objects in "input_variables" directory
  TIter next(gDirectory->GetListOfKeys());
  TKey *key;
  char fname[200];
  while ((key = (TKey*)next())) {

    // make sure, that we only look at histograms
    TClass *cl = gROOT->GetClass(key->GetClassName());
    if (!cl->InheritsFrom("TH1")) continue;
    TH1 *sig = (TH1*)key->ReadObj();
    TString hname= sig->GetName();

    // check for all signal histograms
    if (hname.Contains("__S")){ // found a new signal plot

      // create new canvas
      if ((c[countCanvas]==NULL) || (countPad>noPad)) {
        cout << "--- Book canvas no: " << countCanvas << endl;
        char cn[20];
        sprintf( cn, "canvas%d", countCanvas+1 );
        c[countCanvas] = new TCanvas( cn, "MVA Input Variables", 
                                      countCanvas*50+300, countCanvas*20, width, height ); 
        // style
        c[countCanvas]->SetBorderMode(0);
        c[countCanvas]->SetFillColor(10);

        c[countCanvas]->Divide(xPad,yPad);
        countPad = 1;
      }       

      // save canvas to file
      c[countCanvas]->cd(countPad);
      countPad++;
      if (countPad > noPad) {
        c[countCanvas]->Update();
        sprintf( fname, "plots/variables_c%i", countCanvas+1 );
        TMVAGlob::imgconv( c[countCanvas], &fname[0] );
        countCanvas++;
      }

      // find the corredponding backgrouns histo
      TString bgname = hname;
      bgname.ReplaceAll("__S","__B");
      TH1 *bgd = (TH1*)gDirectory->Get(bgname);
      if (bgd == NULL){
          cout << "ERROR!!! couldn't find backgroung histo for" << hname << endl;
        exit;
      }
      // this is set but not stored during plot creation in MVA_Factory
      TMVAGlob::SetSignalAndBackgroundStyle( sig, bgd );

      // chop off "signal" 
      TString title(sig->GetTitle());
      title.ReplaceAll("signal","");
      sig->SetTitle( TString("MVA input variable: ") + title );

      TMVAGlob::SetFrameStyle( sig, 1.2 );

      // finally plot and overlay       
      Float_t sc = 1.1;
      if (countPad==2) sc = 1.3;
      sig->SetMaximum( TMath::Max( sig->GetMaximum(), bgd->GetMaximum() )*sc );
      sig->Draw();
      bgd->SetLineColor( 2 );
      bgd->SetLineWidth( 1 );
      bgd->SetFillStyle( 3002 );
      bgd->SetFillColor( 46 );
      bgd->Draw("same");
      sig->GetYaxis()->SetTitleOffset( 1.35 );
      sig->GetYaxis()->SetTitle("Normalized");

      // redraw axes
      sig->Draw("sameaxis");

      // Draw legend
      if (countPad==2){
        TLegend *legend= new TLegend( 0.131, 0.762, 0.531, 0.901 );
        legend->AddEntry(sig,"Signal","F");
        legend->AddEntry(bgd,"Background","F");
        legend->Draw("same");
        legend->SetBorderSize(1);
        legend->SetMargin( 0.3 );
      } 
     
    }
  }
  if (countPad <= noPad) {

    c[countCanvas]->Update();
    sprintf( fname, "plots/variables_c%i", countCanvas+1 );
    TMVAGlob::imgconv( c[countCanvas], &fname[0] );
  }
}
