
{
   // --------- S t y l e ---------------------------
   const Bool_t UsePaperStyle = 0;
   // -----------------------------------------------

   gSystem->Load("libMLP");

   // load TMVA shared library created in local release
   TString libTMVA( "../lib/libTMVA.1" );
   gSystem->Load( libTMVA );
 
   // welcome the user
   TMVA::Tools::TMVAWelcomeMessage();
   cout << "TMVAlogon: loaded TMVA library: \"" << libTMVA << "\"" << endl;
 
   // some basic style settings
   TStyle *TMVAStyle = gROOT->GetStyle("Plain"); // our style is based on Plain
   // new TStyle("TMVA","TMVA plots style");
   // the pretty color palette of old
   TMVAStyle->SetPalette(1,0);
 
   // use plain black on white colors
   TMVAStyle->SetFrameBorderMode(0);
   TMVAStyle->SetCanvasBorderMode(0);
   TMVAStyle->SetPadBorderMode(0);
   TMVAStyle->SetPadColor(0);
   TMVAStyle->SetFillStyle(0);

   TMVAStyle->SetLegendBorderSize(0);
 
   // title properties
   // TMVAStyle->SetTitleW(.4);
   // TMVAStyle->SetTitleH(.10);
   // MVAStyle->SetTitleX(.5);
   // TMVAStyle->SetTitleY(.9);
   TMVAStyle->SetTitleFillColor(33 + 150);
   if (!UsePaperStyle) {
      TMVAStyle->SetFrameFillColor(19 + 150);
      TMVAStyle->SetCanvasColor(21 + 150);
   }

   // set the paper & margin sizes
   TMVAStyle->SetPaperSize(20,26);
   TMVAStyle->SetPadTopMargin(0.10);
   TMVAStyle->SetPadRightMargin(0.05);
   TMVAStyle->SetPadBottomMargin(0.11);
   TMVAStyle->SetPadLeftMargin(0.12);
  
   // use bold lines and markers
   TMVAStyle->SetMarkerStyle(21);
   TMVAStyle->SetMarkerSize(0.3);
   TMVAStyle->SetHistLineWidth(1.85);
   TMVAStyle->SetLineStyleString(2,"[12 12]"); // postscript dashes
 
   // do not display any of the standard histogram decorations
   TMVAStyle->SetOptTitle(1);
   TMVAStyle->SetTitleH(0.052);

   TMVAStyle->SetOptStat(0);
   TMVAStyle->SetOptFit(0);
 
   // put tick marks on top and RHS of plots
   TMVAStyle->SetPadTickX(1);
   TMVAStyle->SetPadTickY(1);

   gROOT->SetStyle("Plain"); 
   cout << "TMVAlogon: use " << gStyle->GetName() << " style with \"Pretty Color Palette\"" << endl;
   cout << endl;
}
