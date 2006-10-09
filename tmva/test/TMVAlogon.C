
{

   // load TMVA shared library created in local release
   TString libTMVA( "../lib/libTMVA.1.so" );
   gSystem->Load( libTMVA );
 
   // welcome the user
   TMVA::Tools::TMVAWelcomeMessage();
   cout << "--- TMVAlogon: loaded TMVA library: \"" << libTMVA << "\"" << endl;
 
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
   TMVAStyle->SetCanvasColor(0);
   TMVAStyle->SetTitleFillColor(0);
   //TMVAStyle->SetStatColor(0);
   TMVAStyle->SetFillStyle(0);
   TMVAStyle->SetLegendBorderSize(0);
 
   // set the paper & margin sizes
   TMVAStyle->SetPaperSize(20,26);
   TMVAStyle->SetPadTopMargin(0.10);
   TMVAStyle->SetPadRightMargin(0.05);
   TMVAStyle->SetPadBottomMargin(0.11);
   TMVAStyle->SetPadLeftMargin(0.12);
 
   // use large Times-Roman fonts
   //    TMVAStyle->SetTextFont(132);
   //    TMVAStyle->SetTitleFont(132,"x");
   //    TMVAStyle->SetTitleFont(132,"y");
   //    TMVAStyle->SetTextSize(0.08);
   //    TMVAStyle->SetLabelFont(132,"x");
   //    TMVAStyle->SetLabelFont(132,"y");
   //    TMVAStyle->SetLabelFont(132,"z");
   //    TMVAStyle->SetLabelSize(0.05,"x");
   //    TMVAStyle->SetTitleSize(0.06,"x");
   //    TMVAStyle->SetLabelSize(0.05,"y");
   //    TMVAStyle->SetTitleSize(0.06,"y");
   //    TMVAStyle->SetLabelSize(0.05,"z");
   //    TMVAStyle->SetTitleSize(0.06,"z");
 
   // use bold lines and markers
   TMVAStyle->SetMarkerStyle(21);
   TMVAStyle->SetMarkerSize(0.3);
   TMVAStyle->SetHistLineWidth(1.85);
   TMVAStyle->SetLineStyleString(2,"[12 12]"); // postscript dashes
 
   // do not display any of the standard histogram decorations
   TMVAStyle->SetOptTitle(1);
   TMVAStyle->SetTitleH(0.052);
   //   TMVAStyle->SetTitleBorderSize(0);

   //    cout << "--- TMVAlogon: use \"Plain\" style with \"Pretty Color Palette\"" << endl;
   TMVAStyle->SetOptStat(0);
   TMVAStyle->SetOptFit(0);
 
   // put tick marks on top and RHS of plots
   TMVAStyle->SetPadTickX(1);
   TMVAStyle->SetPadTickY(1);

   gROOT->SetStyle("Plain"); 
   cout << "--- TMVAlogon: use " << gStyle->GetName() << " style with \"Pretty Color Palette\"" << endl;
}
