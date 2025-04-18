/// \file
/// \ingroup tutorial_graphics
/// \notebook
/// \preview This tutorial illustrates the special contour options.
///
///   - "AITOFF"     : Draw a contour via an AITOFF projection
///   - "MERCATOR"   : Draw a contour via an Mercator projection
///   - "SINUSOIDAL" : Draw a contour via an Sinusoidal projection
///   - "PARABOLIC"  : Draw a contour via an Parabolic projection
///   - "MOLLWEIDE"  : Draw a contour via an Mollweide projection
///
/// \macro_image
/// \macro_code
///
/// \author Olivier Couet (from an original macro sent by Ernst-Jan Buis)

TCanvas *earth()
{

   gStyle->SetOptTitle(1);
   gStyle->SetOptStat(0);

   TCanvas *c1 = new TCanvas("c1", "earth_projections", 700, 1000);
   c1->Divide(2, 3);

   TH2F *ha = new TH2F("ha", "Aitoff", 180, -180, 180, 179, -89.5, 89.5);
   TH2F *hm = new TH2F("hm", "Mercator", 180, -180, 180, 161, -80.5, 80.5);
   TH2F *hs = new TH2F("hs", "Sinusoidal", 180, -180, 180, 181, -90.5, 90.5);
   TH2F *hp = new TH2F("hp", "Parabolic", 180, -180, 180, 181, -90.5, 90.5);
   TH2F *hw = new TH2F("hw", "Mollweide", 180, -180, 180, 181, -90.5, 90.5);

   TString dat = gROOT->GetTutorialDir();
   dat.Append("/visualisation/graphics/earth.dat");
   dat.ReplaceAll("/./", "/");

   ifstream in;
   in.open(dat.Data());
   Float_t x, y;
   while (1) {
      in >> x >> y;
      if (!in.good())
         break;
      ha->Fill(x, y, 1);
      hm->Fill(x, y, 1);
      hs->Fill(x, y, 1);
      hp->Fill(x, y, 1);
      hw->Fill(x, y, 1);
   }
   in.close();

   c1->cd(1);
   ha->Draw("aitoff");
   c1->cd(2);
   hm->Draw("mercator");
   c1->cd(3);
   hs->Draw("sinusoidal");
   c1->cd(4);
   hp->Draw("parabolic");
   c1->cd(5);
   hw->Draw("mollweide");

   return c1;
}
