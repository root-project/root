#include <iostream>
#include <fstream>
#include <vector>

#include <cmath>

#include <TMath.h>
#include <Math/SpecFuncMathCore.h>

#include <TApplication.h>

#include <TCanvas.h>
#include <TH2F.h>
#include <TGraph.h>
#include <TLegend.h>

const double ERRORLIMIT = 1E-8;
const double MIN = -2.5;
const double MAX = +2.5;
const double INCREMENT = 0.01;
const int ARRAYSIZE = (int) (( MAX - MIN ) / INCREMENT) + 1;

bool showGraphics = false;
//bool verbose = false;
using namespace std;

TGraph* drawPoints(Double_t x[], Double_t y[], int color, int style = 1)
{
   TGraph* g = new TGraph(ARRAYSIZE, x, y);
   g->SetLineColor(color);
   g->SetLineStyle(style);
   g->SetLineWidth(3);
   g->Draw("SAME");

   return g;
}

int testSpecFuncGamma()
{
   vector<Double_t> x( ARRAYSIZE );
   vector<Double_t> yg( ARRAYSIZE );
   vector<Double_t> ymtg( ARRAYSIZE );
   vector<Double_t> yga( ARRAYSIZE );
   vector<Double_t> ymga( ARRAYSIZE );
   vector<Double_t> ylng( ARRAYSIZE );
   vector<Double_t> ymlng( ARRAYSIZE );

   Double_t a = 0.56;

   int status = 0;

   //ofstream cout ("values.txt");

   unsigned int index = 0;
   for ( double i = MIN; i < MAX; i += INCREMENT )
   {
//       cout << "i:"; cout.width(5); cout << i
//            << " index: "; cout.width(5); cout << index
//            << " TMath::Gamma(x): "; cout.width(10); cout << TMath::Gamma(i)
//            << " ROOT::Math::tgamma(x): "; cout.width(10); cout << ROOT::Math::tgamma(i)
//            << " TMath::Gamma(a, x): "; cout.width(10); cout << TMath::Gamma(a, i)
//            << " ROOT::Math::Inc_Gamma(a, x): "; cout.width(10); cout << ROOT::Math::inc_gamma(a, i)
//            << " TMath::LnGamma(x): "; cout.width(10); cout << TMath::LnGamma(i)
//            << " ROOT::Math::lgamma(x): "; cout.width(10); cout << ROOT::Math::lgamma(i)
//            << endl;

      x[index] = i;
      yg[index] = TMath::Gamma(i);
      ymtg[index] = ROOT::Math::tgamma(i);
      // take the infinity values out of the error checking!
      if ( std::fabs(yg[index]) < 1E+12 && std::fabs( yg[index] - ymtg[index] ) > ERRORLIMIT )
      {
         cout << "i " << i
              << " yg[index] " << yg[index]
              << " ymtg[index] " << ymtg[index]
              << " " << std::fabs( yg[index] - ymtg[index] )
              << endl;
         status += 1;
      }

      yga[index] = TMath::Gamma(a, i);
      ymga[index] = ROOT::Math::inc_gamma(a, i);
      if ( std::fabs( yga[index] - ymga[index] ) > ERRORLIMIT )
      {
         cout << "i " << i
              << " yga[index] " << yga[index]
              << " ymga[index] " << ymga[index]
              << " " << std::fabs( yga[index] - ymga[index] )
              << endl;
         status += 1;
      }

      ylng[index] = TMath::LnGamma(i);
      ymlng[index] = ROOT::Math::lgamma(i);
      if ( std::fabs( ylng[index] - ymlng[index] ) > ERRORLIMIT )
      {
         cout << "i " << i
              << " ylng[index] " << ylng[index]
              << " ymlng[index] " << ymlng[index]
              << " " << std::fabs( ylng[index] - ymlng[index] )
              << endl;
         status += 1;
      }

      index += 1;
   }

   if ( showGraphics )
   {

      TCanvas* c1 = new TCanvas("c1", "Two Graphs", 600, 400);
      TH2F* hpx = new TH2F("hpx", "Two Graphs(hpx)", ARRAYSIZE, MIN, MAX, ARRAYSIZE, -1,5);
      hpx->SetStats(kFALSE);
      hpx->Draw();

      TGraph* gg    = drawPoints(&x[0], &yg[0], 1);
      TGraph* gmtg  = drawPoints(&x[0], &ymtg[0], 2, 7);
      TGraph* gga   = drawPoints(&x[0], &yga[0], 3);
      TGraph* gmga  = drawPoints(&x[0], &ymga[0], 4, 7);
      TGraph* glng  = drawPoints(&x[0], &ylng[0], 5);
      TGraph* gmlng = drawPoints(&x[0], &ymlng[0], 6, 7);

      TLegend* legend = new TLegend(0.61,0.52,0.86,0.86);
      legend->AddEntry(gg,    "TMath::Gamma()");
      legend->AddEntry(gmtg,  "ROOT::Math::tgamma()");
      legend->AddEntry(gga,   "TMath::GammaI()");
      legend->AddEntry(gmga,  "ROOT::Math::inc_gamma()");
      legend->AddEntry(glng,  "TMath::LnGamma()");
      legend->AddEntry(gmlng, "ROOT::Math::lgamma()");
      legend->Draw();

      c1->Show();
   }

   cout << "Test Done!" << endl;

   return status;
}


int main(int argc, char **argv)
{
  // Parse command line arguments
  for (Int_t i=1 ;  i<argc ; i++) {
     std::string arg = argv[i] ;
     if (arg == "-g") {
      showGraphics = true;
     }
     if (arg == "-v") {
      showGraphics = true;
      //verbose = true;
     }
     if (arg == "-h") {
        cerr << "Usage: " << argv[0] << " [-g] [-v]\n";
        cerr << "  where:\n";
        cerr << "     -g : graphics mode\n";
        cerr << "     -v : verbose  mode";
        cerr << endl;
        return -1;
     }
   }

   TApplication* theApp = 0;
   if ( showGraphics )
      theApp = new TApplication("App",&argc,argv);

   int status = testSpecFuncGamma();

   if ( showGraphics )
   {
      theApp->Run();
      delete theApp;
      theApp = 0;
   }

   return status;
}
