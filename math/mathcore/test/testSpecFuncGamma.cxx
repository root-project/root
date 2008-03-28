#include <iostream>
#include <fstream>
#include <vector>

#include <TMath.h>
#include <Math/SpecFuncMathCore.h>

#include <TApplication.h>

#include <TCanvas.h>
#include <TH2F.h>
#include <TGraph.h>
#include <TLegend.h>

const double MIN = -2.5;
const double MAX = +2.5;
const double INCREMENT = 0.01;
const int ARRAYSIZE = (int) (( MAX - MIN ) / INCREMENT);
inline int arrayindex(double i) { return ARRAYSIZE - (int) ( (MAX - i) / INCREMENT ) -1 ; };

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

void testSpecFuncGamma() 
{
   vector<Double_t> x( ARRAYSIZE );
   vector<Double_t> yg( ARRAYSIZE );
   vector<Double_t> ymtg( ARRAYSIZE );
   vector<Double_t> yga( ARRAYSIZE );
   vector<Double_t> ymga( ARRAYSIZE );
   vector<Double_t> ylng( ARRAYSIZE );
   vector<Double_t> ymlng( ARRAYSIZE );

   Double_t a = 0.56;

   //ofstream cout ("values.txt");

   for ( double i = MIN; i < MAX; i += INCREMENT )
   {
//       cout << "i:"; cout.width(5); cout << i 
//            << " index: "; cout.width(5); cout << arrayindex(i) 
//            << " TMath::Gamma(x): "; cout.width(10); cout << TMath::Gamma(i)
//            << " ROOT::Math::tgamma(x): "; cout.width(10); cout << ROOT::Math::tgamma(i)
//            << " TMath::Gamma(a, x): "; cout.width(10); cout << TMath::Gamma(a, i)
//            << " ROOT::Math::Inc_Gamma(a, x): "; cout.width(10); cout << ROOT::Math::inc_gamma(a, i)
//            << " TMath::LnGamma(x): "; cout.width(10); cout << TMath::LnGamma(i)
//            << " ROOT::Math::lgamma(x): "; cout.width(10); cout << ROOT::Math::lgamma(i)
//            << endl;

      x[arrayindex(i)] = i;
      yg[arrayindex(i)] = TMath::Gamma(i);
      ymtg[arrayindex(i)] = ROOT::Math::tgamma(i);
      yga[arrayindex(i)] = TMath::Gamma(a, i);
      ymga[arrayindex(i)] = ROOT::Math::inc_gamma(a, i);
      ylng[arrayindex(i)] = TMath::LnGamma(i);
      ymlng[arrayindex(i)] = ROOT::Math::lgamma(i);
   }

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

   cout << "Test Done!" << endl;

   return;
}


int main(int argc, char **argv) 
{
   TApplication theApp("App",&argc,argv);
   testSpecFuncGamma();
   theApp.Run();

   return 0;
}
