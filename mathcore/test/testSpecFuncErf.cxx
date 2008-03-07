#include <iostream>
#include <fstream>
#include <vector>

#include <TMath.h>
#include <Math/SpecFunc.h>

// #include "SpecFuncCephes.h"

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

void testSpecFuncErf() 
{
   vector<Double_t> x( ARRAYSIZE );
   vector<Double_t> yerf( ARRAYSIZE );
   vector<Double_t> ymerf( ARRAYSIZE );
   vector<Double_t> yerfc( ARRAYSIZE );
   vector<Double_t> ymerfc( ARRAYSIZE );
   vector<Double_t> yierf( ARRAYSIZE );
   vector<Double_t> yierfc( ARRAYSIZE );
//    vector<Double_t> yndtri( ARRAYSIZE );

   ofstream outputFile ("values.txt");

   for ( double i = MIN; i < MAX; i += INCREMENT )
   {
      outputFile << "i:"; outputFile.width(5); outputFile << i 
           << " index: "; outputFile.width(5); outputFile << arrayindex(i) 
           << " TMath::Erf(x): "; outputFile.width(10); outputFile << TMath::Erf(i)
           << " ROOT::Math::erf(x): "; outputFile.width(10); outputFile << ROOT::Math::erf(i)
           << " TMath::Erfc(x): "; outputFile.width(10); outputFile << TMath::Erfc(i)
           << " ROOT::Math::erfc(x): "; outputFile.width(10); outputFile << ROOT::Math::erfc(i)
           << " TMath::ErfInverse(x): "; outputFile.width(10); outputFile << TMath::ErfInverse(i)
           << " TMath::ErfcInverse(x): "; outputFile.width(10); outputFile << TMath::ErfcInverse(i)
//            << " ROOT::Math::Cephes::ndtri(x): "; outputFile.width(10); outputFile << ROOT::Math::Cephes::ndtri(i)
           << endl;

      x[arrayindex(i)] = i;
      yerf[arrayindex(i)] = TMath::Erf(i);
      ymerf[arrayindex(i)] = ROOT::Math::erf(i);
      yerfc[arrayindex(i)] = TMath::Erfc(i);
      ymerfc[arrayindex(i)] = ROOT::Math::erfc(i);
      yierf[arrayindex(i)] = TMath::ErfInverse(i);
      yierfc[arrayindex(i)] = TMath::ErfcInverse(i);
//       yndtri[arrayindex(i)] = ROOT::Math::Cephes::ndtri(i);
   }

   TCanvas* c1 = new TCanvas("c1", "Two Graphs", 600, 400); 
   TH2F* hpx = new TH2F("hpx", "Two Graphs(hpx)", ARRAYSIZE, MIN, MAX, ARRAYSIZE, -1,2);
   hpx->SetStats(kFALSE);
   hpx->Draw();

   TGraph* gerf   = drawPoints(&x[0], &yerf[0], 14);
   TGraph* gmerf  = drawPoints(&x[0], &ymerf[0], 5, 7);
   TGraph* gerfc  = drawPoints(&x[0], &yerfc[0], 2);
   TGraph* gmerfc = drawPoints(&x[0], &ymerfc[0], 3, 7);
//   drawPoints(&x[0], &yierf[0], 21);
//   drawPoints(&x[0], &yierfc[0], 28);
//   drawPoints(&x[0], &yndtri[0], 9);

   TLegend* legend = new TLegend(0.61,0.62,0.86,0.86);
   legend->AddEntry(gerf,   "TMath::Erf()");
   legend->AddEntry(gmerf,  "ROOT:Math::erf()");
   legend->AddEntry(gerfc,  "TMath::Erfc()");
   legend->AddEntry(gmerfc, "ROOT::Math::erfInverse()");
   legend->Draw();

   c1->Show();

   cout << "Test Done!" << endl;

   return;
}


int main(int argc, char **argv)
{
   TApplication theApp("App",&argc,argv);
   testSpecFuncErf();
   theApp.Run();

   return 0;
}
