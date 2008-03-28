#include <iostream>
#include <fstream>
#include <vector>

#include <TMath.h>
#include <Math/SpecFunc.h>

#include <TApplication.h>

#include <TCanvas.h>
#include <TH2F.h>
#include <TGraph.h>
#include <TLegend.h>

const double MIN = 0;
const double MAX = 1;
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

void testSpecFuncBetaI() 
{
   vector<Double_t> x( ARRAYSIZE );
   vector<Double_t> yb( ARRAYSIZE );
   vector<Double_t> ymb( ARRAYSIZE );

//    ofstream outputFile ("values.txt");

   TCanvas* c1 = new TCanvas("c1", "Two Graphs", 600, 400); 
   TH2F* hpx = new TH2F("hpx", "Two Graphs(hpx)", ARRAYSIZE, MIN, MAX, ARRAYSIZE, 0, 5);
   hpx->SetStats(kFALSE);
   hpx->Draw();

   int color = 2;

   TGraph *gb, *gmb;
   double b = 0.2, a= 0.9;
   cout << "** b = " << b << " **" << endl;
   for ( double i = MIN; i < MAX; i += INCREMENT )
   {
      cout << "i:"; cout.width(5); cout << i 
           << " index: "; cout.width(5); cout << arrayindex(i) 
           << " TMath::BetaIncomplete(x,a,b): "; cout.width(10); cout << TMath::BetaIncomplete(i,a,b)
           << " ROOT::Math::inc_beta(a,a,b): "; cout.width(10); cout << ROOT::Math::inc_beta(i,a,b)
           << endl;
      
      x[arrayindex(i)] = i;
      yb[arrayindex(i)] = TMath::BetaIncomplete(i,a,b);
      ymb[arrayindex(i)] = ROOT::Math::inc_beta(i,a,b);
   }
   
   gb = drawPoints(&x[0], &yb[0], color++);
   gmb = drawPoints(&x[0], &ymb[0], color++, 7);

   TLegend* legend = new TLegend(0.61,0.72,0.86,0.86);
   legend->AddEntry(gb, "TMath::BetaIncomplete()");
   legend->AddEntry(gmb, "ROOT::Math::inc_beta()");
   legend->Draw();

   c1->Show();

   cout << "Test Done!" << endl;

   return;
}

int main(int argc, char **argv) 
{
   TApplication theApp("App",&argc,argv);
   testSpecFuncBetaI();
   theApp.Run();

   return 0;
}
