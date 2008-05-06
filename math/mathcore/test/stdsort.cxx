#include <iostream>
#include <algorithm>
#include <ctime>
#include <vector>

#include "TStopwatch.h"
#include "TMath.h"
#include "TRandom2.h"

#include <TApplication.h>
#include <TCanvas.h>
#include <TH2F.h>
#include <TGraph.h>
#include <TLegend.h>
#include <TAxis.h>

using namespace std;

const int npass = 100000;
const int maxint = 20;
const int minsize = 20;
const int maxsize = 500;
const int increment = 10;
const int arraysize = (maxsize-minsize)/10 + 1;

bool showGraphics = true;

template<typename T> 
struct Compare { 

   Compare(const T *  d) : fData(d) {}

   bool operator()(int i1, int i2) { 
      return fData[i1] > fData[i2];
   }

   const T * fData; 
};

template <typename T> void testSort(const int n, double* tTMath, double* tStd)
{
   vector<T> k(n);

   vector<T> index(n);
   TStopwatch t; 

   TRandom2 r( time( 0 ) );
   for ( Int_t i = 0; i < n; i++) {
      k[i] = (T) r.Integer( maxint ); 
//      cout << k[i] << ' ' << endl;
   }

   t.Start(); 
   for (int j = 0; j < npass; ++j) { 
//    for(Int_t i = 0; i < n; i++) { index[i] = i; }
      TMath::Sort(n,&k[0],&index[0],kTRUE);  
   }
   t.Stop(); 
   *tTMath = t.RealTime();
   cout << "\nTMath Sort\n";
   cout << "TMath::Sort time :\t\t " << t.RealTime() << endl;

   cout << "\nSort using indices" << endl;
  
   t.Start(); 
   for (int j = 0; j < npass; ++j) { 
      for(Int_t i = 0; i < n; i++) { index[i] = i; }
      std::sort(&index[0],&index[n], Compare<T>(&k[0]) );
   }
   t.Stop(); 
   *tStd = t.RealTime();
   std::cout << "std::sort using indices:\t " << t.RealTime() << std::endl;
}

void stdsort() 
{
   vector<double> tM( arraysize );
   vector<double> tS( arraysize );
   vector<double> index( arraysize );

   //cout << (maxsize-minsize)/10 + 1 << endl;

   for ( int i = minsize; i <= maxsize; i += increment)
   {
      testSort<Int_t>(i, &tM[(i-minsize)/10], &tS[(i-minsize)/10]);
      index[(i-minsize)/10] = i;
   }

   for ( int i = minsize; i <= maxsize; i += increment)
      cout << tM[(i-minsize)/10] << ' ' << tS[(i-minsize)/10] << endl;

   if ( showGraphics )
   {
      TCanvas* c1 = new TCanvas("c1", "Comparision of Sorting Time", 600, 400);
      TH2F* hpx = new TH2F("hpx", "Comparision of Sorting Time", arraysize, minsize, maxsize, arraysize, 0,tM[arraysize-1]);
      hpx->SetStats(kFALSE);
      hpx->Draw();
      
      TGraph* gM = new TGraph(arraysize, &index[0], &tM[0]);
      gM->SetLineColor(2);
      gM->SetLineWidth(3);
      gM->SetTitle("TMath::Sort()");
      gM->Draw("SAME");
      
      TGraph* gS = new TGraph(arraysize, &index[0], &tS[0]);
      gS->SetLineColor(3);
      gS->SetLineWidth(3);
      gS->SetTitle("std::sort()");
      gS->Draw("SAME");
      
      TLegend* legend = new TLegend(0.15,0.72,0.4,0.86);
      legend->AddEntry(gM, "TMath::Sort()");
      legend->AddEntry(gS, "std::sort()");
      legend->Draw();
      
      hpx->GetXaxis()->SetTitle("Array Size");
      hpx->GetYaxis()->SetTitle("Time");
      
      
      c1->Show();
   }

}

int main(int argc, char **argv)
{
   if ( argc > 1 && argc != 2 )
   {
      cerr << "Usage: " << argv[0] << " [-ng]\n";
      cerr << "  where:\n";
      cerr << "     -ng : no graphics mode";
      cerr << endl;
      exit(1);
   }

   if ( argc == 2 && strcmp( argv[1], "-ng") == 0 ) 
   {
      showGraphics = false;
   }

   TApplication* theApp = 0;
   if ( showGraphics )
      theApp = new TApplication("App",&argc,argv);

   stdsort();

   if ( showGraphics )
   {
      theApp->Run();
      delete theApp;
      theApp = 0;
   }

   return 0;
}
