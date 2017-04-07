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
//#include <TSystem.h>

using namespace std;

const int npass0 = 200000;
const int maxint = 20;
const int minsize = 10;
const int maxsize = 100000;
const int increment = 2;
const int arraysize = 20;

bool showGraphics = false;
bool verbose = false;
//std::string plotFile;

namespace {
template<typename T>
struct Compare {

   Compare(const T *  d) : fData(d) {}

   bool operator()(int i1, int i2) {
      return fData[i1] > fData[i2];
   }

   const T * fData;
};
}

template <typename T> bool testSort(const int n, double* tTMath, double* tStd)
{

   std::cout << "Testing Sort of array - size n = " << n <<  "\t(Time / call in microsec.) " << std::endl;

   vector<T> k(n);

   vector<T> index(n);
   vector<T> index2(n);
   TStopwatch t;

   TRandom2 r( time( 0 ) );
   for ( Int_t i = 0; i < n; i++) {
      k[i] = (T) r.Integer( maxint );
//      cout << k[i] << ' ' << endl;
   }

   //int npass = npass0/std::log10(10*n/minsize);
   int npass = npass0/(n/minsize);

   t.Start();
   for (int j = 0; j < npass; ++j) {
//    for(Int_t i = 0; i < n; i++) { index[i] = i; }
      TMath::Sort(n,&k[0],&index[0],kTRUE);
   }
   t.Stop();
   *tTMath = t.RealTime()/npass*1.E6;
   cout << "TMath::Sort time :\t\t " << *tTMath << endl;


   t.Start();
   for (int j = 0; j < npass; ++j) {
      for(Int_t i = 0; i < n; i++) { index2[i] = i; }
      std::sort(&index2[0],&index2[0]+n, Compare<T>(&k[0]) );
   }
   t.Stop();
   *tStd = t.RealTime()/npass*1.E6;
   std::cout << "std::sort time (using indices):\t " << *tStd << std::endl;

   // test results
   bool ok = true;
   for(Int_t i = 0; i < n && ok; i++) {
      ok &= (index[i] == index2[i]);
      if (!ok) Error("test sort","Different values found at i= %d - index1 = %d - index2 = %d",i,index[i],index2[i]);
   }
   return ok;
}

bool stdsort()
{
   vector<double> tM( arraysize );
   vector<double> tS( arraysize );
   vector<double> index( arraysize );

   //cout << (maxsize-minsize)/10 + 1 << endl;

   bool ok = true;
   int j = 0; int i = minsize;
   while ( i <= maxsize && j < arraysize)
   {
      ok &= testSort<Int_t>(i, &tM[j], &tS[j]);
      index[j] = i;
      j++;
      i *= increment;
   }
   int ntest = j;

   if (verbose) {
      cout << " TMATH - time  ---  std time " << std::endl;
      for ( i = 0; i < ntest; ++i) {
         cout << " size = " << index[i] << " :  " << tM[i] << ' ' << tS[i] << endl;
      }
   }

   if ( showGraphics )
   {
      TCanvas* c1 = new TCanvas("c1", "Comparision of Sorting Time", 600, 400);

      TGraph* gM = new TGraph(ntest, &index[0], &tM[0]);
      gM->SetLineColor(2);
      gM->SetLineWidth(3);
      gM->SetMarkerStyle(20);
      gM->SetTitle("TMath::Sort()");
      gM->Draw("ALP");

      TGraph* gS = new TGraph(arraysize, &index[0], &tS[0]);
      gS->SetLineColor(3);
      gS->SetLineWidth(3);
      gS->SetTitle("std::sort()");
      gS->Draw("SAME");

      TLegend* legend = new TLegend(0.15,0.72,0.4,0.86);
      legend->AddEntry(gM, "TMath::Sort()");
      legend->AddEntry(gS, "std::sort()");
      legend->Draw();

      TH1 * hpx = gM->GetHistogram();
      hpx->GetXaxis()->SetTitle("Array Size");
      hpx->GetYaxis()->SetTitle("Time");
      hpx->SetTitle("Comparison of Sorting Time");


      c1->Show();

   }

   return ok;
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
      verbose = true;
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
   if ( showGraphics ) {
      theApp = new TApplication("App",&argc,argv);
//      plotFile = std::string(gSystem->TempDirectory())+std::string("/test-stdsort.pdf");
   }


   bool ok = stdsort();

   if ( showGraphics )
   {
      theApp->Run();
      delete theApp;
      theApp = 0;
   }

   return (ok) ? 0 : 1;
}
