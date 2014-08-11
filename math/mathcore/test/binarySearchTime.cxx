#include <iostream>
#include <ctime>
#include <cstring>
#include <vector>

#include <TRandom2.h>
#include <TMath.h>
#include <TStopwatch.h>

#include <TApplication.h>
#include <TCanvas.h>
#include <TH2F.h>
#include <TGraph.h>
#include <TLegend.h>
#include <TAxis.h>

using namespace std;

const int npass0 = 200000;
const int maxint = 100;//20;
const int minsize = 10;//20;
const int maxsize = 1000000;//500;
const int increment = 10;  // increment factor (multiplicative)
const int arraysize = int(std::log10(double(maxsize/minsize)))+1;

bool showGraphics = false;
bool verbose = false;

template <typename T> bool testBinarySearch(const int n, double* tTMath, double* tStd)
{
   std::cout << "Testing size n = " << n <<  "\t(Time / call in microsec.) " << std::endl;

   vector<T> k(n);
   TStopwatch t;
   TRandom2 r( time( 0 ) );
   for ( Int_t i = 0; i < n; i++) {
      k[i] = (T) r.Integer( maxint );
   }

   std::sort(k.begin(), k.end());

   int npass = npass0/std::log10(double(10*n/minsize));

   int s1 = 0;
   t.Start();
   for (int j = 0; j < npass; ++j) {
      for ( T elem = 0; elem < maxint; ++elem ) {
         Long_t index = TMath::BinarySearch((Long_t) n, &k[0], elem);
         s1 += index;
      }
   }
   t.Stop();
   *tTMath = t.RealTime()/npass*1.E6;
   cout << "TMath::BinarySearch time :\t " << *tTMath << endl;
//   cout << "sum " << s1 << endl;

   int s2 = 0;
   t.Start();
   for (int j = 0; j < npass; ++j) {
      for ( T elem = 0; elem < maxint; ++elem ) {
         T* pind;
         pind = std::lower_bound(&k[0], &k[n], elem);
         Long_t index2 = ((*pind == elem)? (pind - &k[0]): ( pind - &k[0] - 1));
         s2+= index2;
      }
   }
   t.Stop();
   *tStd = t.RealTime()/double(npass)*1.E6;
   std::cout << "std::binary_search time:\t " << *tStd << '\n' << std::endl;
//   cout << "sum " << s2 << endl;
   if (s1 != s2) {
      Error("testBinarySearch","Different results obtained for size n = %d  - s1 = %d s2 = %d",n,s1,s2);
      return false;
   }
   return true;

}

bool binarySearchTime()
{
   vector<double> tM( arraysize );
   vector<double> tS( arraysize );
   vector<double> index( arraysize );

   //cout << (maxsize-minsize)/10 + 1 << endl;

   bool ok = true;
   int j = 0; int i = minsize;
   while ( i <= maxsize)
   {
      ok &= testBinarySearch<Double_t>(i, &tM[j], &tS[j]);
      index[j] = i;
      j++;
      i *= increment;
   }
   int ntest = j;

   if (verbose) {
      cout << " TMATH - time  ---  std time " << std::endl;
      for ( int i = 0; i < ntest; ++i) {
         cout << " size = " << index[i] << " :  " << tM[i] << ' ' << tS[i] << endl;
      }
   }

   if ( showGraphics )
   {
      TCanvas* c1 = new TCanvas("c1", "Comparision of Searching Time", 600, 400);
      c1->SetLogx(true);

      TGraph* gM = new TGraph(arraysize, &index[0], &tM[0]);
      gM->SetLineColor(2);
      gM->SetLineWidth(3);
      gM->SetMarkerStyle(20);
      gM->SetTitle("TMath::BinarySearch()");
      gM->Draw("ALP");

      TGraph* gS = new TGraph(arraysize, &index[0], &tS[0]);
      gS->SetLineColor(3);
      gS->SetLineWidth(3);
      gS->SetMarkerStyle(20);
      gS->SetTitle("std::binary_search()");
      gS->Draw("SAME");

      TLegend* legend = new TLegend(0.15,0.72,0.4,0.86);
      legend->AddEntry(gM, "TMath::BinarySearch()");
      legend->AddEntry(gS, "std::binary_search()");
      legend->Draw();

      gM->SetTitle("Comparision of Searching Time");
      gM->GetXaxis()->SetTitle("Array Size");
      gM->GetYaxis()->SetTitle("Time");


      c1->Show();
   }

   if (ok)
      cout << "Test done!" << endl;
   else
      cout << "Error: Test Failed!" << endl;
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
   if ( showGraphics )
      theApp = new TApplication("App",&argc,argv);


   bool ok = binarySearchTime();

   if ( showGraphics )
   {
      theApp->Run();
      delete theApp;
      theApp = 0;
   }

   return (ok) ? 0 : 1;
}
