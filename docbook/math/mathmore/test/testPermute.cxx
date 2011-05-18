#include <iostream>
#include <ctime>
#include <algorithm>
#include <vector>

#include <gsl/gsl_permutation.h>

#include <TRandom2.h>
#include <TMath.h>
#include <TStopwatch.h>

#include <TApplication.h>
#include <TCanvas.h>
#include <TH2F.h>
#include <TGraph.h>
#include <TLegend.h>
#include <TAxis.h>

bool showGraphics = true;

const int npass = 2;
const int minsize = 5;
const int maxsize = 12;
const int maxint = 5000;
const int arraysize = (maxsize-minsize) + 1;

using namespace std;

 ostream& operator <<(ostream& os, const vector<Int_t>& v)
{
   os << "[ ";
   for ( vector<Int_t>::const_iterator i = v.begin(); i != v.end() ; ++i) {
      os << *i << ' ';
   }
   os << ']';

   return os;
}

void initArray(Int_t n, vector<Int_t>& array)
{
   TRandom2 r( time( 0 ) );
   for ( Int_t i = 0; i < n; i++) {
      array[i] = r.Integer( maxint ); 
   }
   sort(array.begin(), array.end());
}

bool checkPermute()
{
   const Int_t n = minsize;

   vector<Int_t> original(n);
   vector<Int_t> vM(n);
   vector<Int_t> vS(n);

   bool equals = true;

   initArray(n, original);

   // TMATH
   copy(original.begin(), original.end(), vM.begin());
   copy(original.begin(), original.end(), vS.begin());
   //cout << original << vM << vS << endl;

   while ( TMath::Permute(n, &vM[0]) ) {
      std::next_permutation(&vS[0], &vS[n]);
      //cout << vM << vS << endl;
      equals &= equal(vM.begin(), vM.end(), vS.begin());
   }

   TMath::Permute(n, &vM[0]);
   std::next_permutation(vS.begin(), vS.end());
   //cout << "kFALSE: " << vM << vS << endl;

   return equals;
}

void permuteTime(const int n, double* tTMath, double* tStd)
{
   vector<Int_t> original(n);
   vector<Int_t> v(n);

   TStopwatch t; 

   initArray(n, original);

   // TMATH
   t.Start(); 
   for (int j = 0; j < npass; ++j) { 
      copy(original.begin(), original.end(), v.begin());
      while ( TMath::Permute(n, &v[0]) ) {}
   }
   t.Stop(); 
   *tTMath =  t.RealTime();
   cout << "TMath::Permute time :\t " << t.RealTime();

   // STD
   t.Start();
   for (int j = 0; j < npass; ++j) { 
      copy(original.begin(), original.end(), v.begin());
      while ( std::next_permutation(&v[0], &v[n]) ) {}
   }
   t.Stop();
   *tStd = t.RealTime();
   cout << "  std::next_permutation time :\t " << t.RealTime();
}

void testGSLPermute(Int_t n, double* tGSL)
{
   TStopwatch t; 

   gsl_permutation *original = gsl_permutation_alloc(n);
   gsl_permutation *p = gsl_permutation_alloc(n);

   gsl_permutation_init(original);

   t.Start();
   for (int j = 0; j < npass; ++j) { 
      gsl_permutation_memcpy(p, original);
      while ( gsl_permutation_next(p) == GSL_SUCCESS )
      {
//          gsl_permutation_fprintf(stdout, p, " %u");
//          fprintf(stdout, "\n");
      }
   }
   t.Stop();
   *tGSL = t.RealTime();
   cout << "  gsl_permutation_next time :\t " << t.RealTime();

   gsl_permutation_free(p);
   gsl_permutation_free(original);
}

int testPermute() 
{
   int status = 0;
   bool equals;

   vector<double> tM( arraysize );
   vector<double> tS( arraysize );
   vector<double> tG( arraysize );
   vector<double> index( arraysize );

   equals = checkPermute();

   cout << "checkPermute()...." 
        << (equals? "OK" : "FAILED")
        << endl;

   status += (equals == false);

   for ( int i = minsize; i <= maxsize; i += 1)
   {
      permuteTime(i, &tM[ i - minsize ], &tS[ i -minsize ]);
      testGSLPermute(i, &tG[ i - minsize ]);
      index[ i - minsize ] = i;
      cout << endl;
   }

   for ( int i = minsize; i <= maxsize; i += 1)
      cout << tM[ i - minsize ] << ' ' << tS[ i - minsize ] << ' ' << tG[ i - minsize ] << endl;

   if ( showGraphics )
   {
      TCanvas* c1 = new TCanvas("c1", "Comparision of Permutation Time", 600, 400);
      TH2F* hpx = new TH2F("hpx", "Comparision of Permutation Time", arraysize, minsize, maxsize, arraysize, tM[0],tS[arraysize-1]);
      hpx->SetStats(kFALSE);
      hpx->Draw();
      
      TGraph* gM = new TGraph(arraysize, &index[0], &tM[0]);
      gM->SetLineColor(2);
      gM->SetLineWidth(3);
      gM->SetTitle("TMath::Permute()");
      gM->Draw("SAME");
      
      TGraph* gS = new TGraph(arraysize, &index[0], &tS[0]);
      gS->SetLineColor(3);
      gS->SetLineWidth(3);
      gS->SetTitle("std::next_permutation()");
      gS->Draw("SAME");
      
      TGraph* gG = new TGraph(arraysize, &index[0], &tG[0]);
      gG->SetLineColor(4);
      gG->SetLineWidth(3);
      gG->SetTitle("gsl_permutation_next()");
      gG->Draw("SAME");
      
      TLegend* legend = new TLegend(0.15,0.72,0.4,0.86);
      legend->AddEntry(gM, "TMath::Permute()");
      legend->AddEntry(gS, "std::next_permutation()");
      legend->AddEntry(gG, "gsl_permutation_next()");
      legend->Draw();

      hpx->GetXaxis()->SetTitle("Array Size");
      hpx->GetYaxis()->SetTitle("Time");
      
      
      c1->Show();
   }

   cout << "Test Done!" << endl;

   return status;
}

int main(int argc, char **argv)
{
   int status = 0;

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


   status += testPermute();

   if ( showGraphics )
   {
      theApp->Run();
      delete theApp;
      theApp = 0;
   }

   return status;
}
