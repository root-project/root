#include "TH2.h"
#include "TF2.h"
#include "TCanvas.h"
#include "TApplication.h"

#include "TMath.h"
#include "Fit/SparseData.h"
#include "HFitInterface.h"
#include "Fit/Fitter.h"
#include "Math/WrappedMultiTF1.h"

#include <iostream>
#include <iterator>
#include <algorithm>
#include <functional>

using namespace std;


double minRange[3] = { -5., -5., -5.};
double maxRange[3] = {  5.,  5.,  5.};
int       nbins[3] = {10 , 10 , 100 };

bool showGraphics = false;

ostream& operator << (ostream& out, ROOT::Fit::BinData& bd)
{
   const unsigned int ndim( bd.NDim() );
   const unsigned int npoints( bd.NPoints() );
   for ( unsigned int i = 0; i < npoints; ++i )
   {
      double value = 0, error = 0;
      const double *x = bd.GetPoint(i, value, error);
      for ( unsigned int j = 0; j < ndim; ++j )
      {
         out << " x[" << j << "]: " << x[j];
      }
      out << " value: " << value;
      out << " error: " << error;
      out << endl;
   }
   return out;
}

int findBin(ROOT::Fit::BinData& bd, const double *x)
{
   const unsigned int ndim = bd.NDim();
   const unsigned int npoints = bd.NPoints();

   for ( unsigned int i = 0; i < npoints; ++i )
   {
      double value1 = 0, error1 = 0;
      const double *x1 = bd.GetPoint(i, value1, error1);
      bool thisIsIt = true;
      for ( unsigned int j = 0; j < ndim; ++j )
      {
         thisIsIt &= fabs(x1[j] - x[j]) < 1E-15;
      }
      if ( thisIsIt ) return i;
   }

   cout << "ERROR FINDING BIN!" << endl;
   return -1;
}

bool operator ==(ROOT::Fit::BinData& bd1, ROOT::Fit::BinData& bd2)
{
   const unsigned int ndim = bd1.NDim();
   const unsigned int npoints = bd1.NPoints();
//    unsigned int npoints2 = 0;

   bool equals = true;

   cout << "Equals" << endl;

   for ( unsigned int i = 0; i < npoints && equals; ++i )
   {
      double value1 = 0, error1 = 0 ;
      const double *x1 = bd1.GetPoint(i, value1, error1);

      int bin = findBin(bd2, x1);

      double value2 = 0, error2 = 0;
      const double *x2 = bd2.GetPoint(bin, value2, error2);

      equals &= ( value1 == value2 );
      cout << " v: " << equals;
      equals &= ( error1 == error2 );
      cout << " e: " << equals;
      for ( unsigned int j = 0; j < ndim; ++j )
      {
         equals &= fabs(x1[j] - x2[j]) < 1E-15;
         cout << " x[" << j << "]: " << equals;
      }

      cout << " bd1: ";
      std::copy(x1, &x1[ndim], ostream_iterator<double>(cout, " "));
      cout << " value:" << value1;
      cout << " error:" << error1;

      cout << " bd2: ";
      std::copy(x2, &x2[ndim], ostream_iterator<double>(cout, " "));
      cout << " value:" << value2;
      cout << " error:" << error2;

      cout << " equals: " << equals;

      cout << endl;
   }

   return equals;
}


void OneDimension()
{
   TH1D* h = new TH1D("h1", "h1-title", nbins[0], minRange[0], maxRange[0]);
   h->FillRandom("gaus", 40);


   ROOT::Fit::BinData bd;
   ROOT::Fit::FillData(bd, h);

   cout << bd << endl;

   double min[] = { minRange[0] };
   double max[] = { maxRange[0] };
   ROOT::Fit::SparseData sd(1, min,max);
   ROOT::Fit::FillData(sd, h);
   ROOT::Fit::BinData bd2;
   sd.GetBinData(bd2);

   cout << bd2 << endl;

   cout << " equals : ";
   bool ok = (bd == bd2);

   cout << "One Dimension test ............\t";
   if (ok)
      cout << "OK\n";
   else
      cout << "FAILED\n";


   if (showGraphics) {
      h->Draw();
      gPad->Update();
   }
}

double gaus2D(double *x, double *p)
{
   return p[0]*TMath::Gaus(x[0],p[1],p[2]) * TMath::Gaus(x[1],p[3],p[4]);
}


void TwoDimensions()
{
   TH2D* h = new TH2D("h2", "h2-title",
                      nbins[0], minRange[0], maxRange[0],
                      nbins[1], minRange[1], maxRange[1]);

   TF2* f2 = new TF2("gaus2D", gaus2D,
                     minRange[0],maxRange[0], minRange[1], maxRange[1], 5);
   double initialPars[] = {300,0.,2.,0.,3.};
   f2->SetParameters(initialPars);

   h->FillRandom("gaus2D",20);

   ROOT::Fit::BinData bd;
   ROOT::Fit::FillData(bd, h);

   cout << bd << endl;

   double min[] = { minRange[0], minRange[1] };
   double max[] = { maxRange[0], maxRange[1] };
   ROOT::Fit::SparseData sd(2, min, max);

   ROOT::Fit::FillData(sd, h);
   ROOT::Fit::BinData bd2;
   sd.GetBinData(bd2);

   cout << bd2 << endl;

   cout << " equals : ";
   bool ok = (bd == bd2);

   if (showGraphics) {
      new TCanvas();
      h->Draw("lego2");
      gPad->Update();
   }

   cout << "Two Dimension test............\t";
   if (ok)
      cout << "OK\n";
   else
      cout << "FAILED\n";
}

int main(int argc, char** argv)
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

   if (showGraphics)
      theApp = new TApplication("App",&argc,argv);

   cout << "\nONE DIMENSION" << endl;
   OneDimension();
   cout << "\nTWO DIMENSIONS" << endl;
   TwoDimensions();

   if (showGraphics) {
      theApp->Run();
      delete theApp;
      theApp = 0;
   }

   return 0;
}

