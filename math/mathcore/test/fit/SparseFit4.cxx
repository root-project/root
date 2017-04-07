#include "THnSparse.h"
#include "TH2.h"
#include "TF2.h"
#include "TF3.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TMath.h"

#include "Fit/SparseData.h"
#include "HFitInterface.h"
#include "Fit/Fitter.h"
#include "Math/WrappedMultiTF1.h"

#include "TRandom2.h"

#include <iostream>
#include <iterator>
#include <algorithm>

#include <list>
#include <vector>

#include <cmath>
#include <cassert>

using namespace std;

bool showGraphics = false;

TRandom2 r(17);
Int_t bsize[] = { 10, 10, 10 };
Double_t x_min[] = { 0., 0., 0. };
Double_t x_max[] = { 10., 10., 10. };

double gaus1D(double *x, double *p)
{
   return p[0]*TMath::Gaus(x[0],p[1],p[2]);
}

double gaus2D(double *x, double *p)
{
   return p[0]*TMath::Gaus(x[0],p[1],p[2]) * TMath::Gaus(x[1],p[3],p[4]);
}

double gaus3D(double *x, double *p)
{
   return p[0] * TMath::Gaus(x[0],p[1],p[2])
               * TMath::Gaus(x[1],p[3],p[4])
               * TMath::Gaus(x[2],p[5],p[6]);
}

double pol2D(double *x, double *p)
{
   return p[0]*x[0]+ p[1]*x[0]*x[0] + p[2]*x[1] + p[3]*x[1]*x[1] + p[4];
}

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

   cout << "NO ENCONTRADO!";
   copy(x, x+ndim, ostream_iterator<double>(cout, " " ));
   cout  << endl;

   return -1;
}

bool operator ==(ROOT::Fit::BinData& bd1, ROOT::Fit::BinData& bd2)
{
   const unsigned int ndim = bd1.NDim();
   const unsigned int npoints = bd1.NPoints();

   bool equals = true;

   cout << "Equals" << endl;

   for ( unsigned int i = 0; i < npoints && equals; ++i )
   {
      double value1 = 0, error1 = 0;
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

void fillSparse(THnSparse* s, TF1* f, int nEvents = 5)
{
   const unsigned int ndim = s->GetNdimensions();

   for ( Int_t e = 0; e < nEvents; ++e ) {
      Double_t *points = new Double_t[ndim];
      for ( UInt_t i = 0; i < ndim; ++ i )
         points[i] = r.Uniform( x_min[0] * .9 , x_max[0] * 1.1 );
      double value = gRandom->Poisson( f->EvalPar(points));
      s->Fill(points, value );
      cout << value << " " << s->GetNbins() << endl;
      delete [] points;
   }
}

void DoFit(THnSparse* s, TF1* f, ROOT::Fit::BinData& bd)
{
   ///////////////// CREATE THE SPARSE DATA
   cout << "DoFit: dim = " << s->GetNdimensions() << " - Retrieving the Sparse Data Structure" << endl;
   //ROOT::Fit::SparseData d(s);
   ROOT::Fit::SparseData d(s->GetNdimensions(), x_min, x_max);
   ROOT::Fit::FillData(d, s, 0);
   d.GetBinData(bd);

   ///////////////// FITS
   // Fit preparation
   bool ret;
   ROOT::Fit::Fitter fitter;
   ROOT::Math::WrappedMultiTF1 wf2(*f);
   ROOT::Math::IParametricFunctionMultiDim & if2 = wf2;
   fitter.Config().SetMinimizer("Minuit2");

   ROOT::Fit::DataOptions opt;
   opt.fUseEmpty = true;
   opt.fIntegral = true;

   /////////////////////////////////////////////////////////////////////////
   cout << "\n ******* Likelihood with BinData and NoZeros *******" << endl;
   ROOT::Fit::BinData bdNoZeros;
   d.GetBinDataNoZeros(bdNoZeros);
   ret = fitter.LikelihoodFit(bdNoZeros, if2, true);
   fitter.Result().Print(std::cout);
   if (!ret)
      std::cout << "Fit Failed " << std::endl;

   /////////////////////////////////////////////////////////////////////////
   cout << "\n ******* Likelihood with BinData with Zeros *******" << endl;
   ROOT::Fit::BinData bdWithZeros(opt);
   d.GetBinDataIntegral(bdWithZeros);
   ret = fitter.LikelihoodFit(bdWithZeros, if2, true);
   fitter.Result().Print(std::cout);
   if (!ret)
      std::cout << "Fit Failed " << std::endl;

   /////////////////////////////////////////////////////////////////////////
}

void fitSparse1D()
{

   std::cout << "1D SPARSE FIT \n" << std::endl;

   const unsigned int ndim = 1;

   THnSparseD* s1 = new THnSparseD("mfND-s1", "s1-Title", ndim, bsize, x_min, x_max);

   TF1* f = new TF1("func1D", gaus1D, x_min[0] - 2, x_max[0] + 2, 3);
   f->SetParameters(10., 5., 2.);

   fillSparse(s1,f,5);

   cout << "1D Fit : Retrieving the Sparse Data Structure" << endl;
   //ROOT::Fit::SparseData d(s1);
   ROOT::Fit::SparseData d(ndim, x_min, x_max);
   ROOT::Fit::FillData(d, s1, 0);
}

void fitSparse2D()
{

   std::cout << "2D SPARSE FIT \n" << std::endl;
   const unsigned int ndim = 2;

   THnSparseD* s1 = new THnSparseD("mfND-s1", "s1-Title", ndim, bsize, x_min, x_max);

   TF2* f = new TF2("func2D", gaus2D, x_min[0],x_max[0], x_min[1], x_max[1], 5);
   f->SetParameters(40,5,2,5,1);

   for (int ix=1; ix <= bsize[0]; ++ix) {
      for (int iy=1; iy <= bsize[1]; ++iy) {
         double x = s1->GetAxis(0)->GetBinCenter(ix);
         double y = s1->GetAxis(1)->GetBinCenter(iy);

         double value = gRandom->Poisson( f->Eval(x,y) );
         if ( value )
         {
            double points[] = {x,y};
            s1->Fill( points, value );
         }
      }
   }

   ROOT::Fit::BinData bd;
   DoFit(s1, f, bd);

   TH2D* h2 = new TH2D("2D Blanked Hist Fit", "h1-title",
                       bsize[0], x_min[0], x_max[0],
                       bsize[1], x_min[1], x_max[1]);
   //cout << "Filling second histogram" << endl;
   for ( unsigned int i = 0; i < bd.NPoints(); ++i )
   {
      const double* x;
      double value, error;
      x = bd.GetPoint(i, value, error);
      value = (value)?value:-10;
      h2->Fill(x[0], x[1], value);
   }

   if (showGraphics) {
      h2->Draw("colZ");
      f->Draw("SAME");
      gPad->Update();
   }
}

void fitSparse3D()
{
   const unsigned int ndim = 3;

   std::cout << "3D SPARSE FIT \n" << std::endl;

   THnSparseD* s1 = new THnSparseD("mfND-s1", "s1-Title", ndim, bsize, x_min, x_max);

   TF3* f = new TF3("func3D", gaus3D,
                    x_min[0],x_max[0],
                    x_min[1],x_max[1],
                    x_min[2],x_max[2],
                    7);
   f->SetParameters(100,5,2,5,1,5,2);

   for (int ix=1; ix <= bsize[0]; ++ix) {
      for (int iy=1; iy <= bsize[1]; ++iy) {
         for (int iz=1; iz <= bsize[2]; ++iz) {
            double x = s1->GetAxis(0)->GetBinCenter(ix);
            double y = s1->GetAxis(1)->GetBinCenter(iy);
            double z = s1->GetAxis(2)->GetBinCenter(iz);
            double value = gRandom->Poisson( f->Eval(x,y,z) );

            if ( value )
            {
               double points[] = {x,y,z};
               s1->Fill( points, value );
            }
         }
      }
   }

   ROOT::Fit::BinData bd;
   DoFit(s1, f, bd);
}


int main(int argc, char** argv)
{

   bool do3dfit = false;

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
     if (arg == "-3d") {
      do3dfit = true;
     }
     if (arg == "-h") {
        cerr << "Usage: " << argv[0] << " [-g] [-v]\n";
        cerr << "  where:\n";
        cerr << "     -g  :  graphics mode\n";
        cerr << "     -v  :  verbose  mode\n";
        cerr << "     -3d :  3d fit";
        cerr << endl;
        return -1;
     }
   }


   TApplication* theApp = 0;

   if ( showGraphics )
      theApp = new TApplication("App",&argc,argv);

   // fitSparse1D();
   fitSparse2D();
   if (do3dfit) fitSparse3D();

   cout << "C'est fini!" << endl;

   if ( showGraphics ) {
      theApp->Run();
      delete theApp;
      theApp = 0;
   }

   return 0;
}

