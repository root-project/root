#include "TH2.h"
#include "TH3.h"
#include "TF2.h"
#include "TF3.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TMath.h"

#include "Fit/SparseData.h"
#include "HFitInterface.h"
#include "Fit/Fitter.h"
#include "Math/WrappedMultiTF1.h"

#include "TRandom.h"

#include <iostream>
#include <iterator>
#include <algorithm>

#include <list>
#include <vector>

#include <cmath>
#include <cassert>

using namespace std;

const bool __DRAW__ = 1;

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
      double value, error;
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
      double value1, error1;
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
      double value1, error1;
      const double *x1 = bd1.GetPoint(i, value1, error1);

      int bin = findBin(bd2, x1);

      double value2 = 0, error2;
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

void fit3DHist()
{
   vector<double> min(3); min[0] = 0.;  min[1] = 0.;   min[2] = 0.;
   vector<double> max(3); max[0] = 10.; max[1] = 10.;  max[2] = 10.;
   vector<int> nbins(3); nbins[0] = 10; nbins[1] = 10; nbins[2] = 10;
   
   TH3D* h1 = new TH3D("3D Original Hist Fit", "h1-title", 
                       nbins[0], min[0], max[0], 
                       nbins[1], min[1], max[1],
                       nbins[2], min[2], max[2]);
   TH3D* h2 = new TH3D("3D Blanked Hist Fit", "h1-title",  
                       nbins[0], min[0], max[0], 
                       nbins[1], min[1], max[1],
                       nbins[2], min[2], max[2]);
   
   TF3* f1 = new TF3("func3D", gaus3D, 
                     min[0],max[0], 
                     min[1],max[1],
                     min[2],max[2],
                     7);
   double initialPars[] = {20,5,2,5,1,5,2};
//    TF1* f1 = new TF1("func3D", pol2D, 
//                      min[0],max[0], min[1], max[1], 5);
//    double initialPars[] = {1,0.,0.5,0.,0.};
   f1->SetParameters(initialPars);
//    f1->FixParameter(1,0.);
//    f1->FixParameter(3,0.);
   
   
   for (int ix=1; ix <= h1->GetNbinsX(); ++ix) { 
      for (int iy=1; iy <= h1->GetNbinsY(); ++iy) { 
         for (int iz=1; iz <= h1->GetNbinsZ(); ++iz) { 
            double x = h1->GetXaxis()->GetBinCenter(ix);
            double y = h1->GetYaxis()->GetBinCenter(iy);
            double z = h1->GetZaxis()->GetBinCenter(iz);
            
            h1->SetBinContent( ix, iy, iz, gRandom->Poisson( f1->Eval(x,y,z) ) );
         }
      }
   }

///////////////// CREATE THE SPARSE DATA
   cout << "Retrieving the Sparse Data Structure" << endl;
   ROOT::Fit::SparseData d(min,max);
   ROOT::Fit::FillData(d, h1, 0);
   ROOT::Fit::BinData bd;
   d.GetBinData(bd);

   
   cout << "Filling second histogram" << endl;
   for ( unsigned int i = 0; i < bd.NPoints(); ++i )
   {
      const double* x;
      double value, error;
      x = bd.GetPoint(i, value, error);
      value = (value)?value:-10;
      h2->Fill(x[0], x[1], x[2], value);
   }


   ///////////////// FITS
   // Fit preparation
   bool ret;
   ROOT::Fit::Fitter fitter;
   ROOT::Math::WrappedMultiTF1 wf1(*f1);
   ROOT::Math::IParametricFunctionMultiDim & if1 = wf1;
   fitter.Config().SetMinimizer("Minuit2");
   //fitter.Config().MinimizerOptions().SetPrintLevel(3);

   /////////////////////////////////////////////////////////////////////////
   cout << "\n ******* Chi2Fit with Original BinData *******" << endl;
   ROOT::Fit::BinData bdOriginal;
   ROOT::Fit::FillData(bdOriginal, h1, 0);
   ret = fitter.LikelihoodFit(bdOriginal, if1);
   fitter.Result().Print(std::cout); 
   if (!ret)  
      std::cout << "Fit Failed " << std::endl;

   /////////////////////////////////////////////////////////////////////////
   cout << "\n ******* Chi2Fit with Original BinData with Ceros*******" << endl;
   ROOT::Fit::DataOptions opt;
   opt.fUseEmpty = true;
   opt.fIntegral = true;
   ROOT::Fit::BinData bdOriginalWithCeros(opt);
   ROOT::Fit::FillData(bdOriginalWithCeros, h1, 0);
   ret = fitter.LikelihoodFit(bdOriginalWithCeros, if1);
   fitter.Result().Print(std::cout); 
   if (!ret)  
      std::cout << "Fit Failed " << std::endl;

   /////////////////////////////////////////////////////////////////////////
   cout << "\n ******* Chi2Fit with BinData and NoCeros *******" << endl;
   ROOT::Fit::BinData bdNoCeros;
   d.GetBinDataNoZeros(bdNoCeros);
   ret = fitter.LikelihoodFit(bdNoCeros, if1);
   fitter.Result().Print(std::cout); 
   if (!ret)  
      std::cout << "Fit Failed " << std::endl;

//    cout << "bdOriginal:\n" << bdOriginal << endl;
//    cout << "bdNoCeros:\n" << *bdNoCeros << endl;
//    cout << "Equals: " << (bdOriginal == *bdNoCeros) << endl;
   /////////////////////////////////////////////////////////////////////////
   cout << "\n ******* Chi2Fit with BinData with Ceros *******" << endl;
   ROOT::Fit::BinData bdWithCeros(opt);
   d.GetBinDataIntegral(bdWithCeros);
   ret = fitter.LikelihoodFit(bdWithCeros, if1);
   fitter.Result().Print(std::cout); 
   if (!ret)  
      std::cout << "Fit Failed " << std::endl;

//    cout << "bdOriginal:\n" << bdOriginal << endl;
//    cout << "bdWithCeros:\n" << bdWithCeros << endl;
//    cout << "Equals: " << (bdOriginal == bdWithCeros) << endl;

   /////////////////////////////////////////////////////////////////////////

  
   TCanvas* c = new TCanvas("Histogram 3D");
   c->Divide(1,2);
   c->cd(1);
   h1->Draw("lego");
   c->cd(2);
   h2->Draw("lego");
}

void fit2DHist()
{
   vector<double> min(2); min[0] = 0.;  min[1] = 0.;
   vector<double> max(2); max[0] = 10.; max[1] = 10.;
   vector<int> nbins(2); nbins[0] = 10; nbins[1] = 10;
   
   TH2D* h1 = new TH2D("2D Original Hist Fit", "h1-title", nbins[0], min[0], max[0], nbins[1], min[1], max[1]);
   TH2D* h2 = new TH2D("2D Blanked Hist Fit", "h1-title",  nbins[0], min[0], max[0], nbins[1], min[1], max[1]);
   
   TF2* f2 = new TF2("func2D", gaus2D, 
                     min[0],max[0], min[1], max[1], 5);
   double initialPars[] = {20,5,2,5,1};
//    TF2* f2 = new TF2("func2D", pol2D, 
//                      min[0],max[0], min[1], max[1], 5);
//    double initialPars[] = {1,0.,0.5,0.,0.};
   f2->SetParameters(initialPars);
//    f2->FixParameter(1,0.);
//    f2->FixParameter(3,0.);
   
   
   for (int ix=1; ix <= h1->GetNbinsX(); ++ix) { 
      for (int iy=1; iy <= h1->GetNbinsY(); ++iy) { 
         double x=  h1->GetXaxis()->GetBinCenter(ix);
         double y= h1->GetYaxis()->GetBinCenter(iy);
         
         h1->SetBinContent( ix, iy, gRandom->Poisson( f2->Eval(x,y) ) );
      }
   }

///////////////// CREATE THE SPARSE DATA
   cout << "Retrieving the Sparse Data Structure" << endl;
   ROOT::Fit::SparseData d(min,max);
   ROOT::Fit::FillData(d, h1, 0);
   ROOT::Fit::BinData bd;
   d.GetBinData(bd);

   
   cout << "Filling second histogram" << endl;
   for ( unsigned int i = 0; i < bd.NPoints(); ++i )
   {
      const double* x;
      double value, error;
      x = bd.GetPoint(i, value, error);
      value = (value)?value:-10;
      h2->Fill(x[0], x[1], value);
   }

   ///////////////// FITS
   // Fit preparation
   bool ret;
   ROOT::Fit::Fitter fitter;
   ROOT::Math::WrappedMultiTF1 wf2(*f2);
   ROOT::Math::IParametricFunctionMultiDim & if2 = wf2;
   fitter.Config().SetMinimizer("Minuit2");
   //fitter.Config().MinimizerOptions().SetPrintLevel(3);

   /////////////////////////////////////////////////////////////////////////
   cout << "\n ******* Chi2Fit with Original BinData *******" << endl;
   ROOT::Fit::BinData bdOriginal;
   ROOT::Fit::FillData(bdOriginal, h1, 0);
   ret = fitter.LikelihoodFit(bdOriginal, if2);
   fitter.Result().Print(std::cout); 
   if (!ret)  
      std::cout << "Fit Failed " << std::endl;

   /////////////////////////////////////////////////////////////////////////
   cout << "\n ******* Chi2Fit with Original BinData with Ceros*******" << endl;
   ROOT::Fit::DataOptions opt;
   opt.fUseEmpty = true;
   opt.fIntegral = true;
   ROOT::Fit::BinData bdOriginalWithCeros(opt);
   ROOT::Fit::FillData(bdOriginalWithCeros, h1, 0);
   ret = fitter.LikelihoodFit(bdOriginalWithCeros, if2);
   fitter.Result().Print(std::cout); 
   if (!ret)  
      std::cout << "Fit Failed " << std::endl;

   /////////////////////////////////////////////////////////////////////////
   cout << "\n ******* Chi2Fit with BinData and NoCeros *******" << endl;
   ROOT::Fit::BinData bdNoCeros;
   d.GetBinDataNoZeros(bdNoCeros);
   ret = fitter.LikelihoodFit(bdNoCeros, if2);
   fitter.Result().Print(std::cout); 
   if (!ret)  
      std::cout << "Fit Failed " << std::endl;

//    cout << "bdOriginal:\n" << bdOriginal << endl;
//    cout << "bdNoCeros:\n" << *bdNoCeros << endl;
//    cout << "Equals: " << (bdOriginal == *bdNoCeros) << endl;
   /////////////////////////////////////////////////////////////////////////
   cout << "\n ******* Chi2Fit with BinData with Ceros *******" << endl;
   ROOT::Fit::BinData bdWithCeros(opt);
   d.GetBinDataIntegral(bdWithCeros);
   ret = fitter.LikelihoodFit(bdWithCeros, if2);
   fitter.Result().Print(std::cout); 
   if (!ret)  
      std::cout << "Fit Failed " << std::endl;

//    cout << "bdOriginal:\n" << bdOriginal << endl;
//    cout << "bdWithCeros:\n" << *bdWithCeros << endl;
//    cout << "Equals: " << (bdOriginal == *bdWithCeros) << endl;

   /////////////////////////////////////////////////////////////////////////

  
   TCanvas* c = new TCanvas("Histogram 2D");
   c->Divide(2,2);
   c->cd(1);
   h1->Draw("colz");
   c->cd(2);
   h1->Draw("text");
   c->cd(3);
   h2->Draw("colz");
   c->cd(4);
   h2->Draw("text");
}

// void fit1DHist()
// {
//    vector<double> min(1);
//    min[0] = 0.;

//    vector<double> max(1);
//    max[0] = 10.;

//    vector<int> nbins(1);
//    nbins[0] = 10;

//    TH1D* h1 = new TH1D("1D Original Hist Fit", "h1-Original", nbins[0], min[0], max[0]);
//    TH1D* h2 = new TH1D("1D Blanked Hist Fit",  "h1-Blanked",  nbins[0], min[0], max[0]);

//    TF1* f1 = new TF1("MyGaus", "[0]*TMath::Gaus([1],[2])", min[0], max[0]);
//    f1->SetParameters(10., 5., 2.);

//    h1->FillRandom("MyGaus",1000);

//    cout << "Retrieving the Sparse Data Structure" << endl;
//    ROOT::Fit::SparseData d(h1);
//    ROOT::Fit::FillData(d, h1, 0);
//    ROOT::Fit::BinData* bd = d.GetBinData();

//    cout << "Filling second histogram" << endl;
//    for ( unsigned int i = 0; i < bd->NPoints(); ++i)
//    {
//       const double* x;
//       double value, error;
//       x = bd->GetPoint(i, value, error);
//       value = (value)?value:-10;
//       h2->Fill(x[0], value);
//    }

//    TCanvas* c = new TCanvas("Histogram 2D");
//    c->Divide(1,2);
//    c->cd(1);
//    h1->Draw("lego2Z");
//    c->cd(2);
//    h2->Draw("lego2Z");

//    // Fit preparation
//    bool ret;
//    ROOT::Fit::Fitter fitter;
//    ROOT::Math::WrappedMultiTF1 wf1(*f1);
//    fitter.Config().SetMinimizer("TMinuit");

//    cout << "\n ******* Chi2Fit with Original BinData *******" << endl;
//    ROOT::Fit::BinData bdOriginal;
//    ROOT::Fit::FillData(bdOriginal, h1, 0);
//    ret = fitter.Fit(bdOriginal, wf1);
//    fitter.Result().Print(std::cout); 
//    if (!ret)  
//       std::cout << "Fit Failed " << std::endl;

//    cout << "\n ******* Chi2Fit with BinData and NoCeros *******" << endl;
//    ROOT::Fit::BinData* bdNoCeros = d.GetBinDataNoCeros();

//    cout << "bdOriginal:\n" << bdOriginal << endl;
//    cout << "bdNoCeros:\n" << *bdNoCeros << endl;

//    cout << "Equals: " << (bdOriginal == *bdNoCeros) << endl;
   
//    ret = fitter.Fit(*bdNoCeros, wf1);
//    fitter.Result().Print(std::cout); 
//    if (!ret)  
//       std::cout << "Fit Failed " << std::endl;


//    delete bd;
// }

   int main(int argc, char** argv)
{
   TApplication* theApp = 0;

   if ( __DRAW__ )
      theApp = new TApplication("App",&argc,argv);
   
   fit3DHist();
//    fit2DHist();
//    fit1DHist();
  
   if ( __DRAW__ ) {
      theApp->Run();
      delete theApp;
      theApp = 0;
   }
   
   return 0;
}

