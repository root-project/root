#include "TH1.h"
#include "TH2.h"
#include "TF1.h"
#include "TF2.h"
#include "TGraphErrors.h"
#include "TGraphAsymmErrors.h"
#include "TGraph2D.h"
#include "TSystem.h"
#include "TRandom3.h"
#include "TROOT.h"
#include "TVirtualFitter.h"

#include "Fit/BinData.h"
#include "Fit/UnBinData.h"
#include "THFitInterface.h"
#include "TGraphFitInterface.h"
#include "Fit/Fitter.h"

#include "Math/WrappedMultiTF1.h"
#include "Math/WrappedParamFunction.h"
#include "Math/WrappedTF1.h"
//#include "Math/Polynomial.h"

#include <string>
#include <iostream>

// print the data
void printData(const ROOT::Fit::BinData & data) {
   std::cout << "Bin data, point  size is " << data.PointSize() << " data dimension is " << data.NDim() << " type is " << data.GetErrorType() << std::endl;
   for (unsigned int i = 0; i < data.Size(); ++i) { 
      if (data.GetErrorType() == ROOT::Fit::BinData::kNoError) 
         std::cout << data.Coords(i)[0] << "   " << data.Value(i) << std::endl; 
      else if (data.GetErrorType() == ROOT::Fit::BinData::kValueError) 
         std::cout << data.Coords(i)[0] << "   " << data.Value(i) << " +/-  " << data.Error(i) << std::endl; 
      else if (data.GetErrorType() == ROOT::Fit::BinData::kCoordError) {
         double ey = 0; 
         const double * ex = data.GetPointError(i,ey); 
         std::cout << data.Coords(i)[0] <<  " +/-  " << ex[0] <<  "   " << data.Value(i) << " +/-  " << ey << std::endl; 
      }
      else if (data.GetErrorType() == ROOT::Fit::BinData::kAsymError) { 
         double eyl = 0; double eyh = 0;  
         const double * ex = data.GetPointError(i,eyl,eyh); 
         std::cout << data.Coords(i)[0] <<  " +/-  " << ex[0] <<  "   " << data.Value(i) << " -  " << eyl << " + " << eyh << std::endl; 
      }

   }
   std::cout << "\ndata size is " << data.Size() << std::endl;
}    
void printData(const ROOT::Fit::UnBinData & data) {
   for (unsigned int i = 0; i < data.Size(); ++i) { 
      std::cout << data.Coords(i)[0] << "\t"; 
   }
   std::cout << "\ndata size is " << data.Size() << std::endl;
}    

int testHisto1DFit() { 


   std::string fname("gaus");
   TF1 * func = (TF1*)gROOT->GetFunction(fname.c_str());
   func->SetParameter(0,10);
   func->SetParameter(1,0);
   func->SetParameter(2,3.0);

   TRandom3 rndm;

   // fill an histogram 
   TH1D * h1 = new TH1D("h1","h1",30,-5.,5.);
//      h1->FillRandom(fname.c_str(),100);
   for (int i = 0; i <1000; ++i) 
      h1->Fill( rndm.Gaus(0,1) );

   h1->Print();
   //h1->Draw();

//    gSystem->Load("libMinuit2");
//    gSystem->Load("libFit");


   // ROOT::Fit::DataVector<ROOT::Fit::BinPoint> dv; 
   
   ROOT::Fit::BinData d; 
   ROOT::Fit::FillData(d,h1,func);


   printData(d);

   // create the function
   ROOT::Math::WrappedMultiTF1 f(*func); 
   double p[3] = {100,0,3.}; 
   f.SetParameters(p); 

   // create the fitter 

   ROOT::Fit::Fitter fitter; 

   bool ret = fitter.Fit(d, f);
   if (ret)  
      fitter.Result().Print(std::cout); 
   else {
      std::cout << "Chi2 Fit Failed " << std::endl;
      return -1; 
   }
   // compare with TH1::Fit
   TVirtualFitter::SetDefaultFitter("Minuit2"); 
   std::cout << "\n******************************\n\t TH1::Fit Result \n" << std::endl; 
   func->SetParameters(p);   
   h1->Fit(func);

   // test using binned likelihood 
   std::cout << "\n\nTest Binned Likelihood Fit" << std::endl; 

   ret = fitter.LikelihoodFit(d, f);
   f.SetParameters(p); 
   if (ret)  
      fitter.Result().Print(std::cout); 
   else {
      std::cout << "Binned Likelihood Fit Failed " << std::endl;
      return -1; 
   }
   // compare with TH1::Fit
   std::cout << "\n******************************\n\t TH1::Fit Result \n" << std::endl; 
   func->SetParameters(p);   
   h1->Fit(func,"L");
   std::cout << "Equivalent Chi2 from TF1::Fit " << func->GetChisquare() << std::endl;

   std::cout << "\n\nTest Chi2 Fit using integral option" << std::endl; 

   // need to re-create data
   ROOT::Fit::DataOptions opt; 
   opt.fIntegral = true; 
   ROOT::Fit::BinData d2(opt); 
   ROOT::Fit::FillData(d2,h1,func); 

   f.SetParameters(p); 
   ret = fitter.Fit(d2, f);
   if (ret)  
      fitter.Result().Print(std::cout); 
   else {
      std::cout << "Integral Chi2 Fit Failed " << std::endl;
      return -1; 
   }
   // compare with TH1::Fit
   std::cout << "\n******************************\n\t TH1::Fit Result \n" << std::endl; 
   func->SetParameters(p);   
   h1->Fit(func,"I");

   f.SetParameters(p); 
   ret = fitter.LikelihoodFit(d2, f);
   if (ret)  
      fitter.Result().Print(std::cout); 
   else {
      std::cout << "Integral Likelihood Fit Failed " << std::endl;
      return -1; 
   }
   // compare with TH1::Fit
   std::cout << "\n******************************\n\t TH1::Fit Result \n" << std::endl; 
   func->SetParameters(p);   
   h1->Fit(func,"IL");
   std::cout << "Equivalent Chi2 from TF1::Fit " << func->GetChisquare() << std::endl;


   

   
   // redo chi2fit
   std::cout << "\n\nRedo Chi2 Hist Fit" << std::endl; 
   f.SetParameters(p);   
   ret = fitter.Fit(d, f);
   if (ret)  
      fitter.Result().Print(std::cout); 
   else {
      std::cout << "Chi2 Fit Failed " << std::endl;
      return -1; 
   }




   // test grapherrors fit 
   std::cout << "\n\nTest Same Fit from a TGraphErrors - no coord errors" << std::endl; 
   TGraphErrors gr(h1); 
   ROOT::Fit::BinData dg; 
   ROOT::Fit::FillData(dg,&gr);

   f.SetParameters(p);   
   ret = fitter.Fit(dg, f);
   if (ret)  
      fitter.Result().Print(std::cout); 
   else {
      std::cout << "Chi2 Graph Errors Fit Failed " << std::endl;
      return -1; 
   }

   // fit using error on X
   std::cout << "\n\nTest Same Fit from a TGraphErrors - use coord errors" << std::endl; 
   ROOT::Fit::BinData dger; 
   dger.Opt().fCoordErrors = true;  // use coordinate errors
   ROOT::Fit::FillData(dger,&gr);

   f.SetParameters(p);   
   ret = fitter.Fit(dger, f);
   if (ret)  
      fitter.Result().Print(std::cout); 
   else {
      std::cout << "Chi2 Graph Errors Fit Failed " << std::endl;
      return -1; 
   }

   // compare with TGraphErrors::Fit
   std::cout << "\n******************************\n\t TGraphErrors::Fit Result \n" << std::endl; 
   func->SetParameters(p);   
   gr.Fit(func); 
   std::cout << "Ndf of TGraphErrors::Fit  = " << func->GetNDF() << std::endl;


   // test graph fit (errors are 1) do a re-normalization
   std::cout << "\n\nTest Same Fit from a TGraph" << std::endl; 
   fitter.Config().SetNormErrors(true);
   TGraph gr2(h1); 
   ROOT::Fit::BinData dg2; 
   ROOT::Fit::FillData(dg2,&gr2);

   f.SetParameters(p);   
   ret = fitter.Fit(dg2, f);
   if (ret)  
      fitter.Result().Print(std::cout); 
   else {
      std::cout << "Chi2 Graph Fit Failed " << std::endl;
      return -1; 
   }


   // compare with TGraph::Fit (no errors)
   std::cout << "\n******************************\n\t TGraph::Fit Result \n" << std::endl; 
   func->SetParameters(p);   
   gr2.Fit(func); 
   std::cout << "Ndf of TGraph::Fit = " << func->GetNDF() << std::endl;


   // reddo chi2fit using Fumili
   std::cout << "\n\nRedo Chi2 Hist Fit using FUMILI" << std::endl; 
   f.SetParameters(p);   
   fitter.Config().SetMinimizer("Minuit2","Fumili");
   ret = fitter.Fit(d, f);
   if (ret)  
      fitter.Result().Print(std::cout); 
   else {
      std::cout << "Chi2 Fit Failed " << std::endl;
      return -1; 
   }


   return 0;
}


class Func1D : public ROOT::Math::IParamFunction { 
public:
   void SetParameters(const double *p) { std::copy(p,p+NPar(),fp);}
   const double * Parameters() const { return fp; }
   ROOT::Math::IGenFunction * Clone() const { 
      Func1D * f =  new Func1D(); 
      f->SetParameters(fp);
      return f;
   };
   unsigned int NPar() const { return 3; }
private:
   double DoEval( double x) const { 
      return fp[0]*x*x + fp[1]*x + fp[2]; 
   }
   double fp[3];
   
};

// gradient 2D function
class GradFunc2D : public ROOT::Math::IParamMultiGradFunction { 
public:
   void SetParameters(const double *p) { std::copy(p,p+NPar(),fp);}
   const double * Parameters() const { return fp; }
   ROOT::Math::IMultiGenFunction * Clone() const { 
      GradFunc2D * f =  new GradFunc2D(); 
      f->SetParameters(fp);
      return f;
   };
   unsigned int NDim() const { return 2; }
   unsigned int NPar() const { return 5; }

   void ParameterGradient( const double * x, double * grad) const { 
      grad[0] = x[0]*x[0]; 
      grad[1] = x[0];
      grad[2] = x[1]*x[1]; 
      grad[3] = x[1];
      grad[4] = 1; 
   }

private:
   double DoEval( const double *x) const { 
      return fp[0]*x[0]*x[0] + fp[1]*x[0] + fp[2]*x[1]*x[1] + fp[3]*x[1] + fp[4]; 
   }
   double DoDerivative(const double *x, unsigned int icoord = 0) const { 
      assert(icoord <= 1); 
      if (icoord == 0) 
         return 2. * fp[0] * x[0] + fp[1];
      else 
         return 2. * fp[2] * x[1] + fp[3];
   }
   double DoParameterDerivative(const double * x, unsigned int ipar) const { 
      std::vector<double> grad(NPar());
      ParameterGradient(x, &grad[0] ); 
      return grad[ipar]; 
   }

   double fp[5];
   
};

int testHisto1DPolFit() { 



   std::string fname("pol2");
   TF1 * func = (TF1*)gROOT->GetFunction(fname.c_str());
   func->SetParameter(0,1.);
   func->SetParameter(1,2.);
   func->SetParameter(2,3.0);

   TRandom3 rndm;

   // fill an histogram 
   TH1D * h2 = new TH1D("h2","h2",30,-5.,5.);
//      h1->FillRandom(fname.c_str(),100);
   for (int i = 0; i <1000; ++i) 
      h2->Fill( func->GetRandom() );

   // fill fit data
   ROOT::Fit::BinData d; 
   ROOT::Fit::FillData(d,h2,func);


   printData(d);

   // create the function
   Func1D f; 

   double p[3] = {100,0,3.}; 
   f.SetParameters(p); 


   // create the fitter 
   //std::cout << "Fit parameter 2  " << f.Parameters()[2] << std::endl;
   std::cout << "\n\nTest histo polynomial fit (Minuit2)" << std::endl; 

   ROOT::Fit::Fitter fitter; 
   bool ret = fitter.Fit(d, f);
   if (ret)  
      fitter.Result().Print(std::cout); 
   else {
      std::cout << " Fit Failed " << std::endl;
      return -1; 
   }

   // compare with TH1::Fit
   std::cout << "\n******************************\n\t TH1::Fit(pol2) Result with TMinuit \n" << std::endl; 
   func->SetParameters(p);   
   h2->Fit(func,"F"); 

   std::cout << "\n\nTest histo polynomial linear fit " << std::endl; 

   ROOT::Math::WrappedTF1 pf(*func); 
   //ROOT::Math::Polynomial pf(2); 
   pf.SetParameters(p);

   fitter.Config().SetMinimizer("Linear");
   ret = fitter.Fit(d, pf);
   if (ret)  
      fitter.Result().Print(std::cout); 
   else {
      std::cout << " Fit Failed " << std::endl;
      return -1; 
   }

   // compare with TH1::Fit
   std::cout << "\n******************************\n\t TH1::Fit(pol2) Result with TLinearFitter \n" << std::endl; 
   func->SetParameters(p);   
   h2->Fit(func); 

   

   return 0; 
}

int testHisto2DFit() { 

   // fit using a 2d parabola (test also gradient)


   std::string fname("pol2");
   TF2 * func = new TF2("f2d",ROOT::Math::ParamFunctor(GradFunc2D() ), -5.,5.,-5,5,5);
   double p0[5] = { 1.,2.,0.5,1.,3. }; 
   func->SetParameters(p0);
   assert(func->GetNpar() == 5); 

   TRandom3 rndm;

   // fill an histogram 
   TH2D * h2 = new TH2D("h2d","h2d",30,-5.,5.,30,-5.,5.);
//      h1->FillRandom(fname.c_str(),100);
   for (int i = 0; i <1000; ++i) {
      double x,y = 0;
      func->GetRandom2(x,y);
      h2->Fill(x,y);
   }
   // fill fit data
   ROOT::Fit::BinData d; 
   ROOT::Fit::FillData(d,h2,func);


   //printData(d);

   // create the function
   GradFunc2D f; 

   double p[5] = { 2.,1.,1,2.,100. }; 
   f.SetParameters(p); 


   // create the fitter 

   ROOT::Fit::Fitter fitter; 
   bool ret = fitter.Fit(d, f);
   if (ret)  
      fitter.Result().Print(std::cout); 
   else {
      std::cout << "Gradient Fit Failed " << std::endl;
      return -1; 
   }

   // test without gradient
   std::cout <<"\ntest result without using gradient" << std::endl;
   ROOT::Math::WrappedParamFunction<GradFunc2D *> f2(&f,2,5,p);
   ret = fitter.Fit(d, f2);
   if (ret)  
      fitter.Result().Print(std::cout); 
   else {
      std::cout << " Chi2 Fit Failed " << std::endl;
      return -1; 
   }

   // test binned likelihood gradient
   std::cout <<"\ntest result using gradient and binned likelihood" << std::endl;
   f.SetParameters(p); 
   ret = fitter.LikelihoodFit(d, f);
   if (ret)  
      fitter.Result().Print(std::cout); 
   else {
      std::cout << "Gradient Bin Likelihood  Fit Failed " << std::endl;
      return -1; 
   }

   // test with linear fitter 
   std::cout <<"\ntest result using linear fitter" << std::endl;
   fitter.Config().SetMinimizer("Linear");
   f.SetParameters(p); 
   ret = fitter.Fit(d, f);
   if (ret)  
      fitter.Result().Print(std::cout); 
   else {
      std::cout << "Linear 2D Fit Failed " << std::endl;
      return -1; 
   }

   // test fitting using TGraph2D
   TGraph2D g2(h2);

   std::cout <<"\ntest using TGraph2D" << std::endl;
   ROOT::Fit::BinData d2; 
   ROOT::Fit::FillData(d2,&g2,func);
   //g2.Dump();
   std::cout << "data size from graph " << d2.Size() <<  std::endl; 

   f2.SetParameters(p); 
   fitter.Config().SetMinimizer("Minuit2");
   ret = fitter.Fit(d2, f2);
   if (ret)  
      fitter.Result().Print(std::cout); 
   else {
      std::cout << " TGraph2D Fit Failed " << std::endl;
      return -1; 
   }
   // compare with TGraph2D::Fit
   std::cout << "\n******************************\n\t TGraph::Fit Result \n" << std::endl; 
   func->SetParameters(p);   
   g2.Fit(func);

   std::cout <<"\ntest using TGraph2D and gradient function" << std::endl;
   f.SetParameters(p); 
   ret = fitter.Fit(d2, f);
   if (ret)  
      fitter.Result().Print(std::cout); 
   else {
      std::cout << " TGraph2D Grad Fit Failed " << std::endl;
      return -1; 
   }



   return 0; 
}


int testUnBin1DFit() { 

   int iret = 0;

   std::string fname("gausn");
   TF1 * func = (TF1*)gROOT->GetFunction(fname.c_str());

   TRandom3 rndm;

   int n = 100;
   ROOT::Fit::UnBinData d(n); 

   for (int i = 0; i <n; ++i) 
      d.Add( rndm.Gaus(0,1) );
  

   // printData(d);

   // create the function
   ROOT::Math::WrappedMultiTF1 f(*func); 
   double p[3] = {1,2,10.}; 
   f.SetParameters(p); 

   // create the fitter 
   //std::cout << "Fit parameters  " << f.Parameters()[2] << std::endl;

   ROOT::Fit::Fitter fitter; 
   fitter.SetFunction(f);
   std::cout << "fix parameter 0 " << " to value " << f.Parameters()[0] << std::endl;
   fitter.Config().ParSettings(0).Fix();
   // set range in sigma sigma > 0
   std::cout << "set lower range to 0 for sigma " << std::endl;
   fitter.Config().ParSettings(2).SetLowerLimit(0);

#ifdef DEBUG
   fitter.Config().MinimizerOptions().SetPrintLevel(3);
#endif

//    double x[1]; x[0] = 0.; 
//    std::cout << "fval " << f(x) << std::endl;
//    x[0] = 1.; std::cout << "fval " << f(x) << std::endl;

   bool ret = fitter.Fit(d);
   if (ret)  
      fitter.Result().Print(std::cout); 
   else {
      std::cout << "Unbinned Likelihood Fit Failed " << std::endl; 
      iret |= 1;
   }

   std::cout << "\n\nRedo Fit using FUMILI" << std::endl; 
   f.SetParameters(p);   
   fitter.Config().SetMinimizer("Fumili2");
   // need to set function first (need to change this)
   fitter.SetFunction(f);
   fitter.Config().ParSettings(0).Fix(); //need to re-do it
   // set range in sigma sigma > 0
   fitter.Config().ParSettings(2).SetLowerLimit(0);

   ret = fitter.Fit(d);
   if (ret)  
      fitter.Result().Print(std::cout); 
   else {
      std::cout << "Unbinned Likelihood Fit using FUMILI Failed " << std::endl;      
      iret |= 1;
   }


   return iret; 
}

int testGraphFit() { 
   
   int iret = 0; 

   // simple test of fitting a Tgraph 

   double x[5] = {1,2,3,4,5}; 
   double y[5] = {2.1, 3.5, 6.5, 8.8, 9.5};
   double ex[5] = {.3,.3,.3,.3,.3};
   double ey[5] = {.5,.5,.5,.5,.5};
   double eyl[5] = {.2,.2,.2,.2,.2};
   double eyh[5] = {.8,.8,.8,.8,.8};

   std::cout << "\n********************************************************\n";
   std::cout << "Test simple fit of Tgraph of 5 points" << std::endl;
   std::cout << "\n********************************************************\n";


   double p[2] = {1,1}; 
   TF1 * func = new TF1("f","pol1",0,10);
   func->SetParameters(p);

   ROOT::Math::WrappedMultiTF1 f(*func); 
   f.SetParameters(p); 

   ROOT::Fit::Fitter fitter; 
   fitter.SetFunction(f);

      
   std::cout <<"\ntest TGraph (no errors) " << std::endl;
   TGraph gr(5, x,y);  

   ROOT::Fit::BinData dgr; 
   ROOT::Fit::FillData(dgr,&gr);

   //printData(dgr);

   f.SetParameters(p);   
   bool ret = fitter.Fit(dgr, f);
   if (ret)  
      fitter.Result().Print(std::cout); 
   else {
      std::cout << "Chi2 Graph Fit Failed " << std::endl;
      return -1; 
   }

   // compare with TGraph::Fit
   std::cout << "\n******************************\n\t TGraph::Fit Result \n" << std::endl; 
   func->SetParameters(p);   
   gr.Fit(func,"F"); // use Minuit  


   std::cout <<"\ntest TGraphErrors  " << std::endl;
   TGraphErrors grer(5, x,y,ex,ey);  

   ROOT::Fit::BinData dgrer; 
   dgrer.Opt().fCoordErrors = true; 
   ROOT::Fit::FillData(dgrer,&grer);
   
   //printData(dgrer);

   f.SetParameters(p);   
   ret = fitter.Fit(dgrer, f);
   if (ret)  
      fitter.Result().Print(std::cout); 
   else {
      std::cout << "Chi2 Graph Fit Failed " << std::endl;
      return -1; 
   }

   // compare with TGraph::Fit
   std::cout << "\n******************************\n\t TGraphErrors::Fit Result \n" << std::endl; 
   func->SetParameters(p);   
   grer.Fit(func,"F"); // use Minuit  

   std::cout <<"\ntest TGraphAsymmErrors  " << std::endl;
   TGraphAsymmErrors graer(5, x,y,ex,ex,eyl, eyh);  

   ROOT::Fit::BinData dgraer; 
   // option error on coordinate and asymmetric on values
   dgraer.Opt().fCoordErrors = true; 
   dgraer.Opt().fAsymErrors = true; 
   ROOT::Fit::FillData(dgraer,&graer);
   //printData(dgraer);

   f.SetParameters(p);   
   ret = fitter.Fit(dgraer, f);
   if (ret)  
      fitter.Result().Print(std::cout); 
   else {
      std::cout << "Chi2 Graph Fit Failed " << std::endl;
      return -1; 
   }

   // compare with TGraph::Fit
   std::cout << "\n******************************\n\t TGraphAsymmErrors::Fit Result \n" << std::endl; 
   func->SetParameters(p);   
   graer.Fit(func,"F"); // use Minuit  



   return iret; 
}


template<typename Test> 
int testFit(Test t, std::string name) { 
   std::cout << name << "\n\t\t";  
   int iret = t();
   std::cout << "\n" << name << ":\t\t";  
   if (iret == 0) 
      std::cout << "OK" << std::endl;  
   else 
      std::cout << "Failed" << std::endl;  
   return iret; 
}

int main() { 

   int iret = 0; 
   iret |= testFit( testHisto1DFit, "Histogram1D Fit");
   iret |= testFit( testHisto1DPolFit, "Histogram1D Polynomial Fit");
   iret |= testFit( testHisto2DFit, "Histogram2D Gradient Fit");
   iret |= testFit( testUnBin1DFit, "Unbin 1D Fit");
   iret |= testFit( testGraphFit, "Graph 1D Fit");
   return iret; 
}
   
