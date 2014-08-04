#include "TH1.h"
#include "TF1.h"
#include "TF2.h"
#include "TMath.h"
#include "TSystem.h"
#include "TRandom3.h"
#include "TTree.h"
#include "TROOT.h"

#include "Fit/BinData.h"
#include "Fit/UnBinData.h"
//#include "Fit/BinPoint.h"
#include "Fit/Fitter.h"
#include "HFitInterface.h"

#include "Math/IParamFunction.h"
#include "Math/WrappedTF1.h"
#include "Math/WrappedMultiTF1.h"
#include "Math/WrappedParamFunction.h"
#include "Math/MultiDimParamFunctionAdapter.h"

#include "TGraphErrors.h"

#include "TStyle.h"

#include "TSeqCollection.h"

#include "Math/Polynomial.h"
#include "Math/DistFunc.h"

#include <string>
#include <iostream>

#include "TStopwatch.h"

#include "TVirtualFitter.h"
// #include "TFitterFumili.h"
// #include "TFumili.h"

#include "GaussFunction.h"

// #include "RooDataHist.h"
// #include "RooDataSet.h"
// #include "RooRealVar.h"
// #include "RooGaussian.h"
// #include "RooMinuit.h"
// #include "RooChi2Var.h"
// #include "RooGlobalFunc.h"
// #include "RooFitResult.h"
// #include "RooProdPdf.h"

#include <cassert>

#include "MinimizerTypes.h"

#ifdef USE_VC
#include "Vc/Vc"
//#include "Vc/Allocator"
//Vc_DECLARE_ALLOCATOR(Vc::double_v)
#endif

//#define USE_AOS

#ifdef USE_VDT
#include "vdtMath.h"
#endif

//#define DEBUG

int nfit;
const int N = 20;
double iniPar[2*N];
int ndimObs;
int ndimPars;

void printData(const ROOT::Fit::UnBinData & data) {
   for (unsigned int i = 0; i < data.Size(); ++i) {
      std::cout << data.Coords(i)[0] << "\t";
   }
   std::cout << "\ndata size is " << data.Size() << std::endl;
}

void printResult(int iret) {
   std::cout << "\n************************************************************\n";
   std::cout << "Test\t\t\t\t";
   if (iret == 0) std::cout << "OK";
   else std::cout << "FAILED";
   std::cout << "\n************************************************************\n";
}

bool USE_BRANCH = false;
ROOT::Fit::UnBinData * FillUnBinData(TTree * tree, bool copyData = true, unsigned int dim = 1 ) {

   // fill the unbin data set from a TTree
   ROOT::Fit::UnBinData * d = 0;
   // for the large tree
   if (std::string(tree->GetName()) == "t2") {
      d = new ROOT::Fit::UnBinData();
      // large tree
      unsigned int n = tree->GetEntries();
#ifdef DEBUG
      std::cout << "number of unbin data is " << n << " of dim " << N << std::endl;
#endif
      d->Initialize(n,N);
      TBranch * bx = tree->GetBranch("x");
      double vx[N];
      bx->SetAddress(vx);
      std::vector<double>  m(N);
      for (int unsigned i = 0; i < n; ++i) {
         bx->GetEntry(i);
         d->Add(vx);
         for (int j = 0; j < N; ++j)
            m[j] += vx[j];
      }

#ifdef DEBUG
      std::cout << "average values of means :\n";
      for (int j = 0; j < N; ++j)
         std::cout << m[j]/n << "  ";
      std::cout << "\n";
#endif

      return d;
   }
   if (USE_BRANCH)
   {
      d = new ROOT::Fit::UnBinData();
      unsigned int n = tree->GetEntries();
      //std::cout << "number of unbin data is " << n << std::endl;

      if (dim == 2) {
         d->Initialize(n,2);
         TBranch * bx = tree->GetBranch("x");
         TBranch * by = tree->GetBranch("y");
         double v[2];
         bx->SetAddress(&v[0]);
         by->SetAddress(&v[1]);
         for (int unsigned i = 0; i < n; ++i) {
            bx->GetEntry(i);
            by->GetEntry(i);
            d->Add(v);
         }
      }
      else if (dim == 1) {
         d->Initialize(n,1);
         TBranch * bx = tree->GetBranch("x");
         double v[1];
         bx->SetAddress(&v[0]);
         for (int unsigned i = 0; i < n; ++i) {
            bx->GetEntry(i);
            d->Add(v);
         }
      }

      return d;

      //printData(d);
   }
   else {
      tree->SetEstimate(tree->GetEntries());

      // use TTREE::Draw
      if (dim == 2) {
         tree->Draw("x:y",0,"goff");  // goff is used to turn off the graphics
         double * x = tree->GetV1();
         double * y = tree->GetV2();

         if (x == 0 || y == 0) {
            USE_BRANCH= true;
            return FillUnBinData(tree, true, dim);
         }

         // use array pre-allocated in tree->Draw . This is faster
         //assert(x != 0);
         unsigned int n = tree->GetSelectedRows();

         if (copyData) {
            d = new ROOT::Fit::UnBinData(n,2);
            double vx[2];
            for (int unsigned i = 0; i < n; ++i) {
               vx[0] = x[i];
               vx[1] = y[i];
               d->Add(vx);
            }
         }
         else  // use data pointers directly
            d = new ROOT::Fit::UnBinData(n,x,y);

      }
      else if ( dim == 1) {

            tree->Draw("x",0,"goff");  // goff is used to turn off the graphics
            double * x = tree->GetV1();

            if (x == 0) {
               USE_BRANCH= true;
               return FillUnBinData(tree, true, dim);
            }
            unsigned int n = tree->GetSelectedRows();

            if (copyData) {
               d = new ROOT::Fit::UnBinData(n,1);
               for (int unsigned i = 0; i < n; ++i) {
                  d->Add(x[i]);
               }
            }
            else
               d = new ROOT::Fit::UnBinData(n,x);
         }
      return d;
   }

   //std::copy(x,x+n, d.begin() );
   return 0;
}




// print the data
template <class T>
void printData(const T & data) {
   for (typename T::const_iterator itr = data.begin(); itr != data.end(); ++itr) {
      std::cout << itr->Coords()[0] << "   " << itr->Value() << "   " << itr->Error() << std::endl;
   }
   std::cout << "\ndata size is " << data.Size() << std::endl;
}


// new likelihood function for unbinned data
using namespace ROOT::Fit;

template <class F>
struct VecNLL {

   VecNLL( ROOT::Fit::UnBinData & data, F & f) :
      fData(&data), fFunc(f)
   { } //GetData(); }

   // void GetData()  {

   //    const  UnBinData & data = *fData;
   //    //const ROOT::Math::IParamMultiFunction & func = *fFunc; //fNLL.ModelFunction();

   //    unsigned int n = data.Size();

   //    fX = std::vector<T>(n);
   //    fY = std::vector<T>(n);
   //    for (int i = 0; i<n; i++) {
   //       fX[i] = *(data.Coords(i));
   //       fY[i] = *(data.Coords(i)+1);
   //    }

   // }


   // re-implement likelihood evaluation
   double operator() (const double * p) const {

      const  UnBinData & data = *fData;
      //const ROOT::Math::IParamMultiFunction & func = *fFunc; //fNLL.ModelFunction();


      unsigned int n = data.Size();

#ifdef DEBUG
      std::cout << "\n\nFit data size = " << n << std::endl;
      std::cout << "func pointer is " << typeid(func).name() << std::endl;
#endif

      //unsigned int nRejected = 0;


#ifndef USE_VC
#if !defined(USE_VDT) && !defined(USE_AOS)
      double logl = 0;
      for (unsigned int i = 0; i < n; i++) {

         const double * x = data.Coords(i);

         double fval = fFunc(x,p);
         //double logval =  ROOT::Math::Util::EvalLog( fval);
         double logval =  std::log( fval);

         logl += logval;

         // if (i < 4) {
         //    std::cout << x[0] << std::endl;
         //    std::cout << logval << std::endl;
         //    std::cout << logl << std::endl << std::endl;
         // }

      }

      //    std::cout << "Scal:  par = " << p[0] << "  " << p[1] << " log l = " << logl << std::endl;
      return - logl;
#else
      // VDT implementation
      double logl = 0;
      std::vector<double> fval(n);
      std::vector<double> logval(n);

#ifndef USE_AOS
      for (unsigned int i = 0; i < n; i++) {

         const double * x = data.Coords(i);

         fval[i] = fFunc(x,p);
      }
#else
      //std::vector<double> x(n * data.NDim() );
      const double * x = data.Coords(0);

      fFunc(n, data.NDim(), x, p, &fval[0] );

#endif

      // for (unsigned int i = 0; i < n; i++) {


         //double logval =  ROOT::Math::Util::EvalLog( fval);
#ifdef USE_VDT
      vdt::fast_logv(n, &fval[0], &logval[0]);
#endif

      for (unsigned int i = 0; i < n; i++) {

#ifndef USE_VDT
         logval[i] = std::log( fval[i]);
#endif

         logl += logval[i];
      }


         // if (i < 4) {
         //    std::cout << x[0] << std::endl;
         //    std::cout << logval << std::endl;
         //    std::cout << logl << std::endl << std::endl;
         // }


      //    std::cout << "Scal:  par = " << p[0] << "  " << p[1] << " log l = " << logl << std::endl;
      return - logl;

#endif
#else
      // VC implementation

      Vc::double_v logl = 0.0;

      std::vector <Vc::double_v> x(ndimObs);

      for (unsigned int i = 0; i < n; i +=Vc::double_v::Size) {
         for (unsigned int j = 0; j < data.NDim(); ++j) {
            for (int k = 0; k< Vc::double_v::Size; ++k) {
               int ipoint = i + k;
               x[j][k] = *(data.Coords(ipoint)+j);
            }
         }

         Vc::double_v logval = std::log( fFunc ( &x[0], p ) );
         logl += logval;

         // if (i == 0) {
         //    std::cout << x[0] << std::endl;
         //    std::cout << logval << std::endl;
         //    std::cout << logl << std::endl << std::endl;;
         // }
      }
      double ret = 0;
      for (int k = 0; k< Vc::double_v::Size; ++k) {
         ret += logl[k];
      }

//      std::cout << "Vc:: par = " << p[0] << "  " << p[1] << " log l = " << ret << std::endl;

      return -ret;
#endif

   }

   // members

   ROOT::Fit::UnBinData * fData;
   F fFunc;

};



// fitting using new fitter
typedef ROOT::Math::IParamMultiFunction Func;
template <class MinType, class T>
int DoBinFit(T * hist, Func & func, bool debug = false, bool useGrad = false) {

   //std::cout << "Fit histogram " << std::endl;

   ROOT::Fit::BinData d;
   ROOT::Fit::FillData(d,hist);

   //printData(d);

   // create the fitter

   ROOT::Fit::Fitter fitter;
   fitter.Config().SetMinimizer(MinType::name().c_str(),MinType::name2().c_str());

   if (debug)
      fitter.Config().MinimizerOptions().SetPrintLevel(3);


   // create the function
   if (!useGrad) {

      // use simply TF1 wrapper
      //ROOT::Math::WrappedMultiTF1 f(*func);
      //ROOT::Math::WrappedTF1 f(*func);
      fitter.SetFunction(func);

   } else { // only for gaus fits
      // use function gradient
#ifdef USE_MATHMORE_FUNC
   // use mathmore for polynomial
      ROOT::Math::Polynomial pol(2);
      assert(pol.NPar() == func->GetNpar());
      pol.SetParameters(func->GetParameters() );
      ROOT::Math::WrappedParamFunction<ROOT::Math::Polynomial> f(pol,1,func->GetParameters(),func->GetParameters()+func->GetNpar() );
#endif
      GaussFunction f;
      f.SetParameters(func.Parameters());
      fitter.SetFunction(f);
   }


   bool ret = fitter.Fit(d);
   if (!ret) {
      std::cout << " Fit Failed " << std::endl;
      return -1;
   }
   if (debug)
      fitter.Result().Print(std::cout);
   return 0;
}

// unbin fit
template <class MinType, class T>
int DoUnBinFit(T * tree, Func & func, bool debug = false, bool copyData = false ) {

   ROOT::Fit::UnBinData * d  = FillUnBinData(tree, copyData, func.NDim() );
   // need to have done Tree->Draw() before fit
   //FillUnBinData(d,tree);

   //std::cout << "data size type and size  is " << typeid(*d).name() <<  "   " << d->Size() << std::endl;
   if (debug) {
   if (copyData)
      std::cout << "\tcopy data in FitData\n";
   else
      std::cout << "\tre-use original data \n";
   }


   //printData(d);

   // create the fitter
   //std::cout << "Fit parameter 2  " << f.Parameters()[2] << std::endl;

   ROOT::Fit::Fitter fitter;
   fitter.Config().SetMinimizer(MinType::name().c_str(),MinType::name2().c_str());

   if (debug)
      fitter.Config().MinimizerOptions().SetPrintLevel(3);

   // set tolerance 1 for tree to be same as in TTTreePlayer::UnBinFIt
   fitter.Config().MinimizerOptions().SetTolerance(1);


   // create the function

   // need to fix param 0 , normalization in the unbinned fits
   //fitter.Config().ParSettings(0).Fix();

   fitter.SetFunction(func);
   bool ret = fitter.Fit(*d);

   if (!ret) {
      std::cout << " Fit Failed " << std::endl;
      return -1;
   }
   if (debug)
      fitter.Result().Print(std::cout);

   delete d;

   return 0;

}


template <class MinType, class T, class VFunc>
int DoUnBinFitVec(T * tree, VFunc & func, int ndim, int npar, const double * p0, bool debug = false, bool copyData = false ) {

   ROOT::Fit::UnBinData * d  = FillUnBinData(tree, copyData, ndim );
   // need to have done Tree->Draw() before fit
   //FillUnBinData(d,tree);

   //std::cout << "data size type and size  is " << typeid(*d).name() <<  "   " << d->Size() << std::endl;
   if (debug) {
   if (copyData)
      std::cout << "\tcopy data in FitData\n";
   else
      std::cout << "\tre-use original data \n";
   }


   //printData(d);

   // create the fitter
   //std::cout << "Fit parameter 2  " << f.Parameters()[2] << std::endl;

   ROOT::Fit::Fitter fitter;
   fitter.Config().SetMinimizer(MinType::name().c_str(),MinType::name2().c_str());

   if (debug)
      fitter.Config().MinimizerOptions().SetPrintLevel(3);

   // set tolerance 1 for tree to be same as in TTTreePlayer::UnBinFIt
   fitter.Config().MinimizerOptions().SetTolerance(1);


   // create the function

   // need to fix param 0 , normalization in the unbinned fits
   //fitter.Config().ParSettings(0).Fix();

   VecNLL<VFunc> fcn(*d, func);
   bool ret = fitter.FitFCN(npar, fcn, p0);

   if (!ret) {
      std::cout << " Fit Failed " << std::endl;
      return -1;
   }
   if (debug)
      fitter.Result().Print(std::cout);

   std::cout << "use vec nll : nll = " << fitter.Result().MinFcnValue() << std::endl;


   delete d;

   return 0;

}


template <class MinType>
int DoFit(TTree * tree, Func & func, bool debug = false, bool copyData = false ) {
   return DoUnBinFit<MinType, TTree>(tree, func, debug, copyData);
}

template <class MinType>
int DoFit(TH1 * h1, Func & func, bool debug = false, bool copyData = false ) {
   return DoBinFit<MinType, TH1>(h1, func, debug, copyData);
}
template <class MinType>
int DoFit(TGraph * gr, Func & func, bool debug = false, bool copyData = false ) {
   return DoBinFit<MinType, TGraph>(gr, func, debug, copyData);
}

template <class MinType, class F>
int DoFitVec(TTree * tree, F & func, int n1, int n2, const double * p, bool debug = false, bool copyData = false ) {
   return DoUnBinFitVec<MinType, TTree>(tree, func, n1,n2,p,debug, copyData);
}

template <class MinType, class F>
int DoFitVec(TH1 * h1, F & func, int, int , const double *, bool debug = false, bool copyData = false ) {
   return DoBinFit<MinType, TH1>(h1, func, debug, copyData);
}
template <class MinType, class F>
int DoFitVec(TGraph * gr, F & func, int, int , const double *, bool debug = false, bool copyData = false ) {
   return DoBinFit<MinType, TGraph>(gr, func, debug, copyData);
}


template <class MinType, class FitObj, class FuncObj>
int FitUsingNewFitter(FitObj * fitobj, FuncObj  func, bool useGrad=false) {

   std::cout << "\n************************************************************\n";
   std::cout << "\tFit using new Fit::Fitter  " << typeid(*fitobj).name() << std::endl;
   std::cout << "\tMinimizer is " << MinType::name() << "  " << MinType::name2() << " func dim = " << ndimObs << std::endl;

   int iret = 0;
   TStopwatch w; w.Start();

#define USE_VECNLL
#ifndef USE_VECNLL

#ifdef DEBUG
   // std::cout << "initial Parameters " << iniPar << "  " << *iniPar << "   " <<  *(iniPar+1) << std::endl;
   func.SetParameters(iniPar);
   iret |= DoFit<MinType>(fitobj,func,true, useGrad);
   if (iret != 0) {
      std::cout << "Fit failed " << std::endl;
   }

#else
   for (int i = 0; i < nfit; ++i) {
      func.SetParameters(iniPar);
      iret = DoFit<MinType>(fitobj,func, false, useGrad);
      if (iret != 0) {
         std::cout << "Fit failed " << std::endl;
         break;
      }
   }
#endif

#else

   // use vectorized function

   for (int i = 0; i < nfit; ++i) {
      iret = DoFitVec<MinType>(fitobj,func, ndimObs, ndimPars, iniPar, false, useGrad);
      if (iret != 0) {
         std::cout << "Fit failed " << std::endl;
         break;
      }
   }

#endif

   w.Stop();
   std::cout << "\nTime: \t" << w.RealTime() << " , " << w.CpuTime() << std::endl;
   std::cout << "\n************************************************************\n";

   return iret;
}



double poly2(const double *x, const double *p) {
   return p[0] + (p[1]+p[2]*x[0] ) * x[0];
}

int testPolyFit() {

   int iret = 0;


   std::cout << "\n\n************************************************************\n";
   std::cout << "\t POLYNOMIAL FIT\n";
   std::cout << "************************************************************\n";

   std::string fname("pol2");
   //TF1 * func = (TF1*)gROOT->GetFunction(fname.c_str());
   TF1 * f1 = new TF1("pol2",fname.c_str(),-5,5.);

   f1->SetParameter(0,1);
   f1->SetParameter(1,0.0);
   f1->SetParameter(2,1.0);


   // fill an histogram
   TH1D * h1 = new TH1D("h1","h1",20,-5.,5.);
//      h1->FillRandom(fname.c_str(),100);
   for (int i = 0; i <1000; ++i)
      h1->Fill( f1->GetRandom() );

   //h1->Print();
   //h1->Draw();
   iniPar[0] = 2.; iniPar[1] = 2.; iniPar[2] = 2.;


   // dummy for testing
   //iret |= FitUsingNewFitter<DUMMY>(h1,f1);

   // use simply TF1 wrapper
   //ROOT::Math::WrappedMultiTF1 f2(*f1);
   ROOT::Math::WrappedParamFunction<> f2(&poly2,1,iniPar,iniPar+3);


   // if Minuit2 is later than TMinuit on Interl is much slower , why ??
   iret |= FitUsingNewFitter<MINUIT2>(h1,f2);
   iret |= FitUsingNewFitter<TMINUIT>(h1,f2);

   // test with linear fitter
   // for this test need to pass a multi-dim function
   ROOT::Math::WrappedTF1 wf(*f1);
   ROOT::Math::MultiDimParamGradFunctionAdapter lfunc(wf);
   iret |= FitUsingNewFitter<LINEAR>(h1,lfunc,true);

   // test with a graph

   gStyle->SetErrorX(0.); // to seto zero error on X
   TGraphErrors * gr = new TGraphErrors(h1);

   iret |= FitUsingNewFitter<MINUIT2>(gr,f2);


   std::cout << "\n-----> test now TGraphErrors with errors in X coordinates\n\n";
   // try with error in X
   gStyle->SetErrorX(0.5); // to set zero error on X
   TGraphErrors * gr2 = new TGraphErrors(h1);

   iret |= FitUsingNewFitter<MINUIT2>(gr2,f2);

   printResult(iret);

   return iret;
}

template <class T>
struct GausFunctions {
   typedef T value_type;

   static T gaussian(const T *x, const double *p) {
   //return p[0]*TMath::Gaus(x[0],p[1],p[2]);
   T tmp = (x[0]-p[1])/p[2];
   return p[0] * std::exp(-tmp*tmp/2);
}

   static T gausnorm(const T *x, const double *p) {
   //return p[0]*TMath::Gaus(x[0],p[1],p[2]);
   T invsig = 1./p[1];
   T tmp = (x[0]-p[0]) * invsig;
   const T sqrt_2pi = 1./std::sqrt(2.* 3.14159 );
   return std::exp(-0.5 * tmp*tmp ) * sqrt_2pi * invsig;
}
   static T gausnorm2D(const T *x, const double *p) {
      //return p[0]*TMath::Gaus(x[0],p[1],p[2]);
      return gausnorm(x,p)*gausnorm(x+1,p+2);
   }
   static T gausnormN(const T *x, const double *p) {
   //return p[0]*TMath::Gaus(x[0],p[1],p[2]);
      T f = 1.0;
      for (int i = 0; i < N; ++i)
         f *= gausnorm(x+i,p+2*i);

      return f;
   }

   static void gausnorm_v(unsigned int n, unsigned int stride,  const double *x, const double *p, double * res) {

      double invsig = 1./p[1];
      std::vector<double> arg(n);
      for (unsigned int i = 0; i< n; ++i){
         double tmp = (x[i*stride]-p[0]) * invsig;
         arg[i] = -0.5 *tmp*tmp;
      }
      const double sqrt_2pi = 1./std::sqrt(2.* 3.14159 );
#ifdef USE_VDT
      vdt::fast_expv(n,&arg[0],res);
      for (unsigned int i = 0; i< n; ++i){
#else
      for (unsigned int i = 0; i< n; ++i){
         res[i] = std::exp(arg[i]);
#endif
         res[i] *= sqrt_2pi * invsig;
      }

   }

   static void gausnorm2D_v(unsigned int n, unsigned int stride,const double *x, const double *p, double * res) {
      std::vector<double> tmp(n);
      gausnorm_v(n,2,x,p,&tmp[0]);
      gausnorm_v(n,2,x+1,p+2,res);
      for (unsigned int i = 0; i< n; ++i){
         res[i] *= tmp[i];
      }
   }
   static void gausnormN_v(unsigned int n, unsigned int stride,const double *x, const double *p, double * res) {
      std::vector<double> tmp(n);
      gausnorm_v(n,stride,x,p,res);
      for (int j = 1; j < stride; ++j) {
         gausnorm_v(n,stride,x+j,p+2*j,&tmp[0]);
         for (unsigned int i = 0; i< n; ++i){
            res[i] *= tmp[i];
         }
      }
   }

};

double gaussian (const double *x, const double *p) {
   return GausFunctions<double>::gaussian(x,p);
}
double gausnorm (const double *x, const double *p) {
   return GausFunctions<double>::gausnorm(x,p);
}
double gausnorm2D (const double *x, const double *p) {
   return GausFunctions<double>::gausnorm2D(x,p);
}
double gausnormN (const double *x, const double *p) {
   return GausFunctions<double>::gausnormN(x,p);
}

int testGausFit() {

   int iret = 0;

   std::cout << "\n\n************************************************************\n";
   std::cout << "\t GAUSSIAN FIT\n";
   std::cout << "************************************************************\n";



   //std::string fname = std::string("gaus");
   //TF1 * func = (TF1*)gROOT->GetFunction(fname.c_str());
   //TF1 * f1 = new TF1("gaus",fname.c_str(),-5,5.);
   //TF1 * f1 = new TF1("gaussian",gaussian,-5,5.,3);
   //f2->SetParameters(0,1,1);

   // fill an histogram
   int nbin = 10000;
   TH1D * h2 = new TH1D("h2","h2",nbin,-5.,5.);
//      h1->FillRandom(fname.c_str(),100);
   for (int i = 0; i < 10000000; ++i)
      h2->Fill( gRandom->Gaus(0,10) );

   iniPar[0] = 100.; iniPar[1] = 2.; iniPar[2] = 2.;


   // use simply TF1 wrapper
   //ROOT::Math::WrappedMultiTF1 f2(*f1);
   ROOT::Math::WrappedParamFunction<> f2(&gaussian,1,iniPar,iniPar+3);


   iret |= FitUsingNewFitter<MINUIT2>(h2,f2);
   iret |= FitUsingNewFitter<TMINUIT>(h2,f2);

//    iret |= FitUsingNewFitter<GSL_PR>(h2,f2);




    iret |= FitUsingNewFitter<GSL_FR>(h2,f2);
    iret |= FitUsingNewFitter<GSL_PR>(h2,f2);
    iret |= FitUsingNewFitter<GSL_BFGS>(h2,f2);
    iret |= FitUsingNewFitter<GSL_BFGS2>(h2,f2);


   // test also fitting a TGraphErrors with histogram data
   gStyle->SetErrorX(0.); // to seto zero error on X
   TGraphErrors * gr = new TGraphErrors(h2);


   iret |= FitUsingNewFitter<MINUIT2>(gr,f2);

   // try with error in X
   gStyle->SetErrorX(0.5); // to seto zero error on X
   TGraphErrors * gr2 = new TGraphErrors(h2);

   iret |= FitUsingNewFitter<MINUIT2>(gr2,f2);



//#ifdef LATER
   // test using grad function
   std::cout << "\n\nTest Using pre-calculated gradients\n\n";
   bool useGrad=true;
   iret |= FitUsingNewFitter<MINUIT2>(h2,f2,useGrad);
   iret |= FitUsingNewFitter<TMINUIT>(h2,f2,useGrad);
   iret |= FitUsingNewFitter<GSL_FR>(h2,f2,useGrad);
   iret |= FitUsingNewFitter<GSL_PR>(h2,f2,useGrad);
   iret |= FitUsingNewFitter<GSL_BFGS>(h2,f2,useGrad);
   iret |= FitUsingNewFitter<GSL_BFGS2>(h2,f2,useGrad);


   // test LS algorithm
   std::cout << "\n\nTest Least Square algorithms\n\n";
   iret |= FitUsingNewFitter<GSL_NLS>(h2,f2);
   iret |= FitUsingNewFitter<FUMILI2>(h2,f2);
   iret |= FitUsingNewFitter<TFUMILI>(h2,f2);

//    iret |= FitUsingTFit<TH1,FUMILI2>(h2,f1);
//    iret |= FitUsingTFit<TH1,TFUMILI>(h2,f1);
//#endif

   //iret |= FitUsingRooFit(h2,f1);

   printResult(iret);

   return iret;
}

int testTreeFit() {

   std::cout << "\n\n************************************************************\n";
   std::cout << "\t UNBINNED TREE (GAUSSIAN)  FIT\n";
   std::cout << "************************************************************\n";


   TTree t1("t1","a simple Tree with simple variables");
   double  x, y;
   Int_t ev;
   t1.Branch("x",&x,"x/D");
   t1.Branch("y",&y,"y/D");
//          t1.Branch("pz",&pz,"pz/F");
//          t1.Branch("random",&random,"random/D");
   t1.Branch("ev",&ev,"ev/I");

   //fill the tree
   int nrows = 10000;
#ifdef TREE_FIT2D
   nrows = 10000;
#endif
   for (Int_t i=0;i<nrows;i++) {
      gRandom->Rannor(x,y);
      x *= 2; x += 1.;
      y *= 3; y -= 2;

      ev = i;
      t1.Fill();

   }
   //t1.Draw("x"); // to select fit variable

   //TF1 * f1 = new TF1("gausnorm", gausnorm, -10,10, 2);
   //TF2 * f2 = new TF2("gausnorm2D", gausnorm2D, -10,10, -10,10, 4);

   ROOT::Math::WrappedParamFunction<> wf1(&gausnorm,1,iniPar,iniPar+2);
   ROOT::Math::WrappedParamFunction<> wf2(&gausnorm2D,2,iniPar,iniPar+4);


   iniPar[0] = 0;
   iniPar[1] = 1;
   iniPar[2] = 0;
   iniPar[3] = 1;

   // use simply TF1 wrapper
   //ROOT::Math::WrappedMultiTF1 f2(*f1);

   int iret = 0;

   // fit 1D first



   // iret |= FitUsingNewFitter<MINUIT2>(&t1,wf1,false); // not copying the data
   // iret |= FitUsingNewFitter<TMINUIT>(&t1,wf1,false); // not copying the data


   ndimObs = wf1.NDim();
   ndimPars = wf1.NPar();

#ifdef USE_AOS
   iret |= FitUsingNewFitter<MINUIT2>(&t1,&GausFunctions<double>::gausnorm_v,true); // copying the data
#else


#ifndef USE_VC
   iret |= FitUsingNewFitter<MINUIT2>(&t1,wf1,true); // copying the data
   iret |= FitUsingNewFitter<MINUIT2>(&t1,&GausFunctions<double>::gausnorm,true); // copying the data
#else
   iret |= FitUsingNewFitter<MINUIT2>(&t1,&GausFunctions<Vc::double_v>::gausnorm,true); // copying the data
#endif
//   iret |= FitUsingNewFitter<TMINUIT>(&t1,wf1,true); // copying the data

#endif
   // fit 2D

   ndimObs = wf2.NDim();
   ndimPars = wf2.NPar();


#ifndef USE_AOS
#ifndef USE_VC
   iret |= FitUsingNewFitter<MINUIT2>(&t1,wf2, true);
   iret |= FitUsingNewFitter<MINUIT2>(&t1,&GausFunctions<double>::gausnorm2D, true);
#else
   iret |= FitUsingNewFitter<MINUIT2>(&t1,&GausFunctions<Vc::double_v>::gausnorm2D, true);
#endif

#else
   iret |= FitUsingNewFitter<MINUIT2>(&t1,&GausFunctions<double>::gausnorm2D_v,true); // copying the data
#endif

   //iret |= FitUsingNewFitter<MINUIT2>(&t1,wf2, false);




   printResult(iret);
   return iret;

}

int testLargeTreeFit(int nevt = 1000) {



   std::cout << "\n\n************************************************************\n";
   std::cout << "\t UNBINNED TREE (GAUSSIAN MULTI-DIM)  FIT\n";
   std::cout << "************************************************************\n";

   TTree t1("t2","a large Tree with simple variables");
   double  x[N];
   Int_t ev;
   t1.Branch("x",x,"x[20]/D");
   t1.Branch("ev",&ev,"ev/I");

   //fill the tree
   TRandom3 r;
   for (Int_t i=0;i<nevt;i++) {
      for (int j = 0;  j < N; ++j) {
         double mu = double(j)/10.;
         double s  = 1.0 + double(j)/10.;
         x[j] = r.Gaus(mu,s);
      }

      ev = i;
      t1.Fill();

   }
   //t1.Draw("x"); // to select fit variable


   for (int i = 0; i <N; ++i) {
      iniPar[2*i] = 0;
      iniPar[2*i+1] = 1;
   }

   // use simply TF1 wrapper
   //ROOT::Math::WrappedMultiTF1 f2(*f1);

   ROOT::Math::WrappedParamFunction<> f2(&gausnormN,N,2*N,iniPar);

    ndimObs = f2.NDim();
    ndimPars = f2.NPar();


   int iret = 0;


#ifndef USE_VC

#ifndef USE_AOS
   iret |= FitUsingNewFitter<MINUIT2>(&t1,f2);
   iret |= FitUsingNewFitter<MINUIT2>(&t1,&GausFunctions<double>::gausnormN,true);
#else
   iret |= FitUsingNewFitter<MINUIT2>(&t1,&GausFunctions<double>::gausnormN_v,true);
#endif
   // iret |= FitUsingNewFitter<GSL_BFGS2>(&t1,f2);
#else
  iret |= FitUsingNewFitter<MINUIT2>(&t1,&GausFunctions<Vc::double_v>::gausnormN,true);
#endif


   printResult(iret);
   return iret;


}


int testFitPerf() {

   int iret = 0;




#ifdef DEBUG
   nfit = 1;
#else
   nfit = 10;
#endif
  iret |= testTreeFit();

  nfit = 1;
  iret |= testLargeTreeFit(2000);


#ifdef LATER

#ifndef DEBUG
   nfit = 10;
#endif
  iret |= testGausFit();



#ifndef DEBUG
   nfit = 1000;
#endif
   iret |= testPolyFit();


#endif


 //return iret;



   if (iret != 0)
      std::cerr << "testFitPerf :\t FAILED " << std::endl;
   else
      std::cerr << "testFitPerf :\t OK " << std::endl;
   return iret;
}

int main() {
   return testFitPerf();
}

