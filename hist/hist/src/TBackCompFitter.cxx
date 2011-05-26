#include "TROOT.h"
#include "TBackCompFitter.h"


#include "TMethodCall.h"
#include "TInterpreter.h"

#include "Math/Util.h"

#include <iostream>
#include <cassert>

//needed by GetCondifenceLevel
#include "Math/IParamFunction.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TMath.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TGraph2DErrors.h"
#include "TMultiGraph.h"
#include "HFitInterface.h"
#include "Math/Minimizer.h"
#include "Fit/BinData.h"
#include "Fit/UnBinData.h"
#include "Fit/PoissonLikelihoodFCN.h"
#include "Fit/LogLikelihoodFCN.h"
#include "Fit/Chi2FCN.h"
#include "Fit/FcnAdapter.h"
#include "TFitResult.h"

//#define DEBUG 1


//______________________________________________________________________________
/**
   Backward compatible implementation of TVirtualFitter using the the class ROOT::Fit::Fitter.
   This class is created after fitting an histogram (TH1), TGraph or TTree and provides in addition to the 
   methods of the TVirtualFitter hooks to access the fit result class (ROOT::Fit::FitResult), the fit configuration
   (ROOT::Fit::FitConfig) or the fit data (ROOT::Fit::FitData) using
   <pre>
   TBackCompFitter * fitter = (TBackCompFitter *) TVirtualFitter::GetFitter();
   ROOT::Fit::FitResult & result = fitter->GetFitResult();
   result.Print(std::cout);
   </pre>
   
   Methods for getting the confidence level or contours are also provided. 
   Note that after a new calls to TH1::Fit (or similar) the class will be deleted and all reference to the FitResult, FitConfig 
   or minimizer will be invalid. One could eventually copying  the class before issuing a new fit to avoid deleting this information
*/



ClassImp(TBackCompFitter);



TBackCompFitter::TBackCompFitter( ) : 
   fMinimizer(0), 
   fObjFunc(0),
   fModelFunc(0)
{
   // Constructur needed by TVirtualFitter interface. Same behavior as default constructor.
   // initialize setting name and the global pointer
   SetName("BCFitter");
}

TBackCompFitter::TBackCompFitter(std::auto_ptr<ROOT::Fit::Fitter> fitter, std::auto_ptr<ROOT::Fit::FitData>  data) : 
   fFitData(data),
   fFitter(fitter),
   fMinimizer(0),
   fObjFunc(0),
   fModelFunc(0)
{
   // constructor used after having fit using directly ROOT::Fit::Fitter
   // will create a dummy fitter copying configuration and parameter settings
   SetName("LastFitter");
}




TBackCompFitter::~TBackCompFitter() { 
   // data are own here
   //if (fFitData) delete fFitData; 

   if (fMinimizer) delete fMinimizer; 
   if (fObjFunc) delete fObjFunc; 
   if (fModelFunc) delete fModelFunc;
}

Double_t TBackCompFitter::Chisquare(Int_t npar, Double_t *params) const {
   // do chisquare calculations in case of likelihood fits 
   // do evaluation a the minimum only 
   const std::vector<double> & minpar = fFitter->Result().Parameters(); 
   assert (npar == (int) minpar.size() ); 
   double diff = 0; 
   double s = 0; 
   for (int i =0; i < npar; ++i) { 
      diff += std::abs( params[i] - minpar[i] );  
      s += minpar[i]; 
   }

   if (diff > s * 1.E-12 ) Warning("Chisquare","given parameter values are not at minimum - chi2 at minimum is returned"); 
   return fFitter->Result().Chi2(); 
}

void TBackCompFitter::Clear(Option_t*) {
   // clear resources for consecutive fits

   // need to do something here ??? to be seen
   
   
}





Int_t TBackCompFitter::ExecuteCommand(const char *command, Double_t *args, Int_t nargs){	
   // execute the command (Fortran Minuit compatible interface)
   
#ifdef DEBUG
   std::cout<<"Execute command= "<<command<<std::endl;
#endif

   int nfcn = GetMaxIterations();  
   double edmval = GetPrecision(); 

   // set also number of parameters in obj function
   DoSetDimension(); 

   TString scommand(command); 
   scommand.ToUpper();

   // MIGRAD 
   if (scommand.Contains("MIG")) {
      if (nargs > 0) nfcn = int ( args[0] );
      if (nargs > 1) edmval = args[1];
      if (!fObjFunc) { 
         Error("ExecuteCommand","FCN must set before executing this command"); 
         return -1; 
      }

      fFitter->Config().SetMinimizer(GetDefaultFitter(), "Migrad");
      bool ret = fFitter->FitFCN(*fObjFunc); 
      return  (ret) ? 0 : -1;
      
      
      
   } 
   //Minimize
   if (scommand.Contains("MINI")) {

      if (nargs > 0) nfcn = int ( args[0] );
      if (nargs > 1) edmval = args[1];

      fFitter->Config().SetMinimizer(GetDefaultFitter(), "Minimize");
      if (!fObjFunc) { 
         Error("ExecuteCommand","FCN must set before executing this command"); 
         return -1; 
      }
      bool ret = fFitter->FitFCN(*fObjFunc); 
      return  (ret) ? 0 : -1;
   }
   //Simplex
   if (scommand.Contains("SIM")) {
      
      if (nargs > 0) nfcn = int ( args[0] );
      if (nargs > 1) edmval = args[1];
      if (!fObjFunc) { 
         Error("ExecuteCommand","FCN must set before executing this command"); 
         return -1; 
      }

      fFitter->Config().SetMinimizer(GetDefaultFitter(), "Simplex");
      bool ret = fFitter->FitFCN(*fObjFunc); 
      return  (ret) ? 0 : -1;
   }
   //SCan
   if (scommand.Contains("SCA")) {
      
      if (nargs > 0) nfcn = int ( args[0] );
      if (nargs > 1) edmval = args[1];
      if (!fObjFunc) { 
         Error("ExecuteCommand","FCN must set before executing this command"); 
         return -1; 
      }

      fFitter->Config().SetMinimizer(GetDefaultFitter(), "Scan");
      bool ret = fFitter->FitFCN(*fObjFunc); 
      return  (ret) ? 0 : -1;
   }
   // MINOS 
   else if (scommand.Contains("MINO"))   {

      if (fFitter->Config().MinosErrors() ) return 0; 

      if (!fObjFunc) { 
         Error("ExecuteCommand","FCN must set before executing this command"); 
         return -1; 
      }
      // do only MINOS. need access to minimizer. For the moment re-run fitting with minos options 
      fFitter->Config().SetMinosErrors(true);
      // set new parameter values

      fFitter->Config().SetMinimizer(GetDefaultFitter(), "Migrad"); // redo -minimization with Minos
      bool ret = fFitter->FitFCN(*fObjFunc); 
      return  (ret) ? 0 : -1;

   } 
   //HESSE
   else if (scommand.Contains("HES"))   {

      if (fFitter->Config().ParabErrors() ) return 0; 

      if (!fObjFunc) { 
         Error("ExecuteCommand","FCN must set before executing this command"); 
         return -1; 
      }

      // do only HESSE. need access to minimizer. For the moment re-run fitting with hesse options 
      fFitter->Config().SetParabErrors(true);
      fFitter->Config().SetMinimizer(GetDefaultFitter(), "Migrad"); // redo -minimization with Minos
      bool ret = fFitter->FitFCN(*fObjFunc); 
      return  (ret) ? 0 : -1;
   } 
   
   // FIX 
   else if (scommand.Contains("FIX"))   {
      for(int i = 0; i < nargs; i++) {
         FixParameter(int(args[i])-1);
      }
      return 0;
   } 
   // SET LIMIT (upper and lower)
   else if (scommand.Contains("SET LIM"))   {
      if (nargs < 3) { 
         Error("ExecuteCommand","Invalid parameters given in SET LIMIT");
         return -1; 
      }
      int ipar = int(args[0]);
      if (!ValidParameterIndex(ipar) )  return -1;   
      double low = args[1];
      double up = args[2];
      fFitter->Config().ParSettings(ipar).SetLimits(low,up);
      return 0; 
   } 
   // SET PRINT
   else if (scommand.Contains("SET PRIN"))   {
      if (nargs < 1) return -1;  
      fFitter->Config().MinimizerOptions().SetPrintLevel(int(args[0]) );
      return 0; 
   } 
   // SET ERR
   else if (scommand.Contains("SET ERR"))   {
      if (nargs < 1) return -1;  
      fFitter->Config().MinimizerOptions().SetPrintLevel(int( args[0]) );
      return 0; 
   } 
   // SET STRATEGY
   else if (scommand.Contains("SET STR"))   {
      if (nargs < 1) return -1;  
      fFitter->Config().MinimizerOptions().SetStrategy(int(args[0]) );
      return 0; 
   } 
   //SET GRAD (not impl.) 
   else if (scommand.Contains("SET GRA"))   {
      //     not yet available
      //     fGradient = true;
      return -1;
   } 
   //SET NOW (not impl.) 
   else if (scommand.Contains("SET NOW"))   {
      //     no warning (works only for TMinuit)
      //     fGradient = true;
      return -1;
   } 
   // CALL FCN
   else if (scommand.Contains("CALL FCN"))   {
      //     call fcn function (global pointer to free function)

      if (nargs < 1 || fFCN == 0 ) return -1;
      int npar = fObjFunc->NDim();
      // use values in fit result if existing  otherwise in ParameterSettings
      std::vector<double> params(npar); 
      for (int i = 0; i < npar; ++i) 
         params[i] = GetParameter(i); 

      double fval = 0;
      (*fFCN)(npar, 0, fval, &params[0],int(args[0]) ) ;
      return 0; 
   } 
   else {
      // other commands passed 
      Error("ExecuteCommand","Invalid or not supported command given %s",command);
      return -1;
   }
   
   
}

bool TBackCompFitter::ValidParameterIndex(int ipar) const  { 
   // check if ipar is a valid parameter index
   int nps  = fFitter->Config().ParamsSettings().size(); 
   if (ipar  < 0 || ipar >= nps ) { 
      std::string msg = ROOT::Math::Util::ToString(ipar) + " is an invalid Parameter index";
      Error("ValidParameterIndex","%s",msg.c_str());
      return false;
   } 
   return true; 
}

void TBackCompFitter::FixParameter(Int_t ipar) {
   // fix the paramter
   //   std::cout<<"FixParameter"<<std::endl;
   if (ValidParameterIndex(ipar) )    
      fFitter->Config().ParSettings(ipar).Fix();
}



void TBackCompFitter::GetConfidenceIntervals(Int_t n, Int_t ndim, const Double_t *x, Double_t *ci, Double_t cl)
{
//Computes point-by-point confidence intervals for the fitted function
//Parameters:
//n - number of points
//ndim - dimensions of points
//x - points, at which to compute the intervals, for ndim > 1 
//    should be in order: (x0,y0, x1, y1, ... xn, yn)
//ci - computed intervals are returned in this array
//cl - confidence level, default=0.95
//NOTE, that the intervals are approximate for nonlinear(in parameters) models
   
   if (!fFitter->Result().IsValid()) { 
      Error("GetConfidenceIntervals","Cannot compute confidence intervals with an invalide fit result");
      return; 
   }
   
   fFitter->Result().GetConfidenceIntervals(n,ndim,1,x,ci,cl);         
}

void TBackCompFitter::GetConfidenceIntervals(TObject *obj, Double_t cl)
{
//Computes confidence intervals at level cl. Default is 0.95
//The TObject parameter can be a TGraphErrors, a TGraph2DErrors or a TH1,2,3.
//For Graphs, confidence intervals are computed for each point,
//the value of the graph at that point is set to the function value at that
//point, and the graph y-errors (or z-errors) are set to the value of
//the confidence interval at that point.
//For Histograms, confidence intervals are computed for each bin center
//The bin content of this bin is then set to the function value at the bin
//center, and the bin error is set to the confidence interval value.
//NOTE: confidence intervals are approximate for nonlinear models!
//
//Allowed combinations:
//Fitted object               Passed object
//TGraph                      TGraphErrors, TH1
//TGraphErrors, AsymmErrors   TGraphErrors, TH1
//TH1                         TGraphErrors, TH1
//TGraph2D                    TGraph2DErrors, TH2
//TGraph2DErrors              TGraph2DErrors, TH2
//TH2                         TGraph2DErrors, TH2
//TH3                         TH3

   if (!fFitter->Result().IsValid() ) { 
      Error("GetConfidenceIntervals","Cannot compute confidence intervals with an invalide fit result");
      return; 
   }

   // get data dimension from fit object
   int datadim = 1; 
   TObject * fitobj = GetObjectFit(); 
   if (!fitobj) { 
      Error("GetConfidenceIntervals","Cannot compute confidence intervals without a fitting object");
      return; 
   }

   if (fitobj->InheritsFrom(TGraph2D::Class())) datadim = 2; 
   if (fitobj->InheritsFrom(TH1::Class())) { 
      TH1 * h1 = dynamic_cast<TH1*>(fitobj); 
      assert(h1 != 0); 
      datadim = h1->GetDimension(); 
   } 

   if (datadim == 1) { 
      if (!obj->InheritsFrom(TGraphErrors::Class()) && !obj->InheritsFrom(TH1::Class() ) )  {
         Error("GetConfidenceIntervals", "Invalid object passed for storing confidence level data, must be a TGraphErrors or a TH1");
         return; 
      }
   } 
   if (datadim == 2) { 
      if (!obj->InheritsFrom(TGraph2DErrors::Class()) && !obj->InheritsFrom(TH2::Class() ) )  {
         Error("GetConfidenceIntervals", "Invalid object passed for storing confidence level data, must be a TGraph2DErrors or a TH2");
         return; 
      }
   }
   if (datadim == 3) { 
      if (!obj->InheritsFrom(TH3::Class() ) )  {
         Error("GetConfidenceIntervals", "Invalid object passed for storing confidence level data, must be a TH3");
         return; 
      }
   }

   // fill bin data (for the moment use all ranges) according to object passed
   ROOT::Fit::BinData data; 
   data.Opt().fUseEmpty = true; // need to use all bins of given histograms
   // call appropriate function according to type of object
   if (obj->InheritsFrom(TGraph::Class()) ) 
      ROOT::Fit::FillData(data, dynamic_cast<TGraph *>(obj) ); 
   else if (obj->InheritsFrom(TGraph2D::Class()) ) 
      ROOT::Fit::FillData(data, dynamic_cast<TGraph2D *>(obj) ); 
//    else if (obj->InheritsFrom(TMultiGraph::Class()) ) 
//       ROOT::Fit::FillData(data, dynamic_cast<TMultiGraph *>(obj) ); 
   else if (obj->InheritsFrom(TH1::Class()) ) 
      ROOT::Fit::FillData(data, dynamic_cast<TH1 *>(obj) ); 
   

   unsigned int n = data.Size(); 

   std::vector<double> ci( n ); 

   fFitter->Result().GetConfidenceIntervals(data,&ci[0],cl);         

   const ROOT::Math::IParamMultiFunction * func =  fFitter->Result().FittedFunction(); 
   assert(func != 0); 

   // fill now the object with cl data
   for (unsigned int i = 0; i < n; ++i) {
      const double * x = data.Coords(i); 
      double y = (*func)( x ); // function is evaluated using its  parameters

      if (obj->InheritsFrom(TGraphErrors::Class()) ) { 
         TGraphErrors * gr = dynamic_cast<TGraphErrors *> (obj); 
         assert(gr != 0); 
         gr->SetPoint(i, *x, y); 
         gr->SetPointError(i, 0, ci[i]); 
      }
      if (obj->InheritsFrom(TGraph2DErrors::Class()) ) { 
         TGraph2DErrors * gr = dynamic_cast<TGraph2DErrors *> (obj); 
         assert(gr != 0); 
         gr->SetPoint(i, x[0], x[1], y); 
         gr->SetPointError(i, 0, 0, ci[i]); 
      }
      if (obj->InheritsFrom(TH1::Class()) ) { 
         TH1 * h1 = dynamic_cast<TH1 *> (obj); 
         assert(h1 != 0); 
         int ibin = 0; 
         if (datadim == 1) ibin = h1->FindBin(*x); 
         if (datadim == 2) ibin = h1->FindBin(x[0],x[1]); 
         if (datadim == 3) ibin = h1->FindBin(x[0],x[1],x[2]); 
         h1->SetBinContent(ibin, y); 
         h1->SetBinError(ibin, ci[i]); 
      }
   }

}

Double_t* TBackCompFitter::GetCovarianceMatrix() const {
   // get the error matrix in a pointer to a NxN array.  
   // excluding the fixed parameters 

   unsigned int nfreepar =   GetNumberFreeParameters();
   unsigned int ntotpar =   GetNumberTotalParameters();
   
   if (fCovar.size() !=  nfreepar*nfreepar ) 
      fCovar.resize(nfreepar*nfreepar);

   if (!fFitter->Result().IsValid() ) { 
      Warning("GetCovarianceMatrix","Invalid fit result");
      return 0; 
   }

   unsigned int l = 0; 
   for (unsigned int i = 0; i < ntotpar; ++i) { 
      if (fFitter->Config().ParSettings(i).IsFixed() ) continue;
      unsigned int m = 0; 
      for (unsigned int j = 0; j < ntotpar; ++j) {
         if (fFitter->Config().ParSettings(j).IsFixed() ) continue;
         unsigned int index = nfreepar*l + m;
         assert(index < fCovar.size() );
         fCovar[index] = fFitter->Result().CovMatrix(i,j);
         m++;
      }
      l++;
   }
   return &(fCovar.front());
}

Double_t TBackCompFitter::GetCovarianceMatrixElement(Int_t i, Int_t j) const {
   // get error matrix element (return all zero if matrix is not available)

   unsigned int np2 = fCovar.size();
   unsigned int npar = GetNumberFreeParameters(); 
   if ( np2 == 0 || np2 != npar *npar ) { 
      double * c = GetCovarianceMatrix();
      if (c == 0) return 0;  
   }
   return fCovar[i*npar + j];  
}


Int_t TBackCompFitter::GetErrors(Int_t ipar,Double_t &eplus, Double_t &eminus, Double_t &eparab, Double_t &globcc) const {
   // get fit errors 

   if (!ValidParameterIndex(ipar) )   return -1; 
   
   const ROOT::Fit::FitResult & result = fFitter->Result(); 
   if (!result.IsValid() ) { 
      Warning("GetErrors","Invalid fit result");
      return -1; 
   }

   eparab = result.Error(ipar); 
   eplus = result.UpperError(ipar); 
   eminus = result.LowerError(ipar); 
   globcc = result.GlobalCC(ipar); 
   return 0;
}

Int_t TBackCompFitter::GetNumberTotalParameters() const {
   // number of total parameters 
   return fFitter->Result().NTotalParameters();  
}
Int_t TBackCompFitter::GetNumberFreeParameters() const {
   // number of variable parameters
   return fFitter->Result().NFreeParameters();  
}


Double_t TBackCompFitter::GetParError(Int_t ipar) const {
   // parameter error
   if (fFitter->Result().IsEmpty() ) {
      if (ValidParameterIndex(ipar) )  return  fFitter->Config().ParSettings(ipar).StepSize();
      else return 0; 
   }
   return fFitter->Result().Error(ipar);  
}

Double_t TBackCompFitter::GetParameter(Int_t ipar) const {
   // parameter value
   if (fFitter->Result().IsEmpty() ) {
      if (ValidParameterIndex(ipar) )  return  fFitter->Config().ParSettings(ipar).Value();
      else return 0; 
   }
   return fFitter->Result().Value(ipar);  
}

Int_t TBackCompFitter::GetParameter(Int_t ipar,char *name,Double_t &value,Double_t &verr,Double_t &vlow, Double_t &vhigh) const {
   // get all parameter info (name, value, errors) 
   if (!ValidParameterIndex(ipar) )    {
      return -1; 
   }
   const std::string & pname = fFitter->Config().ParSettings(ipar).Name(); 
   const char * c = pname.c_str(); 
   std::copy(c,c + pname.size(),name);

   if (fFitter->Result().IsEmpty() ) { 
      value = fFitter->Config().ParSettings(ipar).Value(); 
      verr  = fFitter->Config().ParSettings(ipar).Value();  // error is step size in this case 
      vlow  = fFitter->Config().ParSettings(ipar).LowerLimit();  // vlow is lower limit in this case 
      vhigh   = fFitter->Config().ParSettings(ipar).UpperLimit();  // vlow is lower limit in this case 
      return 1; 
   }
   else { 
      value =  fFitter->Result().Value(ipar);  
      verr = fFitter->Result().Error(ipar);  
      vlow = fFitter->Result().LowerError(ipar);  
      vhigh = fFitter->Result().UpperError(ipar);  
   }
   return 0; 
}

const char *TBackCompFitter::GetParName(Int_t ipar) const {
   //   return name of parameter ipar
   if (!ValidParameterIndex(ipar) )    {
      return 0; 
   }
   return fFitter->Config().ParSettings(ipar).Name().c_str(); 
}

Int_t TBackCompFitter::GetStats(Double_t &amin, Double_t &edm, Double_t &errdef, Int_t &nvpar, Int_t &nparx) const {
   // get fit statistical information
   const ROOT::Fit::FitResult & result = fFitter->Result(); 
   amin = result.MinFcnValue(); 
   edm = result.Edm(); 
   errdef = fFitter->Config().MinimizerOptions().ErrorDef(); 
   nvpar = result.NFreeParameters();  
   nparx = result.NTotalParameters();  
   return 0;
}

Double_t TBackCompFitter::GetSumLog(Int_t) {
   //   sum of log . Un-needed
   Warning("GetSumLog","Dummy  method - returned 0"); 
   return 0.;
}


Bool_t TBackCompFitter::IsFixed(Int_t ipar) const {
   // query if parameter ipar is fixed
   if (!ValidParameterIndex(ipar) )    {
      return false; 
   }
   return fFitter->Config().ParSettings(ipar).IsFixed(); 
}


void TBackCompFitter::PrintResults(Int_t level, Double_t ) const {
   // print the fit result
   // use PrintResults function in case of Minuit for old -style printing
   if (fFitter->GetMinimizer() && fFitter->Config().MinimizerType() == "Minuit")
      fFitter->GetMinimizer()->PrintResults();
   else { 
      if (level > 0) fFitter->Result().Print(std::cout); 
      if (level > 1)  fFitter->Result().PrintCovMatrix(std::cout);    
   }
   // need to print minos errors and globalCC + other info
}

void TBackCompFitter::ReleaseParameter(Int_t ipar) {
   // release a fit parameter
   if (ValidParameterIndex(ipar) )    
      fFitter->Config().ParSettings(ipar).Release(); 
}



void TBackCompFitter::SetFitMethod(const char *) {
   // set fit method (i.e. chi2 or likelihood)
   // according to the method the appropriate FCN function will be created   
   Info("SetFitMethod","non supported method");
}

Int_t TBackCompFitter::SetParameter(Int_t ipar,const char *parname,Double_t value,Double_t verr,Double_t vlow, Double_t vhigh) {
   // set (add) a new fit parameter passing initial value,  step size (verr) and parametr limits
   // if vlow > vhigh the parameter is unbounded
   // if the stepsize (verr) == 0 the parameter is treated as fixed   

   std::vector<ROOT::Fit::ParameterSettings> & parlist = fFitter->Config().ParamsSettings(); 
   if ( ipar >= (int) parlist.size() ) parlist.resize(ipar+1); 
   ROOT::Fit::ParameterSettings ps(parname, value, verr); 
   if (verr == 0) ps.Fix(); 
   if (vlow < vhigh) ps.SetLimits(vlow, vhigh); 
   parlist[ipar] = ps; 
   return 0; 
}

// static method evaluating FCN
// void TBackCompFitter::FCN( int &, double * , double & f, double * x , int /* iflag */) { 
//    // get static instance of fitter
//    TBackCompFitter * fitter = dynamic_cast<TBackCompFitter *>(TVirtualFitter::GetFitter()); 
//    assert(fitter); 
//    if (fitter->fObjFunc == 0) fitter->RecreateFCN(); 
//    assert(fitter->fObjFunc);
//    f = (*(fitter.fObjFunc) )(x);
// }

void TBackCompFitter::ReCreateMinimizer() { 
   // Recreate a minimizer instance using the function and data 
   // set objective function in minimizers function to re-create FCN from stored data object and fit options
   assert(fFitData.get());

   // case of standard fits (not made fia Fitter::FitFCN) 
   if (fFitter->Result().FittedFunction() != 0) {

      if (fModelFunc) delete fModelFunc; 
      fModelFunc =  dynamic_cast<ROOT::Math::IParamMultiFunction *>((fFitter->Result().FittedFunction())->Clone());
      assert(fModelFunc);

      // create fcn functions, should consider also gradient case
      const ROOT::Fit::BinData * bindata = dynamic_cast<const ROOT::Fit::BinData *>(fFitData.get()); 
      if (bindata) { 
         if (GetFitOption().Like ) 
            fObjFunc = new ROOT::Fit::PoissonLikelihoodFCN<ROOT::Math::IMultiGenFunction>(*bindata, *fModelFunc);
         else
            fObjFunc = new ROOT::Fit::Chi2FCN<ROOT::Math::IMultiGenFunction>(*bindata, *fModelFunc);
      }
      else { 
         const ROOT::Fit::UnBinData * unbindata = dynamic_cast<const ROOT::Fit::UnBinData *>(fFitData.get()); 
         assert(unbindata); 
         fObjFunc = new ROOT::Fit::LogLikelihoodFCN<ROOT::Math::IMultiGenFunction>(*unbindata, *fModelFunc);
      }
   }
   

   // recreate the minimizer
   fMinimizer = fFitter->Config().CreateMinimizer(); 
   if (fMinimizer == 0) { 
      Error("SetMinimizerFunction","cannot create minimizer %s",fFitter->Config().MinimizerType().c_str() );
   }
   else {
      if (!fObjFunc) {
         Error("SetMinimizerFunction","Object Function pointer is NULL");
      }
      else 
         fMinimizer->SetFunction(*fObjFunc);
   }
  
} 



void TBackCompFitter::SetFCN(void (*fcn)(Int_t &, Double_t *, Double_t &f, Double_t *, Int_t))
{
   // override setFCN to use the Adapter to Minuit2 FCN interface
   //*-*-*-*-*-*-*To set the address of the minimization function*-*-*-*-*-*-*-*
   //*-*          ===============================================
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   fFCN = fcn;
   if (fObjFunc) delete fObjFunc;
   fObjFunc = new ROOT::Fit::FcnAdapter(fFCN);
   DoSetDimension(); 
}

// need for interactive environment


// global functions needed by interpreter 


//______________________________________________________________________________
void InteractiveFCNm2(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag)
{
   //*-*-*-*-*-*-*Static function called when SetFCN is called in interactive mode
   //*-*          ===============================================
   
   // get method call from static instance
   TMethodCall *m  = (TVirtualFitter::GetFitter())->GetMethodCall();
   if (!m) return;
   
   Long_t args[5];
   args[0] = (Long_t)&npar;
   args[1] = (Long_t)gin;
   args[2] = (Long_t)&f;
   args[3] = (Long_t)u;
   args[4] = (Long_t)flag;
   m->SetParamPtrs(args);
   Double_t result;
   m->Execute(result);
}


//______________________________________________________________________________
void TBackCompFitter::SetFCN(void *fcn)
{
   //*-*-*-*-*-*-*To set the address of the minimization function*-*-*-*-*-*-*-*
   //*-*          ===============================================
   //     this function is called by CINT instead of the function above
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   
   if (!fcn) return;
   
   const char *funcname = gCint->Getp2f2funcname(fcn);
   if (funcname) {
      fMethodCall = new TMethodCall();
      fMethodCall->InitWithPrototype(funcname,"Int_t&,Double_t*,Double_t&,Double_t*,Int_t");
   }
   fFCN = InteractiveFCNm2;
   // set the static instance (required by InteractiveFCNm)
   TVirtualFitter::SetFitter(this); 
   
   if (fObjFunc) delete fObjFunc;
   fObjFunc = new ROOT::Fit::FcnAdapter(fFCN);
   DoSetDimension(); 
}

void TBackCompFitter::SetObjFunction(ROOT::Math::IMultiGenFunction   * fcn) { 
   // set the objective function for fitting
   // Needed if fitting directly using TBackCompFitter class
   // The class clones a copy of the function and manages it
   if (fObjFunc) delete fObjFunc;
   fObjFunc = fcn->Clone(); 
}


void TBackCompFitter::DoSetDimension() { 
   // Private method to set dimension in objective function
   if (!fObjFunc) return; 
   ROOT::Fit::FcnAdapter * fobj = dynamic_cast<ROOT::Fit::FcnAdapter*>(fObjFunc); 
   assert(fobj != 0); 
   int ndim = fFitter->Config().ParamsSettings().size(); 
   if (ndim != 0) fobj->SetDimension(ndim); 
}

ROOT::Math::IMultiGenFunction * TBackCompFitter::GetObjFunction( ) const { 
   // return a pointer to the objective function (FCN) 
   // If fitting directly using TBackCompFitter the pointer is managed by the class,
   // which has been set previously when calling SetObjFunction or SetFCN
   // Otherwise if the class is used in the backward compatible mode (e.g. after having fitted a TH1) 
   // the return pointer will be valid after fitting and as long a new fit will not be done. 
   if (fObjFunc) return fObjFunc;
   return fFitter->GetFCN(); 
}

ROOT::Math::Minimizer * TBackCompFitter::GetMinimizer( ) const { 
   // return a pointer to the minimizer.  
   // the return pointer will be valid after fitting and as long a new fit will not be done. 
   // For keeping a minimizer pointer the method ReCreateMinimizer() could eventually be used  
   if (fMinimizer) return fMinimizer;
   return fFitter->GetMinimizer();
}

TFitResult * TBackCompFitter::GetTFitResult( ) const {
   // return a new copy of the TFitResult object which needs to be deleted later by the user
   if (!fFitter.get() ) return 0; 
   return new TFitResult( fFitter->Result() );
}

//________________________________________________________________________________
bool TBackCompFitter::Scan(unsigned int ipar, TGraph * gr, double xmin, double xmax )
{
   //     scan parameter ipar between value of xmin and xmax
   //     a graph must be given which will be on return filled with the scan resul 
   //     If the graph size is zero, a default size n = 40 will be used
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


   if (!gr) return false; 
   ROOT::Math::Minimizer * minimizer = fFitter->GetMinimizer(); 
   if (!minimizer) {
      Error("Scan","Minimizer is not available - cannot scan before fitting");
      return false;
   }


   unsigned int npoints = gr->GetN(); 
   if (npoints == 0)  { 
      npoints = 40; 
      gr->Set(npoints);
   }
   bool ret = minimizer->Scan( ipar, npoints, gr->GetX(), gr->GetY(), xmin, xmax); 
   if ((int) npoints < gr->GetN() ) gr->Set(npoints); 
   return ret; 
}
   
// bool  TBackCompFitter::Scan2D(unsigned int ipar, unsigned int jpar, TGraph2D * gr, 
//                       double xmin = 0, double xmax = 0, double ymin = 0, double ymax = 0) { 
//    //     scan the parameters ipar between values of [xmin,xmax] and 
//    //     jpar between values of [ymin,ymax] and 
//    //     a graph2D must be given which will be on return filled with the scan resul 
//    //     If the graph size is zero, a default size n = 20x20 will be used
//    //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

//    if (!gr) return false; 
//    if (!fMinimizer) {
//       Error("Scan","Minimizer is not available - cannot scan before fitting");
//       return false;
//    }
//    unsigned int npoints = gr->GetN(); 
//    if (npoints == 0)  { 
//       npoints = 40; 
//       gr->Set(npoints);
//    }
//    // to be implemented 
//    for (unsigned int ix = 0; ix < npoints; ++ix) {       
//       return fMinimizer->Scan( ipar, npoints, gr->GetX(), gr->GetY(), xmin, xmax); 

// }

bool  TBackCompFitter::Contour(unsigned int ipar, unsigned int jpar, TGraph * gr, double confLevel) { 
   //  create a 2D contour around the minimum for the parameter ipar and jpar
   // if a minimum does not exist or is invalid it will return false
   // on exit a TGraph is filled with the contour points 
   // the number of contur points is determined by the size of the TGraph. 
   // if the size is zero a default number of points = 20 is used 
   // pass optionally the confidence level, default is 0.683
   // it is assumed that ErrorDef() defines the right error definition 
   // (i.e 1 sigma error for one parameter). If not the confidence level are scaled to new level

   if (!gr) return false; 
   ROOT::Math::Minimizer * minimizer = fFitter->GetMinimizer(); 
   if (!minimizer) {
      Error("Scan","Minimizer is not available - cannot scan before fitting");
      return false;
   }

   // get error level used for fitting
   double upScale = fFitter->Config().MinimizerOptions().ErrorDef();

   double upVal = TMath::ChisquareQuantile( confLevel, 2);  // 2 is number of parameter we do the contour
   
   // set required error definition in minimizer
   minimizer->SetErrorDef (upScale * upVal);    

   unsigned int npoints = gr->GetN(); 
   if (npoints == 0)  { 
      npoints = 40; 
      gr->Set(npoints);
   }
   bool ret =  minimizer->Contour( ipar, jpar, npoints, gr->GetX(), gr->GetY()); 
   if ((int) npoints < gr->GetN() ) gr->Set(npoints); 

   // restore the error level used for fitting
   minimizer->SetErrorDef ( upScale);

   return ret; 
}


