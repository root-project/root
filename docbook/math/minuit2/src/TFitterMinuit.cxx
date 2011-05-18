#include "TROOT.h"
#include "TFitterMinuit.h"
#include "TF1.h"
#include "TH1.h"
#include "TGraph.h"

#include "TChi2FCN.h"
#include "TChi2ExtendedFCN.h"
#include "TBinLikelihoodFCN.h"
#include "TInterpreter.h"
#include "TError.h"


#include "Minuit2/MnMigrad.h"
#include "Minuit2/MnMinos.h"
#include "Minuit2/MnHesse.h"
#include "Minuit2/MinuitParameter.h"
#include "Minuit2/MnPrint.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/VariableMetricMinimizer.h"
#include "Minuit2/SimplexMinimizer.h"
#include "Minuit2/CombinedMinimizer.h"
#include "Minuit2/ScanMinimizer.h"

#include <iomanip>

using namespace ROOT::Minuit2;

#ifndef ROOT_TMethodCall
#include "TMethodCall.h"
#endif

//#define DEBUG 1

//#define OLDFCN_INTERFACE
#ifdef OLDFCN_INTERFACE
extern void H1FitChisquare(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
extern void H1FitLikelihood(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
extern void Graph2DFitChisquare(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
extern void MultiGraphFitChisquare(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
#endif

//______________________________________________________________________________
/**
Interface to the new C++ Minuit package (MINUIT2) for ROOT. 
It implements the TVirtualFitter interface using Minuit2
For more information on the new C++ Minuit, see 
BEGIN_HTML
 See:
<ul>
<li><a href="http://www.cern.ch/mathlibs/sw/Minuit2/html/index.html">Online doc for Minuit2 classes</a></li>
<li><a href="http://seal.web.cern.ch/seal/documents/minuit/mnusersguide.pdf">C++ Minuit Users Guide
    </a></li>
<li><a href="http://seal.cern.ch/documents/minuit/mntutorial.pdf">Minuit Tutorial on Function Minimization</a>, describing the Minuit algorithm</li>
<li><a href="http://seal.cern.ch/documents/minuit/mnerror.pdf">The Interpretation of Errors in Minuit</a></li>
</ul>
<p>
Minuit2 can be set as the default fitter to be used in method lik TH1::Fit, by doing 
<pre>
TVirtualFitter::SetDefaultFitter("Minuit2");
</pre>
This class can be used also directly by providing for the objective function either a global C function, like in TMinuit, or by passing a function class implementing the ROOT::Minuit2::FCNBase interface and used via the SetMinuitFCN method 
END_HTML
*/



ClassImp(TFitterMinuit);

TFitterMinuit* gMinuit2 = 0;

TFitterMinuit::TFitterMinuit() : fErrorDef(1.) , fEDMVal(0.), fGradient(false),
fState(MnUserParameterState()), fMinosErrors(std::vector<MinosError>()), fMinimizer(0), fMinuitFCN(0), fDebug(1), fStrategy(1), fMinTolerance(0) {
   // Default constructor . Srategy and tolerance set to default values.
   Initialize();
}


TFitterMinuit::TFitterMinuit(Int_t /* maxpar */) : fErrorDef(1.) , fEDMVal(0.), fGradient(false), fState(MnUserParameterState()), fMinosErrors(std::vector<MinosError>()), fMinimizer(0), fMinuitFCN(0), fDebug(1), fStrategy(1), fMinTolerance(0) { 
   // Constructur needed by TVirtualFitter interface. Same behavior as default constructor.
   Initialize();
}

void TFitterMinuit::Initialize() {
   // initialize setting name and the global pointer
   SetName("Minuit2");
   gMinuit2 = this;
   gROOT->GetListOfSpecials()->Add(gMinuit2);
   
}


void TFitterMinuit::CreateMinimizer(EMinimizerType type) { 
   // create the minimizer engine and register the plugin in ROOT 
#ifdef DEBUG
   std::cout<<"TFitterMinuit:create minimizer of type "<< type << std::endl;
#endif
   if (fMinimizer) delete fMinimizer;
   
   switch (type) { 
      case kMigrad: 
         SetMinimizer( new VariableMetricMinimizer() );
         return;
      case kSimplex: 
         SetMinimizer( new SimplexMinimizer() );
         return;
      case kCombined: 
         SetMinimizer( new CombinedMinimizer() );
         return;
      case kScan: 
         SetMinimizer( new ScanMinimizer() );
         return;
      case kFumili: 
         std::cout << "TFitterMinuit::Error - Fumili Minimizer must be created from TFitterFumili " << std::endl;
         SetMinimizer(0);
         return;
      default: 
         //migrad minimizer
         SetMinimizer( new VariableMetricMinimizer() );
   }
}



TFitterMinuit::~TFitterMinuit() { 
// destructor - deletes the minimizer and minuit fcn
// if using TVirtualFitter one should use Clear() and not delete() 

#ifdef DEBUG
   std::cout << "delete minimizer and FCN" << std::endl;
#endif
   if (fMinuitFCN) delete fMinuitFCN; 
   if (fMinimizer) delete fMinimizer; 
   // delete minuit2 pointer from TROOT
   gROOT->GetListOfSpecials()->Remove(this);
   if (gMinuit2 == this) gMinuit2 = 0;
   
}

Double_t TFitterMinuit::Chisquare(Int_t npar, Double_t *params) const {
   // do chisquare calculations in case of likelihood fits 
   const TBinLikelihoodFCN * fcn = dynamic_cast<const TBinLikelihoodFCN *> (GetMinuitFCN() ); 
   if (fcn == 0) return 0;  
   std::vector<double> p(npar); 
   for (int i = 0; i < npar; ++i) 
      p[i] = params[i];
   return fcn->Chi2(p);
}

void TFitterMinuit::Clear(Option_t*) {
   // clear resources for consecutive fits
   
   //std::cout<<"clear "<<std::endl;
   
   fErrorDef = 1.; 
   fEDMVal = 0; 
   fGradient = false; 
   State() = MnUserParameterState();
   fMinosErrors.clear();
   //fDebug = 1;  
   fStrategy = 1;  
   fMinTolerance = 0;
   fCovar.clear();
   
//    if (fMinuitFCN) { 
//       delete fMinuitFCN;
//       fMinuitFCN = 0; 
//    }
   if (fMinimizer) { 
      delete fMinimizer; 
      fMinimizer = 0; 
   }
   
}



FunctionMinimum TFitterMinuit::DoMinimization( int nfcn, double edmval)  {
   // perform minimization using Minuit2 function
   // use always strategy 1 (2 is not yet fully tested)

   assert(GetMinuitFCN() != 0 );
   assert(GetMinimizer() != 0 );

   fMinuitFCN->SetErrorDef(fErrorDef); // set the error def

   if (fDebug >=1) { 
      std::cout << "TFitterMinuit - Minimize with max iterations = " << nfcn << " edmval = " << edmval << " errorDef = " << fMinuitFCN->Up() << std::endl;
   } 

   
   return GetMinimizer()->Minimize(*GetMinuitFCN(), State(), MnStrategy(fStrategy), nfcn, edmval);
}



int  TFitterMinuit::Minimize( int nfcn, double edmval)  { 
   // minimize (call DoMinimization() and analyze the result
   
   // min tolerance
   edmval = std::max(fMinTolerance, edmval);

   // switch off debugging if requested 
   int prevLevel = gErrorIgnoreLevel; 
   if (fDebug < 0)  // switch off printing of info messages in Minuit2
      gErrorIgnoreLevel = 1001;
   
   FunctionMinimum min = DoMinimization(nfcn,edmval);

   if (fDebug < 0) gErrorIgnoreLevel = prevLevel; // restore previous debug level 

   fState = min.UserState();
   return ExamineMinimum(min);
}

Int_t TFitterMinuit::ExecuteCommand(const char *command, Double_t *args, Int_t nargs){	
   // execute the command (Fortran Minuit compatible interface)
   
#ifdef DEBUG
   std::cout<<"Execute command= "<<command<<std::endl;
#endif

   
   // MIGRAD 
   if (strncmp(command,"MIG",3) == 0 || strncmp(command,"mig",3)  == 0) {
      int nfcn = 0; // default values for Minuit
      double edmval = 0.1;
      //     nfcn = GetMaxIterations();
      //     edmval = GetPrecision();
      if (nargs > 0) nfcn = int ( args[0] );
      if (nargs > 1) edmval = args[1];
      
      // create migrad minimizer
      CreateMinimizer(kMigrad);
      return  Minimize(nfcn, edmval);
      
      //     if(fGradient) {
      //       MnMigrad migrad(theFcn, State());
      //       theMinimum = migrad(theNFcn, fEDMVal);
      //       State() = theMinimum.userParameters();
      //     } else {
      //     std::cout<<"State(): "<<State()<<std::endl;
      //     std::cout<<"State(): "<<State()<<std::endl;
      //     }      
      
      
      
   } 
   //Minimize
   if (strncmp(command,"MINI",4) == 0 || strncmp(command,"mini",4)  == 0) {
      
      int nfcn = 0; // default values for Minuit
      double edmval = 0.1;
      if (nargs > 0) nfcn = int ( args[0] );
      if (nargs > 1) edmval = args[1];
      // create combined minimizer
      CreateMinimizer(kCombined);
      return Minimize(nfcn, edmval);
   }
   //Simplex
   if (strncmp(command,"SIM",3) == 0 || strncmp(command,"sim",3)  == 0) {
      
      int nfcn = 0; // default values for Minuit
      double edmval = 0.1;
      if (nargs > 0) nfcn = int ( args[0] );
      if (nargs > 1) edmval = args[1];
      // create combined minimizer
      CreateMinimizer(kSimplex);
      return Minimize(nfcn, edmval);
   }
   //SCan
   if (strncmp(command,"SCA",3) == 0 || strncmp(command,"sca",3)  == 0) {
      
      int nfcn = 0; // default values for Minuit
      double edmval = 0.1;
      if (nargs > 0) nfcn = int ( args[0] );
      if (nargs > 1) edmval = args[1];
      // create combined minimizer
      CreateMinimizer(kScan);
      return Minimize(nfcn, edmval);
   }
   // MINOS 
   else if (strncmp(command,"MINO",4) == 0 || strncmp(command,"mino",4)  == 0) {

      // switch off debugging if requested 
      int prevLevel = gErrorIgnoreLevel; 
      if (fDebug < 0)  // switch off printing of info messages in Minuit2
         gErrorIgnoreLevel = 1001;
      
      // recall minimize using default nfcn and edmval
      // should use maybe FunctionMinimum from previous call to migrad
      // need to keep a pointer to function minimum in the class
      FunctionMinimum min = DoMinimization();
      if (!min.IsValid() ) { 
         std::cout << "TFitterMinuit::MINOS failed due to invalid function minimum" << std::endl;
         if (fDebug < 0) gErrorIgnoreLevel = prevLevel; // restore previous debug level 
         return -10;
      }
      MnMinos minos( *fMinuitFCN, min);
      fMinosErrors.clear();
      for(unsigned int i = 0; i < State().VariableParameters(); i++) {
         if (fDebug>=3) std::cout << "Running Minos for parameter (ext#) " << State().ExtOfInt(i) << std::endl;
         MinosError me = minos.Minos(State().ExtOfInt(i));
         // print error message in Minos
         if (fDebug >= 0) {
            if ( !me.IsValid() )  
               std::cout << "Error running Minos for parameter " << State().ExtOfInt(i) << std::endl; 
         }
         if (fDebug >= 1) {
            if (!me.LowerValid() )  
               std::cout << "Minos:  Invalid lower error for parameter " << State().ExtOfInt(i) << std::endl; 
            if(me.AtLowerLimit()) 
               std::cout << "Minos:  Parameter  is at Lower limit."<<std::endl;
            if(me.AtLowerMaxFcn())
               std::cout << "Minos:  Maximum number of function calls exceeded when running for lower error" <<std::endl;   
            if(me.LowerNewMin() )
               std::cout << "Minos:  New Minimum found while running Minos for lower error" <<std::endl;     
            
            if (!me.UpperValid() )  
               std::cout << "Minos:  Invalid upper error for parameter " << State().ExtOfInt(i) << std::endl; 
            if(me.AtUpperLimit()) 
               std::cout << "Minos:  Parameter  is at Upper limit."<<std::endl;
            if(me.AtUpperMaxFcn())
               std::cout << "Minos:  Maximum number of function calls exceeded when running for upper error" <<std::endl;   
            if(me.UpperNewMin() )
               std::cout << "Minos:  New Minimum found while running Minos for upper error" <<std::endl;     
            
         }
         
         
         fMinosErrors.push_back(me);
      }
      if (fDebug >= 3) {
         for(std::vector<MinosError>::const_iterator ime = fMinosErrors.begin();
             ime != fMinosErrors.end(); ime++) 
            std::cout<<*ime<<std::endl;
      }

      if (fDebug < 0) gErrorIgnoreLevel = prevLevel; // restore previous debug level 

      return 0;
   } 
   //HESSE
   else if (strncmp(command,"HES",3) == 0 || strncmp(command,"hes",3)  == 0) {

      // switch off debugging if requested 
      int prevLevel = gErrorIgnoreLevel; 
      if (fDebug < 0)  // switch off printing of info messages in Minuit2
         gErrorIgnoreLevel = 1001;

      MnHesse hesse( GetStrategy() ); 
      // update the state
      const FCNBase * fcn = GetMinuitFCN();
      assert(fcn != 0);
      fState = hesse(*fcn, State(),100000 );

      if (fDebug < 0) gErrorIgnoreLevel = prevLevel; // restore previous debug level 

      if (!fState.IsValid() ) {
         std::cout << "TFitterMinuit::Hesse calculation failed " << std::endl;
         return -10;
      }
      return 0; 
   } 
   
   // FIX 
   else if (strncmp(command,"FIX",3) == 0 || strncmp(command,"fix",3)  == 0) {
      for(int i = 0; i < nargs; i++) {
         FixParameter(int(args[i])-1);
      }
      return 0;
   } 
   // SET LIMIT (uper and lower)
   else if (strncmp(command,"SET LIM",7) == 0 || strncmp(command,"set lim",7)  == 0) {
      assert(nargs >= 3);
      int ipar = int(args[0]);
      double low = args[1];
      double up = args[2];
      State().SetLimits(ipar, low, up);
      return 0; 
   } 
   // SET PRINT
   else if (strncmp(command,"SET PRINT",9) == 0 || strncmp(command,"set print",9)  == 0) {
      fDebug = int(args[0]);
      // if (fDebug >= 0) fDebug = 3;  // use print level of 3 (by default) - turn off with setprint(-1)
#ifdef DEBUG
      fDebug = 3;
#endif
      return 0; 
   } 
   // SET ERR
   else if (strncmp(command,"SET Err",7) == 0 || strncmp(command,"set err",7)  == 0) {
      fErrorDef = args[0];
      return 0; 
   } 
   // SET STRATEGY
   else if (strncmp(command,"SET STR",7) == 0 || strncmp(command,"set str",7)  == 0) {
      fStrategy = int(args[0]);
      return 0; 
   } 
   //SET GRAD (not impl.) 
   else if (strncmp(command,"SET GRA",7) == 0 || strncmp(command,"set gra",7)  == 0) {
      //     not yet available
      //     fGradient = true;
      return -1;
   } 
   // CALL FCN
   else if (strncmp(command,"CALL FCN",8) == 0 || strncmp(command,"call fcn",8)  == 0) {
      //     call fcn function 
      if (nargs < 1 || fFCN == 0) return -1;
      const std::vector<double> & params = State().Params();
      std::cout << State() << std::endl;
      int npar = params.size();
      double fval = 0; 
      (*fFCN)(npar, 0, fval, const_cast<double *>(&params.front()),int(args[0]) ) ;
      return 0; 
   } 
   else {
      // other commands passed 
      return 0;
   }
   
   
}



int  TFitterMinuit::ExamineMinimum(const FunctionMinimum & min) {  
   /// study the function minimum      
   
   // debug ( print all the states) 
   if (fDebug >= 3) { 
#ifdef LATER     
      const std::vector<MinimumState>& iterationStates = min.States();
      std::cout << "Number of iterations " << iterationStates.size() << std::endl;
      for (unsigned int i = 0; i <  iterationStates.size(); ++i) {
         //std::cout << iterationStates[i] << std::endl;                                                                       
         const MinimumState & st =  iterationStates[i];
         std::cout << "----------> Iteration " << i << std::endl;
         int pr = std::cout.precision(18);
         std::cout << "            FVAL = " << st.Fval() << " Edm = " << st.Edm() << " Nfcn = " << st.NFcn() << std::endl;
         std::cout.precision(pr);
         std::cout << "            Error matrix change = " << st.Error().Dcovar() << std::endl;
         std::cout << "            Internal parameters : ";
         for (int j = 0; j < st.size() ; ++j) std::cout << " p" << j << " = " << st.Vec()(j);
         std::cout << std::endl;
      }
#endif
   }
   // print result 
   if (min.IsValid() ) {
      if (fDebug >=1 ) { 
         std::cout << "Minimum Found" << std::endl; 
         int pr = std::cout.precision(18);
         std::cout << "FVAL  = " << State().Fval() << std::endl;
         std::cout << "Edm   = " << State().Edm() << std::endl;
         std::cout.precision(pr);
         std::cout << "Nfcn  = " << State().NFcn() << std::endl;
         std::vector<double> par = State().Params();
         std::vector<double> err = State().Errors();
         for (unsigned int i = 0; i < State().Params().size(); ++i) 
            std::cout << State().Parameter(i).Name() << "\t  = " << par[i] << "\t  +/-  " << err[i] << std::endl; 
      }
      return 0;
   }
   else { 
      if (fDebug >= 1)  {
         std::cout << "TFitterMinuit::Minimization DID not converge !" << std::endl; 
         std::cout << "FVAL  = " << State().Fval() << std::endl;
         std::cout << "Edm   = " << State().Edm() << std::endl;
         std::cout << "Nfcn  = " << State().NFcn() << std::endl;
      }
      if (min.HasMadePosDefCovar() ) { 
         if (fDebug >= 1) std::cout << "      Covar was made pos def" << std::endl;
         return -11; 
      }
      if (min.HesseFailed() ) { 
         if (fDebug >= 1) std::cout << "      Hesse is not valid" << std::endl;
         return -12; 
      }
      if (min.IsAboveMaxEdm() ) { 
         if (fDebug >= 1) std::cout << "      Edm is above max" << std::endl;
         return -13; 
      }
      if (min.HasReachedCallLimit() ) { 
         if (fDebug >= 1) std::cout << "      Reached call limit" << std::endl;
         return -14; 
      }
      return -10; 
   }
   return 0;
}

void TFitterMinuit::FixParameter(Int_t ipar) {
   // fix the paramter
   //   std::cout<<"FixParameter"<<std::endl;
   State().Fix(ipar);
}

Double_t* TFitterMinuit::GetCovarianceMatrix() const {
   // get the error matrix in a pointer to a NxN array.  
   // Since Minuit2 stores only the independent element need to copy in a 
   // cached vector
   unsigned int npar =  State().Covariance().Nrow();
   if ( int(npar) != GetNumberFreeParameters() ) { 
      // can happen if fit failes that npar is zero
      std::cout << "TFitterMinuit::GetCovarianceMatrix  Error - return null pointer" << std::endl;
      return 0; 
   }
   if (fCovar.size() !=  npar ) 
      fCovar.resize(npar*npar);
   
   for (unsigned int i = 0; i < npar; ++i) { 
      for (unsigned int j = 0; j < npar; ++j) {
         fCovar[j + npar*i] = State().Covariance()(i,j);
      }
   }
   return &(fCovar.front());
}

Double_t TFitterMinuit::GetCovarianceMatrixElement(Int_t i, Int_t j) const {
   // get error matrix element
   return State().Covariance()(i,j);
}


Int_t TFitterMinuit::GetErrors(Int_t ipar,Double_t &eplus, Double_t &eminus, Double_t &eparab, Double_t &globcc) const {
   // get fit errors 
   //   std::cout<<"GetError"<<std::endl;
   eplus = 0.;
   eminus = 0.; 

   MinuitParameter mpar = State().Parameters().Parameter(ipar);
   if (mpar.IsFixed() || mpar.IsConst() )  return 0;
   if(fMinosErrors.empty()) return 0; 
   
   
   unsigned int nintern = State().IntOfExt(ipar);
   eplus = fMinosErrors[nintern].Upper();
   eminus = fMinosErrors[nintern].Lower();
   
   eparab = State().Error(ipar);
   globcc = State().GlobalCC().GlobalCC()[ipar];
   return 0;
}

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// here is example of deficit of new Minuit interface
//////////////////////////////////////////////
Int_t TFitterMinuit::GetNumberTotalParameters() const {
   // number of total parameters (ugly interface)
   return State().Parameters().Parameters().size();
}
Int_t TFitterMinuit::GetNumberFreeParameters() const {
   // number of variable parameters
   return State().Parameters().VariableParameters();
}


Double_t TFitterMinuit::GetParError(Int_t ipar) const {
   // parameter error
   //   std::cout<<"GetParError"<<std::endl;
   return State().Error(ipar);
}

Double_t TFitterMinuit::GetParameter(Int_t ipar) const {
   // parameter value
   //   std::cout<<"GetParameter"<<std::endl;
   return State().Value(ipar);
}

Int_t TFitterMinuit::GetParameter(Int_t ipar,char *name,Double_t &value,Double_t &verr,Double_t &vlow, Double_t &vhigh) const {
   // get all parameter info (name, value, errors) 
   //   std::cout<<"GetParameter(Int_t ipar,char"<<std::endl;
   const MinuitParameter& mp = State().Parameter(ipar);
   //   std::cout<<"i= "<<ipar<<" verr= "<<mp.error()<<std::endl;
   std::string mpName = mp.Name();
   std::copy(mpName.c_str(), mpName.c_str() + mpName.size(), name);
   value = mp.Value();
   verr = mp.Error();
   vlow = mp.LowerLimit();
   vhigh = mp.UpperLimit();
   return 0;
}

const char *TFitterMinuit::GetParName(Int_t ipar) const {
   //   return name of parameter ipar
   const MinuitParameter& mp = State().Parameter(ipar);
   return mp.Name();
}

Int_t TFitterMinuit::GetStats(Double_t &amin, Double_t &edm, Double_t &errdef, Int_t &nvpar, Int_t &nparx) const {
   // get fit statistical information
   //   std::cout<<"GetStats"<<std::endl;
   amin = State().Fval();
   edm = State().Edm();
   errdef = fErrorDef;
   nvpar = State().VariableParameters();
   nparx = State().Parameters().Parameters().size();
   return 0;
}

Double_t TFitterMinuit::GetSumLog(Int_t) {
   //   std::cout<<"GetSumLog"<<std::endl;
   return 0.;
}

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Bool_t TFitterMinuit::IsFixed(Int_t ipar) const {
   // query if parameter ipar is fixed
   return State().Parameter(ipar).IsFixed();
}


void TFitterMinuit::PrintResults(Int_t level, Double_t) const {
   // print the fit result
   
   //   std::cout<<"PrintResults"<<std::endl;
   //   std::cout<<State().parameters()<<std::endl;
   if (fDebug >= 0 || level > 3) { 
      std::cout<<State()<<std::endl;
   }
   else { 
      // do no print covariance matrix
      if(!State().IsValid()) {
         std::cout <<  std::endl;
         std::cout << "WARNING: Minimum  is not valid."<<std::endl;
         std::cout <<  std::endl;
      }
   
      std::cout << "# of function calls: "<<State().NFcn()<<std::endl;
      std::cout << "function Value: "<< std::setprecision(12) << State().Fval()<<std::endl;
      std::cout << "expected distance to the Minimum (edm): "<< std::setprecision(8) << State().Edm()<<std::endl;
      std::cout << "fitted parameters: "<<State().Parameters()<<std::endl;
   }

   // print errors 
   if (level > 3) { 
      for(std::vector<MinosError>::const_iterator ime = fMinosErrors.begin();
          ime != fMinosErrors.end(); ime++) {
         std::cout<<*ime<<std::endl;
      }
   }
}

void TFitterMinuit::ReleaseParameter(Int_t ipar) {
   // release a fit parameter
   
   //   std::cout<<"ReleaseParameter"<<std::endl;
   State().Release(ipar);
}



void TFitterMinuit::SetFitMethod(const char *name) {
   // set fit method (i.e. chi2 or likelihood)
   // according to the method the appropriate FCN function will be created   
      
#ifdef DEBUG
   std::cout<<"SetFitMethod to "<< name << std::endl;
#endif
   if (!strcmp(name,"H1FitChisquare")) {
      // old way of passing
#ifdef OLDFCN_INTERFACE 
      SetFCN(H1FitChisquare);
#else 
      // call function (because overloaded by derived class)
      CreateChi2FCN();
#endif
      return;
   }
   if (!strcmp(name,"GraphFitChisquare")) {
#ifdef OLDFCN_INTERFACE 
      SetFCN(GraphFitChisquare);
#else 
      // use for graph extended chi2 to include error in X coordinate
      if (!GetFitOption().W1) 
         CreateChi2ExtendedFCN( );
      else
         CreateChi2FCN( );
#endif
      return;
   }
   if (!strcmp(name, "Graph2DFitChisquare")) {
#ifdef OLDFCN_INTERFACE 
      SetFCN(Graph2DFitChisquare);
#else 
      // use for graph extended chi2 to include error in X and Y coordinates
      //     if (!GetFitOption().W1) {
      //       CreateChi2ExtendedFCN( );
      //      else
      CreateChi2FCN( );
#endif
      return;
      }
   if (!strcmp(name, "MultiGraphFitChisquare")) {
      fErrorDef = 1.;
#ifdef OLDFCN_INTERFACE 
      SetFCN(MultiGraphFitChisquare);
#else 
      CreateChi2FCN();
#endif
      return;
   }
   if (!strcmp(name, "H1FitLikelihood")) {
      // old way of passing
#ifdef OLDFCN_INTERFACE 
      SetFCN(H1FitLikelihood);
#else 
      CreateBinLikelihoodFCN();    
#endif  
      return;
   }
   
   std::cout << "TFitterMinuit::fit method " << name << " is not  supported !" << std::endl; 
   
   assert(fMinuitFCN != 0);
   
   
   
   // 
   //   } else {
   //     SetFCN(H1FitChisquare);
   //   }
   // TFitterMinuit manages the data
   
   
}

Int_t TFitterMinuit::SetParameter(Int_t,const char *parname,Double_t value,Double_t verr,Double_t vlow, Double_t vhigh) {
   // set (add) a new fit parameter passing initial value,  step size (verr) and parametr limits
   // if vlow > vhigh the parameter is unbounded
   // if the stepsize (verr) == 0 the parameter is treated as fixed   

#ifdef DEBUG
   std::cout<<"SetParameter";
   std::cout << parname<<" value = " << value << " verr= "<<verr<<std::endl;
#endif
   if(vlow < vhigh) { 
      State().Add(parname, value, verr, vlow, vhigh);
   }
   else
      State().Add(parname, value, verr);
   
   // treat constant parameter as fixed 
   if (verr == 0)  State().Fix(parname);
   return 0;
}


void TFitterMinuit::SetFCN(void (*fcn)(Int_t &, Double_t *, Double_t &f, Double_t *, Int_t))
{
   // override setFCN to use the Adapter to Minuit2 FCN interface
   //*-*-*-*-*-*-*To set the address of the minimization function*-*-*-*-*-*-*-*
   //*-*          ===============================================
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   fFCN = fcn;
   if (fMinuitFCN) delete fMinuitFCN;
   fMinuitFCN = new TFcnAdapter(fFCN);
}

// need for interactive environment


// global functions needed by interpreter 


//______________________________________________________________________________
void Minuit2InteractiveFCN(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag)
{
   //*-*-*-*-*-*-*Static function called when SetFCN is called in interactive mode
   //*-*          ===============================================
   
   TMethodCall *m  = gMinuit2->GetMethodCall();
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
void TFitterMinuit::SetFCN(void *fcn)
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
   fFCN = Minuit2InteractiveFCN;
   gMinuit2 = this; //required by InteractiveFCNm
   
   if (fMinuitFCN) delete fMinuitFCN;
   fMinuitFCN = new TFcnAdapter(fFCN);
}

void TFitterMinuit::SetMinuitFCN(  FCNBase * f) { 
   // class takes the ownership of the passed pointer
   // so needs to delete previous one 
   if (fMinuitFCN) delete fMinuitFCN;
   fMinuitFCN = f; 
}

void TFitterMinuit::CreateChi2FCN() { 
   // create a chi2 FCN object
   SetMinuitFCN(new TChi2FCN( *this ) );
}


void TFitterMinuit::CreateChi2ExtendedFCN() { 
   // create an extended chi2 FCN object 
   // used in the case of errors both on the coordinates and the value (case of a graph fit)
   SetMinuitFCN(new TChi2ExtendedFCN( *this ) );
}

void TFitterMinuit::CreateBinLikelihoodFCN() { 
   // create a binned likelihood FCN
   SetMinuitFCN(new TBinLikelihoodFCN( *this ) );
}
