//
// implementation file for class GSLMCIntegrator
// Author: Magdalena Slawinska
// 
//

#include "Math/IFunctionfwd.h"
#include "Math/IFunction.h"
#include <vector>

#include "GSLMonteFunctionWrapper.h"

#include "Math/GSLMCIntegrator.h"
#include "GSLMCIntegrationWorkspace.h"
#include "GSLRngWrapper.h"


#include "gsl/gsl_monte_vegas.h"
#include "gsl/gsl_monte_miser.h"
#include "gsl/gsl_monte_plain.h"



namespace ROOT {
namespace Math {


                      
// constructor
      
                   
      
GSLMCIntegrator::GSLMCIntegrator(MCIntegration::Type type, double absTol, double relTol, unsigned int calls):
   fType(type),
   fMode(MCIntegration::kIMPORTANCE),
   fAbsTol(absTol),
   fRelTol(relTol),
   fDim(0),
   //fr(r),
   fCalls(calls),
   fResult(0),fError(0),fStatus(-1),
   fWorkspace(0),
   fFunction(0)
{
   // constructor of GSL MCIntegrator.Vegas MC is set as default integration type
   //set Workspace according to type
   
   //set random number generator
   fRng = new GSLRngWrapper();      
   fRng->Allocate();
   
}

GSLMCIntegrator::GSLMCIntegrator(const char * type, double absTol, double relTol, unsigned int calls):
   fMode(MCIntegration::kIMPORTANCE),
   fAbsTol(absTol),
   fRelTol(relTol),
   fDim(0),
   //fr(r),
   fCalls(calls),
   fResult(0),fError(0),fStatus(-1),
   fWorkspace(0),
   fFunction(0)
{
   // constructor of GSL MCIntegrator. Vegas MC is set as default integration type
   std::string typeName(type); 
   if (typeName == "PLAIN")
      fType =  MCIntegration::kPLAIN;
   else if (typeName == "MISER")
      fType =  MCIntegration::kMISER;
   else 
      fType =  MCIntegration::kVEGAS;  // default
   
   //set random number generator
   fRng = new GSLRngWrapper();      
   fRng->Allocate();
   
}
       
 
       //maybe to be added later; for various rules within basic methods
      //GSLIntegrator(const Integration::Type type, const Integration::GKRule rule, double absTol = 1.E-9, double relTol = 1E-6, unsigned int size = 1000);
      
      
GSLMCIntegrator::~GSLMCIntegrator()
{
   // delete workspace 
   if (fWorkspace) delete fWorkspace;
   if (fRng != 0) delete fRng;
   if (fFunction != 0) delete fFunction;
   fRng = 0;
   
}
      
      
// disable copy ctrs
  
         
GSLMCIntegrator::GSLMCIntegrator(const GSLMCIntegrator &) : 
   VirtualIntegratorMultiDim()
{}

GSLMCIntegrator & GSLMCIntegrator::operator=(const GSLMCIntegrator &) { return *this; }
      
   
         
         
              
         
void GSLMCIntegrator::SetFunction(const IMultiGenFunction &f)
{
   // method to set the a generic integration function
   if(fFunction == 0) fFunction = new  GSLMonteFunctionWrapper();
   fFunction->SetFunction(f);
   fDim = f.NDim();
} 
      
void GSLMCIntegrator::SetFunction( GSLMonteFuncPointer f,  unsigned int dim, void * p  )
{
   // method to set the a generic integration function
   if(fFunction == 0) fFunction = new  GSLMonteFunctionWrapper();
   fFunction->SetFuncPointer( f );
   fFunction->SetParams ( p );
   fDim = dim;
}



double GSLMCIntegrator::Integral(const double* a, const double* b)
{
   // evaluate the Integral of a over the defined interval (a[],b[])
   assert(fRng != 0);
   gsl_rng* fr = fRng->Rng();
   assert(fr != 0);
   if (!CheckFunction()) return 0;  

   // initialize by  creating the right WS 
   // (if dimension and type are different than previous calculation)
   DoInitialize(); 

   if ( fType == MCIntegration::kVEGAS) 
   {
      GSLVegasIntegrationWorkspace * ws = dynamic_cast<GSLVegasIntegrationWorkspace *>(fWorkspace); 
      assert(ws != 0);
      if(fMode == MCIntegration::kIMPORTANCE) ws->GetWS()->mode = GSL_VEGAS_MODE_IMPORTANCE;
      else if(fMode == MCIntegration::kSTRATIFIED) ws->GetWS()->mode = GSL_VEGAS_MODE_STRATIFIED;
      else if(fMode == MCIntegration::kIMPORTANCE_ONLY) ws->GetWS()->mode = GSL_VEGAS_MODE_IMPORTANCE_ONLY;

      fStatus = gsl_monte_vegas_integrate( fFunction->GetFunc(), (double *) a, (double*) b , fDim, fCalls, fr, ws->GetWS(),  &fResult, &fError);
   }
   else if (fType ==  MCIntegration::kMISER) 
   {
      GSLMiserIntegrationWorkspace * ws = dynamic_cast<GSLMiserIntegrationWorkspace *>(fWorkspace); 
      assert(ws != 0); 
      fStatus = gsl_monte_miser_integrate( fFunction->GetFunc(), (double *) a, (double *) b , fDim, fCalls, fr, ws->GetWS(),  &fResult, &fError);
   }
   else if (fType ==  MCIntegration::kPLAIN) 
   {
      GSLPlainIntegrationWorkspace * ws = dynamic_cast<GSLPlainIntegrationWorkspace *>(fWorkspace); 
      assert(ws != 0); 
      fStatus = gsl_monte_plain_integrate( fFunction->GetFunc(), (double *) a, (double *) b , fDim, fCalls, fr, ws->GetWS(),  &fResult, &fError);
   }
   /**/
   else 
   {
      
      fResult = 0;
      fError = 0;
      fStatus = -1;
      std::cerr << "GSLIntegrator - Error: Unknown integration type" << std::endl;
      throw std::exception(); 
   }
   
   return fResult;
   
}

     
double GSLMCIntegrator::Integral(const GSLMonteFuncPointer & f, unsigned int dim, double* a, double* b, void * p )
{
   // evaluate the Integral for a function f over the defined interval (a[],b[])
   SetFunction(f,dim,p);
   return Integral(a,b);
}


/* to be added later           
   double GSLMCIntegrator::Integral(GSLMonteFuncPointer f, void * p, double* a, double* b)
   {

   }
      
*/
//MCIntegration::Type GSLMCIntegrator::MCType() const {return fType;}

/**
   return  the Result of the last Integral calculation
*/
double GSLMCIntegrator::Result() const { return fResult; }
      
/**
   return the estimate of the absolute Error of the last Integral calculation
*/
double GSLMCIntegrator::Error() const { return fError; }
      
/**
   return the Error Status of the last Integral calculation
*/
int GSLMCIntegrator::Status() const { return fStatus; }
      
      
// setter for control Parameters  (getters are not needed so far )
      
/**
   set the desired relative Error
*/
void GSLMCIntegrator::SetRelTolerance(double relTol){ this->fRelTol = relTol; }
           
/**
   set the desired absolute Error
*/
void GSLMCIntegrator::SetAbsTolerance(double absTol){ this->fAbsTol = absTol; }
           
void GSLMCIntegrator::SetGenerator(GSLRngWrapper* r){ this->fRng = r; } 
      
void GSLMCIntegrator::SetType (MCIntegration::Type type)
{
      fType=type;
}

void GSLMCIntegrator::DoInitialize ( )
{
   //    initialize by setting  integration type 

   

   if (fWorkspace != 0) { 
      if (fDim == fWorkspace->NDim() && fType == fWorkspace->Type() ) 
         return; // can use previously existing ws

      // otherwise delete previously existing ws and create a new one
      delete fWorkspace; 
   }
 
   if(fType  ==  ROOT::Math::MCIntegration::kVEGAS)
   {
      
      fWorkspace = new GSLVegasIntegrationWorkspace(fDim);
	  
   }

   else if (fType ==  ROOT::Math::MCIntegration::kMISER) 
   {

      fWorkspace = new GSLMiserIntegrationWorkspace(fDim);
   }
   else if (fType ==  ROOT::Math::MCIntegration::kPLAIN)   
   {

      fWorkspace = new GSLPlainIntegrationWorkspace(fDim);
   }
   else 
   {
      std::cerr << "GSLIntegrator - Error: Unknown integration type" << std::endl;
      throw std::exception(); 
   }
}  


void GSLMCIntegrator::SetMode(MCIntegration::Mode mode)
{
   //   set integration mode for VEGAS method
   if(fType ==  ROOT::Math::MCIntegration::kVEGAS)
   {  fMode = mode; }

   else std::cerr << "Mode not matching integration type";
}



void GSLMCIntegrator::SetParameters(const VegasParameters &p)
{
   // set method parameters
   if (fType ==  MCIntegration::kVEGAS) 
   {
      GSLVegasIntegrationWorkspace * ws = dynamic_cast<GSLVegasIntegrationWorkspace *>(fWorkspace); 
      assert(ws != 0);
      ws->SetParameters(p);
   }
   else 
      std::cerr << "GSLIntegrator - Error: Parameters not mathing integration type" << std::endl;
}

void GSLMCIntegrator::SetParameters(const MiserParameters &p)
{
   // set method parameters
   if (fType ==  MCIntegration::kMISER) 
   {
      GSLMiserIntegrationWorkspace * ws = dynamic_cast<GSLMiserIntegrationWorkspace *>(fWorkspace); 
      assert(ws != 0); 
      ws->SetParameters(p);
   }
   else
      std::cerr << "GSLIntegrator - Error: Parameters not mathing integration type" << std::endl;
}

	


//----------- methods specific for VEGAS

double GSLMCIntegrator::Sigma()
{
   // returns the error sigma from the last iteration of the VEGAS algorithm
   if(fType == MCIntegration::kVEGAS)
   {
      GSLVegasIntegrationWorkspace * ws = dynamic_cast<GSLVegasIntegrationWorkspace *>(fWorkspace);
      assert (ws != 0);
      return ws->GetWS()->sigma;
   }
   else 
   {  
      std::cerr << "Parameter not mathcing integration type";
      return 0;
   }

}


/**
*/  
double GSLMCIntegrator::ChiSqr()
{
   //   returns chi-squared per degree of freedom for the estimate of the integral
   if(fType == MCIntegration::kVEGAS)
   {
      GSLVegasIntegrationWorkspace * ws = dynamic_cast<GSLVegasIntegrationWorkspace *>(fWorkspace);
      assert(ws != 0);
      return ws->GetWS()->chisq;
   }
   else 
   {
      std::cerr << "Parameter not mathcing integration type";
      return 0;
   }
}



bool GSLMCIntegrator::CheckFunction() 
{ 
   // internal method to check validity of GSL function pointer
   return true;
   /*
   // check if a function has been previously set.
   if (fFunction->IsValid()) return true; 
   fStatus = -1; fResult = 0; fError = 0;
   std::cerr << "GS:Integrator - Error : Function has not been specified " << std::endl; 
   return false; */
}
      
 
    

} // namespace Math
} // namespace ROOT



