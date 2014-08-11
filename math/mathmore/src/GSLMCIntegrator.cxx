// @(#)root/mathmore:$Id$
// Author: Magdalena Slawinska  08/2007

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2007 ROOT Foundation,  CERN/PH-SFT                   *
  *                                                                    *
  * This library is free software; you can redistribute it and/or      *
  * modify it under the terms of the GNU General Public License        *
  * as published by the Free Software Foundation; either version 2     *
  * of the License, or (at your option) any later version.             *
  *                                                                    *
  * This library is distributed in the hope that it will be useful,    *
  * but WITHOUT ANY WARRANTY; without even the implied warranty of     *
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU   *
  * General Public License for more details.                           *
  *                                                                    *
  * You should have received a copy of the GNU General Public License  *
  * along with this library (see file COPYING); if not, write          *
  * to the Free Software Foundation, Inc., 59 Temple Place, Suite      *
  * 330, Boston, MA 02111-1307 USA, or contact the author.             *
  *                                                                    *
  **********************************************************************/
//
// implementation file for class GSLMCIntegrator
// Author: Magdalena Slawinska
//
//

#include "Math/IFunctionfwd.h"
#include "Math/IFunction.h"
#include "Math/Error.h"
#include <vector>

#include "GSLMonteFunctionWrapper.h"

#include "Math/GSLMCIntegrator.h"
#include "GSLMCIntegrationWorkspace.h"
#include "GSLRngWrapper.h"

#include <algorithm>
#include <functional>
#include <ctype.h>   // need to use c version of tolower defined here


#include "gsl/gsl_monte_vegas.h"
#include "gsl/gsl_monte_miser.h"
#include "gsl/gsl_monte_plain.h"



namespace ROOT {
namespace Math {



// constructors

// GSLMCIntegrator::GSLMCIntegrator():
//    fResult(0),fError(0),fStatus(-1),
//    fWorkspace(0),
//    fFunction(0)
// {
//    // constructor of GSL MCIntegrator.Vegas MC is set as default integration type
//    //set random number generator
//    fRng = new GSLRngWrapper();
//    fRng->Allocate();
//    // use the default options
//    ROOT::Math::IntegratorMultiDimOptions opts;  // this create the default options
//    SetOptions(opts);
// }


GSLMCIntegrator::GSLMCIntegrator(MCIntegration::Type type, double absTol, double relTol, unsigned int calls):
   fType(type),
   fDim(0),
   fCalls((calls > 0)  ? calls : IntegratorMultiDimOptions::DefaultNCalls()),
   fAbsTol((absTol >0) ? absTol : IntegratorMultiDimOptions::DefaultAbsTolerance() ),
   fRelTol((relTol >0) ? relTol : IntegratorMultiDimOptions::DefaultRelTolerance() ),
   fResult(0),fError(0),fStatus(-1),
   fWorkspace(0),
   fFunction(0)
{
   // constructor of GSL MCIntegrator using enumeration as type
   SetType(type);
   //set random number generator
   fRng = new GSLRngWrapper();
   fRng->Allocate();
   // use the default options for the needed extra parameters
   // use the default options for the needed extra parameters
   if (fType == MCIntegration::kVEGAS) {
      IOptions * opts = IntegratorMultiDimOptions::FindDefault("VEGAS");
      if (opts != 0) SetParameters( VegasParameters(*opts) );
   }
   else  if (fType == MCIntegration::kMISER) {
      IOptions * opts = IntegratorMultiDimOptions::FindDefault("MISER");
      if (opts != 0)  SetParameters( MiserParameters(*opts) );
   }

}

GSLMCIntegrator::GSLMCIntegrator(const char * type, double absTol, double relTol, unsigned int calls):
   fDim(0),
   fCalls(calls),
   fAbsTol(absTol),
   fRelTol(relTol),
   fResult(0),fError(0),fStatus(-1),
   fWorkspace(0),
   fFunction(0)
{
   // constructor of GSL MCIntegrator. Vegas MC is set as default integration type if type == 0
   SetTypeName(type);

   //set random number generator
   fRng = new GSLRngWrapper();
   fRng->Allocate();
   // use the default options for the needed extra parameters
   if (fType == MCIntegration::kVEGAS) {
      IOptions * opts = IntegratorMultiDimOptions::FindDefault("VEGAS");
      if (opts != 0) SetParameters( VegasParameters(*opts) );
   }
   else  if (fType == MCIntegration::kMISER) {
      IOptions * opts = IntegratorMultiDimOptions::FindDefault("MISER");
      if (opts != 0)  SetParameters( MiserParameters(*opts) );
   }

}



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
   // create workspace according to the type
   fType=type;
   if (fWorkspace != 0) {
      if (type == fWorkspace->Type() ) return;
      delete fWorkspace;  // delete because is a different type
      fWorkspace = 0;
   }
   //create Workspace according to type
   if (type == MCIntegration::kPLAIN) {
      fWorkspace = new  GSLPlainIntegrationWorkspace();
   }
   else if (type == MCIntegration::kMISER) {
      fWorkspace = new  GSLMiserIntegrationWorkspace();
   }
   else {
       // default: use  VEGAS
      if (type != MCIntegration::kVEGAS) {
         MATH_WARN_MSG("GSLMCIntegration","Invalid integration type : use Vegas as default");
         fType =  MCIntegration::kVEGAS;
      }
      fWorkspace = new  GSLVegasIntegrationWorkspace();
   }
}

void GSLMCIntegrator::SetTypeName(const char * type)
{
   // set the integration type using a string
   std::string typeName = (type!=0) ? type : "VEGAS";
   if (type == 0) MATH_INFO_MSG("GSLMCIntegration::SetTypeName","use default Vegas integrator method");
   std::transform(typeName.begin(), typeName.end(), typeName.begin(), (int(*)(int)) toupper );

   MCIntegration::Type integType =  MCIntegration::kVEGAS;  // default

   if (typeName == "PLAIN") {
      integType =  MCIntegration::kPLAIN;
   }
   else if (typeName == "MISER") {
      integType =  MCIntegration::kMISER;
   }
   else if (typeName != "VEGAS")  {
      MATH_WARN_MSG("GSLMCIntegration::SetTypeName","Invalid integration type : use Vegas as default");
   }

   // create the fWorkspace object
   if (integType != fType) SetType(integType);
}


void GSLMCIntegrator::SetMode(MCIntegration::Mode mode)
{
   //   set integration mode for VEGAS method
   if(fType ==  ROOT::Math::MCIntegration::kVEGAS)
   {
      GSLVegasIntegrationWorkspace * ws = dynamic_cast<GSLVegasIntegrationWorkspace *>(fWorkspace);
      assert(ws != 0);
      if(mode == MCIntegration::kIMPORTANCE) ws->GetWS()->mode = GSL_VEGAS_MODE_IMPORTANCE;
      else if(mode == MCIntegration::kSTRATIFIED) ws->GetWS()->mode = GSL_VEGAS_MODE_STRATIFIED;
      else if(mode == MCIntegration::kIMPORTANCE_ONLY) ws->GetWS()->mode = GSL_VEGAS_MODE_IMPORTANCE_ONLY;
   }

   else std::cerr << "Mode not matching integration type";
}

void GSLMCIntegrator::SetOptions(const ROOT::Math::IntegratorMultiDimOptions & opt)
{
   //   set integration options
   SetTypeName(opt.Integrator().c_str() );
   SetAbsTolerance( opt.AbsTolerance() );
   SetRelTolerance( opt.RelTolerance() );
   fCalls = opt.NCalls();

   //std::cout << fType << "   " <<  MCIntegration::kVEGAS << std::endl;

   // specific options
   ROOT::Math::IOptions * extraOpt = opt.ExtraOptions();
   if (extraOpt) {
      if (fType ==  MCIntegration::kVEGAS ) {
         VegasParameters p(*extraOpt);
         SetParameters(p);
      }
      else if (fType ==  MCIntegration::kMISER ) {
         MiserParameters p(fDim); // if possible pass dimension
         p = (*extraOpt);
         SetParameters(p);
      }
      else {
         MATH_WARN_MSG("GSLMCIntegrator::SetOptions","Invalid options set for the chosen integration type - ignore them");
      }
   }
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
      MATH_ERROR_MSG("GSLIntegrator::SetParameters"," Parameters not matching integration type");
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
      MATH_ERROR_MSG("GSLIntegrator::SetParameters"," Parameters not matching integration type");
}


void GSLMCIntegrator::DoInitialize ( )
{
   //    initialize by setting  integration type

   if (fWorkspace == 0) return;
   if (fDim == fWorkspace->NDim() && fType == fWorkspace->Type() )
      return; // can use previously existing ws

   // otherwise clear workspace
   fWorkspace->Clear();
   // and create a new one
   fWorkspace->Init(fDim);
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

const char * GSLMCIntegrator::GetTypeName() const {
   if (fType == MCIntegration::kVEGAS) return "VEGAS";
   if (fType == MCIntegration::kMISER) return "MISER";
   if (fType == MCIntegration::kPLAIN) return "PLAIN";
   return "UNDEFINED";
}

ROOT::Math::IntegratorMultiDimOptions  GSLMCIntegrator::Options() const {
   IOptions * extraOpts = ExtraOptions();
   ROOT::Math::IntegratorMultiDimOptions opt(extraOpts);
   opt.SetAbsTolerance(fAbsTol);
   opt.SetRelTolerance(fRelTol);
   opt.SetNCalls(fCalls);
   opt.SetWKSize(0);
   opt.SetIntegrator(GetTypeName() );
   return opt;
}

ROOT::Math::IOptions *  GSLMCIntegrator::ExtraOptions() const {
   if (!fWorkspace) return 0;
   return fWorkspace->Options();
}


} // namespace Math
} // namespace ROOT



