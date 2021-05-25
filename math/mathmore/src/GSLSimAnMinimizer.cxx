// @(#)root/mathmore:$Id$
// Author: L. Moneta Wed Dec 20 17:16:32 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class GSLSimAnMinimizer

#include "Math/GSLSimAnMinimizer.h"
#include "Math/WrappedParamFunction.h"
#include "Math/Error.h"

#include "Math/MinimTransformFunction.h"
#include "Math/MultiNumGradFunction.h"   // needed to use transformation function
#include "Math/FitMethodFunction.h"
#include "Math/GenAlgoOptions.h"

#include <iostream>
#include <cassert>

namespace ROOT {

   namespace Math {



// GSLSimAnMinimizer implementation

GSLSimAnMinimizer::GSLSimAnMinimizer( int /* ROOT::Math::EGSLSimAnMinimizerType type */ ) :
   BasicMinimizer()
{
   // Constructor implementation : create GSLMultiFit wrapper object

   SetMaxIterations(100);
   SetPrintLevel(0);
}

GSLSimAnMinimizer::~GSLSimAnMinimizer () {
}


bool GSLSimAnMinimizer::Minimize() {
   // set initial parameters of the minimizer
   int debugLevel = PrintLevel();

   if (debugLevel >=1 ) std::cout <<"Minimize using GSLSimAnMinimizer " << std::endl;

   const ROOT::Math::IMultiGenFunction * function = ObjFunction();
   if (function == 0) {
      MATH_ERROR_MSG("GSLSimAnMinimizer::Minimize","Function has not been set");
      return false;
   }

   // vector of internal values (copied by default)
   unsigned int npar = NPar();
   std::vector<double> xvar;
   std::vector<double> steps(StepSizes(),StepSizes()+npar);

   // needed for the transformation
   MultiNumGradFunction * gradFunc = new MultiNumGradFunction( *function );
   gradFunc->SetOwnership();

   MinimTransformFunction * trFunc  = CreateTransformation(xvar, gradFunc );
   // ObjFunction() will return now the new transformed function

   if (trFunc) {
      // transform also the step sizes
      trFunc->InvStepTransformation(X(), StepSizes(), &steps[0]);
      steps.resize( trFunc->NDim() );
   }

   assert (xvar.size() == steps.size() );


#ifdef DEBUG
   for (unsigned int i = 0; i < npar ; ++i) {
      std::cout << "x  = " << xvar[i] << " steps " << steps[i] << "  x " << X()[i] << std::endl;
   }
   std::cout << "f(x) = " <<  (*ObjFunction())(&xvar.front() ) << std::endl;
   std::cout << "f(x) not transf = " <<  (*function)( X() ) << std::endl;
   if (trFunc) std::cout << "ftrans(x) = " <<  (*trFunc) (&xvar.front() ) << std::endl;
#endif

   // output vector
   std::vector<double> xmin(xvar.size() );


   int iret = fSolver.Solve(*ObjFunction(), &xvar.front(), &steps.front(), &xmin[0], (debugLevel > 1) );

   SetMinValue( (*ObjFunction())(&xmin.front() ) );

   SetFinalValues(&xmin.front());


   if (debugLevel >=1 ) {
      if (iret == 0)
         std::cout << "GSLSimAnMinimizer: Minimum Found" << std::endl;
      else
         std::cout << "GSLSimAnMinimizer: Error in solving" << std::endl;

      int pr = std::cout.precision(18);
      std::cout << "FVAL         = " << MinValue() << std::endl;
      std::cout.precision(pr);
      for (unsigned int i = 0; i < NDim(); ++i)
         std::cout << VariableName(i) << "\t  = " << X()[i] << std::endl;
   }


   return ( iret == 0) ? true : false;
}


unsigned int GSLSimAnMinimizer::NCalls() const {
   // return number of function calls
   const ROOT::Math::MinimTransformFunction * tfunc = dynamic_cast<const ROOT::Math::MinimTransformFunction *>(ObjFunction());
   const ROOT::Math::MultiNumGradFunction * f = 0;
   if (tfunc) f = dynamic_cast<const ROOT::Math::MultiNumGradFunction *>(tfunc->OriginalFunction());
   else
      f = dynamic_cast<const ROOT::Math::MultiNumGradFunction *>(ObjFunction());
   if (f) return f->NCalls();
   return 0;
}

ROOT::Math::MinimizerOptions  GSLSimAnMinimizer::Options() const {
   ROOT::Math::MinimizerOptions opt;
   opt.SetMinimizerType("GSLSimAn");
   // set dummy values since those are not used 
   opt.SetTolerance(-1);
   opt.SetPrintLevel(0);
   opt.SetMaxIterations(-1);
   opt.SetMaxFunctionCalls(0);
   opt.SetStrategy(-1);
   opt.SetErrorDef(0);
   opt.SetPrecision(0);
   opt.SetMinimizerAlgorithm("");
   
   const GSLSimAnParams & params = MinimizerParameters(); 

   ROOT::Math::GenAlgoOptions simanOpt;
   simanOpt.SetValue("n_tries",params.n_tries);
   simanOpt.SetValue("iters_fixed_T",params.iters_fixed_T);
   simanOpt.SetValue("step_size",params.step_size);
   simanOpt.SetValue("k",params.k);
   simanOpt.SetValue("t_initial",params.t_initial);
   simanOpt.SetValue("mu_t",params.mu_t);
   simanOpt.SetValue("t_min",params.t_min);

   opt.SetExtraOptions(simanOpt);
   return opt;
}

void GSLSimAnMinimizer::SetOptions(const ROOT::Math::MinimizerOptions & opt) {

   // get the specific siman options
   const ROOT::Math::IOptions * simanOpt = opt.ExtraOptions();
   if (!simanOpt) {
      MATH_WARN_MSG("GSLSimAnMinimizer::SetOptions", "No specific sim. annealing minimizer options are provided. No options are set");
      return;
   }
   GSLSimAnParams params; 
   simanOpt->GetValue("n_tries",params.n_tries);
   simanOpt->GetValue("iters_fixed_T",params.iters_fixed_T);
   simanOpt->GetValue("step_size",params.step_size);
   simanOpt->GetValue("k",params.k);
   simanOpt->GetValue("t_initial",params.t_initial);
   simanOpt->GetValue("mu_t",params.mu_t);
   simanOpt->GetValue("t_min",params.t_min);

   SetParameters(params);
} 


   } // end namespace Math

} // end namespace ROOT

