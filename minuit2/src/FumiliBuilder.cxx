// @(#)root/minuit2:$Name:  $:$Id: FumiliBuilder.cxx,v 1.1 2005/11/29 14:43:31 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/FumiliBuilder.h"
#include "Minuit2/FumiliStandardMaximumLikelihoodFCN.h"
#include "Minuit2/GradientCalculator.h"
#include "Minuit2/MinimumState.h"
#include "Minuit2/MinimumError.h"
#include "Minuit2/FunctionGradient.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnLineSearch.h"
#include "Minuit2/MinimumSeed.h"
#include "Minuit2/MnFcn.h"
#include "Minuit2/MnMachinePrecision.h"
#include "Minuit2/MnPosDef.h"
#include "Minuit2/MnParabolaPoint.h"
#include "Minuit2/LaSum.h"
#include "Minuit2/LaProd.h"
#include "Minuit2/MnStrategy.h"
#include "Minuit2/MnHesse.h"


#include "Minuit2/MnPrint.h" 

//#define DEBUG 1
#ifdef DEBUG
#ifndef WARNINGMSG
#define WARNINGMSG
#endif
#endif


namespace ROOT {

   namespace Minuit2 {




double inner_product(const LAVector&, const LAVector&);

FunctionMinimum FumiliBuilder::Minimum(const MnFcn& fcn, const GradientCalculator& gc, const MinimumSeed& seed, const MnStrategy& strategy, unsigned int maxfcn, double edmval) const {


  edmval *= 0.0001;


#ifdef DEBUG
  std::cout<<"FumiliBuilder convergence when edm < "<<edmval<<std::endl;
#endif

  if(seed.Parameters().Vec().size() == 0) {
    return FunctionMinimum(seed, fcn.Up());
  }


//   double edm = Estimator().Estimate(seed.Gradient(), seed.Error());
  double edm = seed.State().Edm();

  FunctionMinimum min(seed, fcn.Up() );

  if(edm < 0.) {
    std::cout<<"FumiliBuilder: initial matrix not pos.def."<<std::endl;
    //assert(!seed.Error().IsPosDef());
    return min;
  }

  std::vector<MinimumState> result;
//   result.reserve(1);
  result.reserve(8);

  result.push_back( seed.State() );

  // do actual iterations


  // try first with a maxfxn = 50% of maxfcn 
  // FUmili in principle needs much less iterations
  int maxfcn_eff = int(0.5*maxfcn);
  int ipass = 0;
  double edmprev = 1;
  
  do { 

    
    min = Minimum(fcn, gc, seed, result, maxfcn_eff, edmval);
    // second time check for validity of function Minimum 
    if (ipass > 0) { 
      if(!min.IsValid()) {
#ifdef WARNINGMSG
	std::cout<<"FumiliBuilder: FunctionMinimum is invalid."<<std::endl;
#endif
	return min;
      }
    }

    // resulting edm of minimization
    edm = result.back().Edm();

    if( (strategy.Strategy() == 2) || 
	(strategy.Strategy() == 1 && min.Error().Dcovar() > 0.05) ) {

#ifdef DEBUG
      std::cout<<"MnMigrad will verify convergence and Error matrix. "<< std::endl;
      std::cout<<"dcov is =  "<<  min.Error().Dcovar() << std::endl;
#endif

      MinimumState st = MnHesse(strategy)(fcn, min.State(), min.Seed().Trafo());
      result.push_back( st );
    
      // check edm 
      edm = st.Edm();
#ifdef DEBUG
      std::cout << "edm after Hesse calculation " << edm << std::endl;
#endif

      // break the loop if edm is NOT getting smaller 
      if (ipass > 0 && edm >= edmprev) { 
#ifdef WARNINGMSG
	std::cout << "FumiliBuilder: Exit iterations, no improvements after Hesse. edm is  " << edm << " previous " << edmprev << std::endl;
#endif
	break; 
      } 
      if (edm > edmval) { 
#ifdef DEBUG
	std::cout << "FumiliBuilder: Tolerance is not sufficient - edm is " << edm << " requested " << edmval 
		  << " continue the minimization" << std::endl;
#endif
      }
    }

    // end loop on iterations
    // ? need a maximum here (or max of function calls is enough ? ) 
    // continnue iteration (re-calculate funciton Minimum if edm IS NOT sufficient) 
    // no need to check that hesse calculation is done (if isnot done edm is OK anyway)
    // count the pass to exit second time when function Minimum is invalid
    // increase by 20% maxfcn for doing some more tests
    if (ipass == 0) maxfcn_eff = maxfcn;
    ipass++;
    edmprev = edm; 

  }  while (edm > edmval );



  // Add latest state (Hessian calculation)
  min.Add( result.back() );

  return min;

}

FunctionMinimum FumiliBuilder::Minimum(const MnFcn& fcn, const GradientCalculator& gc, const MinimumSeed& seed, std::vector<MinimumState>& result, unsigned int maxfcn, double edmval) const {



  /*
    Three options were possible:
    
    1) create two parallel and completely separate hierarchies, in which case
    the FumiliMinimizer would NOT inherit from ModularFunctionMinimizer, 
    FumiliBuilder would not inherit from MinimumBuilder etc

    2) Use the inheritance (base classes of ModularFunctionMinimizer,
    MinimumBuilder etc), but recreate the member functions Minimize() and 
    Minimum() respectively (naming them for example minimize2() and 
    minimum2()) so that they can take FumiliFCNBase as Parameter instead FCNBase
    (otherwise one wouldn't be able to call the Fumili-specific methods).

    3) Cast in the daughter classes derived from ModularFunctionMinimizer,
    MinimumBuilder.

    The first two would mean to duplicate all the functionality already existent,
    which is a very bad practice and Error-prone. The third one is the most
    elegant and effective solution, where the only constraint is that the user
    must know that he has to pass a subclass of FumiliFCNBase to the FumiliMinimizer 
    and not just a subclass of FCNBase.
    BTW, the first two solutions would have meant to recreate also a parallel
    structure for MnFcn...
  **/
  //  const FumiliFCNBase* tmpfcn =  dynamic_cast<const FumiliFCNBase*>(&(fcn.Fcn()));

  const MnMachinePrecision& prec = seed.Precision();

  const MinimumState & initialState = result.back();

  double edm = initialState.Edm();


#ifdef DEBUG
  std::cout << "\n\nDEBUG FUMILI Builder  \nSEED State: "  
	    << " Parameter " << seed.State().Vec()       
	    << " Gradient " << seed.Gradient().Vec() 
	    << " Inv Hessian " << seed.Error().InvHessian()  
	    << " edm = " << seed.State().Edm() 
            << " maxfcn = " << maxfcn 
            << " tolerance = " << edmval 
	    << std::endl; 
#endif


  // iterate until edm is small enough or max # of iterations reached
  edm *= (1. + 3.*seed.Error().Dcovar());
  MnLineSearch lsearch;
  MnAlgebraicVector step(seed.Gradient().Vec().size());

  // initial lambda Value
  double lambda = 0.001; 
  //double lambda = 0.0; 


  do {   
    
//     const MinimumState& s0 = result.back();
    MinimumState s0 = result.back();
    
    step = -1.*s0.Error().InvHessian()*s0.Gradient().Vec();


#ifdef DEBUG
    std::cout << "\n\n---> Iteration - " << result.size() 
	      << "\nFval = " << s0.Fval() << " numOfCall = " << fcn.NumOfCalls() 
	      << "\nInternal Parameter values " << s0.Vec() 
	      << " Newton step " << step << std::endl; 
#endif
 
    double gdel = inner_product(step, s0.Gradient().Grad());
    if(gdel > 0.) {
      std::cout<<"FumiliBuilder: matrix not pos.def."<<std::endl;
      std::cout<<"gdel > 0: "<<gdel<<std::endl;
      MnPosDef psdf;
      s0 = psdf(s0, prec);
      step = -1.*s0.Error().InvHessian()*s0.Gradient().Vec();
      gdel = inner_product(step, s0.Gradient().Grad());
      std::cout<<"gdel: "<<gdel<<std::endl;
      if(gdel > 0.) {
	result.push_back(s0);
	return FunctionMinimum(seed, result, fcn.Up());
      }
    }


//     MnParabolaPoint pp = lsearch(fcn, s0.Parameters(), step, gdel, prec);

//     if(fabs(pp.y() - s0.Fval()) < prec.Eps()) {
//       std::cout<<"FumiliBuilder: no improvement"<<std::endl;
//       break; //no improvement
//     }


//     MinimumParameters p(s0.Vec() + pp.x()*step, pp.y());

    // if taking a full step 

    // take a full step

    MinimumParameters p(s0.Vec() + step,  fcn( s0.Vec() + step ) );

    // check that taking the full step does not do crazy things 
    // in that case do a line search (use a cut if 10)  
    if ( p.Fval() > 10*s0.Fval()  ) {
      MnParabolaPoint pp = lsearch(fcn, s0.Parameters(), step, gdel, prec);

      if(fabs(pp.y() - s0.Fval()) < prec.Eps()) {
	//std::cout<<"FumiliBuilder: no improvement"<<std::endl;
	break; //no improvement
      }
      p =  MinimumParameters(s0.Vec() + pp.x()*step, pp.y() );
    }

#ifdef DEBUG
    std::cout << "Before Gradient " << fcn.NumOfCalls() << std::endl; 
#endif
        
    FunctionGradient g = gc(p, s0.Gradient());
 
#ifdef DEBUG   
    std::cout << "After Gradient " << fcn.NumOfCalls() << std::endl; 
#endif

    //FunctionGradient g = gc(s0.Parameters(), s0.Gradient()); 


    // move Error updator after Gradient since the Value is cached inside

    MinimumError e = ErrorUpdator().Update(s0, p, gc, lambda);


    edm = Estimator().Estimate(g, s0.Error());

    
#ifdef DEBUG
    std::cout << "Updated new point: \n " 
              << " Parameter " << p.Vec()       
	      << " Gradient " << g.Vec() 
	      << " InvHessian " << e.Matrix() 
	      << " Hessian " << e.Hessian() 
	      << " edm = " << edm << std::endl << std::endl;
#endif

    if(edm < 0.) {
#ifdef WARNINGMSG
      std::cout<<"FumiliBuilder: matrix not pos.def."<<std::endl;
      std::cout<<"edm < 0"<<std::endl;
#endif
      MnPosDef psdf;
      s0 = psdf(s0, prec);
      edm = Estimator().Estimate(g, s0.Error());
      if(edm < 0.) {
	result.push_back(s0);
	return FunctionMinimum(seed, result, fcn.Up());
      }
    } 
 
    // check lambda according to step 
    if ( p.Fval() < s0.Fval()  ) 
      // fcn is decreasing along the step
      lambda *= 0.1;
    else 
      lambda *= 10; 

#ifdef DEBUG
    std::cout <<  " finish iteration- " << result.size() << " lambda =  "  << lambda << " f1 = " << p.Fval() << " f0 = " << s0.Fval() << " num of calls = " << fcn.NumOfCalls() << " edm " << edm << std::endl; 
#endif
  

    //std::cout << "FumiliBuilder DEBUG e.Matrix()" << e.Matrix() << std::endl;
    //std::cout << "DEBUG FumiliBuilder e.Hessian()" << e.Hessian() << std::endl;

    result.push_back(MinimumState(p, e, g, edm, fcn.NumOfCalls())); 

    //std::cout << "FumiliBuilder DEBUG " << FunctionMinimum(seed, result, fcn.Up()) << std::endl;
 

    edm *= (1. + 3.*e.Dcovar());

    /**std::cout << "DEBUG FumiliBuilder edm: " << edm << " edmval: " << edmval <<
      " fcn.numOfCalls: " << fcn.NumOfCalls() << " maxfcn: " << maxfcn << 
      " condition: " << (edm > edmval && fcn.NumOfCalls() < maxfcn) <<std::endl;
    */


  } while(edm > edmval && fcn.NumOfCalls() < maxfcn);
  


  if(fcn.NumOfCalls() >= maxfcn) {
#ifdef WARNINGMSG
    std::cout<<"FumiliBuilder: call limit exceeded."<<std::endl;
#endif
    return FunctionMinimum(seed, result, fcn.Up(), FunctionMinimum::MnReachedCallLimit());
  }

  if(edm > edmval) {
    if(edm < fabs(prec.Eps2()*result.back().Fval())) {
#ifdef WARNINGMSG
      std::cout<<"FumiliBuilder: machine accuracy limits further improvement."<<std::endl;
#endif
      return FunctionMinimum(seed, result, fcn.Up());
    } else if(edm < 10.*edmval) {
      return FunctionMinimum(seed, result, fcn.Up());
    } else {
#ifdef WARNINGMSG
      std::cout<<"FumiliBuilder: finishes without convergence."<<std::endl;
      std::cout<<"FumiliBuilder: edm= "<<edm<<" requested: "<<edmval<<std::endl;
#endif
      return FunctionMinimum(seed, result, fcn.Up(), FunctionMinimum::MnAboveMaxEdm());
    }
  }
//   std::cout<<"result.back().Error().Dcovar()= "<<result.back().Error().Dcovar()<<std::endl;

#ifdef DEBUG
  std::cout << "Exiting succesfully FumiliBuilder \n" 
	    << "NFCalls = " << fcn.NumOfCalls() 
	    << "\nFval = " <<  result.back().Fval() 
	    << "\nedm = " << edm << " requested = " << edmval << std::endl; 
#endif

  return FunctionMinimum(seed, result, fcn.Up());
}

  }  // namespace Minuit2

}  // namespace ROOT
