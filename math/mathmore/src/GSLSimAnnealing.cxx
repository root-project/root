// @(#)root/mathmore:$Id$
// Author: L. Moneta Thu Jan 25 11:13:48 2007

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class GSLSimAnnealing

#include "Math/GSLSimAnnealing.h"

#include "gsl/gsl_siman.h"

#include "Math/IFunction.h"
#include "Math/GSLRndmEngines.h"
#include "GSLRngWrapper.h"


#include <cassert>
#include <iostream>
#include <cmath>
#include <vector>

namespace ROOT {

   namespace Math {


// implementation of GSLSimAnFunc

GSLSimAnFunc::GSLSimAnFunc(const ROOT::Math::IMultiGenFunction & func, const double * x) :
   fX( std::vector<double>(x, x + func.NDim() ) ),
   fScale( std::vector<double>(func.NDim() )),
   fFunc(&func)
{
   // set scale factors to 1
   fScale.assign(fScale.size(), 1.);
}

GSLSimAnFunc::GSLSimAnFunc(const ROOT::Math::IMultiGenFunction & func, const double * x, const double * scale) :
   fX( std::vector<double>(x, x + func.NDim() ) ),
   fScale( std::vector<double>(scale, scale + func.NDim() ) ),
   fFunc(&func)
{}


double GSLSimAnFunc::Energy() const {
   // evaluate the energy
   return   (*fFunc)(&fX.front() );
}

void GSLSimAnFunc::Step(const GSLRandomEngine & random, double maxstep) {
   // x  -> x + Random[-step,step]     for each coordinate
   unsigned int ndim = NDim();
   for (unsigned int i = 0; i < ndim; ++i) {
      double urndm = random();
      double sstep = maxstep * fScale[i];
      fX[i] +=  2 * sstep * urndm - sstep;
   }
}


double GSLSimAnFunc::Distance(const GSLSimAnFunc & f) const {
   // calculate the distance with respect onother configuration
   const std::vector<double> & x = fX;
   const std::vector<double> & y = f.X();
   unsigned int n = x.size();
   assert (n == y.size());
   if (n > 1) {
      double d2 = 0;
      for (unsigned int i = 0; i < n; ++i)
         d2 += ( x[i] - y[i] ) * ( x[i] - y[i] );
      return std::sqrt(d2);
   }
   else
      // avoid doing a sqrt for 1 dim
      return std::abs( x[0] - y[0] );
}

void GSLSimAnFunc::Print() {
   // print the position  x in standard std::ostream
   // GSL prints also niter-  ntrials - temperature and then the energy and energy min value (from 1.10)
   std::cout << "\tx = ( ";
   unsigned n = NDim();
   for (unsigned int i = 0; i < n-1; ++i) {
      std::cout << fX[i] << " , ";
   }
   std::cout << fX.back() << " )\t";
   // energy us printed by GSL (and also end-line)
   std::cout << "E  / E_best = ";   // GSL print then E and E best
}

GSLSimAnFunc &  GSLSimAnFunc::FastCopy(const GSLSimAnFunc & rhs) {
   // copy only the information which is changed during the process
   // in this case only the x values
   std::copy(rhs.fX.begin(), rhs.fX.end(), fX.begin() );
   return *this;
}



// definition and implementations of the static functions required by GSL

namespace GSLSimAn {


   double E( void * xp) {
      // evaluate the energy given a state xp
      GSLSimAnFunc * fx = reinterpret_cast<GSLSimAnFunc *> (xp);
      assert (fx != 0);
      return fx->Energy();
   }

   void Step( const gsl_rng * r, void * xp, double step_size) {
      // change xp according to  the random step
      GSLSimAnFunc * fx = reinterpret_cast<GSLSimAnFunc *> (xp);
      assert (fx != 0);
      // create GSLRandomEngine class
      // cast away constness (we make sure we don't delete (no call to Terminate() )
      GSLRngWrapper  rng(const_cast<gsl_rng *>(r));
      GSLRandomEngine random(&rng);
      // wrapper classes random and rng must exist during call to Step()
      fx->Step(random, step_size);
   }

   double Dist( void * xp, void * yp) {
      // calculate the distance between two configuration
      GSLSimAnFunc * fx = reinterpret_cast<GSLSimAnFunc *> (xp);
      GSLSimAnFunc * fy = reinterpret_cast<GSLSimAnFunc *> (yp);

      assert (fx != 0);
      assert (fy != 0);
      return fx->Distance(*fy);
   }

   void Print(void * xp) {
      // print the position  xp
      // GSL prints also first niter-  ntrials - temperature and then the energy
      GSLSimAnFunc * fx = reinterpret_cast<GSLSimAnFunc *> (xp);
      assert (fx != 0);
      fx->Print();
   }

// static function to pass to GSL copy - create and destroy the object

   void Copy( void * source, void * dest) {
      GSLSimAnFunc * fx = reinterpret_cast<GSLSimAnFunc *> (source);
      assert (fx != 0);
      GSLSimAnFunc * gx = reinterpret_cast<GSLSimAnFunc *> (dest);
      assert (gx != 0);
      gx->FastCopy(*fx);
   }

   void * CopyCtor( void * xp) {
      GSLSimAnFunc * fx = reinterpret_cast<GSLSimAnFunc *> (xp);
      assert (fx != 0);
      return static_cast<void *> ( fx->Clone() );
   }

   void Destroy( void * xp) {
      GSLSimAnFunc * fx = reinterpret_cast<GSLSimAnFunc *> (xp);
      assert (fx != 0);
      delete fx;
   }

}

// implementation of GSLSimAnnealing class


GSLSimAnnealing::GSLSimAnnealing()
{
   // Default constructor implementation.
}



// function for solving (from a Genfunction interface)

int GSLSimAnnealing::Solve(const ROOT::Math::IMultiGenFunction & func, const double * x0, const double * scale, double * xmin, bool debug) {
   // solve the simulated annealing problem given starting point and objective function interface


   // initial conditions
   GSLSimAnFunc   fx(func, x0, scale);

   int iret =  Solve(fx, debug);

   if (iret == 0) {
      // copy value of the minimum in xmin
      std::copy(fx.X().begin(), fx.X().end(), xmin);
   }
   return iret;

}

int GSLSimAnnealing::Solve(GSLSimAnFunc & fx, bool debug) {
   // solve the simulated annealing problem given starting point and GSLSimAnfunc object

   gsl_rng * r = gsl_rng_alloc(gsl_rng_mt19937);



   gsl_siman_params_t simanParams;

   // parameters for the simulated annealing
   // copy them in GSL structure

   simanParams.n_tries =        fParams.n_tries;     /* how many points to try for each step */
   simanParams.iters_fixed_T =  fParams.iters_fixed_T;  /* how many iterations at each temperature? */
   simanParams.step_size =      fParams.step_size;     /* max step size in the random walk */
   // the following parameters are for the Boltzmann distribution */
   simanParams.k =              fParams.k;
   simanParams.t_initial =      fParams.t_initial;
   simanParams.mu_t =           fParams.mu;
   simanParams.t_min =          fParams.t_min;


   if (debug)
      gsl_siman_solve(r, &fx, &GSLSimAn::E, &GSLSimAn::Step, &GSLSimAn::Dist,
                   &GSLSimAn::Print, &GSLSimAn::Copy, &GSLSimAn::CopyCtor , &GSLSimAn::Destroy, 0, simanParams );

   else
      gsl_siman_solve(r, &fx, &GSLSimAn::E, &GSLSimAn::Step, &GSLSimAn::Dist,
                   0, &GSLSimAn::Copy, &GSLSimAn::CopyCtor , &GSLSimAn::Destroy, 0, simanParams );

   return 0;

}




   } // end namespace Math

} // end namespace ROOT

