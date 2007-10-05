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


#include <cassert> 
#include <iostream> 
#include <cmath> 
#include <vector>

namespace ROOT { 

   namespace Math { 


/**
   structure to hold objetive function and variables passed to gsl 
   sim annealing functions
 */
class GSLSimAnFunc { 
public: 

   GSLSimAnFunc(const ROOT::Math::IMultiGenFunction & func, const double * x) : 
      fX( std::vector<double>(x, x + func.NDim() ) ), 
      fScale( std::vector<double>(func.NDim() )), 
      fFunc(&func)
   {
      // set scale factors to 1
      fScale.assign(fScale.size(), 1.);
   }

   GSLSimAnFunc(const ROOT::Math::IMultiGenFunction & func, const double * x, const double * scale) : 
      fX( std::vector<double>(x, x + func.NDim() ) ), 
      fScale( std::vector<double>(scale, scale + func.NDim() ) ), 
      fFunc(&func) 
   {}

   double operator()() const { 
      return   (*fFunc)(&fX.front() ); 
   }

   void SetX(const double * x) { 
      std::copy(x, x+ fX.size(), fX.begin() );
   }

   template <class IT> 
   void SetX(IT begin, IT end) { 
      std::copy(begin, end, fX.begin() );
   }

   unsigned int NDim() const { return fX.size(); } 

   double X(unsigned int i) const { return fX[i]; }

   const std::vector<double> &  X() const { return fX; }

   double Scale(unsigned int i) const { return fScale[i]; }

   void SetX(unsigned int i, double x) { fX[i] = x; } 

   // use compiler generated  copy ctror and assignment operators 

private: 

   std::vector<double>  fX; 
   std::vector<double>  fScale; 
   const ROOT::Math::IMultiGenFunction * fFunc;
   
}; 


// static functions required by GSL 
namespace GSLSimAn { 


   double E( void * xp) { 
      // evaluate the energy given a state xp 
      GSLSimAnFunc * fx = reinterpret_cast<GSLSimAnFunc *> (xp); 
      assert (fx != 0);
      return (*fx)(); 
   }

   void Step( const gsl_rng * r, void * xp, double step_size) { 
      // change xp according to  the random step 
      GSLSimAnFunc * fx = reinterpret_cast<GSLSimAnFunc *> (xp); 
      assert (fx != 0);
      
      unsigned int ndim = fx->NDim();
      for (unsigned int i = 0; i < ndim; ++i) { 
         double u = gsl_rng_uniform(r); 
         double xold = fx->X(i);
         double sstep = step_size * fx->Scale(i);
         double xnew = u * 2 * sstep - sstep + xold; 
         fx->SetX(i, xnew);
      }
   }
 
   double Dist( void * xp, void * yp) { 
      // calculate the distance between two configuration
      GSLSimAnFunc * fx = reinterpret_cast<GSLSimAnFunc *> (xp); 
      GSLSimAnFunc * fy = reinterpret_cast<GSLSimAnFunc *> (yp); 
      
      assert (fx != 0);
      assert (fy != 0);
      
      // change step size 
      
      return std::fabs( (*fx)() - (*fy)() ); 
   }

   void Print(void * xp) { 
      // print a configuration xp
      GSLSimAnFunc * fx = reinterpret_cast<GSLSimAnFunc *> (xp); 
      assert (fx != 0);
      
      std::cout << "x = ( "; 
      for (unsigned int i = 0; i < fx->NDim()-1; ++i) { 
         std::cout << fx->X(i) << " , "; 
      }
      std::cout << fx->X(fx->NDim()-1) << " )\n";
      std::cout << "f(x) = " << (*fx)()  << std::endl;
   }   

// static function to pass to GSL copy - create and destroy the object 
   
   void Copy( void * source, void * dest) { 
      GSLSimAnFunc * fx = reinterpret_cast<GSLSimAnFunc *> (source); 
      assert (fx != 0);
      GSLSimAnFunc * gx = reinterpret_cast<GSLSimAnFunc *> (dest); 
      assert (gx != 0);
      gx->operator=(*fx); 
   }

   void * CopyCtor( void * xp) { 
      GSLSimAnFunc * fx = reinterpret_cast<GSLSimAnFunc *> (xp); 
      assert (fx != 0);
      return static_cast<void *> ( new GSLSimAnFunc(*fx) ); 
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



// function for solving 
int GSLSimAnnealing::Solve(const ROOT::Math::IMultiGenFunction & func, const double * x0, const double * scale, double * xmin, bool debug) { 
   // solve the simulated annealing problem given starting point and function
   
   gsl_rng * r = gsl_rng_alloc(gsl_rng_mt19937); 

   // initial conditions
   GSLSimAnFunc   fx(func, x0, scale); 

   gsl_siman_params_t simanParams; 
   // parameters for the simulated annealing

   simanParams.n_tries =    200;     /* how many points to try for each step */
   simanParams.iters_fixed_T =  10;  /* how many iterations at each temperature? */ 
   simanParams.step_size =   10;     /* max step size in the random walk */
   // the following parameters are for the Boltzmann distribution */
   simanParams.k = 1.0; 
   simanParams.t_initial =  0.002; 
   simanParams.mu_t =  1.005; 
   simanParams.t_min = 2.0E-6;


   if (debug) 
      gsl_siman_solve(r, &fx, &GSLSimAn::E, &GSLSimAn::Step, &GSLSimAn::Dist, 
                   &GSLSimAn::Print, &GSLSimAn::Copy, &GSLSimAn::CopyCtor , &GSLSimAn::Destroy, 0, simanParams );

   else 
      gsl_siman_solve(r, &fx, &GSLSimAn::E, &GSLSimAn::Step, &GSLSimAn::Dist, 
                   0, &GSLSimAn::Copy, &GSLSimAn::CopyCtor , &GSLSimAn::Destroy, 0, simanParams );


   // copy value of the minimum
   std::copy(fx.X().begin(), fx.X().end(), xmin);

   return 0; 
}




   } // end namespace Math

} // end namespace ROOT

