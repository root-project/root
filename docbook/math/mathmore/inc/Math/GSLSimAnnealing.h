// @(#)root/mathmore:$Id$
// Author: L. Moneta Thu Jan 25 11:13:48 2007

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
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

// Header file for class GSLSimAnnealing

#ifndef ROOT_Math_GSLSimAnnealing
#define ROOT_Math_GSLSimAnnealing

#include "Math/IFunctionfwd.h"

#include <vector>

namespace ROOT { 

   namespace Math { 

      class GSLRandomEngine;

//_____________________________________________________________________________
/**
   GSLSimAnFunc class description. 
   Interface class for the  objetive function to be used in simulated annealing 
   If user wants to re-implement some of the methods (like the one defining the metric) which are used by the
   the simulated annealing algorithm must build a user derived class.
   NOTE: Derived classes must re-implement the assignment and copy constructor to call them of the parent class

   @ingroup MultiMin
 */
class GSLSimAnFunc { 
public: 

   /**
      construct from an interface of a multi-dimensional function
    */
   GSLSimAnFunc(const ROOT::Math::IMultiGenFunction & func, const double * x);  

   /**
      construct from an interface of a multi-dimensional function
      Use optionally a scale factor (for each coordinate) which can  be used to scale the step sizes
      (this is used for example by the minimization algorithm)
    */
   GSLSimAnFunc(const ROOT::Math::IMultiGenFunction & func, const double * x, const double * scale);  
   
protected: 

   /**
      derived classes might need to re-define completly the class 
    */
   GSLSimAnFunc() : 
      fFunc(0)
   {}

public: 


   /// virtual distructor (no operations) 
   virtual ~GSLSimAnFunc() { } //


   /**
      fast copy method called by GSL simuated annealing internally
      copy only the things which have been changed 
      must be re-implemented by derived classes if needed
   */
   virtual GSLSimAnFunc & FastCopy(const GSLSimAnFunc & f); 


   /**
      clone method. Needs to be re-implemented by the derived classes for deep  copying 
    */
   virtual GSLSimAnFunc * Clone() const { 
      return new GSLSimAnFunc(*this); 
   }

   /**
      evaluate the energy ( objective function value) 
      re-implement by derived classes if needed to be modified
    */
   virtual double Energy() const; 

   /**
      change the x[i] value using a random value urndm generated between [0,1]
      up to a maximum value maxstep
      re-implement by derived classes if needed to be modified      
    */
   virtual void Step(const GSLRandomEngine & r, double maxstep);

   /**
      calculate the distance (metric) between  this one and another configuration
      Presently a cartesian metric is used. 
      re-implement by derived classes if needed to be modified      
    */
   virtual double Distance(const GSLSimAnFunc & func) const; 

   /**
      print the position in the standard output ostream 
      GSL prints in addition n iteration, n function calls, temperature and energy
      re-implement by derived classes if necessary
    */
   virtual void Print();

   /** 
       change the x values (used by sim annealing to take a step)
    */ 
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

//_____________________________________________________
/** 
    structure holding the simulated annealing parameters

   @ingroup MultiMin
*/
struct GSLSimAnParams {

   // constructor with some default values
   GSLSimAnParams() {  
      n_tries =    200;    
      iters_fixed_T =  10; 
      step_size =   10;    
      // the following parameters are for the Boltzmann distribution */
      k = 1.0; 
      t_initial =  0.002; 
      mu =  1.005; 
      t_min = 2.0E-6;
   }
 

   int n_tries;            // number of points to try for each step
   int iters_fixed_T;      // number of iterations at each temperature
   double step_size;       // max step size used in random walk
   /// parameters for the Boltzman distribution
   double k; 
   double t_initial; 
   double mu; 
   double t_min; 
}; 

//___________________________________________________________________________
/** 
   GSLSimAnnealing class for performing  a simulated annealing search of 
   a multidimensional function

   @ingroup MultiMin
*/ 
class GSLSimAnnealing {

public: 

   /** 
      Default constructor
   */ 
   GSLSimAnnealing ();

   /** 
      Destructor (no operations)
   */ 
   ~GSLSimAnnealing ()  {}  

private:
   // usually copying is non trivial, so we make this unaccessible

   /** 
      Copy constructor
   */ 
   GSLSimAnnealing(const GSLSimAnnealing &) {} 

   /** 
      Assignment operator
   */ 
   GSLSimAnnealing & operator = (const GSLSimAnnealing & rhs)  {
      if (this == &rhs) return *this;  // time saving self-test
      return *this;
   }

public: 


   /**
      solve the simulated annealing given a multi-dim function, the initial vector parameters 
      and a vector containing the scaling factors for the parameters 
   */
   int Solve(const ROOT::Math::IMultiGenFunction & func, const double * x0, const double * scale, double * xmin, bool debug = false); 

   /**
      solve the simulated annealing given a GSLSimAnFunc object 
      The object will contain the initial state at the beginning and the final minimum state at the end
   */
   int Solve(GSLSimAnFunc & func, bool debug = false); 


   GSLSimAnParams & Params() { return fParams; }
   const GSLSimAnParams & Params() const { return fParams; }
   

protected: 


private: 

   GSLSimAnParams fParams; // parameters for GSLSimAnnealig

}; 

   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_GSLSimAnnealing */
