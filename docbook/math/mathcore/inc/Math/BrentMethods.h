#ifndef ROOT_Math_BrentMethods
#define ROOT_Math_BrentMethods

#ifndef ROOT_Math_IFunctionfwd
#include "Math/IFunctionfwd.h"
#endif


namespace ROOT {
namespace Math {

namespace BrentMethods { 

/**
     Grid search implementation, used to bracket the minimum and later
     use Brent's method with the bracketed interval
     The step of the search is set to (xmax-xmin)/fNpx
     type: 0-returns MinimumX
           1-returns Minimum
           2-returns MaximumX
           3-returns Maximum
           4-returns X corresponding to fy

*/

   double MinimStep(const IGenFunction* f, int type, double &xmin, double &xmax, double fy, int npx = 100, bool useLog = false);

   /**
      Finds a minimum of a function, if the function is unimodal  between xmin and xmax
      This method uses a combination of golden section search and parabolic interpolation
      Details about convergence and properties of this algorithm can be
      found in the book by R.P.Brent "Algorithms for Minimization Without Derivatives"
      or in the "Numerical Recipes", chapter 10.2
      convergence is reached using  tolerance = 2 *( epsrel * abs(x) + epsabs)
   
      type: 0-returns MinimumX
            1-returns Minimum
            2-returns MaximumX
            3-returns Maximum
            4-returns X corresponding to fy

      if ok=true the method has converged. 
      Maxiter returns the actual  number of iteration performed

   */

   double MinimBrent(const IGenFunction* f, int type, double &xmin, double &xmax, double xmiddle, double fy, bool &ok, int &niter, double epsabs = 1.E-8, double epsrel = 1.E-10, int maxiter = 100  );
   

} // end namespace BrentMethods
} // end namespace Math
} // ned namespace ROOT

#endif
