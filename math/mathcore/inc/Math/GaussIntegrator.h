// @(#)root/mathcore:$Id$
// Authors: David Gonzalez Maline    01/2008

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for GaussIntegrator
//
// Created by: David Gonzalez Maline  : Wed Jan 16 2008
//

#ifndef ROOT_Math_GaussIntegrator
#define ROOT_Math_GaussIntegrator

#ifndef ROOT_Math_IFunction
#include "Math/IFunction.h"
#endif

#ifndef ROOT_Math_VirtualIntegrator
#include "Math/VirtualIntegrator.h"
#endif


namespace ROOT {
namespace Math {


//___________________________________________________________________________________________
/**
   User class for performing function integration.

   It will use the Gauss Method for function integration in a given interval.
   This class is implemented from TF1::Integral().

   @ingroup Integration

 */

class GaussIntegrator: public VirtualIntegratorOneDim {


public:

   /** Destructor */
   virtual ~GaussIntegrator();

   /** Default Constructor. */
   GaussIntegrator(double absTol = 0, double relTol = 0);


   /** Static function: set the fgAbsValue flag.
       By default TF1::Integral uses the original function value to compute the integral
       However, TF1::Moment, CentralMoment require to compute the integral
       using the absolute value of the function.
   */
   void AbsValue(bool flag);


   // Implementing VirtualIntegrator Interface

   /** Set the desired relative Error. */
   virtual void SetRelTolerance (double eps) { fEpsRel = eps; }

   /** This method is not implemented. */
   virtual void SetAbsTolerance (double eps) { fEpsAbs = eps; }

   /** Returns the result of the last Integral calculation. */
   double Result () const;

   /** Return the estimate of the absolute Error of the last Integral calculation. */
   double Error () const;

   /** return the status of the last integration - 0 in case of success */
   int Status () const;

   // Implementing VirtualIntegratorOneDim Interface

   /**
     Returns Integral of function between a and b.
     Based on original CERNLIB routine DGAUSS by Sigfried Kolbig
     converted to C++ by Rene Brun

     This function computes, to an attempted specified accuracy, the value
     of the integral.

    Method:
       For any interval [a,b] we define g8(a,b) and g16(a,b) to be the 8-point
       and 16-point Gaussian quadrature approximations to
   Begin_Latex
      I = #int^{b}_{a} f(x)dx
   End_Latex
      and define
   Begin_Latex
      r(a,b) = #frac{#||{g_{16}(a,b)-g_{8}(a,b)}}{1+#||{g_{16}(a,b)}}
   End_Latex
      Then,
   Begin_Latex
      G = #sum_{i=1}^{k}g_{16}(x_{i-1},x_{i})
   End_Latex
      where, starting with x0 = A and finishing with xk = B,
      the subdivision points xi(i=1,2,...) are given by
   Begin_Latex
      x_{i} = x_{i-1} + #lambda(B-x_{i-1})
   End_Latex
   Begin_Latex
      #lambda
   End_Latex
      is equal to the first member of the
      sequence 1,1/2,1/4,... for which r(xi-1, xi) < EPS.
      If, at any stage in the process of subdivision, the ratio
  Begin_Latex
      q = #||{#frac{x_{i}-x_{i-1}}{B-A}}
  End_Latex
      is so small that 1+0.005q is indistinguishable from 1 to
      machine accuracy, an error exit occurs with the function value
      set equal to zero.

   Accuracy:
      The user provides absolute and relative error bounds (epsrel and epsabs) and the
      algorithm will stop when the estimated error is less than the epsabs OR is less
      than |I| * epsrel.
      Unless there is severe cancellation of positive and negative values of
      f(x) over the interval [A,B], the relative error may be considered as
      specifying a bound on the <I>relative</I> error of I in the case
      |I|&gt;1, and a bound on the absolute error in the case |I|&lt;1. More
      precisely, if k is the number of sub-intervals contributing to the
      approximation (see Method), and if
      Begin_Latex
      I_{abs} = #int^{B}_{A} #||{f(x)}dx
      End_Latex
      then the relation
   Begin_Latex
    #frac{#||{G-I}}{I_{abs}+k} < EPS
   End_Latex
      will nearly always be true, provided the routine terminates without
      printing an error message. For functions f having no singularities in
      the closed interval [A,B] the accuracy will usually be much higher than
      this.

    Error handling:
      The requested accuracy cannot be obtained (see Method).
      The function value is set equal to zero.

    Note 1:
      Values of the function f(x) at the interval end-points A and B are not
      required. The subprogram may therefore be used when these values are
      undefined
   */
   double Integral (double a, double b);

   /** Returns Integral of function on an infinite interval.
      This function computes, to an attempted specified accuracy, the value of the integral:
   Begin_Latex
      I = #int^{#infinity}_{-#infinity} f(x)dx
   End_Latex
      Usage:
        In any arithmetic expression, this function has the approximate value
        of the integral I.

      The integral is mapped onto [0,1] using a transformation then integral computation is surrogated to DoIntegral.
   */
   double Integral ();

   /** Returns Integral of function on an upper semi-infinite interval.
      This function computes, to an attempted specified accuracy, the value of the integral:
   Begin_Latex
      I = #int^{#infinity}_{A} f(x)dx
   End_Latex
      Usage:
        In any arithmetic expression, this function has the approximate value
        of the integral I.
        - A: lower end-point of integration interval.

      The integral is mapped onto [0,1] using a transformation then integral computation is surrogated to DoIntegral.
   */
   double IntegralUp (double a);

   /** Returns Integral of function on a lower semi-infinite interval.
       This function computes, to an attempted specified accuracy, the value of the integral:
   Begin_Latex
      I = #int^{B}_{#infinity} f(x)dx
   End_Latex
      Usage:
         In any arithmetic expression, this function has the approximate value
         of the integral I.
         - B: upper end-point of integration interval.

      The integral is mapped onto [0,1] using a transformation then integral computation is surrogated to DoIntegral.
   */
   double IntegralLow (double b);


   /** Set integration function (flag control if function must be copied inside).
       \@param f Function to be used in the calculations.
   */
   void SetFunction (const IGenFunction &);

   /** This method is not implemented. */
   double Integral (const std::vector< double > &pts);

   /** This method is not implemented. */
   double IntegralCauchy (double a, double b, double c);

   ///  get the option used for the integration
   virtual ROOT::Math::IntegratorOneDimOptions Options() const;

   // set the options
   virtual void SetOptions(const ROOT::Math::IntegratorOneDimOptions & opt);

private:

   /**
      Integration surrugate method. Return integral of passed function in  interval [a,b]
      Derived class (like GaussLegendreIntegrator)  can re-implement this method to modify to use
      an improved algorithm
   */
   virtual double DoIntegral (double a, double b, const IGenFunction* func);

protected:

   static bool fgAbsValue;          // AbsValue used for the calculation of the integral
   double fEpsRel;                  // Relative error.
   double fEpsAbs;                  // Absolute error.
   bool fUsedOnce;                  // Bool value to check if the function was at least called once.
   double fLastResult;              // Result from the last stimation.
   double fLastError;               // Error from the last stimation.
   const IGenFunction* fFunction;   // Pointer to function used.

};

/**
   Auxillary inner class for mapping infinite and semi-infinite integrals
*/
class IntegrandTransform : public IGenFunction {
public:
   enum ESemiInfinitySign {kMinus = -1, kPlus = +1};
   IntegrandTransform(const IGenFunction* integrand);
   IntegrandTransform(const double boundary, ESemiInfinitySign sign, const IGenFunction* integrand);
   double operator()(double x) const;
   double DoEval(double x) const;
   IGenFunction* Clone() const;
private:
   ESemiInfinitySign fSign;
   const IGenFunction* fIntegrand;
   double fBoundary;
   bool fInfiniteInterval;
   double DoEval(double x, double boundary, int sign) const;
};



} // end namespace Math

} // end namespace ROOT

#endif /* ROOT_Math_GaussIntegrator */
