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

#ifndef ROOT_Math_GaussLegendreIntegrator
#define ROOT_Math_GaussLegendreIntegrator

#include <Math/IFunction.h>
#include <Math/VirtualIntegrator.h>

namespace ROOT {
namespace Math {

//___________________________________________________________________________________________
/**
   User class for performing function integration. 

   It will use the Gauss-Legendre Method for function integration in a given interval. 
   This class is implemented from TF1::Integral().

   @ingroup Integration
  
 */

class GaussLegendreIntegrator: public VirtualIntegratorOneDim {
public:

   /** Basic contructor of GaussLegendreIntegrator. 
       \@param num Number of desired points to calculate the integration.
       \@param eps Desired relative error.
   */
   GaussLegendreIntegrator(int num = 10 ,double eps=1e-12);

   /** Default Destructor */
   virtual ~GaussLegendreIntegrator();
   
   /** Set the number of points used in the calculation of the
       integral */
   void SetNumberPoints(int num);

   /** Returns the arrays x and w containing the abscissa and weight of
       the Gauss-Legendre n-point quadrature formula.
       
       Gauss-Legendre: W(x)=1 -1<x<1 
                       (j+1)P_{j+1} = (2j+1)xP_j-jP_{j-1}
   */
   void GetWeightVectors(double *x, double *w);

   // Implementing VirtualIntegrator Interface

   /** Set the desired relative Error. */
   void SetRelTolerance (double);

   /** Absolute Tolerance is not used in this class. */
   void SetAbsTolerance (double);

   /** Returns the result of the last integral calculation. */
   double Result () const;

   /** Return the estimate of the absolute Error of the last Integral calculation. */
   double Error () const;

   /** This method is not implemented. */
   int Status () const;

   // Implementing VirtualIntegratorOneDim Interface

   /** Gauss-Legendre integral, see CalcGaussLegendreSamplingPoints. */
   double Integral (double a, double b);

   /** Set integration function (flag control if function must be copied inside).
       \@param f Function to be used in the calculations.
       \@param copy Indicates whether the function has to be copied.
   */
   void SetFunction (const IGenFunction &, bool copy=false);

   /** This method is not implemented. */
   double Integral ();

   /** This method is not implemented. */
   double IntegralUp (double a);

   /**This method is not implemented. */
   double IntegralLow (double b);

   /** This method is not implemented. */
   double Integral (const std::vector< double > &pts);

   /** This method is not implemented. */
   double IntegralCauchy (double a, double b, double c);   

private:
   // Middle functions

   /** 
      Type: unsafe but fast interface filling the arrays x and w (static method)
     
      Given the number of sampling points this routine fills the arrays x and w
      of length num, containing the abscissa and weight of the Gauss-Legendre
      n-point quadrature formula.
     
      Gauss-Legendre: W(x)=1  -1<x<1
                      (j+1)P_{j+1} = (2j+1)xP_j-jP_{j-1}
     
      num is the number of sampling points (>0)
      x and w are arrays of size num
      eps is the relative precision
     
      If num<=0 or eps<=0 no action is done.
     
      Reference: Numerical Recipes in C, Second Edition
   */
   void CalcGaussLegendreSamplingPoints();

protected:
   int fNum;                         // Number of points used in the stimation of the integral.
   double* fX;                       // Abscisa of the points used.
   double* fW;                       // Weights of the points used.
   double fEpsilon;                  // Desired relative error.
   bool fUsedOnce;                   // Bool value to check if the function was at least called once.
   double fLastResult;               // Result from the last stimation.
   double fLastError;                // Error from the last stimation.
   const IGenFunction* fFunction;    // Pointer to function used.
   bool fFunctionCopied;             // Bool value to check if the function was copied when set.

};

} // end namespace Math
   
} // end namespace ROOT

#endif /* ROOT_Math_GaussLegendreIntegrator */
