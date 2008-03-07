// @(#)root/mathcore:$Id$
// Authors: David Gonzalez Maline    01/2008 

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

#include "Math/GaussLegendreIntegrator.h"
#include <cmath>
#include <string.h>

namespace ROOT {
namespace Math {

GaussLegendreIntegrator::GaussLegendreIntegrator(int num, double eps)
{
   // Basic contructor of GaussLegendreIntegrator.

   fEpsilon = eps;
   fNum = num;
   fX = 0;
   fW = 0;
   fLastResult = fLastError = 0;
   fUsedOnce = false;
   fFunctionCopied = false;
   fFunction = 0;

   CalcGaussLegendreSamplingPoints();
}

GaussLegendreIntegrator::~GaussLegendreIntegrator()
{
   if ( fFunction != 0 && fFunctionCopied )
      delete fFunction;

   delete [] fX;
   delete [] fW;
}

void GaussLegendreIntegrator::SetNumberPoints(int num)
{
   // Set the number of points used in the calculation of the
   // integral
   fNum = num;
   CalcGaussLegendreSamplingPoints();
}

void GaussLegendreIntegrator::GetWeightVectors(double *x, double *w)
{
   // Returns the arrays x and w containing the abscissa and weight of
   // the Gauss-Legendre n-point quadrature formula.
   //
   // Gauss-Legendre: W(x)=1 -1<x<1 
   //                 (j+1)P_{j+1} = (2j+1)xP_j-jP_{j-1}

   memcpy(x, fX, fNum);
   memcpy(w, fW, fNum);
}


double GaussLegendreIntegrator::Integral(double a, double b)
{
   // Gauss-Legendre integral, see CalcGaussLegendreSamplingPoints

   if (fNum<=0 || fX == 0 || fW == 0)
      return 0;

   const double a0 = (b + a)/2;
   const double b0 = (b - a)/2;

   double xx[1];

   double result = 0.0;
   for (int i=0; i<fNum; i++)
   {
      xx[0] = a0 + b0*fX[i];
      result += fW[i] * (*fFunction)(xx);
   }

   fLastResult = result*b0;
   return fLastResult;
}
   

void GaussLegendreIntegrator::SetRelTolerance (double eps)
{
   // Set the desired relative Error

   fEpsilon = eps;
   CalcGaussLegendreSamplingPoints();
}

void GaussLegendreIntegrator::SetAbsTolerance (double)
{
   // Absolute Tolerance is not used in this class.

   MATH_ERROR_MSG("ROOT::Math::GausIntegratorOneDim", "There is no Absolute Tolerance!");
}

double GaussLegendreIntegrator::Result () const
{
   // Returns the result of the last Integral calculation

   if (!fUsedOnce)
      MATH_ERROR_MSG("ROOT::Math::GausIntegratorOneDim", "You must calculate the result at least once!");

   return fLastResult;
}

double GaussLegendreIntegrator::Error() const
{
   // This method is not implemented.

   return fLastError;
   // TODO
}

int GaussLegendreIntegrator::Status() const
{
   // This method is not implemented.

   return 0;
   // TODO
}

void GaussLegendreIntegrator::SetFunction (const IGenFunction & function, bool copy)
{
   // Set integration function (flag control if function must be
   // copied inside)

   if ( copy )
      fFunction = function.Clone();
   else
      fFunction = &function;

   fFunctionCopied = copy;
}


double GaussLegendreIntegrator::Integral ()
{
   // This method is not implemented.
   return 0.0;
}

double GaussLegendreIntegrator::IntegralUp (double /*a*/)
{
   // This method is not implemented.
   return 0.0;
}

double GaussLegendreIntegrator::IntegralLow (double /*b*/)
{
   // This method is not implemented.
   return 0.0;
}

double GaussLegendreIntegrator::Integral (const std::vector< double > &/*pts*/)
{
   // This method is not implemented.
   return 0.0;
}

double GaussLegendreIntegrator::IntegralCauchy (double /*a*/, double /*b*/, double /*c*/)
{
   // This method is not implemented.
   return 0.0;
}

void GaussLegendreIntegrator::CalcGaussLegendreSamplingPoints()
{
   // Type: unsafe but fast interface filling the arrays x and w (static method)
   //
   // Given the number of sampling points this routine fills the arrays x and w
   // of length num, containing the abscissa and weight of the Gauss-Legendre
   // n-point quadrature formula.
   //
   // Gauss-Legendre: W(x)=1  -1<x<1
   //                 (j+1)P_{j+1} = (2j+1)xP_j-jP_{j-1}
   //
   // num is the number of sampling points (>0)
   // x and w are arrays of size num
   // eps is the relative precision
   //
   // If num<=0 or eps<=0 no action is done.
   //
   // Reference: Numerical Recipes in C, Second Edition

   if (fNum<=0 || fEpsilon<=0)
      return;

   if ( fX == 0 )
      delete [] fX;

   if ( fW == 0 )
      delete [] fW;

   fX = new double[fNum];
   fW = new double[fNum];

   // The roots of symmetric is the interval, so we only have to find half of them
   const unsigned int m = (fNum+1)/2;

   double z, pp, p1,p2, p3;

   // Loop over the disired roots
   for (unsigned int i=0; i<m; i++) {
      z = std::cos(3.14159265358979323846*(i+0.75)/(fNum+0.5));

      // Starting with the above approximation to the i-th root, we enter
      // the main loop of refinement by Newton's method
      do {
         p1=1.0;
         p2=0.0;

         // Loop up the recurrence relation to get the Legendre
         // polynomial evaluated at z
         for (int j=0; j<fNum; j++)
         {
            p3 = p2;
            p2 = p1;
            p1 = ((2.0*j+1.0)*z*p2-j*p3)/(j+1.0);
         }
         // p1 is now the desired Legendre polynomial. We next compute pp, its
         // derivative, by a standard relation involving also p2, the polynomial
         // of one lower order
         pp = fNum*(z*p1-p2)/(z*z-1.0);
         // Newton's method
         z -= p1/pp;

      } while (std::fabs(p1/pp) > fEpsilon);

      // Put root and its symmetric counterpart
      fX[i]       = -z;
      fX[fNum-i-1] =  z;

      // Compute the weight and put its symmetric counterpart
      fW[i]       = 2.0/((1.0-z*z)*pp*pp);
      fW[fNum-i-1] = fW[i];
   }
}


}
}
