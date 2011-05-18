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


#ifndef ROOT_Math_GaussIntegrator
#include "Math/GaussIntegrator.h"
#endif


namespace ROOT {
namespace Math {

//___________________________________________________________________________________________
/**
   User class for performing function integration. 

   It will use the Gauss-Legendre Method for function integration in a given interval. 
   This class is implemented from TF1::Integral().

   @ingroup Integration
  
 */

class GaussLegendreIntegrator: public GaussIntegrator {
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

   /** Set the desired relative Error. */
   virtual void SetRelTolerance (double);

   /** This method is not implemented. */
   virtual void SetAbsTolerance (double);


   /** Returns the arrays x and w containing the abscissa and weight of
       the Gauss-Legendre n-point quadrature formula.
       
       Gauss-Legendre: W(x)=1 -1<x<1 
                       (j+1)P_{j+1} = (2j+1)xP_j-jP_{j-1}
   */
   void GetWeightVectors(double *x, double *w) const;

   int GetNumberPoints() const { return fNum; }

   /** 
       return number of function evaluations in calculating the integral 
       This is equivalent to the number of points
   */
   int NEval() const { return fNum; }


   ///  get the option used for the integration 
   virtual ROOT::Math::IntegratorOneDimOptions Options() const; 
   
   // set the options 
   virtual void SetOptions(const ROOT::Math::IntegratorOneDimOptions & opt); 

private:
   
   /**
      Integration surrugate method. Return integral of passed function in  interval [a,b]
      Reimplement method of GaussIntegrator using CalcGaussLegendreSamplingPoints
   */
   virtual double DoIntegral (double a, double b, const IGenFunction* func);

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

};

} // end namespace Math
   
} // end namespace ROOT

#endif /* ROOT_Math_GaussLegendreIntegrator */
