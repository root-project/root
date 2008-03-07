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

#include <Math/IFunction.h>
#include <Math/VirtualIntegrator.h>

namespace ROOT {
namespace Math {

class GaussLegendreIntegrator: public VirtualIntegratorOneDim {
public:
   GaussLegendreIntegrator(int num = 10 ,double eps=1e-12);
   virtual ~GaussLegendreIntegrator();
   
   void SetNumberPoints(int num);
   void GetWeightVectors(double *x, double *w);

   // Implementing VirtualIntegrator Interface
   void SetRelTolerance (double);
   void SetAbsTolerance (double);
   double Result () const;
   double Error () const;
   int Status () const;

   // Implementing VirtualIntegratorOneDim Interface
   double Integral (double a, double b);
   void SetFunction (const IGenFunction &, bool copy=false);
   double Integral ();
   double IntegralUp (double a);
   double IntegralLow (double b);
   double Integral (const std::vector< double > &pts);
   double IntegralCauchy (double a, double b, double c);

private:
   // Middle functions
   void CalcGaussLegendreSamplingPoints();

protected:
   int fNum;
   double* fX;
   double* fW;
   double fEpsilon;
   bool fUsedOnce;
   double fLastResult;
   double fLastError;
   const IGenFunction* fFunction;
   bool fFunctionCopied;

};

}
}
