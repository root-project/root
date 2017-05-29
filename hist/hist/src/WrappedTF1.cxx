// @(#)root/mathmore:$Id$
// Author: L. Moneta Wed Sep  6 09:52:26 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// implementation file for class WrappedTF1 and WrappedMultiTF1

#include "Math/WrappedTF1.h"
#include "Math/WrappedMultiTF1.h"
#include "TClass.h"   // needed to copy the TF1 pointer

#include <cmath>


namespace ROOT {

   namespace Math {

      namespace Internal {
         double DerivPrecision(double eps)
         {
            static double gEPs = 0.001; // static value for epsilon used in derivative calculations
            if (eps > 0)
               gEPs = eps;
            return gEPs;
         }

         TF1 *CopyTF1Ptr(const TF1 *funcToCopy)
         {
            TF1 *fnew = (TF1 *) funcToCopy->IsA()->New();
            funcToCopy->Copy(*fnew);
            return fnew;
         }
      }

      WrappedTF1::WrappedTF1(TF1 &f)  :
         fLinear(false),
         fPolynomial(false),
         fFunc(&f),
         fX()
         //fParams(f.GetParameters(),f.GetParameters()+f.GetNpar())
      {
         // constructor from a TF1 function pointer.

         // init the pointers for CINT
         //if (fFunc->GetMethodCall() )  fFunc->InitArgs(fX, &fParams.front() );
         if (fFunc->GetMethodCall())  fFunc->InitArgs(fX, 0);
         // distinguish case of polynomial functions and linear functions
         if (fFunc->GetNumber() >= 300 && fFunc->GetNumber() < 310) {
            fLinear = true;
            fPolynomial = true;
         }
         // check that in case function is linear the linear terms are not zero
         if (fFunc->IsLinear()) {
            int ip = 0;
            fLinear = true;
            while (fLinear && ip < fFunc->GetNpar())  {
               fLinear &= (fFunc->GetLinearPart(ip) != 0) ;
               ip++;
            }
         }
      }

      WrappedTF1::WrappedTF1(const WrappedTF1 &rhs) :
         BaseFunc(),
         BaseGradFunc(),
         IGrad(),
         fLinear(rhs.fLinear),
         fPolynomial(rhs.fPolynomial),
         fFunc(rhs.fFunc),
         fX()
         //fParams(rhs.fParams)
      {
         // copy constructor
         fFunc->InitArgs(fX, 0);
      }

      WrappedTF1 &WrappedTF1::operator = (const WrappedTF1 &rhs)
      {
         // assignment operator
         if (this == &rhs) return *this;  // time saving self-test
         fLinear = rhs.fLinear;
         fPolynomial = rhs.fPolynomial;
         fFunc = rhs.fFunc;
         fFunc->InitArgs(fX, 0);
         //fParams = rhs.fParams;
         return *this;
      }

      void  WrappedTF1::ParameterGradient(double x, const double *par, double *grad) const
      {
         // evaluate the derivative of the function with respect to the parameters
         if (!fLinear) {
            // need to set parameter values
            fFunc->SetParameters(par);
            // no need to call InitArgs (it is called in TF1::GradientPar)
            fFunc->GradientPar(&x, grad, GetDerivPrecision());
         } else {
            unsigned int np = NPar();
            for (unsigned int i = 0; i < np; ++i)
               grad[i] = DoParameterDerivative(x, par, i);
         }
      }

      double WrappedTF1::DoDerivative(double  x) const
      {
         // return the function derivatives w.r.t. x

         // parameter are passed as non-const in Derivative
         //double * p =  (fParams.size() > 0) ? const_cast<double *>( &fParams.front()) : 0;
         return  fFunc->Derivative(x, (double *) 0, GetDerivPrecision());
      }

      double WrappedTF1::DoParameterDerivative(double x, const double *p, unsigned int ipar) const
      {
         // evaluate the derivative of the function with respect to the parameters
         //IMPORTANT NOTE: TF1::GradientPar returns 0 for fixed parameters to avoid computing useless derivatives
         //  BUT the TLinearFitter wants to have the derivatives also for fixed parameters.
         //  so in case of fLinear (or fPolynomial) a non-zero value will be returned for fixed parameters

         if (! fLinear) {
            fFunc->SetParameters(p);
            return fFunc->GradientPar(ipar, &x, GetDerivPrecision());
         } else if (fPolynomial) {
            // case of polynomial function (no parameter dependency)
            return std::pow(x, static_cast<int>(ipar));
         } else {
            // case of general linear function (built in TFormula with ++ )
            const TFormula *df = dynamic_cast<const TFormula *>(fFunc->GetLinearPart(ipar));
            assert(df != 0);
            fX[0] = x;
            // hack since TFormula::EvalPar is not const
            return (const_cast<TFormula *>(df))->Eval(x) ;     // derivatives should not depend on parameters since func is linear
         }
      }

      void WrappedTF1::SetDerivPrecision(double eps)
      {
         ::ROOT::Math::Internal::DerivPrecision(eps);
      }

      double WrappedTF1::GetDerivPrecision()
      {
         return ::ROOT::Math::Internal::DerivPrecision(-1);
      }

   } // end namespace Math

} // end namespace ROOT


