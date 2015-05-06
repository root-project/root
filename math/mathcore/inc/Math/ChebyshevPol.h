// @(#)root/mathcore:$Id$
// Author: L. Moneta,    11/2012

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Header file declaring functions for the evaluation of the Chebyshev  //
// polynomials and the ChebyshevPol class which can be used for         //
// creating a TF1.                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Math_ChebyshevPol
#define ROOT_Math_ChebyshevPol

#include <sys/types.h>
#include <cstring>

namespace ROOT {

   namespace Math {

      /// template recursive functions for defining evaluation of Chebyshev polynomials
      ///  T_n(x) and the series S(x) = Sum_i c_i* T_i(x)
      namespace Chebyshev {

         template<int N> double T(double x) {
            return  (2.0 * x * T<N-1>(x)) - T<N-2>(x);
         }

         template<> double T<0> (double );
         template<> double T<1> (double x);
         template<> double T<2> (double x);
         template<> double T<3> (double x);

         template<int N> double Eval(double x, const double * c) {
            return c[N]*T<N>(x) + Eval<N-1>(x,c);
         }

         template<> double Eval<0> (double , const double *c);
         template<> double Eval<1> (double x, const double *c);
         template<> double Eval<2> (double x, const double *c);
         template<> double Eval<3> (double x, const double *c);

      } // end namespace Chebyshev


      // implementation of Chebyshev polynomials using all coefficients
      // needed for creating TF1 functions
      inline double Chebyshev0(double , double c0) {
         return c0;
      }
      inline double Chebyshev1(double x, double c0, double c1) {
         return c0 + c1*x;
      }
      inline double Chebyshev2(double x, double c0, double c1, double c2) {
         return c0 + c1*x + c2*(2.0*x*x - 1.0);
      }
      inline double Chebyshev3(double x, double c0, double c1, double c2, double c3) {
         return c3*Chebyshev::T<3>(x) + Chebyshev2(x,c0,c1,c2);
      }
      inline double Chebyshev4(double x, double c0, double c1, double c2, double c3, double c4) {
         return c4*Chebyshev::T<4>(x) + Chebyshev3(x,c0,c1,c2,c3);
      }
      inline double Chebyshev5(double x, double c0, double c1, double c2, double c3, double c4, double c5) {
         return c5*Chebyshev::T<5>(x) + Chebyshev4(x,c0,c1,c2,c3,c4);
      }
      inline double Chebyshev6(double x, double c0, double c1, double c2, double c3, double c4, double c5, double c6) {
         return c6*Chebyshev::T<6>(x) + Chebyshev5(x,c0,c1,c2,c3,c4,c5);
      }
      inline double Chebyshev7(double x, double c0, double c1, double c2, double c3, double c4, double c5, double c6, double c7) {
         return c7*Chebyshev::T<7>(x) + Chebyshev6(x,c0,c1,c2,c3,c4,c5,c6);
      }
      inline double Chebyshev8(double x, double c0, double c1, double c2, double c3, double c4, double c5, double c6, double c7, double c8) {   
         return c8*Chebyshev::T<8>(x) + Chebyshev7(x,c0,c1,c2,c3,c4,c5,c6,c7);
      }
      inline double Chebyshev9(double x, double c0, double c1, double c2, double c3, double c4, double c5, double c6, double c7, double c8, double c9) {
         return c9*Chebyshev::T<9>(x) + Chebyshev8(x,c0,c1,c2,c3,c4,c5,c6,c7,c8);
      }
      inline double Chebyshev10(double x, double c0, double c1, double c2, double c3, double c4, double c5, double c6, double c7, double c8, double c9, double c10) {
         return c10*Chebyshev::T<10>(x) + Chebyshev9(x,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9);
      }


      // implementation of Chebyshev polynomial with run time parameter
      inline double ChebyshevN(unsigned int n, double x, const double * c) {

         if (n == 0) return Chebyshev0(x,c[0]);
         if (n == 1) return Chebyshev1(x,c[0],c[1]);
         if (n == 2) return Chebyshev2(x,c[0],c[1],c[2]);
         if (n == 3) return Chebyshev3(x,c[0],c[1],c[2],c[3]);
         if (n == 4) return Chebyshev4(x,c[0],c[1],c[2],c[3],c[4]);
         if (n == 5) return Chebyshev5(x,c[0],c[1],c[2],c[3],c[4],c[5]);

         /* do not use recursive formula
            (2.0 * x * Tn(n - 1, x)) - Tn(n - 2, x) ;
            which is too slow for large n
         */

         size_t i;
         double d1 = 0.0;
         double d2 = 0.0;

         // if not in range [-1,1]
         //double y = (2.0 * x - a - b) / (b - a);
         //double y = x;
         double y2 = 2.0 * x;

         for (i = n; i >= 1; i--)
         {
            double temp = d1;
            d1 = y2 * d1 - d2 + c[i];
            d2 = temp;
         }

         return x * d1 - d2 + c[0];
      }


      // implementation of Chebyshev Polynomial class
      // which can be used for building TF1 classes
      class ChebyshevPol {
      public:
         ChebyshevPol(unsigned int n) : fOrder(n) {}

         double operator() (const double *x, const double * coeff) {
            return ChebyshevN(fOrder, x[0], coeff);
         }
      private:
         unsigned int fOrder;
      };



   } // end namespace Math

} // end namespace ROOT



#endif  // ROOT_Math_Chebyshev
