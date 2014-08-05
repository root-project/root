// @(#)root/mathcore:$Id$
// Authors: Andras Zsenei & Lorenzo Moneta   06/2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

#if defined(__sun) || defined(__sgi) || defined(_WIN32) || defined(_AIX)
#define NOT_HAVE_TGAMMA
#endif


#include "SpecFuncCephes.h"


#include <cmath>
#include <limits>

#ifndef PI
#define PI       3.14159265358979323846264338328      /* pi */
#endif

// use cephes for functions which are also in C99
#define USE_CEPHES

// platforms not implemening C99
// #if defined(__sun) || defined(__sgi) || defined(_WIN32) || defined(_AIX)
// #define USE_CEPHES
// #endif


namespace ROOT {
namespace Math {





// (26.x.21.2) complementary error function

double erfc(double x) {


#ifdef USE_CEPHES
   // use cephes implementation
   return ROOT::Math::Cephes::erfc(x);
#else
   return ::erfc(x);
#endif

}


// (26.x.21.1) error function

double erf(double x) {


#ifdef USE_CEPHES
   return ROOT::Math::Cephes::erf(x);
#else
   return ::erf(x);
#endif


}




double lgamma(double z) {

#ifdef USE_CEPHES
   return ROOT::Math::Cephes::lgam(z);
#else
   return ::lgamma(z);
#endif

}




// (26.x.18) gamma function

double tgamma(double x) {

#ifdef USE_CEPHES
   return ROOT::Math::Cephes::gamma(x);
#else
   return ::tgamma(x);
#endif

}

double inc_gamma( double a, double x) {
   return ROOT::Math::Cephes::igam(a,x);
}

double inc_gamma_c( double a, double x) {
   return ROOT::Math::Cephes::igamc(a,x);
}


// [5.2.1.3] beta function
// (26.x.19)

double beta(double x, double y) {
   return std::exp(lgamma(x)+lgamma(y)-lgamma(x+y));
}

double inc_beta( double x, double a, double b) {
   return ROOT::Math::Cephes::incbet(a,b,x);
}

// Sine integral
// Translated from CERNLIB SININT (C336) by B. List 29.4.2010

double sinint(double x) {

   static const double z1 = 1, r8 = z1/8;

   static const double pih = PI/2;

   static const double s[16] = {
     +1.95222097595307108, -0.68840423212571544,
     +0.45518551322558484, -0.18045712368387785,
     +0.04104221337585924, -0.00595861695558885,
     +0.00060014274141443, -0.00004447083291075,
     +0.00000253007823075, -0.00000011413075930,
     +0.00000000418578394, -0.00000000012734706,
     +0.00000000000326736, -0.00000000000007168,
     +0.00000000000000136, -0.00000000000000002};

   static const double p[29] = {
     +0.96074783975203596, -0.03711389621239806,
     +0.00194143988899190, -0.00017165988425147,
     +0.00002112637753231, -0.00000327163256712,
     +0.00000060069211615, -0.00000012586794403,
     +0.00000002932563458, -0.00000000745695921,
     +0.00000000204105478, -0.00000000059502230,
     +0.00000000018322967, -0.00000000005920506,
     +0.00000000001996517, -0.00000000000699511,
     +0.00000000000253686, -0.00000000000094929,
     +0.00000000000036552, -0.00000000000014449,
     +0.00000000000005851, -0.00000000000002423,
     +0.00000000000001025, -0.00000000000000442,
     +0.00000000000000194, -0.00000000000000087,
     +0.00000000000000039, -0.00000000000000018,
     +0.00000000000000008};

   static const double q[25] = {
     +0.98604065696238260, -0.01347173820829521,
     +0.00045329284116523, -0.00003067288651655,
     +0.00000313199197601, -0.00000042110196496,
     +0.00000006907244830, -0.00000001318321290,
     +0.00000000283697433, -0.00000000067329234,
     +0.00000000017339687, -0.00000000004786939,
     +0.00000000001403235, -0.00000000000433496,
     +0.00000000000140273, -0.00000000000047306,
     +0.00000000000016558, -0.00000000000005994,
     +0.00000000000002237, -0.00000000000000859,
     +0.00000000000000338, -0.00000000000000136,
     +0.00000000000000056, -0.00000000000000024,
     +0.00000000000000010};

   double h;
   if (std::abs(x) <= 8) {
      double y = r8*x;
      h = 2*y*y-1;
      double alfa = h+h;
      double b0 = 0;
      double b1 = 0;
      double b2 = 0;
      for (int i = 15; i >= 0; --i) {
        b0 = s[i]+alfa*b1-b2;
        b2 = b1;
        b1 = b0;
      }
      h = y*(b0-b2);
   } else {
      double r = 1/x;
      h = 128*r*r-1;
      double alfa = h+h;
      double b0 = 0;
      double b1 = 0;
      double b2 = 0;
      for (int i = 28; i >= 0; --i) {
         b0 = p[i]+alfa*b1-b2;
         b2 = b1;
         b1 = b0;
      }
      double pp = b0-h*b2;
      b1 = 0;
      b2 = 0;
      for (int i = 24; i >= 0; --i) {
        b0 = q[i]+alfa*b1-b2;
        b2 = b1;
        b1 = b0;
      }
      h = (x > 0 ? pih : -pih)-r*(r*pp*std::sin(x)+(b0-h*b2)*std::cos(x));
   }
   return h;
}

// Real part of the cosine integral
// Translated from CERNLIB COSINT (C336) by B. List 29.4.2010

double cosint(double x) {

   static const double z1 = 1, r32 = z1/32;

   static const double ce = 0.57721566490153286;

   static const double c[16] = {
     +1.94054914648355493, +0.94134091328652134,
     -0.57984503429299276, +0.30915720111592713,
     -0.09161017922077134, +0.01644374075154625,
     -0.00197130919521641, +0.00016925388508350,
     -0.00001093932957311, +0.00000055223857484,
     -0.00000002239949331, +0.00000000074653325,
     -0.00000000002081833, +0.00000000000049312,
     -0.00000000000001005, +0.00000000000000018};

   static const double p[29] = {
     +0.96074783975203596, -0.03711389621239806,
     +0.00194143988899190, -0.00017165988425147,
     +0.00002112637753231, -0.00000327163256712,
     +0.00000060069211615, -0.00000012586794403,
     +0.00000002932563458, -0.00000000745695921,
     +0.00000000204105478, -0.00000000059502230,
     +0.00000000018322967, -0.00000000005920506,
     +0.00000000001996517, -0.00000000000699511,
     +0.00000000000253686, -0.00000000000094929,
     +0.00000000000036552, -0.00000000000014449,
     +0.00000000000005851, -0.00000000000002423,
     +0.00000000000001025, -0.00000000000000442,
     +0.00000000000000194, -0.00000000000000087,
     +0.00000000000000039, -0.00000000000000018,
     +0.00000000000000008};

   static const double q[25] = {
     +0.98604065696238260, -0.01347173820829521,
     +0.00045329284116523, -0.00003067288651655,
     +0.00000313199197601, -0.00000042110196496,
     +0.00000006907244830, -0.00000001318321290,
     +0.00000000283697433, -0.00000000067329234,
     +0.00000000017339687, -0.00000000004786939,
     +0.00000000001403235, -0.00000000000433496,
     +0.00000000000140273, -0.00000000000047306,
     +0.00000000000016558, -0.00000000000005994,
     +0.00000000000002237, -0.00000000000000859,
     +0.00000000000000338, -0.00000000000000136,
     +0.00000000000000056, -0.00000000000000024,
     +0.00000000000000010};

      double h = 0;
      if(x == 0) {
         h = - std::numeric_limits<double>::infinity();
      } else if (std::abs(x) <= 8) {
         h = r32*x*x-1;
         double alfa = h+h;
         double b0 = 0;
         double b1 = 0;
         double b2 = 0;
         for (int i = 15; i >= 0; --i) {
            b0 = c[i]+alfa*b1-b2;
            b2 = b1;
            b1 = b0;
         }
         h = ce+std::log(std::abs(x))-b0+h*b2;
      } else {
         double r = 1/x;
         h = 128*r*r-1;
         double alfa = h+h;
         double b0 = 0;
         double b1 = 0;
         double b2 = 0;
         for (int i = 28; i >= 0; --i) {
            b0 = p[i]+alfa*b1-b2;
            b2 = b1;
            b1 = b0;
         }
         double pp = b0-h*b2;
         b1 = 0;
         b2 = 0;
         for (int i = 24; i >= 0; --i) {
            b0 = q[i]+alfa*b1-b2;
            b2 = b1;
            b1 = b0;
         }
         h = r*((b0-h*b2)*std::sin(x)-r*pp*std::cos(x));
      }
      return h;
}




} // namespace Math
} // namespace ROOT





