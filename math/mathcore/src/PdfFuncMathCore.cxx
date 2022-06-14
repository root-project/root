// @(#)root/mathcore:$Id$
// Authors: Andras Zsenei & Lorenzo Moneta   06/2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/



#include "Math/Math.h"
#include "Math/SpecFuncMathCore.h"
#include <limits>


namespace ROOT {
namespace Math {

   double landau_pdf(double x, double xi, double x0) {
      // LANDAU pdf : algorithm from CERNLIB G110 denlan
      // same algorithm is used in GSL

      static double p1[5] = {0.4259894875,-0.1249762550, 0.03984243700, -0.006298287635,   0.001511162253};
      static double q1[5] = {1.0         ,-0.3388260629, 0.09594393323, -0.01608042283,    0.003778942063};

      static double p2[5] = {0.1788541609, 0.1173957403, 0.01488850518, -0.001394989411,   0.0001283617211};
      static double q2[5] = {1.0         , 0.7428795082, 0.3153932961,   0.06694219548,    0.008790609714};

      static double p3[5] = {0.1788544503, 0.09359161662,0.006325387654, 0.00006611667319,-0.000002031049101};
      static double q3[5] = {1.0         , 0.6097809921, 0.2560616665,   0.04746722384,    0.006957301675};

      static double p4[5] = {0.9874054407, 118.6723273,  849.2794360,   -743.7792444,      427.0262186};
      static double q4[5] = {1.0         , 106.8615961,  337.6496214,    2016.712389,      1597.063511};

      static double p5[5] = {1.003675074,  167.5702434,  4789.711289,    21217.86767,     -22324.94910};
      static double q5[5] = {1.0         , 156.9424537,  3745.310488,    9834.698876,      66924.28357};

      static double p6[5] = {1.000827619,  664.9143136,  62972.92665,    475554.6998,     -5743609.109};
      static double q6[5] = {1.0         , 651.4101098,  56974.73333,    165917.4725,     -2815759.939};

      static double a1[3] = {0.04166666667,-0.01996527778, 0.02709538966};

      static double a2[2] = {-1.845568670,-4.284640743};

      if (xi <= 0) return 0;
      double v = (x - x0)/xi;
      double u, ue, us, denlan;
      if (v < -5.5) {
         u   = std::exp(v+1.0);
         if (u < 1e-10) return 0.0;
         ue  = std::exp(-1/u);
         us  = std::sqrt(u);
         denlan = 0.3989422803*(ue/us)*(1+(a1[0]+(a1[1]+a1[2]*u)*u)*u);
      } else if(v < -1) {
         u   = std::exp(-v-1);
         denlan = std::exp(-u)*std::sqrt(u)*
            (p1[0]+(p1[1]+(p1[2]+(p1[3]+p1[4]*v)*v)*v)*v)/
            (q1[0]+(q1[1]+(q1[2]+(q1[3]+q1[4]*v)*v)*v)*v);
      } else if(v < 1) {
         denlan = (p2[0]+(p2[1]+(p2[2]+(p2[3]+p2[4]*v)*v)*v)*v)/
            (q2[0]+(q2[1]+(q2[2]+(q2[3]+q2[4]*v)*v)*v)*v);
      } else if(v < 5) {
         denlan = (p3[0]+(p3[1]+(p3[2]+(p3[3]+p3[4]*v)*v)*v)*v)/
            (q3[0]+(q3[1]+(q3[2]+(q3[3]+q3[4]*v)*v)*v)*v);
      } else if(v < 12) {
         u   = 1/v;
         denlan = u*u*(p4[0]+(p4[1]+(p4[2]+(p4[3]+p4[4]*u)*u)*u)*u)/
            (q4[0]+(q4[1]+(q4[2]+(q4[3]+q4[4]*u)*u)*u)*u);
      } else if(v < 50) {
         u   = 1/v;
         denlan = u*u*(p5[0]+(p5[1]+(p5[2]+(p5[3]+p5[4]*u)*u)*u)*u)/
            (q5[0]+(q5[1]+(q5[2]+(q5[3]+q5[4]*u)*u)*u)*u);
      } else if(v < 300) {
         u   = 1/v;
         denlan = u*u*(p6[0]+(p6[1]+(p6[2]+(p6[3]+p6[4]*u)*u)*u)*u)/
            (q6[0]+(q6[1]+(q6[2]+(q6[3]+q6[4]*u)*u)*u)*u);
      } else {
         u   = 1/(v-v*std::log(v)/(v+1));
         denlan = u*u*(1+(a2[0]+a2[1]*u)*u);
      }
      return denlan/xi;

   }


} // namespace Math
} // namespace ROOT





