// @(#)root/mathmore:$Id$
// Authors: B. List 29.4.2010


 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2004 ROOT Foundation,  CERN/PH-SFT                   *
  *                                                                    *
  * This library is free software; you can redistribute it and/or      *
  * modify it under the terms of the GNU General Public License        *
  * as published by the Free Software Foundation; either version 2     *
  * of the License, or (at your option) any later version.             *
  *                                                                    *
  * This library is distributed in the hope that it will be useful,    *
  * but WITHOUT ANY WARRANTY; without even the implied warranty of     *
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU   *
  * General Public License for more details.                           *
  *                                                                    *
  * You should have received a copy of the GNU General Public License  *
  * along with this library (see file COPYING); if not, write          *
  * to the Free Software Foundation, Inc., 59 Temple Place, Suite      *
  * 330, Boston, MA 02111-1307 USA, or contact the author.             *
  *                                                                    *
  **********************************************************************/

// Implementation file for class VavilovFast
//
// Created by: blist  at Thu Apr 29 11:19:00 2010
//
// Last update: Thu Apr 29 11:19:00 2010
//


#include "Math/VavilovFast.h"
#include "Math/PdfFuncMathCore.h"
#include "Math/ProbFuncMathCore.h"
#include "Math/SpecFuncMathCore.h"
#include "Math/SpecFuncMathMore.h"

#include <cassert>
#include <iostream>
#include <cmath>
#include <limits>


namespace ROOT {
namespace Math {

VavilovFast *VavilovFast::fgInstance = 0;


VavilovFast::VavilovFast(double kappa, double beta2)
{
   SetKappaBeta2 (kappa, beta2);
}


VavilovFast::~VavilovFast()
{
   // desctructor (clean up resources)
}

void VavilovFast::SetKappaBeta2 (double kappa, double beta2)
{
   // Modified version of void TMath::VavilovSet(Double_t rkappa, Double_t beta2, Bool_t mode, Double_t *WCM, Double_t *AC, Double_t *HC, Int_t &itype, Int_t &npt)
   fKappa = kappa;
   fBeta2 = beta2;

   double BKMNX1 = 0.02, BKMNY1 = 0.05, BKMNX2 = 0.12, BKMNY2 = 0.05,
          BKMNX3 = 0.22, BKMNY3 = 0.05, BKMXX1 = 0.1 , BKMXY1 = 1,
          BKMXX2 = 0.2 , BKMXY2 = 1   , BKMXX3 = 0.3 , BKMXY3 = 1;

   double FBKX1 = 2/(BKMXX1-BKMNX1), FBKX2 = 2/(BKMXX2-BKMNX2),
          FBKX3 = 2/(BKMXX3-BKMNX3), FBKY1 = 2/(BKMXY1-BKMNY1),
          FBKY2 = 2/(BKMXY2-BKMNY2), FBKY3 = 2/(BKMXY3-BKMNY3);

   double FNINV[] = {0, 1, 0.5, 0.33333333, 0.25, 0.2};

   double EDGEC[]= {0, 0, 0.16666667e+0, 0.41666667e-1, 0.83333333e-2,
                    0.13888889e-1, 0.69444444e-2, 0.77160493e-3};

   double U1[] = {0, 0.25850868e+0,  0.32477982e-1, -0.59020496e-2,
                     0.            , 0.24880692e-1,  0.47404356e-2,
                    -0.74445130e-3,  0.73225731e-2,  0.           ,
                     0.11668284e-2,  0.           , -0.15727318e-2,-0.11210142e-2};

   double U2[] = {0, 0.43142611e+0,  0.40797543e-1, -0.91490215e-2,
                     0.           ,  0.42127077e-1,  0.73167928e-2,
                    -0.14026047e-2,  0.16195241e-1,  0.24714789e-2,
                     0.20751278e-2,  0.           , -0.25141668e-2,-0.14064022e-2};

   double U3[] = {0,  0.25225955e+0,  0.64820468e-1, -0.23615759e-1,
                      0.           ,  0.23834176e-1,  0.21624675e-2,
                     -0.26865597e-2, -0.54891384e-2,  0.39800522e-2,
                      0.48447456e-2, -0.89439554e-2, -0.62756944e-2,-0.24655436e-2};

   double U4[] = {0, 0.12593231e+1, -0.20374501e+0,  0.95055662e-1,
                    -0.20771531e-1, -0.46865180e-1, -0.77222986e-2,
                     0.32241039e-2,  0.89882920e-2, -0.67167236e-2,
                    -0.13049241e-1,  0.18786468e-1,  0.14484097e-1};

   double U5[] = {0, -0.24864376e-1, -0.10368495e-2,  0.14330117e-2,
                      0.20052730e-3,  0.18751903e-2,  0.12668869e-2,
                      0.48736023e-3,  0.34850854e-2,  0.           ,
                     -0.36597173e-3,  0.19372124e-2,  0.70761825e-3, 0.46898375e-3};

   double U6[] = {0,  0.35855696e-1, -0.27542114e-1,  0.12631023e-1,
                     -0.30188807e-2, -0.84479939e-3,  0.           ,
                      0.45675843e-3, -0.69836141e-2,  0.39876546e-2,
                     -0.36055679e-2,  0.           ,  0.15298434e-2, 0.19247256e-2};

   double U7[] = {0, 0.10234691e+2, -0.35619655e+1,  0.69387764e+0,
                    -0.14047599e+0, -0.19952390e+1, -0.45679694e+0,
                     0.           ,  0.50505298e+0};
   double U8[] = {0,  0.21487518e+2, -0.11825253e+2,  0.43133087e+1,
                     -0.14500543e+1, -0.34343169e+1, -0.11063164e+1,
                     -0.21000819e+0,  0.17891643e+1, -0.89601916e+0,
                      0.39120793e+0,  0.73410606e+0,  0.           ,-0.32454506e+0};

   double V1[] = {0, 0.27827257e+0, -0.14227603e-2,  0.24848327e-2,
                     0.           ,  0.45091424e-1,  0.80559636e-2,
                    -0.38974523e-2,  0.           , -0.30634124e-2,
                     0.75633702e-3,  0.54730726e-2,  0.19792507e-2};

   double V2[] = {0, 0.41421789e+0, -0.30061649e-1,  0.52249697e-2,
                     0.           ,  0.12693873e+0,  0.22999801e-1,
                    -0.86792801e-2,  0.31875584e-1, -0.61757928e-2,
                     0.           ,  0.19716857e-1,  0.32596742e-2};

   double V3[] = {0, 0.20191056e+0, -0.46831422e-1,  0.96777473e-2,
                    -0.17995317e-2,  0.53921588e-1,  0.35068740e-2,
                    -0.12621494e-1, -0.54996531e-2, -0.90029985e-2,
                     0.34958743e-2,  0.18513506e-1,  0.68332334e-2,-0.12940502e-2};

   double V4[] = {0, 0.13206081e+1,  0.10036618e+0, -0.22015201e-1,
                     0.61667091e-2, -0.14986093e+0, -0.12720568e-1,
                     0.24972042e-1, -0.97751962e-2,  0.26087455e-1,
                    -0.11399062e-1, -0.48282515e-1, -0.98552378e-2};

   double V5[] = {0, 0.16435243e-1,  0.36051400e-1,  0.23036520e-2,
                    -0.61666343e-3, -0.10775802e-1,  0.51476061e-2,
                     0.56856517e-2, -0.13438433e-1,  0.           ,
                     0.           , -0.25421507e-2,  0.20169108e-2,-0.15144931e-2};

   double V6[] = {0, 0.33432405e-1,  0.60583916e-2, -0.23381379e-2,
                     0.83846081e-3, -0.13346861e-1, -0.17402116e-2,
                     0.21052496e-2,  0.15528195e-2,  0.21900670e-2,
                    -0.13202847e-2, -0.45124157e-2, -0.15629454e-2, 0.22499176e-3};

   double V7[] = {0, 0.54529572e+1, -0.90906096e+0,  0.86122438e-1,
                     0.           , -0.12218009e+1, -0.32324120e+0,
                     -0.27373591e-1,  0.12173464e+0,  0.           ,
                     0.           ,  0.40917471e-1};

   double V8[] = {0, 0.93841352e+1, -0.16276904e+1,  0.16571423e+0,
                     0.           , -0.18160479e+1, -0.50919193e+0,
                    -0.51384654e-1,  0.21413992e+0,  0.           ,
                     0.           ,  0.66596366e-1};

   double W1[] = {0, 0.29712951e+0,  0.97572934e-2,  0.           ,
                    -0.15291686e-2,  0.35707399e-1,  0.96221631e-2,
                    -0.18402821e-2, -0.49821585e-2,  0.18831112e-2,
                     0.43541673e-2,  0.20301312e-2, -0.18723311e-2,-0.73403108e-3};

   double W2[] = {0, 0.40882635e+0,  0.14474912e-1,  0.25023704e-2,
                    -0.37707379e-2,  0.18719727e+0,  0.56954987e-1,
                     0.           ,  0.23020158e-1,  0.50574313e-2,
                     0.94550140e-2,  0.19300232e-1};

   double W3[] = {0, 0.16861629e+0,  0.           ,  0.36317285e-2,
                    -0.43657818e-2,  0.30144338e-1,  0.13891826e-1,
                    -0.58030495e-2, -0.38717547e-2,  0.85359607e-2,
                     0.14507659e-1,  0.82387775e-2, -0.10116105e-1,-0.55135670e-2};

   double W4[] = {0, 0.13493891e+1, -0.26863185e-2, -0.35216040e-2,
                     0.24434909e-1, -0.83447911e-1, -0.48061360e-1,
                     0.76473951e-2,  0.24494430e-1, -0.16209200e-1,
                    -0.37768479e-1, -0.47890063e-1,  0.17778596e-1, 0.13179324e-1};

   double W5[] = {0,  0.10264945e+0,  0.32738857e-1,  0.           ,
                      0.43608779e-2, -0.43097757e-1, -0.22647176e-2,
                      0.94531290e-2, -0.12442571e-1, -0.32283517e-2,
                     -0.75640352e-2, -0.88293329e-2,  0.52537299e-2, 0.13340546e-2};

   double W6[] = {0, 0.29568177e-1, -0.16300060e-2, -0.21119745e-3,
                     0.23599053e-2, -0.48515387e-2, -0.40797531e-2,
                     0.40403265e-3,  0.18200105e-2, -0.14346306e-2,
                    -0.39165276e-2, -0.37432073e-2,  0.19950380e-2, 0.12222675e-2};

   double W8[] = {0,  0.66184645e+1, -0.73866379e+0,  0.44693973e-1,
                      0.           , -0.14540925e+1, -0.39529833e+0,
                     -0.44293243e-1,  0.88741049e-1};

   fItype = 0;
   if (fKappa <0.01 || fKappa >12) {
      std::cerr << "VavilovFast::set: illegal value of kappa=" << kappa << std::endl;
      if (fKappa < 0.01) fKappa = 0.01;
      else if (fKappa > 12) fKappa = 12;
   }

   double DRK[6];
   double DSIGM[6];
   double ALFA[8];
   int j;
   double x, y, xx, yy, x2, x3, y2, y3, xy, p2, p3, q2, q3, pq;
   if (fKappa >= 0.29) {
      fItype = 1;
      fNpt = 100;
      double wk = 1./std::sqrt(fKappa);

      fAC[0] = (-0.032227*fBeta2-0.074275)*fKappa + (0.24533*fBeta2+0.070152)*wk + (-0.55610*fBeta2-3.1579);
      fAC[8] = (-0.013483*fBeta2-0.048801)*fKappa + (-1.6921*fBeta2+8.3656)*wk + (-0.73275*fBeta2-3.5226);
      DRK[1] = wk*wk;
      DSIGM[1] = std::sqrt(fKappa/(1-0.5*fBeta2));
      for (j=1; j<=4; j++) {
         DRK[j+1] = DRK[1]*DRK[j];
         DSIGM[j+1] = DSIGM[1]*DSIGM[j];
         ALFA[j+1] = (FNINV[j]-fBeta2*FNINV[j+1])*DRK[j];
      }
      fHC[0]=std::log(fKappa)+fBeta2+0.42278434;
      fHC[1]=DSIGM[1];
      fHC[2]=ALFA[3]*DSIGM[3];
      fHC[3]=(3*ALFA[2]*ALFA[2] + ALFA[4])*DSIGM[4]-3;
      fHC[4]=(10*ALFA[2]*ALFA[3]+ALFA[5])*DSIGM[5]-10*fHC[2];
      fHC[5]=fHC[2]*fHC[2];
      fHC[6]=fHC[2]*fHC[3];
      fHC[7]=fHC[2]*fHC[5];
      for (j=2; j<=7; j++)
         fHC[j]*=EDGEC[j];
      fHC[8]=0.39894228*fHC[1];
   }
   else if (fKappa >=0.22) {
      fItype = 2;
      fNpt = 150;
      x = 1+(fKappa-BKMXX3)*FBKX3;
      y = 1+(std::sqrt(fBeta2)-BKMXY3)*FBKY3;
      xx = 2*x;
      yy = 2*y;
      x2 = xx*x-1;
      x3 = xx*x2-x;
      y2 = yy*y-1;
      y3 = yy*y2-y;
      xy = x*y;
      p2 = x2*y;
      p3 = x3*y;
      q2 = y2*x;
      q3 = y3*x;
      pq = x2*y2;
      fAC[1] = W1[1] + W1[2]*x + W1[4]*x3 + W1[5]*y + W1[6]*y2 + W1[7]*y3 +
         W1[8]*xy + W1[9]*p2 + W1[10]*p3 + W1[11]*q2 + W1[12]*q3 + W1[13]*pq;
      fAC[2] = W2[1] + W2[2]*x + W2[3]*x2 + W2[4]*x3 + W2[5]*y + W2[6]*y2 +
         W2[8]*xy + W2[9]*p2 + W2[10]*p3 + W2[11]*q2;
      fAC[3] = W3[1] + W3[3]*x2 + W3[4]*x3 + W3[5]*y + W3[6]*y2 + W3[7]*y3 +
         W3[8]*xy + W3[9]*p2 + W3[10]*p3 + W3[11]*q2 + W3[12]*q3 + W3[13]*pq;
      fAC[4] = W4[1] + W4[2]*x + W4[3]*x2 + W4[4]*x3 + W4[5]*y + W4[6]*y2 + W4[7]*y3 +
         W4[8]*xy + W4[9]*p2 + W4[10]*p3 + W4[11]*q2 + W4[12]*q3 + W4[13]*pq;
      fAC[5] = W5[1] + W5[2]*x + W5[4]*x3 + W5[5]*y + W5[6]*y2 + W5[7]*y3 +
         W5[8]*xy + W5[9]*p2 + W5[10]*p3 + W5[11]*q2 + W5[12]*q3 + W5[13]*pq;
      fAC[6] = W6[1] + W6[2]*x + W6[3]*x2 + W6[4]*x3 + W6[5]*y + W6[6]*y2 + W6[7]*y3 +
         W6[8]*xy + W6[9]*p2 + W6[10]*p3 + W6[11]*q2 + W6[12]*q3 + W6[13]*pq;
      fAC[8] = W8[1] + W8[2]*x + W8[3]*x2 + W8[5]*y + W8[6]*y2 + W8[7]*y3 + W8[8]*xy;
      fAC[0] = -3.05;
   } else if (fKappa >= 0.12) {
      fItype = 3;
      fNpt = 200;
      x = 1 + (fKappa-BKMXX2)*FBKX2;
      y = 1 + (std::sqrt(fBeta2)-BKMXY2)*FBKY2;
      xx = 2*x;
      yy = 2*y;
      x2 = xx*x-1;
      x3 = xx*x2-x;
      y2 = yy*y-1;
      y3 = yy*y2-y;
      xy = x*y;
      p2 = x2*y;
      p3 = x3*y;
      q2 = y2*x;
      q3 = y3*x;
      pq = x2*y2;
      fAC[1] = V1[1] + V1[2]*x + V1[3]*x2 + V1[5]*y + V1[6]*y2 + V1[7]*y3 +
         V1[9]*p2 + V1[10]*p3 + V1[11]*q2 + V1[12]*q3;
      fAC[2] = V2[1] + V2[2]*x + V2[3]*x2 + V2[5]*y + V2[6]*y2 + V2[7]*y3 +
         V2[8]*xy + V2[9]*p2 + V2[11]*q2 + V2[12]*q3;
      fAC[3] = V3[1] + V3[2]*x + V3[3]*x2 + V3[4]*x3 + V3[5]*y + V3[6]*y2 + V3[7]*y3 +
         V3[8]*xy + V3[9]*p2 + V3[10]*p3 + V3[11]*q2 + V3[12]*q3 + V3[13]*pq;
      fAC[4] = V4[1] + V4[2]*x + V4[3]*x2 + V4[4]*x3 + V4[5]*y + V4[6]*y2 + V4[7]*y3 +
         V4[8]*xy + V4[9]*p2 + V4[10]*p3 + V4[11]*q2 + V4[12]*q3;
      fAC[5] = V5[1] + V5[2]*x + V5[3]*x2 + V5[4]*x3 + V5[5]*y + V5[6]*y2 + V5[7]*y3 +
         V5[8]*xy + V5[11]*q2 + V5[12]*q3 + V5[13]*pq;
      fAC[6] = V6[1] + V6[2]*x + V6[3]*x2 + V6[4]*x3 + V6[5]*y + V6[6]*y2 + V6[7]*y3 +
         V6[8]*xy + V6[9]*p2 + V6[10]*p3 + V6[11]*q2 + V6[12]*q3 + V6[13]*pq;
      fAC[7] = V7[1] + V7[2]*x + V7[3]*x2 + V7[5]*y + V7[6]*y2 + V7[7]*y3 +
         V7[8]*xy + V7[11]*q2;
      fAC[8] = V8[1] + V8[2]*x + V8[3]*x2 + V8[5]*y + V8[6]*y2 + V8[7]*y3 +
         V8[8]*xy + V8[11]*q2;
      fAC[0] = -3.04;
   } else {
      fItype = 4;
      if (fKappa >=0.02) fItype = 3;
      fNpt = 200;
      x = 1+(fKappa-BKMXX1)*FBKX1;
      y = 1+(std::sqrt(fBeta2)-BKMXY1)*FBKY1;
      xx = 2*x;
      yy = 2*y;
      x2 = xx*x-1;
      x3 = xx*x2-x;
      y2 = yy*y-1;
      y3 = yy*y2-y;
      xy = x*y;
      p2 = x2*y;
      p3 = x3*y;
      q2 = y2*x;
      q3 = y3*x;
      pq = x2*y2;
      if (fItype==3){
         fAC[1] = U1[1] + U1[2]*x + U1[3]*x2 + U1[5]*y + U1[6]*y2 + U1[7]*y3 +
                  U1[8]*xy + U1[10]*p3 + U1[12]*q3 + U1[13]*pq;
         fAC[2] = U2[1] + U2[2]*x + U2[3]*x2 + U2[5]*y + U2[6]*y2 + U2[7]*y3 +
                  U2[8]*xy + U2[9]*p2 + U2[10]*p3 + U2[12]*q3 + U2[13]*pq;
         fAC[3] = U3[1] + U3[2]*x + U3[3]*x2 + U3[5]*y + U3[6]*y2 + U3[7]*y3 +
                  U3[8]*xy + U3[9]*p2 + U3[10]*p3 + U3[11]*q2 + U3[12]*q3 + U3[13]*pq;
         fAC[4] = U4[1] + U4[2]*x + U4[3]*x2 + U4[4]*x3 + U4[5]*y + U4[6]*y2 + U4[7]*y3 +
                  U4[8]*xy + U4[9]*p2 + U4[10]*p3 + U4[11]*q2 + U4[12]*q3;
         fAC[5] = U5[1] + U5[2]*x + U5[3]*x2 + U5[4]*x3 + U5[5]*y + U5[6]*y2 + U5[7]*y3 +
                  U5[8]*xy + U5[10]*p3 + U5[11]*q2 + U5[12]*q3 + U5[13]*pq;
         fAC[6] = U6[1] + U6[2]*x + U6[3]*x2 + U6[4]*x3 + U6[5]*y + U6[7]*y3 +
                  U6[8]*xy + U6[9]*p2 + U6[10]*p3 + U6[12]*q3 + U6[13]*pq;
         fAC[7] = U7[1] + U7[2]*x + U7[3]*x2 + U7[4]*x3 + U7[5]*y + U7[6]*y2 + U7[8]*xy;
      }
      fAC[8] = U8[1] + U8[2]*x + U8[3]*x2 + U8[4]*x3 + U8[5]*y + U8[6]*y2 + U8[7]*y3 +
               U8[8]*xy + U8[9]*p2 + U8[10]*p3 + U8[11]*q2 + U8[13]*pq;
      fAC[0] = -3.03;
   }

   fAC[9] = (fAC[8] - fAC[0])/fNpt;
   fAC[10] = 1./fAC[9];
   if (fItype == 3) {
      x = (fAC[7]-fAC[8])/(fAC[7]*fAC[8]);
      y = 1./std::log (fAC[8]/fAC[7]);
      p2 = fAC[7]*fAC[7];
      fAC[11] = p2*(fAC[1]*std::exp(-fAC[2]*(fAC[7]+fAC[5]*p2)-
                                    fAC[3]*std::exp(-fAC[4]*(fAC[7]+fAC[6]*p2)))-0.045*y/fAC[7])/(1+x*y*fAC[7]);
      fAC[12] = (0.045+x*fAC[11])*y;
   }
   if (fItype == 4) fAC[13] = 0.995/ROOT::Math::landau_cdf(fAC[8]);

   //
   x = fAC[0];
   fWCM[0] = 0;
   double fl, fu;
   int k;
   fl = Pdf (x);
   for (k=1; k<=fNpt; k++) {
      x += fAC[9];
      fu = Pdf (x);
      fWCM[k] = fWCM[k-1] + fl + fu;
      fl = fu;
   }
   x = 0.5*fAC[9];
   for (k=1; k<=fNpt; k++)
      fWCM[k]*=x;
}

double VavilovFast::Pdf (double x) const
{
   // Modified version of TMath::double VavilovDenEval(Double_t rlam, Double_t *AC, Double_t *HC, Int_t itype);
   //Internal function, called by Vavilov and VavilovSet

   double v = 0;
   if (x < fAC[0] || x > fAC[8])
      return 0;
   int k;
   double h[10];
   if (fItype ==1 ) {
      double fn = 1;
      double xx = (x + fHC[0])*fHC[1];
      h[1] = xx;
      h[2] = xx*xx -1;
      for (k=2; k<=8; k++) {
         fn++;
         h[k+1] = xx*h[k]-fn*h[k-1];
      }
      double s = 1 + fHC[7]*h[9];
      for (k=2; k<=6; k++)
         s += fHC[k]*h[k+1];
      if (s>0) v = fHC[8]*std::exp(-0.5*xx*xx);
   }
   else if (fItype == 2) {
      double xx = x*x;
      v = fAC[1]*std::exp(-fAC[2]*(x+fAC[5]*xx) - fAC[3]*std::exp(-fAC[4]*(x+fAC[6]*xx)));
   }
   else if (fItype == 3) {
      if (x < fAC[7]) {
         double xx = x*x;
         v = fAC[1]*std::exp(-fAC[2]*(x+fAC[5]*xx)-fAC[3]*std::exp(-fAC[4]*(x+fAC[6]*xx)));
      } else {
         double xx = 1./x;
         v = (fAC[11]*xx + fAC[12])*xx;
      }
   }
   else if (fItype == 4) {
      v = fAC[13]*ROOT::Math::landau_pdf(x);
   }
   return v;
}


double VavilovFast::Pdf (double x, double kappa, double beta2) {
   //Returns the value of the Vavilov density function
   //Parameters: 1st - the point were the density function is evaluated
   //            2nd - value of kappa (distribution parameter)
   //            3rd - value of beta2 (distribution parameter)
   //The algorithm was taken from the CernLib function vavden(G115)
   //Reference: A.Rotondi and P.Montagna, Fast Calculation of Vavilov distribution
   //Nucl.Instr. and Meth. B47(1990), 215-224
   //Accuracy: quote from the reference above:
   //"The resuls of our code have been compared with the values of the Vavilov
   //density function computed numerically in an accurate way: our approximation
   //shows a difference of less than 3% around the peak of the density function, slowly
   //increasing going towards the extreme tails to the right and to the left"

   if (kappa != fKappa || beta2 != fBeta2) SetKappaBeta2 (kappa, beta2);
   return Pdf (x);
}

double VavilovFast::Cdf (double x) const {
   // Modified version of Double_t TMath::VavilovI(Double_t x, Double_t kappa, Double_t beta2)
   double xx, v;
   if (x < fAC[0]) v = 0;
   else if (x >= fAC[8]) v = 1;
   else {
      xx = x - fAC[0];
      int k = int (xx*fAC[10]);
      v = fWCM[k] + (xx - k*fAC[9])*(fWCM[k+1]-fWCM[k])*fAC[10];
      if (v > 1) v = 1;
   }
   return v;
}

double VavilovFast::Cdf_c (double x) const {
   return 1-Cdf(x);
}

double VavilovFast::Cdf (double x, double kappa, double beta2) {
   //Returns the value of the Vavilov distribution function
   //Parameters: 1st - the point were the density function is evaluated
   //            2nd - value of kappa (distribution parameter)
   //            3rd - value of beta2 (distribution parameter)
   //The algorithm was taken from the CernLib function vavden(G115)
   //Reference: A.Rotondi and P.Montagna, Fast Calculation of Vavilov distribution
   //Nucl.Instr. and Meth. B47(1990), 215-224
   //Accuracy: quote from the reference above:
   //"The resuls of our code have been compared with the values of the Vavilov
   //density function computed numerically in an accurate way: our approximation
   //shows a difference of less than 3% around the peak of the density function, slowly
   //increasing going towards the extreme tails to the right and to the left"

   if (kappa != fKappa || beta2 != fBeta2) SetKappaBeta2 (kappa, beta2);
   return Cdf (x);
}

double VavilovFast::Cdf_c (double x, double kappa, double beta2) {
   //Returns the value of the Vavilov distribution function
   //Parameters: 1st - the point were the density function is evaluated
   //            2nd - value of kappa (distribution parameter)
   //            3rd - value of beta2 (distribution parameter)
   //The algorithm was taken from the CernLib function vavden(G115)
   //Reference: A.Rotondi and P.Montagna, Fast Calculation of Vavilov distribution
   //Nucl.Instr. and Meth. B47(1990), 215-224
   //Accuracy: quote from the reference above:
   //"The resuls of our code have been compared with the values of the Vavilov
   //density function computed numerically in an accurate way: our approximation
   //shows a difference of less than 3% around the peak of the density function, slowly
   //increasing going towards the extreme tails to the right and to the left"

   if (kappa != fKappa || beta2 != fBeta2) SetKappaBeta2 (kappa, beta2);
   return Cdf_c (x);
}

double VavilovFast::Quantile (double z) const {
   if (z < 0 || z > 1) return std::numeric_limits<double>::signaling_NaN();

   // translated from CERNLIB routine VAVRAN by B. List 14.5.2010

   double t = 2*z/fAC[9];
   double rlam = fAC[0];
   double fl = 0;
   double fu = 0;
   double s = 0;
   double h[10];
   for (int n = 1; n <= fNpt; ++n) {
      rlam += fAC[9];
      if (fItype == 1) {
         double fn = 1;
         double x = (rlam+fHC[0])*fHC[1];
         h[1] = x;
         h[2] = x*x-1;
         for (int k = 2; k <= 8; ++k) {
            ++fn;
            h[k+1] = x*h[k]-fn*h[k-1];
         }
         double y = 1+fHC[7]*h[9];
         for (int k = 2; k <= 6; ++k) {
           y += fHC[k]*h[k+1];
         }
         if (y > 0) fu = fHC[8]*std::exp(-0.5*x*x);
      }
      else if (fItype == 2) {
         double x = rlam*rlam;
         fu = fAC[1]*std::exp(-fAC[2]*(rlam+fAC[5]*x)-
              fAC[3]*std::exp(-fAC[4]*(rlam+fAC[6]*x)));
      }
      else if (fItype == 3) {
         if (rlam < fAC[7]) {
            double x = rlam*rlam;
            fu = fAC[1]*std::exp(-fAC[2]*(rlam+fAC[5]*x)-
                 fAC[3]*std::exp(-fAC[4]*(rlam+fAC[6]*x)));
         } else {
            double x = 1/rlam;
            fu = (fAC[11]*x+fAC[12])*x;
         }
      }
      else {
         fu = fAC[13]*Pdf(rlam);  // in VAVRAN: AC(10) -> difference between VAVRAN and VAVSET
      }
      s += fl+fu;
      if (s > t) break;
      fl = fu;
   }
   double s0 = s-fl-fu;
   double v = rlam-fAC[9];
   if (s > s0) v += fAC[9]*(t-s0)/(s-s0);
   return v;
}

double VavilovFast::Quantile (double z, double kappa, double beta2) {
   if (kappa != fKappa || beta2 != fBeta2) SetKappaBeta2 (kappa, beta2);
   return Quantile (z);
}

double VavilovFast::Quantile_c (double z) const {
   if (z < 0 || z > 1) return std::numeric_limits<double>::signaling_NaN();
   return Quantile (1-z);
}

double VavilovFast::Quantile_c (double z, double kappa, double beta2) {
   if (kappa != fKappa || beta2 != fBeta2) SetKappaBeta2 (kappa, beta2);
   return Quantile_c (z);
}

double VavilovFast::GetLambdaMin() const {
   return fAC[0];
}

double VavilovFast::GetLambdaMax() const {
   return fAC[8];
}

double VavilovFast::GetKappa()     const {
   return fKappa;
}

double VavilovFast::GetBeta2()     const {
   return fBeta2;
}

VavilovFast *VavilovFast::GetInstance() {
   if (!fgInstance) fgInstance = new VavilovFast (1, 1);
   return fgInstance;
}

VavilovFast *VavilovFast::GetInstance(double kappa, double beta2) {
   if (!fgInstance) fgInstance = new VavilovFast (kappa, beta2);
   else if (kappa != fgInstance->fKappa || beta2 != fgInstance->fBeta2) fgInstance->SetKappaBeta2 (kappa, beta2);
   return fgInstance;
}

double vavilov_fast_pdf (double x, double kappa, double beta2) {
   VavilovFast *vavilov = VavilovFast::GetInstance (kappa, beta2);
   return vavilov->Pdf (x);
}

double vavilov_fast_cdf (double x, double kappa, double beta2) {
   VavilovFast *vavilov = VavilovFast::GetInstance (kappa, beta2);
   return vavilov->Cdf (x);
}

double vavilov_fast_cdf_c (double x, double kappa, double beta2) {
   VavilovFast *vavilov = VavilovFast::GetInstance (kappa, beta2);
   return vavilov->Cdf_c (x);
}

double vavilov_fast_quantile (double z, double kappa, double beta2) {
   VavilovFast *vavilov = VavilovFast::GetInstance (kappa, beta2);
   return vavilov->Quantile (z);
}

double vavilov_fast_quantile_c (double z, double kappa, double beta2) {
   VavilovFast *vavilov = VavilovFast::GetInstance (kappa, beta2);
   return vavilov->Quantile_c (z);
}


} // namespace Math
} // namespace ROOT
