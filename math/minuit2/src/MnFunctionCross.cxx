// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/MnFunctionCross.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/FCNBase.h"
#include "Minuit2/MnParabola.h"
#include "Minuit2/MnParabolaPoint.h"
#include "Minuit2/MnParabolaFactory.h"
#include "Minuit2/MnCross.h"
#include "Minuit2/MnMachinePrecision.h"

//#define DEBUG
#include "Minuit2/MnPrint.h"


namespace ROOT {

   namespace Minuit2 {



MnCross MnFunctionCross::operator()(const std::vector<unsigned int>& par, const std::vector<double>& pmid,
                                    const std::vector<double>& pdir, double tlr, unsigned int maxcalls) const {
   // evaluate crossing point where function is equal to MIN + UP,
   // with direction pdir from values pmid
   // tlr indicate tolerance and maxcalls maximum number of calls

//   double edmmax = 0.5*0.001*toler*fFCN.Up();


   unsigned int npar = par.size();
   unsigned int nfcn = 0;
   const MnMachinePrecision& prec = fState.Precision();
   // tolerance used when calling Migrad
   double mgr_tlr = 0.5 * tlr;   // to be consistent with F77 version (for default values of tlr which is 0.1)
   // other olerance values are fixed at 0.01
   tlr = 0.01;
   // convergence when F is within tlf of aim and next prediction
   // of aopt is within tla of previous value of aopt
   double up = fFCN.Up();
   // for finding the point :
   double tlf = tlr*up;
   double tla = tlr;
   unsigned int maxitr = 15;
   unsigned int ipt = 0;
   double aminsv = fFval;
   double aim = aminsv + up;
   //std::cout<<"aim= "<<aim<<std::endl;
   double aopt = 0.;
   bool limset = false;
   std::vector<double> alsb(3, 0.), flsb(3, 0.);

   int printLevel = MnPrint::Level();


#ifdef DEBUG
   std::cout<<"MnFunctionCross for parameter  "<<par.front()<< "fmin = " << aminsv
            << " contur value aim = (fmin + up) = " << aim << std::endl;
#endif


   // find the largest allowed aulim

   double aulim = 100.;
   for(unsigned int i = 0; i < par.size(); i++) {
      unsigned int kex = par[i];
      if(fState.Parameter(kex).HasLimits()) {
         double zmid = pmid[i];
         double zdir = pdir[i];
         if(fabs(zdir) < fState.Precision().Eps()) continue;
         //       double zlim = 0.;
         if(zdir > 0. && fState.Parameter(kex).HasUpperLimit()) {
            double zlim = fState.Parameter(kex).UpperLimit();
            aulim = std::min(aulim, (zlim-zmid)/zdir);
         }
         else if(zdir < 0. && fState.Parameter(kex).HasLowerLimit()) {
            double zlim = fState.Parameter(kex).LowerLimit();
            aulim = std::min(aulim, (zlim-zmid)/zdir);
         }
      }
   }

#ifdef DEBUG
   std::cout<<"Largest allowed aulim "<< aulim << std::endl;
#endif

   if(aulim  < aopt+tla) limset = true;


   MnMigrad migrad(fFCN, fState, MnStrategy(std::max(0, int(fStrategy.Strategy()-1))));

   for(unsigned int i = 0; i < npar; i++) {
#ifdef DEBUG
      std::cout << "MnFunctionCross: Set value for " << par[i] <<  " to " << pmid[i] << std::endl;
#endif
      migrad.SetValue(par[i], pmid[i]);

      if (printLevel > 1) {
         std::cout << "MnFunctionCross: parameter " << i << " set to " << pmid[i] << std::endl;
      }
   }
   // find minimum with respect all the other parameters (n- npar) (npar are the fixed ones)

   FunctionMinimum min0 = migrad(maxcalls, mgr_tlr);
   nfcn += min0.NFcn();

#ifdef DEBUG
   std::cout << "MnFunctionCross: after Migrad on n-1  minimum is " << min0 << std::endl;
#endif

   if(min0.HasReachedCallLimit())
      return MnCross(min0.UserState(), nfcn, MnCross::CrossFcnLimit());
   if(!min0.IsValid()) return MnCross(fState, nfcn);
   if(limset == true && min0.Fval() < aim)
      return MnCross(min0.UserState(), nfcn, MnCross::CrossParLimit());

   ipt++;
   alsb[0] = 0.;
   flsb[0] = min0.Fval();
   flsb[0] = std::max(flsb[0], aminsv + 0.1*up);
   aopt = sqrt(up/(flsb[0]-aminsv)) - 1.;
   if(fabs(flsb[0] - aim) < tlf) return MnCross(aopt, min0.UserState(), nfcn);

   if(aopt > 1.) aopt = 1.;
   if(aopt < -0.5) aopt = -0.5;
   limset = false;
   if(aopt > aulim) {
      aopt = aulim;
      limset = true;
   }
#ifdef DEBUG
   std::cout << "MnFunctionCross: flsb[0] = " << flsb[0] << " aopt =  " << aopt  << std::endl;
#endif

   for(unsigned int i = 0; i < npar; i++) {
#ifdef DEBUG
      std::cout << "MnFunctionCross: Set new value for " << par[i] <<  " from " << pmid[i] << " to " << pmid[i] + (aopt)*pdir[i] << " aopt = " << aopt << std::endl;
#endif
      migrad.SetValue(par[i], pmid[i] + (aopt)*pdir[i]);

      if (printLevel > 1) {
         std::cout << "MnFunctionCross: parameter " << i << " set to " << pmid[i] + (aopt)*pdir[i] << std::endl;
      }

   }

   FunctionMinimum min1 = migrad(maxcalls, mgr_tlr);
   nfcn += min1.NFcn();

#ifdef DEBUG
   std::cout << "MnFunctionCross: after Migrad on n-1  minimum is " << min1 << std::endl;
#endif

   if(min1.HasReachedCallLimit())
      return MnCross(min1.UserState(), nfcn, MnCross::CrossFcnLimit());
   if(!min1.IsValid()) return MnCross(fState, nfcn);
   if(limset == true && min1.Fval() < aim)
      return MnCross(min1.UserState(), nfcn, MnCross::CrossParLimit());

   ipt++;
   alsb[1] = aopt;
   flsb[1] = min1.Fval();
   double dfda = (flsb[1] - flsb[0])/(alsb[1] - alsb[0]);

#ifdef DEBUG
   std::cout << "aopt = " << aopt << " min1Val = " << flsb[1] << " dfda = " << dfda << std::endl;
#endif


L300:
      if(dfda < 0.) {
         // looking for slope of the right sign
#ifdef DEBUG
         std::cout << "MnFunctionCross: dfda < 0 - iterate from " << ipt << " to max of " << maxitr << std::endl;
#endif
         // iterate (max times is maxitr) incrementing aopt

         unsigned int maxlk = maxitr - ipt;
         for(unsigned int it = 0; it < maxlk; it++) {
            alsb[0] = alsb[1];
            flsb[0] = flsb[1];
            // LM: Add + 1, looking at Fortran code it starts from 1 ( see bug #8396)
            aopt = alsb[0] + 0.2*(it+1);
            limset = false;
            if(aopt > aulim) {
               aopt = aulim;
               limset = true;
            }
            for(unsigned int i = 0; i < npar; i++) {
#ifdef DEBUG
      std::cout << "MnFunctionCross: Set new value for " << par[i] <<  " to " << pmid[i] + (aopt)*pdir[i] << " aopt = " << aopt << std::endl;
#endif
               migrad.SetValue(par[i], pmid[i] + (aopt)*pdir[i]);
               if (printLevel > 1) {
                  std::cout << "MnFunctionCross: parameter " << i << " set to " << pmid[i] + (aopt)*pdir[i] << std::endl;
               }

            }
            min1 = migrad(maxcalls, mgr_tlr);
            nfcn += min1.NFcn();

#ifdef DEBUG
   std::cout << "MnFunctionCross: after Migrad on n-1  minimum is " << min1 << std::endl;
   std::cout << "nfcn = " << nfcn << std::endl;
#endif

            if(min1.HasReachedCallLimit())
               return MnCross(min1.UserState(), nfcn, MnCross::CrossFcnLimit());
            if(!min1.IsValid()) return MnCross(fState, nfcn);
            if(limset == true && min1.Fval() < aim)
               return MnCross(min1.UserState(), nfcn, MnCross::CrossParLimit());
            ipt++;
            alsb[1] = aopt;
            flsb[1] = min1.Fval();
            dfda = (flsb[1] - flsb[0])/(alsb[1] - alsb[0]);
            //       if(dfda > 0.) goto L460;

#ifdef DEBUG
   std::cout << "aopt = " << aopt << " min1Val = " << flsb[1] << " dfda = " << dfda << std::endl;
#endif

            if(dfda > 0.) break;
         }
         if(ipt > maxitr) return MnCross(fState, nfcn);
      } //if(dfda < 0.)

L460:

      // dfda > 0: we have two points with the right slope

      aopt = alsb[1] + (aim-flsb[1])/dfda;

#ifdef DEBUG
      std::cout << "MnFunctionCross: dfda > 0 : aopt = " << aopt << std::endl;
#endif

   double fdist = std::min(fabs(aim  - flsb[0]), fabs(aim  - flsb[1]));
   double adist = std::min(fabs(aopt - alsb[0]), fabs(aopt - alsb[1]));
   tla = tlr;
   if(fabs(aopt) > 1.) tla = tlr*fabs(aopt);
   if(adist < tla && fdist < tlf) return MnCross(aopt, min1.UserState(), nfcn);
   if(ipt > maxitr) return MnCross(fState, nfcn);
   double bmin = std::min(alsb[0], alsb[1]) - 1.;
   if(aopt < bmin) aopt = bmin;
   double bmax = std::max(alsb[0], alsb[1]) + 1.;
   if(aopt > bmax) aopt = bmax;

   limset = false;
   if(aopt > aulim) {
      aopt = aulim;
      limset = true;
   }

   for(unsigned int i = 0; i < npar; i++) {
#ifdef DEBUG
      std::cout << "MnFunctionCross: Set new value for " << par[i] <<  " from " << pmid[i] << " to " << pmid[i] + (aopt)*pdir[i] << " aopt = " << aopt << std::endl;
#endif
      migrad.SetValue(par[i], pmid[i] + (aopt)*pdir[i]);
      if (printLevel > 1) {
         std::cout << "MnFunctionCross: parameter " << i << " set to " << pmid[i] + (aopt)*pdir[i] << std::endl;
      }

   }
   FunctionMinimum min2 = migrad(maxcalls, mgr_tlr);
   nfcn += min2.NFcn();

#ifdef DEBUG
   std::cout << "MnFunctionCross: after Migrad on n-1  minimum is " << min2 << std::endl;
   std::cout << "nfcn = " << nfcn << std::endl;
#endif

   if(min2.HasReachedCallLimit())
      return MnCross(min2.UserState(), nfcn, MnCross::CrossFcnLimit());
   if(!min2.IsValid()) return MnCross(fState, nfcn);
   if(limset == true && min2.Fval() < aim)
      return MnCross(min2.UserState(), nfcn, MnCross::CrossParLimit());

   ipt++;
   alsb[2] = aopt;
   flsb[2] = min2.Fval();

   // now we have three points, ask how many < AIM

   double ecarmn = fabs(flsb[2] - aim);
   double ecarmx = 0.;
   unsigned int ibest = 2;
   unsigned int iworst = 0;
   unsigned int noless = 0;

   for(unsigned int i = 0; i < 3; i++) {
      double ecart = fabs(flsb[i] - aim);
      if(ecart > ecarmx) {
         ecarmx = ecart;
         iworst = i;
      }
      if(ecart < ecarmn) {
         ecarmn = ecart;
         ibest = i;
      }
      if(flsb[i] < aim) noless++;
   }

#ifdef DEBUG
   std::cout << "MnFunctionCross: have three points : nless < aim  = " << noless << " ibest = " << ibest << " iworst = " << iworst << std::endl;
#endif

   //std::cout<<"480"<<std::endl;

   // at least one on each side of AIM (contour), fit a parabola
   if(noless == 1 || noless == 2) goto L500;
   // if all three are above AIM, third point must be the closest to AIM, return it
   if(noless == 0 && ibest != 2) return MnCross(fState, nfcn);
   // if all three below and third is not best then the slope has again gone negative,
   // re-iterate and look for positive slope
   if(noless == 3 && ibest != 2) {
      alsb[1] = alsb[2];
      flsb[1] = flsb[2];
#ifdef DEBUG
   std::cout << "MnFunctionCross: all three points below - look again fir positive slope " << std::endl;
#endif
      goto L300;
   }

   // in other case new straight line thru first two points

   flsb[iworst] = flsb[2];
   alsb[iworst] = alsb[2];
   dfda = (flsb[1] - flsb[0])/(alsb[1] - alsb[0]);
#ifdef DEBUG
   std::cout << "MnFunctionCross: new straight line using point 1-2 - dfda =  " << dfda << std::endl;
#endif
   goto L460;

L500:

      do {
         // do parabola fit
         MnParabola parbol = MnParabolaFactory()(MnParabolaPoint(alsb[0], flsb[0]), MnParabolaPoint(alsb[1], flsb[1]), MnParabolaPoint(alsb[2], flsb[2]));
         //   aopt = parbol.X_pos(aim);
         //std::cout<<"alsb1,2,3= "<<alsb[0]<<", "<<alsb[1]<<", "<<alsb[2]<<std::endl;
         //std::cout<<"flsb1,2,3= "<<flsb[0]<<", "<<flsb[1]<<", "<<flsb[2]<<std::endl;

#ifdef DEBUG
   std::cout << "MnFunctionCross: parabola fit: iteration " << ipt  << std::endl;
#endif

         double coeff1 = parbol.C();
         double coeff2 = parbol.B();
         double coeff3 = parbol.A();
         double determ = coeff2*coeff2 - 4.*coeff3*(coeff1 - aim);

#ifdef DEBUG
         std::cout << "MnFunctionCross: parabola fit: a =  " << coeff3  << " b = "
                   << coeff2 << " c = " << coeff1 << " determ = " << determ << std::endl;
#endif
         // curvature is negative
         if(determ < prec.Eps()) return MnCross(fState, nfcn);
         double rt = sqrt(determ);
         double x1 = (-coeff2 + rt)/(2.*coeff3);
         double x2 = (-coeff2 - rt)/(2.*coeff3);
         double s1 = coeff2 + 2.*x1*coeff3;
         double s2 = coeff2 + 2.*x2*coeff3;

#ifdef DEBUG
         std::cout << "MnFunctionCross: parabola fit: x1 =  " << x1  << " x2 = "
                   << x2 << " s1 = " << s1 << " s2 = " << s2 << std::endl;
#endif

#ifdef WARNINGMSG
         if(s1*s2 > 0.)   MN_INFO_MSG("MnFunctionCross problem 1");
#endif
         // find with root is the right one
         aopt = x1;
         double slope = s1;
         if(s2 > 0.) {
            aopt = x2;
            slope = s2;
         }
#ifdef DEBUG
         std::cout << "MnFunctionCross: parabola fit: aopt =  " << aopt  << " slope = "
                   << slope << std::endl;
#endif

         // ask if converged
         tla = tlr;
         if(fabs(aopt) > 1.) tla = tlr*fabs(aopt);

#ifdef DEBUG
         std::cout << "MnFunctionCross: Delta(aopt) =  " << fabs(aopt - alsb[ibest])  << " tla = "
                   << tla << "Delta(F) = " << fabs(flsb[ibest] - aim) << " tlf = " << tlf << std::endl;
#endif


         if(fabs(aopt - alsb[ibest]) < tla && fabs(flsb[ibest] - aim) < tlf)
            return MnCross(aopt, min2.UserState(), nfcn);

         //     if(ipt > maxitr) return MnCross();

         // see if proposed point is in acceptable zone between L and R
         // first find ileft, iright, iout and ibest

         unsigned int ileft = 3;
         unsigned int iright = 3;
         unsigned int iout = 3;
         ibest = 0;
         ecarmx = 0.;
         ecarmn = fabs(aim-flsb[0]);
         for(unsigned int i = 0; i < 3; i++) {
            double ecart = fabs(flsb[i] - aim);
            if(ecart < ecarmn) {
               ecarmn = ecart;
               ibest = i;
            }
            if(ecart > ecarmx) ecarmx = ecart;
            if(flsb[i] > aim) {
               if(iright == 3) iright = i;
               else if(flsb[i] > flsb[iright]) iout = i;
               else {
                  iout = iright;
                  iright = i;
               }
            } else if(ileft == 3) ileft = i;
            else if(flsb[i] < flsb[ileft]) iout = i;
            else {
               iout = ileft;
               ileft = i;
            }
         }

#ifdef DEBUG
         std::cout << "MnFunctionCross: ileft =  " << ileft  << " iright = "
                   << iright << " iout = " << iout << " ibest = " << ibest << std::endl;
#endif

         // avoid keeping a bad point nest time around

         if(ecarmx > 10.*fabs(flsb[iout] - aim))
            aopt = 0.5*(aopt + 0.5*(alsb[iright] + alsb[ileft]));

         // knowing ileft and iright, get acceptable window
         double smalla = 0.1*tla;
         if(slope*smalla > tlf) smalla = tlf/slope;
         double aleft = alsb[ileft] + smalla;
         double aright = alsb[iright] - smalla;

         // move proposed point AOPT into window if necessary
         if(aopt < aleft) aopt = aleft;
         if(aopt > aright) aopt = aright;
         if(aleft > aright) aopt = 0.5*(aleft + aright);

         // see if proposed point outside limits (should be impossible)
         limset = false;
         if(aopt > aulim) {
            aopt = aulim;
            limset = true;
         }

         // evaluate at new point aopt
         for(unsigned int i = 0; i < npar; i++) {
#ifdef DEBUG
      std::cout << "MnFunctionCross: Set new value for " << par[i] <<  " from " << pmid[i] << " to " << pmid[i] + (aopt)*pdir[i] << " aopt = " << aopt << std::endl;
#endif
            migrad.SetValue(par[i], pmid[i] + (aopt)*pdir[i]);
            if (printLevel > 1) {
               std::cout << "MnFunctionCross: parameter " << i << " set to " << pmid[i] + (aopt)*pdir[i] << std::endl;
            }
         }
         min2 = migrad(maxcalls, mgr_tlr);
         nfcn += min2.NFcn();
#ifdef DEBUG
   std::cout << "MnFunctionCross: after Migrad on n-1  minimum is " << min2 << std::endl;
   std::cout << "nfcn = " << nfcn << std::endl;
#endif

         if(min2.HasReachedCallLimit())
            return MnCross(min2.UserState(), nfcn, MnCross::CrossFcnLimit());
         if(!min2.IsValid()) return MnCross(fState, nfcn);
         if(limset == true && min2.Fval() < aim)
            return MnCross(min2.UserState(), nfcn, MnCross::CrossParLimit());

         ipt++;
         // replace odd point with new one (which is the best of three)
         alsb[iout] = aopt;
         flsb[iout] = min2.Fval();
         ibest = iout;
      } while(ipt < maxitr);

   // goto L500;

   return MnCross(fState, nfcn);
}

   }  // namespace Minuit2

}  // namespace ROOT
