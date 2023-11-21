// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/MnContours.h"
#include "Minuit2/MnMinos.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/MnFunctionCross.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/FCNBase.h"
#include "Minuit2/MnCross.h"
#include "Minuit2/MinosError.h"
#include "Minuit2/ContoursError.h"
#include "Minuit2/MnPrint.h"

namespace ROOT {

namespace Minuit2 {

std::vector<std::pair<double, double>> MnContours::
operator()(unsigned int px, unsigned int py, unsigned int npoints) const
{
   // get contour as a pair of (x,y) points passing the parameter index (px, py)  and the number of requested points
   // (>=4)
   ContoursError cont = Contour(px, py, npoints);
   return cont();
}

ContoursError MnContours::Contour(unsigned int px, unsigned int py, unsigned int npoints) const
{
   // calculate the contour passing the parameter index (px, py)  and the number of requested points (>=4)
   // the fcn.UP() has to be set to the required value (see Minuit document on errors)
   assert(npoints > 3);
   unsigned int maxcalls = 100 * (npoints + 5) * (fMinimum.UserState().VariableParameters() + 1);
   unsigned int nfcn = 0;

   MnPrint print("MnContours");
   print.Debug("MnContours: finding ",npoints," contours points for ",px,py," at level ",fFCN.Up()," from value ",fMinimum.Fval());

   std::vector<std::pair<double, double>> result;
   result.reserve(npoints);
   std::vector<MnUserParameterState> states;
   //   double edmmax = 0.5*0.05*fFCN.Up()*1.e-3;

   // double toler = 0.05;
   double toler = 0.1; // use same defaut value as in Minos

   // get first four points running Minos separately on the two parameters
   // and then finding the corresponding minimum in the other
   // P1( exlow, ymin1)  where ymin1 is the parameter value (y) at the minimum of f when x is fixed to exlow
   // P2(xmin1, eylow)  where  xmin1 is the  the parameter value (x) at the minimum of f when y is fixed to eylow
   // P3(exup, ymin2)
   // P4(xmin2, eyup)
   MnMinos minos(fFCN, fMinimum, fStrategy);

   double valx = fMinimum.UserState().Value(px);
   double valy = fMinimum.UserState().Value(py);

   print.Debug("Run Minos to find first 4 contour points. Current minimum is : ",valx,valy);

   MinosError mnex = minos.Minos(px);
   nfcn += mnex.NFcn();
   if (!mnex.IsValid()) {
      print.Error("unable to find first two points");
      return ContoursError(px, py, result, mnex, mnex, nfcn);
   }
   std::pair<double, double> ex = mnex();

   print.Debug("Minos error for p0:  ",ex.first,ex.second);

   MinosError mney = minos.Minos(py);
   nfcn += mney.NFcn();
   if (!mney.IsValid()) {
      print.Error("unable to find second two points");
      return ContoursError(px, py, result, mnex, mney, nfcn);
   }
   std::pair<double, double> ey = mney();

   print.Debug("Minos error for p0:  ",ey.first,ey.second);

   // if Minos is not at limits we can use migrad to find the other corresponding point coordinate
   MnMigrad migrad0(fFCN, fMinimum.UserState(), MnStrategy(std::max(0, int(fStrategy.Strategy() - 1))));


   // start from minimizing in p1 and fixing p0 to Minos value
   migrad0.Fix(px);
   migrad0.SetValue(px, valx + ex.first);
   FunctionMinimum exy_lo = migrad0();
   nfcn += exy_lo.NFcn();
   if (!exy_lo.IsValid()) {
      print.Error("unable to find Lower y Value for x Parameter", px);
      return ContoursError(px, py, result, mnex, mney, nfcn);
   }

   print.Debug("Minimum p1 found for p0 set to ",migrad0.Value(px)," is ",exy_lo.UserState().Value(py),"fcn = ",exy_lo.Fval());

   migrad0.SetValue(px, valx + ex.second);
   FunctionMinimum exy_up = migrad0();
   nfcn += exy_up.NFcn();
   if (!exy_up.IsValid()) {
      print.Error("unable to find Upper y Value for x Parameter", px);
      return ContoursError(px, py, result, mnex, mney, nfcn);
   }
   print.Debug("Minimum p1 found for p0 set to ",migrad0.Value(px)," is ",exy_up.UserState().Value(py),"fcn = ",exy_up.Fval());


   MnMigrad migrad1(fFCN, fMinimum.UserState(), MnStrategy(std::max(0, int(fStrategy.Strategy() - 1))));
   migrad1.Fix(py);
   migrad1.SetValue(py, valy + ey.second);
   FunctionMinimum eyx_up = migrad1();
   nfcn += eyx_up.NFcn();
   if (!eyx_up.IsValid()) {
      print.Error("unable to find Upper x Value for y Parameter", py);
      return ContoursError(px, py, result, mnex, mney, nfcn);
   }
   print.Debug("Minimum p0 found for p1 set to ",migrad1.Value(py)," is ",eyx_up.UserState().Value(px),"fcn = ",eyx_up.Fval());

   migrad1.SetValue(py, valy + ey.first);
   FunctionMinimum eyx_lo = migrad1();
   nfcn += eyx_lo.NFcn();
   if (!eyx_lo.IsValid()) {
      print.Error("unable to find Lower x Value for y Parameter", py);
      return ContoursError(px, py, result, mnex, mney, nfcn);
   }

   print.Debug("Minimum p0 found for p1 set to ",migrad1.Value(py)," is ",eyx_lo.UserState().Value(px),"fcn = ",eyx_lo.Fval());


   double scalx = 1. / (ex.second - ex.first);
   double scaly = 1. / (ey.second - ey.first);

   result.emplace_back(valx + ex.first, exy_lo.UserState().Value(py));
   result.emplace_back(eyx_lo.UserState().Value(px), valy + ey.first);
   result.emplace_back(valx + ex.second, exy_up.UserState().Value(py));
   result.emplace_back(eyx_up.UserState().Value(px), valy + ey.second);

   MnUserParameterState upar = fMinimum.UserState();

   print.Debug("List of first 4 found contour points", '\n', "  Parameter x is", upar.Name(px), '\n', "  Parameter y is", upar.Name(py),
              '\n', result[0], '\n', result[1], '\n', result[2], '\n', result[3]);

   upar.Fix(px);
   upar.Fix(py);

   std::vector<unsigned int> par(2);
   par[0] = px;
   par[1] = py;
   MnFunctionCross cross(fFCN, upar, fMinimum.Fval(), fStrategy);

   // find the remaining points of the contour
   for (unsigned int i = 4; i < npoints; i++) {

      //  find the two neighbouring points with largest separation
      auto idist1 = result.end() - 1;
      auto idist2 = result.begin();
      double dx = idist1->first - (idist2)->first;
      double dy = idist1->second - (idist2)->second;
      double bigdis = scalx * scalx * dx * dx + scaly * scaly * dy * dy;

      for (auto  ipair = result.begin(); ipair != result.end() - 1; ++ipair) {
         double distx = ipair->first - (ipair + 1)->first;
         double disty = ipair->second - (ipair + 1)->second;
         double dist = scalx * scalx * distx * distx + scaly * scaly * disty * disty;
         if (dist > bigdis) {
            bigdis = dist;
            idist1 = ipair;
            idist2 = ipair + 1;
         }
      }

      double a1 = 0.5;
      double a2 = 0.5;
      double sca = 1.;

   L300:

      if (nfcn > maxcalls) {
         print.Error("maximum number of function calls exhausted");
         return ContoursError(px, py, result, mnex, mney, nfcn);
      }

      print.Debug("Find new contour point between points with max sep:  (",idist1->first,", ",idist1->second,") and (",
                                                                           idist2->first,", ",idist2->second,")  with weights ",a1,a2);
      // find next point between the found 2 with max separation
      // start from point situated at the middle (a1,a2=0.5)
      // and direction
      double xmidcr = a1 * idist1->first + a2 * (idist2)->first;
      double ymidcr = a1 * idist1->second + a2 * (idist2)->second;
      // direction is the perpendicular one
      double xdir = (idist2)->second - idist1->second;
      double ydir = idist1->first - (idist2)->first;
      double scalfac = sca * std::max(std::fabs(xdir * scalx), std::fabs(ydir * scaly));
      double xdircr = xdir / scalfac;
      double ydircr = ydir / scalfac;
      std::vector<double> pmid(2);
      pmid[0] = xmidcr;
      pmid[1] = ymidcr;
      std::vector<double> pdir(2);
      pdir[0] = xdircr;
      pdir[1] = ydircr;

      MnCross opt = cross(par, pmid, pdir, toler, maxcalls);
      nfcn += opt.NFcn();
      if (!opt.IsValid()) {
         if(a1 > 0.5) {
         // LM 20/10/23 : remove switch of direction and look instead closer (this is what is done in TMinuit)
         // should we try again closer to P2 (e.g. a1=0.25, a2 = 0.75) if failing?
         //if (sca < 0.) {
            print.Error("unable to find point on Contour", i + 1, '\n', "found only", i, "points");
            return ContoursError(px, py, result, mnex, mney, nfcn);
         }
         a1 = 0.75;
         a2 = 0.25;
         print.Debug("Unable to find point, try closer to p1 with weight values",a1,a2);
         //std::cout<<"*****switch direction"<<std::endl;
         //sca = -1.;
         goto L300;
      }
      double aopt = opt.Value();
      int pos = result.size();
      if (idist2 == result.begin()) {
         result.emplace_back(xmidcr + (aopt)*xdircr, ymidcr + (aopt)*ydircr);
         print.Info(result.back());
      } else {
         print.Info(*idist2);
         auto itr = result.insert(idist2, {xmidcr + (aopt)*xdircr, ymidcr + (aopt)*ydircr});
         pos = std::distance(result.begin(),itr);
      }
      print.Info("Found new contour point - pos: ",pos,result[pos]);
   }

   print.Info("Number of contour points =", result.size());
   print.Debug("List of contour points");
   for (size_t i = 0; i < result.size(); i++)
      print.Debug("point ",i,result[i]);

   return ContoursError(px, py, result, mnex, mney, nfcn);
}

} // namespace Minuit2

} // namespace ROOT
