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

// function cross for a single parameter - copy the state
   MnUserParameterState ustate = fMinimum.UserState();
   MnFunctionCross cross1(fFCN, ustate, fMinimum.Fval(), fStrategy);
   // need to copy the state since we modify it when calling function cross


   auto findCrossValue = [&](unsigned int ipar, double startValue, double direction, bool & status) {
      std::vector<unsigned int> vpar(1, ipar);
      std::vector<double> vmid(1, startValue);
      std::vector<double> vdir(1, direction);
      ustate.Fix(ipar);
      MnCross crossResult = cross1(vpar, vmid, vdir, toler, maxcalls);
      status = crossResult.IsValid();
      print.Debug("result of findCrossValue for par",ipar,"status: ",status," searching from ",vmid[0],"dir",vdir[0],
                  " -> ",crossResult.Value()," fcn = ",crossResult.State().Fval());
      return (status) ?  vmid[0] + crossResult.Value() * vdir[0] : 0;
   };

   // function to find the contour points at the border of p1 after the minimization in p2
   auto findContourPointsAtBorder = [&](unsigned int p1, unsigned int p2, double p1Limit, FunctionMinimum & minp2, bool order) {
      // we are at the limit in p1
      ustate.SetValue(p1, p1Limit);
      ustate.Fix(p1);
      bool ret1 = false;
      bool ret2  = false;
      double deltaFCN = fMinimum.Fval() + fFCN.Up()- minp2.UserState().Fval();
      double pmid =  minp2.UserState().Value(p2);    // starting point for search
      double pdir =  minp2.UserState().Error(p2) * deltaFCN/fFCN.Up(); // direction for the search
      double y1 = findCrossValue(p2, pmid, pdir, ret1);
      double y2 = findCrossValue(p2, pmid, -pdir, ret2);
      if (!ret1 && !ret2) {
         return std::vector<double>();
      }
      std::vector<double> yvalues; yvalues.reserve(2);
      // check if value is
      if (ret1 && !ret2) {
          yvalues.push_back(y1);
      } else if (!ret1 && ret2) {
          yvalues.push_back(y2);
      } else {
         // if points are not the same use both
         if (std::abs(y1-y2) > 0.0001 * fMinimum.UserState().Error(p2)) {
            // order them in increasing/decreasing y depending on order flag
            int orderDir = (order) ? 1 : -1;
            if (orderDir*(y2-y1) > 0) {
               yvalues.push_back(y1);
               yvalues.push_back(y2);
            } else {
               yvalues.push_back(y2);
               yvalues.push_back(y1);
            }
         }
         else
           yvalues.push_back(y1);
      }
      print.Debug("Found contour point at the border: ",ustate.Value(p1)," , ",yvalues[0]);
      if (yvalues.size() > 1) print.Debug("Found additional point at : ",ustate.Value(p1)," , ",yvalues[1]);
      return yvalues;
   };

   // start from minimizing in p1 and fixing p0 to lower Minos value
   std::vector<double> yvalues_xlo(1);
   migrad0.State().Fix(px);
   migrad0.State().SetValue(px, valx + ex.first);
   FunctionMinimum exy_lo = migrad0();
   nfcn += exy_lo.NFcn();
   if (!exy_lo.IsValid()) {
      print.Error("unable to find Lower y Value for x Parameter", px);
      return ContoursError(px, py, result, mnex, mney, nfcn);
   }
   yvalues_xlo[0] = exy_lo.UserState().Value(py);
   print.Debug("Minimum fcn value for px set to ",valx + ex.first," is at py = ",yvalues_xlo[0]," fcn = ",exy_lo.UserState().Fval());
   if (mnex.AtLowerLimit()) {
      // use MnCross to find contour point at borders starting from found minimum point. Order is in decreasing y
      yvalues_xlo = findContourPointsAtBorder(px,py,ustate.Parameter(px).LowerLimit(),exy_lo, false);
      if (yvalues_xlo.empty()) {
         print.Error("unable to find corresponding value for ",py,"when Parameter",px,"is at lower limit: ", ustate.Value(px));
         return ContoursError(px, py, result, mnex, mney, nfcn);
      }
   }

   // now minimize in pi and fix p0 to upper Minos value
   std::vector<double> yvalues_xup(1);
   migrad0.State().SetValue(px, valx + ex.second);
   migrad0.State().Fix(px);
   FunctionMinimum exy_up = migrad0();
   nfcn += exy_up.NFcn();
   if (!exy_up.IsValid()) {
      print.Error("unable to find Upper y Value for x Parameter", px);
      return ContoursError(px, py, result, mnex, mney, nfcn);
   }
   yvalues_xup[0] = exy_up.UserState().Value(py);
   print.Debug("Minimum fcn value for px set to ",valx + ex.second," is at py = ",yvalues_xup[0]," fcn = ",exy_up.UserState().Fval());
   if (mnex.AtUpperLimit()) {
      // use MnCross to find contour point at borders starting from found minimum point. Order is in increasing y
      yvalues_xup = findContourPointsAtBorder(px,py,ustate.Parameter(px).UpperLimit(),exy_up, true);
      if (yvalues_xup.empty()) {
         print.Error("unable to find corresponding value for ",py,"when Parameter",px,"is at upper limit: ", ustate.Value(px));
         return ContoursError(px, py, result, mnex, mney, nfcn);
      }
   }

   // now look for x values when y is fixed

   MnMigrad migrad1(fFCN, fMinimum.UserState(), MnStrategy(std::max(0, int(fStrategy.Strategy() - 1))));
   migrad1.State().Fix(py);
   std::vector<double> xvalues_ylo(1);
   migrad1.State().SetValue(py, valy + ey.first);
   FunctionMinimum eyx_lo = migrad1();
   nfcn += eyx_lo.NFcn();
   if (!eyx_lo.IsValid()) {
      print.Error("unable to find Lower x Value for y Parameter", py);
      return ContoursError(px, py, result, mnex, mney, nfcn);
   }
   xvalues_ylo[0] = eyx_lo.UserState().Value(px);
   print.Debug("Minimum fcn value for py set to ",valy + ey.first," is at px = ",xvalues_ylo[0]," fcn = ",eyx_lo.UserState().Fval());
   if (mney.AtLowerLimit()) {
      // use MnCross to find contour point at borders starting from found minimum point. Order is in increasing x
      xvalues_ylo = findContourPointsAtBorder(py,px,ustate.Parameter(py).LowerLimit(),eyx_lo, true);
      if (xvalues_ylo.empty()) {
         print.Error("unable to find corresponding value for ",px,"when Parameter",py,"is at lower limit: ", ustate.Value(py));
         return ContoursError(px, py, result, mnex, mney, nfcn);
      }
   }

   std::vector<double> xvalues_yup(1);
   migrad1.State().Fix(py);
   migrad1.State().SetValue(py, valy + ey.second);
   FunctionMinimum eyx_up = migrad1();
   nfcn += eyx_up.NFcn();
   if (!eyx_up.IsValid()) {
      print.Error("unable to find Upper x Value for y Parameter", py);
      return ContoursError(px, py, result, mnex, mney, nfcn);
   }
   xvalues_yup[0] = eyx_up.UserState().Value(px);
   print.Debug("Minimum fcn value for py set to ",valy + ey.second," is at px = ",xvalues_yup[0]," fcn = ",eyx_up.UserState().Fval());
   if (mney.AtUpperLimit()) {
      // use MnCross to find contour point at borders starting from found minimum point. Order is in decreasing x
      xvalues_yup = findContourPointsAtBorder(py,px,ustate.Parameter(py).UpperLimit(),eyx_up, false);
      if (xvalues_yup.empty()) {
         print.Error("unable to find corresponding value for ",px,"when Parameter",py,"is at upper limit: ", ustate.Value(py));
         return ContoursError(px, py, result, mnex, mney, nfcn);
      }
   }

   // add the found point, start from low-x and add the extra points when existing (contour at border)
   result.emplace_back(valx + ex.first, yvalues_xlo[0]);
   if (yvalues_xlo.size()==2) result.emplace_back(valx + ex.first, yvalues_xlo[1]);
   // low y
   result.emplace_back(xvalues_ylo[0], valy + ey.first);
   if (xvalues_ylo.size()==2) result.emplace_back(xvalues_ylo[1],valy + ey.first );
   // up x
   result.emplace_back(valx + ex.second, yvalues_xup[0]);
   if (yvalues_xup.size()==2) result.emplace_back(valx + ex.second, yvalues_xup[1]);
   // up y
   result.emplace_back(xvalues_yup[0], valy + ey.second);
   if (xvalues_yup.size()==2) result.emplace_back(xvalues_yup[1],valy + ey.second );


   MnUserParameterState upar = fMinimum.UserState();


   print.Debug([&](std::ostream &os){
                  os << "List of first " << result.size() << " points found: \n";
                  os << "Parameters :   " <<  upar.Name(px) << "\t" <<  upar.Name(py) << std::endl;
                  for (auto & p : result) os << p << std::endl;
   });

   double scalx = 1. / (ex.second - ex.first);
   double scaly = 1. / (ey.second - ey.first);

   upar.Fix(px);
   upar.Fix(py);

   std::vector<unsigned int> par(2);
   par[0] = px;
   par[1] = py;
   MnFunctionCross cross(fFCN, upar, fMinimum.Fval(), fStrategy);

   // find the remaining points of the contour
   unsigned int np0 = result.size();
   for (unsigned int i = np0; i < npoints; i++) {

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
         // need to treat cases where points are at the border
         if (distx == 0. && upar.Parameter(px).HasLimits() &&
            (ipair->first == upar.Parameter(px).LowerLimit() || ipair->first == upar.Parameter(px).UpperLimit() ) )
            dist = 0;
         if (disty == 0. && upar.Parameter(py).HasLimits() &&
            (ipair->second == upar.Parameter(py).LowerLimit() || ipair->second == upar.Parameter(py).UpperLimit() ) )
            dist = 0;

         if (dist > bigdis) {
            bigdis = dist;
            idist1 = ipair;
            idist2 = ipair + 1;
         }
      }

      double a1 = 0.5;
      double a2 = 0.5;
      double sca = 1.;

      bool validPoint = false;

      do {

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

         print.Debug("calling MnCross with pmid: ",pmid[0],pmid[1], "and  pdir ",pdir[0],pdir[1]);
         MnCross opt = cross(par, pmid, pdir, toler, maxcalls);
         nfcn += opt.NFcn();
         if (!opt.IsValid() || opt.AtLimit()) {
            // Exclude also case where we are at limits since point is not in that case
            // on the contour. If we are at limit maybe we should try more?
            if (a1 < 0.3) {
               // here when having tried with a1=0.5, a1 = 0.75 and a1=0.25
               print.Info("Unable to find point on Contour", i + 1, '\n', "found only", i, "points");
               return ContoursError(px, py, result, mnex, mney, nfcn);
            } else if (a1 > 0.5) {
               a1 = 0.25;
               a2 = 0.75;
               print.Debug("Unable to find point, try closer to p2 with weight values",a1,a2);
            } else {
               a1 = 0.75;
               a2 = 0.25;
               print.Debug("Unable to find point, try closer to p1 with weight values",a1,a2);
               //std::cout<<"*****switch direction"<<std::endl;
               //sca = -1.;
            }
         } else {
            // a point is found
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
            print.Info("Found new point - pos: ",pos,result[pos], "fcn = ",opt.State().Fval());
            validPoint = true;
         }
      // loop until a valid point has been found
      } while (!validPoint);
   }  // end loop on points

   print.Info("Number of contour points =", result.size());

   return ContoursError(px, py, result, mnex, mney, nfcn);
}


} // namespace Minuit2

} // namespace ROOT
