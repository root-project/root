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


void PrintContourPoint(const std::pair<double,double> & point)  { 
#ifdef WARNINGMSG 
#ifdef USE_ROOT_ERROR
   std::string msg = "\tx = " + ROOT::Math::Util::ToString(point.first) + "\ty = " + ROOT::Math::Util::ToString(point.first);
   MN_INFO_MSG2("MnContour",msg.c_str());
#else
   std::cout << " x  = " << point.first << "  y = " << point.second << std::endl;
#endif
#endif
}

std::vector<std::pair<double,double> > MnContours::operator()(unsigned int px, unsigned int py, unsigned int npoints) const {
   // get contour as a pair of (x,y) points passing the parameter index (px, py)  and the number of requested points (>=4)
   ContoursError cont = Contour(px, py, npoints);
   return cont();
}

ContoursError MnContours::Contour(unsigned int px, unsigned int py, unsigned int npoints) const {
   // calculate the contour passing the parameter index (px, py)  and the number of requested points (>=4)
   // the fcn.UP() has to be set to the rquired value (see Minuit document on errors)
   assert(npoints > 3);
   unsigned int maxcalls = 100*(npoints+5)*(fMinimum.UserState().VariableParameters()+1);
   unsigned int nfcn = 0;
   
   std::vector<std::pair<double,double> > result; result.reserve(npoints);
   std::vector<MnUserParameterState> states;
   //   double edmmax = 0.5*0.05*fFCN.Up()*1.e-3;    

   //double toler = 0.05;    
   double toler = 0.01; // use same value as in Minos    
   
   //get first four points
   //   std::cout<<"MnContours: get first 4 params."<<std::endl;
   MnMinos minos(fFCN, fMinimum, fStrategy);
   
   double valx = fMinimum.UserState().Value(px);
   double valy = fMinimum.UserState().Value(py);
   
   MinosError mex = minos.Minos(px);
   nfcn += mex.NFcn();
   if(!mex.IsValid()) {
      MN_ERROR_MSG("MnContours is unable to find first two points.");
      return ContoursError(px, py, result, mex, mex, nfcn);
   }
   std::pair<double,double> ex = mex();
   
   MinosError mey = minos.Minos(py);
   nfcn += mey.NFcn();
   if(!mey.IsValid()) {
      MN_ERROR_MSG("MnContours is unable to find second two points.");
      return ContoursError(px, py, result, mex, mey, nfcn);
   }
   std::pair<double,double> ey = mey();
   
   MnMigrad migrad(fFCN, fMinimum.UserState(), MnStrategy(std::max(0, int(fStrategy.Strategy()-1))));
   
   migrad.Fix(px);
   migrad.SetValue(px, valx + ex.second);
   FunctionMinimum exy_up = migrad();
   nfcn += exy_up.NFcn();
   if(!exy_up.IsValid()) {
      MN_ERROR_VAL2("MnContours: unable to find Upper y Value for x Parameter",px);
      return ContoursError(px, py, result, mex, mey, nfcn);
   }
   
   migrad.SetValue(px, valx + ex.first);
   FunctionMinimum exy_lo = migrad();
   nfcn += exy_lo.NFcn();
   if(!exy_lo.IsValid()) {
      MN_ERROR_VAL2("MnContours: unable to find Lower y Value for x Parameter",px);
      return ContoursError(px, py, result, mex, mey, nfcn);
   }
   
   
   MnMigrad migrad1(fFCN, fMinimum.UserState(), MnStrategy(std::max(0, int(fStrategy.Strategy()-1))));
   migrad1.Fix(py);
   migrad1.SetValue(py, valy + ey.second);
   FunctionMinimum eyx_up = migrad1();
   nfcn += eyx_up.NFcn();
   if(!eyx_up.IsValid()) {
      MN_ERROR_VAL2("MnContours: unable to find Upper x Value for y Parameter",py);
      return ContoursError(px, py, result, mex, mey, nfcn);
   }
   
   migrad1.SetValue(py, valy + ey.first);
   FunctionMinimum eyx_lo = migrad1();
   nfcn += eyx_lo.NFcn();
   if(!eyx_lo.IsValid()) {
      MN_ERROR_VAL2("MnContours: unable to find Lower x Value for y Parameter",py);
      return ContoursError(px, py, result, mex, mey, nfcn);
   }
   
   double scalx = 1./(ex.second - ex.first);
   double scaly = 1./(ey.second - ey.first);
   
   result.push_back(std::pair<double,double>(valx + ex.first, exy_lo.UserState().Value(py)));
   result.push_back(std::pair<double,double>(eyx_lo.UserState().Value(px), valy + ey.first));
   result.push_back(std::pair<double,double>(valx + ex.second, exy_up.UserState().Value(py)));
   result.push_back(std::pair<double,double>(eyx_up.UserState().Value(px), valy + ey.second));

   
   
   //   std::cout<<"MnContours: first 4 params finished."<<std::endl;
#ifdef WARNINGMSG
   MN_INFO_MSG("Minuit2::MnContour : List of  found points");
   MN_INFO_VAL2("Minuit2::MnContour : parameter number x",px);
   MN_INFO_VAL2("Minuit2::MnContour : parameter number y",py);
#endif

   for (unsigned int i = 0; i < 4; ++i)
      PrintContourPoint(result[i] ); 
 
   MnUserParameterState upar = fMinimum.UserState();
   upar.Fix(px);
   upar.Fix(py);
   
   std::vector<unsigned int> par(2); par[0] = px; par[1] = py;
   MnFunctionCross cross(fFCN, upar, fMinimum.Fval(), fStrategy);
   
   for(unsigned int i = 4; i < npoints; i++) {
      
      std::vector<std::pair<double,double> >::iterator idist1 = result.end()-1;
      std::vector<std::pair<double,double> >::iterator idist2 = result.begin();
      double dx = idist1->first - (idist2)->first;
      double dy = idist1->second - (idist2)->second;
      double bigdis = scalx*scalx*dx*dx + scaly*scaly*dy*dy;
      
      for(std::vector<std::pair<double,double> >::iterator ipair = result.begin(); ipair != result.end()-1; ipair++) {
         double distx = ipair->first - (ipair+1)->first;
         double disty = ipair->second - (ipair+1)->second;
         double dist = scalx*scalx*distx*distx + scaly*scaly*disty*disty;
         if(dist > bigdis) {
            bigdis = dist;
            idist1 = ipair;
            idist2 = ipair+1;
         }
      }
      
      double a1 = 0.5;
      double a2 = 0.5;
      double sca = 1.;
      
L300:
         
      if(nfcn > maxcalls) {
         MN_ERROR_MSG("MnContours: maximum number of function calls exhausted.");
         return ContoursError(px, py, result, mex, mey, nfcn);
      }
      
      double xmidcr = a1*idist1->first + a2*(idist2)->first;
      double ymidcr = a1*idist1->second + a2*(idist2)->second;
      double xdir = (idist2)->second - idist1->second;
      double ydir = idist1->first - (idist2)->first;
      double scalfac = sca*std::max(fabs(xdir*scalx), fabs(ydir*scaly));
      double xdircr = xdir/scalfac;
      double ydircr = ydir/scalfac;
      std::vector<double> pmid(2); pmid[0] = xmidcr; pmid[1] = ymidcr;
      std::vector<double> pdir(2); pdir[0] = xdircr; pdir[1] = ydircr;
      
      MnCross opt = cross(par, pmid, pdir, toler, maxcalls);
      nfcn += opt.NFcn();
      if(!opt.IsValid()) {
         //       if(a1 > 0.5) {
         if(sca < 0.) {
            MN_ERROR_VAL2("MnContours : unable to find point on Contour",i+1);
            MN_ERROR_VAL2("MnContours : found  only i points",i);
            return ContoursError(px, py, result, mex, mey, nfcn);
         }
         //       a1 = 0.75;
         //       a2 = 0.25;
         //       std::cout<<"*****switch direction"<<std::endl;
         sca = -1.;
         goto L300;
         }
      double aopt = opt.Value();
      if(idist2 == result.begin()) { 
         result.push_back(std::pair<double,double>(xmidcr+(aopt)*xdircr, ymidcr + (aopt)*ydircr));
         PrintContourPoint( result.back() );
      }
      else {
         result.insert(idist2, std::pair<double,double>(xmidcr+(aopt)*xdircr, ymidcr + (aopt)*ydircr));
         PrintContourPoint( *idist2 );
      }
   }   
   return ContoursError(px, py, result, mex, mey, nfcn);
}


   }  // namespace Minuit2

}  // namespace ROOT
