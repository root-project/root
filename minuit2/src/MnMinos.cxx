// @(#)root/minuit2:$Name:  $:$Id: MnMinos.cxx,v 1.3 2006/07/03 22:06:42 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/MnMinos.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/FCNBase.h"
#include "Minuit2/MnFunctionCross.h"
#include "Minuit2/MnCross.h"
#include "Minuit2/MinosError.h"
#include "Minuit2/MnPrint.h"

namespace ROOT {

   namespace Minuit2 {


std::pair<double,double> MnMinos::operator()(unsigned int par, unsigned int maxcalls) const {
   // do Minos analysis given the parameter index returning a pair for (lower,upper) errors
   MinosError mnerr = Minos(par, maxcalls);
   return mnerr();
}

double MnMinos::Lower(unsigned int par, unsigned int maxcalls) const {
   // get lower error for parameter par
   MnUserParameterState upar = fMinimum.UserState();
   double err = fMinimum.UserState().Error(par);
   
   MnCross aopt = Loval(par, maxcalls);
   
   double lower = aopt.IsValid() ? -1.*err*(1.+ aopt.Value()) : (aopt.AtLimit() ? upar.Parameter(par).LowerLimit() : upar.Value(par));
   
   return lower;
}

double MnMinos::Upper(unsigned int par, unsigned int maxcalls) const {
   // upper error for parameter par
   MnCross aopt = Upval(par, maxcalls);
   
   MnUserParameterState upar = fMinimum.UserState();
   double err = fMinimum.UserState().Error(par);
   
   double upper = aopt.IsValid() ? err*(1.+ aopt.Value()) : (aopt.AtLimit() ? upar.Parameter(par).UpperLimit() : upar.Value(par));
   
   return upper;
}

MinosError MnMinos::Minos(unsigned int par, unsigned int maxcalls) const {
   // do full minos error anlysis (lower + upper) for parameter par 
   assert(fMinimum.IsValid());  
   assert(!fMinimum.UserState().Parameter(par).IsFixed());
   assert(!fMinimum.UserState().Parameter(par).IsConst());
   
   MnCross up = Upval(par, maxcalls);
   MnCross lo = Loval(par, maxcalls);
   
   return MinosError(par, fMinimum.UserState().Value(par), lo, up);
}

MnCross MnMinos::Upval(unsigned int par, unsigned int maxcalls) const {
   // get crossing value in the upper parameter direction 
   assert(fMinimum.IsValid());  
   assert(!fMinimum.UserState().Parameter(par).IsFixed());
   assert(!fMinimum.UserState().Parameter(par).IsConst());
   if(maxcalls == 0) {
      unsigned int nvar = fMinimum.UserState().VariableParameters();
      maxcalls = 2*(nvar+1)*(200 + 100*nvar + 5*nvar*nvar);
   }
   
   std::vector<unsigned int> para(1, par);
   
   MnUserParameterState upar = fMinimum.UserState();
   double err = upar.Error(par);
   double val = upar.Value(par) + err;
   std::vector<double> xmid(1, val);
   std::vector<double> xdir(1, err);
   
   double up = fFCN.Up();
   unsigned int ind = upar.IntOfExt(par);
   MnAlgebraicSymMatrix m = fMinimum.Error().Matrix();
   double xunit = sqrt(up/err);
   for(unsigned int i = 0; i < m.Nrow(); i++) {
      if(i == ind) continue;
      double xdev = xunit*m(ind,i);
      unsigned int ext = upar.ExtOfInt(i);
      upar.SetValue(ext, upar.Value(ext) + xdev);
   }
   
   upar.Fix(par);
   upar.SetValue(par, val);
   
   //   double edmmax = 0.5*0.1*fFCN.Up()*1.e-3;
   double toler = 0.1;
   MnFunctionCross cross(fFCN, upar, fMinimum.Fval(), fStrategy);
   
   MnCross aopt = cross(para, xmid, xdir, toler, maxcalls);
   
   //   std::cout<<"aopt= "<<aopt.Value()<<std::endl;
   
#ifdef WARNINGMSG
   if(aopt.AtLimit()) 
      std::cout<<"MnMinos Parameter "<<par<<" is at Upper limit."<<std::endl;
   if(aopt.AtMaxFcn())
      std::cout<<"MnMinos maximum number of function calls exceeded for Parameter "<<par<<std::endl;   
   if(aopt.NewMinimum())
      std::cout<<"MnMinos new Minimum found while looking for Parameter "<<par<<std::endl;     
   if(!aopt.IsValid()) 
      std::cout<<"MnMinos could not find Upper Value for Parameter "<<par<<"."<<std::endl;
#endif
   
   return aopt;
}

MnCross MnMinos::Loval(unsigned int par, unsigned int maxcalls) const {
   // return crossing in the lower parameter direction
   assert(fMinimum.IsValid());  
   assert(!fMinimum.UserState().Parameter(par).IsFixed());
   assert(!fMinimum.UserState().Parameter(par).IsConst());
   if(maxcalls == 0) {
      unsigned int nvar = fMinimum.UserState().VariableParameters();
      maxcalls = 2*(nvar+1)*(200 + 100*nvar + 5*nvar*nvar);
   }
   std::vector<unsigned int> para(1, par);
   
   MnUserParameterState upar = fMinimum.UserState();
   double err = upar.Error(par);
   double val = upar.Value(par) - err;
   std::vector<double> xmid(1, val);
   std::vector<double> xdir(1, -err);
   
   double up = fFCN.Up();
   unsigned int ind = upar.IntOfExt(par);
   MnAlgebraicSymMatrix m = fMinimum.Error().Matrix();
   double xunit = sqrt(up/err);
   for(unsigned int i = 0; i < m.Nrow(); i++) {
      if(i == ind) continue;
      double xdev = xunit*m(ind,i);
      unsigned int ext = upar.ExtOfInt(i);
      upar.SetValue(ext, upar.Value(ext) - xdev);
   }
   
   upar.Fix(par);
   upar.SetValue(par, val);
   
   //   double edmmax = 0.5*0.1*fFCN.Up()*1.e-3;
   double toler = 0.1;
   MnFunctionCross cross(fFCN, upar, fMinimum.Fval(), fStrategy);
   
   MnCross aopt = cross(para, xmid, xdir, toler, maxcalls);
   
   //   std::cout<<"aopt= "<<aopt.Value()<<std::endl;
   
#ifdef WARNINGMSG
   if(aopt.AtLimit()) 
      std::cout<<"MnMinos Parameter "<<par<<" is at Lower limit."<<std::endl;
   if(aopt.AtMaxFcn())
      std::cout<<"MnMinos maximum number of function calls exceeded for Parameter "<<par<<std::endl;   
   if(aopt.NewMinimum())
      std::cout<<"MnMinos new Minimum found while looking for Parameter "<<par<<std::endl;     
   if(!aopt.IsValid()) 
      std::cout<<"MnMinos could not find Lower Value for Parameter "<<par<<"."<<std::endl;
#endif
   
   return aopt;
   
}


   }  // namespace Minuit2

}  // namespace ROOT
