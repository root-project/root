// @(#)root/minuit2:$Id$
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

//#define DEBUG

#if defined(DEBUG) || defined(WARNINGMSG)
#include "Minuit2/MnPrint.h"
#endif


namespace ROOT {

   namespace Minuit2 {


MnMinos::MnMinos(const FCNBase& fcn, const FunctionMinimum& min, unsigned int stra ) :
   fFCN(fcn),
   fMinimum(min),
   fStrategy(MnStrategy(stra))
{
   // construct from FCN + Minimum
   // check if Error definition  has been changed, in case re-update errors
   if (fcn.Up() != min.Up() ) {
#ifdef WARNINGMSG
      MN_INFO_MSG("MnMinos UP value has changed, need to update FunctionMinimum class");
#endif
   }
}

MnMinos::MnMinos(const FCNBase& fcn, const FunctionMinimum& min,  const MnStrategy& stra) :
   fFCN(fcn),
   fMinimum(min),
   fStrategy(stra)
{
   // construct from FCN + Minimum
   // check if Error definition  has been changed, in case re-update errors
   if (fcn.Up() != min.Up() ) {
#ifdef WARNINGMSG
      MN_INFO_MSG("MnMinos UP value has changed, need to update FunctionMinimum class");
#endif
   }
}


std::pair<double,double> MnMinos::operator()(unsigned int par, unsigned int maxcalls, double toler) const {
   // do Minos analysis given the parameter index returning a pair for (lower,upper) errors
   MinosError mnerr = Minos(par, maxcalls,toler);
   return mnerr();
}

double MnMinos::Lower(unsigned int par, unsigned int maxcalls, double toler) const {
   // get lower error for parameter par

   MnCross aopt = Loval(par, maxcalls,toler);

   MinosError mnerr(par, fMinimum.UserState().Value(par), aopt, MnCross());

   return mnerr.Lower();
}

double MnMinos::Upper(unsigned int par, unsigned int maxcalls, double toler) const {
   // upper error for parameter par

   MnCross aopt = Upval(par, maxcalls,toler);

   MinosError mnerr(par, fMinimum.UserState().Value(par), MnCross(), aopt);

   return mnerr.Upper();
}

MinosError MnMinos::Minos(unsigned int par, unsigned int maxcalls, double toler) const {
   // do full minos error anlysis (lower + upper) for parameter par

   MnCross up = Upval(par, maxcalls,toler);
#ifdef DEBUG
   std::cout << "Function calls to find upper error " << up.NFcn() << std::endl;
#endif

   MnCross lo = Loval(par, maxcalls,toler);

#ifdef DEBUG
   std::cout << "Function calls to find lower error " << lo.NFcn() << std::endl;
#endif

   std::cout << "return Minos error " << lo.Value() << "  , " << up.Value() << std::endl;

   return MinosError(par, fMinimum.UserState().Value(par), lo, up);
}


MnCross MnMinos::FindCrossValue(int direction, unsigned int par, unsigned int maxcalls, double toler) const {
   // get crossing value in the parameter direction :
   // direction = + 1 upper value
   // direction = -1 lower value
   // pass now tolerance used for Migrad minimizations

   assert(direction == 1 || direction == -1);

   int printLevel = MnPrint::Level();
   if (printLevel > 2) {
      if (direction == 1)
         std::cout << "--------- MnMinos -------\nDetermination of upper Minos error for parameter "
                   << par << std::endl;
      else
         std::cout << "--------- MnMinos -------\nDetermination of lower Minos error for parameter "
                   << par << std::endl;
   }

   assert(fMinimum.IsValid());
   assert(!fMinimum.UserState().Parameter(par).IsFixed());
   assert(!fMinimum.UserState().Parameter(par).IsConst());

   if(maxcalls == 0) {
      unsigned int nvar = fMinimum.UserState().VariableParameters();
      maxcalls = 2*(nvar+1)*(200 + 100*nvar + 5*nvar*nvar);
   }

   std::vector<unsigned int> para(1, par);

   MnUserParameterState upar = fMinimum.UserState();
   double err = direction * upar.Error(par);
   double val = upar.Value(par) +  err;
   // check if we do not cross limits
   if (direction == 1 && upar.Parameter(par).HasUpperLimit()) {
      val = std::min(val, upar.Parameter(par).UpperLimit());
   }
   if (direction == -1 && upar.Parameter(par).HasLowerLimit()) {
      val = std::max(val, upar.Parameter(par).LowerLimit());
   }
   // recompute err in case it was truncated for the limit
   err = val-upar.Value(par);
   std::vector<double> xmid(1, val);
   std::vector<double> xdir(1, err);

   double up = fFCN.Up();
   unsigned int ind = upar.IntOfExt(par);
   // get error matrix (methods return a copy)
   MnAlgebraicSymMatrix m = fMinimum.Error().Matrix();
   // get internal parameters
   const MnAlgebraicVector & xt = fMinimum.Parameters().Vec();
   //LM:  change to use err**2 (m(i,i) instead of err as in F77 version
   double xunit = sqrt(up/m(ind,ind));
   // LM (29/04/08) bug: change should be done in internal variables
   // set the initial value for the other parmaeters that we are going to fit in MnCross
   for(unsigned int i = 0; i < m.Nrow(); i++) {
      if(i == ind) continue;
      double xdev = xunit*m(ind,i);
      double xnew = xt(i) + direction *  xdev;

      // transform to external values
      unsigned int ext = upar.ExtOfInt(i);

      double unew = upar.Int2ext(i, xnew);

      // take into account limits
      if (upar.Parameter(ext).HasUpperLimit()) {
         unew = std::min(unew, upar.Parameter(ext).UpperLimit());
      }
      if (upar.Parameter(ext).HasLowerLimit()) {
         unew = std::max(unew, upar.Parameter(ext).LowerLimit());
      }

#ifdef DEBUG
      std::cout << "Parameter " << ext << " is set from " << upar.Value(ext) << " to " <<  unew << std::endl;
#endif
      upar.SetValue(ext, unew);
   }

   upar.Fix(par);
   upar.SetValue(par, val);

#ifdef DEBUG
   std::cout << "Parameter " << par << " is fixed and set from " << fMinimum.UserState().Value(par) << " to " << val
   << " delta = " << err << std::endl;
#endif


   MnFunctionCross cross(fFCN, upar, fMinimum.Fval(), fStrategy);
   MnCross aopt = cross(para, xmid, xdir, toler, maxcalls);


#ifdef DEBUG
   std::cout<<"----- MnMinos: aopt value found from MnFunctionCross = "<<aopt.Value()<<std::endl << std::endl;
#endif

#ifdef WARNINGMSG
   const char * par_name = upar.Name(par);
   if(aopt.AtMaxFcn())
      MN_INFO_VAL2("MnMinos maximum number of function calls exceeded for Parameter ",par_name);
   if(aopt.NewMinimum())
      MN_INFO_VAL2("MnMinos new Minimum found while looking for Parameter ",par_name);
   if (direction ==1) {
      if(aopt.AtLimit())
         MN_INFO_VAL2("MnMinos: parameter is at Upper limit.",par_name);
      if(!aopt.IsValid())
         MN_INFO_VAL2("MnMinos: could not find Upper Value for Parameter ",par_name);
   }
   else {
      if(aopt.AtLimit())
         MN_INFO_VAL2("MnMinos: parameter is at Lower limit.",par_name);
      if(!aopt.IsValid())
         MN_INFO_VAL2("MnMinos: could not find Lower Value for Parameter ",par_name);
   }
#endif

   if (printLevel > 2) {
      std::string scanType = (direction == 1) ? "up" : "low";
      std::cout << " ------ end Minos scan for " << scanType << " interval for parameter " << upar.Name(par) << std::endl;
   }

   return aopt;
}

MnCross MnMinos::Upval(unsigned int par, unsigned int maxcalls, double toler) const {
   // return crossing in the lower parameter direction
   return FindCrossValue(1,par,maxcalls,toler);
}

MnCross MnMinos::Loval(unsigned int par, unsigned int maxcalls, double toler) const {
   // return crossing in the lower parameter direction
   return FindCrossValue(-1,par,maxcalls,toler);
}


   }  // namespace Minuit2

}  // namespace ROOT
