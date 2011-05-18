// @(#)root/quadp:$Id$
// Author: Eddy Offermann   May 2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*************************************************************************
 * Parts of this file are copied from the OOQP distribution and          *
 * are subject to the following license:                                 *
 *                                                                       *
 * COPYRIGHT 2001 UNIVERSITY OF CHICAGO                                  *
 *                                                                       *
 * The copyright holder hereby grants you royalty-free rights to use,    *
 * reproduce, prepare derivative works, and to redistribute this software*
 * to others, provided that any changes are clearly documented. This     *
 * software was authored by:                                             *
 *                                                                       *
 *   E. MICHAEL GERTZ      gertz@mcs.anl.gov                             *
 *   Mathematics and Computer Science Division                           *
 *   Argonne National Laboratory                                         *
 *   9700 S. Cass Avenue                                                 *
 *   Argonne, IL 60439-4844                                              *
 *                                                                       *
 *   STEPHEN J. WRIGHT     swright@cs.wisc.edu                           *
 *   Computer Sciences Department                                        *
 *   University of Wisconsin                                             *
 *   1210 West Dayton Street                                             *
 *   Madison, WI 53706   FAX: (608)262-9777                              *
 *                                                                       *
 * Any questions or comments may be directed to one of the authors.      *
 *                                                                       *
 * ARGONNE NATIONAL LABORATORY (ANL), WITH FACILITIES IN THE STATES OF   *
 * ILLINOIS AND IDAHO, IS OWNED BY THE UNITED STATES GOVERNMENT, AND     *
 * OPERATED BY THE UNIVERSITY OF CHICAGO UNDER PROVISION OF A CONTRACT   *
 * WITH THE DEPARTMENT OF ENERGY.                                        *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSolverBase                                                          //
//                                                                      //
// The Solver class contains methods for monitoring and checking the    //
// convergence status of the algorithm, methods to determine the step   //
// length along a given direction, methods to define the starting point,//
// and the solve method that implements the interior-point algorithm    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMath.h"
#include "TQpSolverBase.h"

ClassImp(TQpSolverBase)

//______________________________________________________________________________
TQpSolverBase::TQpSolverBase()
{
// Default constructor

   fSys = 0;

   fDnorm = 0.;

   // define parameters associated with the step length heuristic
   fMutol   = 1.0e-8;
   fArtol   = 1.0e-8;
   fGamma_f = 0.99;
   fGamma_a = 1.0/(1.0-fGamma_f);

   fPhi   = 0.0;

   fMaxit = 100;

   // allocate space to track the sequence of complementarity gaps,
   // residual norms, and merit functions.
   fMu_history      = new Double_t[fMaxit];
   fRnorm_history   = new Double_t[fMaxit];
   fPhi_history     = new Double_t[fMaxit];
   fPhi_min_history = new Double_t[fMaxit];

   fIter = 0;
}


//______________________________________________________________________________
TQpSolverBase::TQpSolverBase(const TQpSolverBase &another) : TObject(another)
{
// Copy constructor

   *this = another;
}


//______________________________________________________________________________
TQpSolverBase::~TQpSolverBase()
{
// Deconstructor

   if (fSys) { delete fSys; fSys = 0; }

   if (fMu_history) { delete [] fMu_history;      fMu_history      = 0; }
   if (fMu_history) { delete [] fRnorm_history;   fRnorm_history   = 0; }
   if (fMu_history) { delete [] fPhi_history;     fPhi_history     = 0; }
   if (fMu_history) { delete [] fPhi_min_history; fPhi_min_history = 0; }
}


//______________________________________________________________________________
void TQpSolverBase::Start(TQpProbBase *formulation,TQpVar *iterate,TQpDataBase *prob,
                          TQpResidual *resid,TQpVar *step)
{
// Implements a default starting-point heuristic. While interior-point theory
//  places fairly loose restrictions on the choice of starting point, the choice 
// of heuristic can significantly affect the robustness and efficiency of the
//  algorithm.

   this->DefStart(formulation,iterate,prob,resid,step);
}


//______________________________________________________________________________
void TQpSolverBase::DefStart(TQpProbBase * /* formulation */,TQpVar *iterate,
                             TQpDataBase *prob,TQpResidual *resid,TQpVar *step)
{
// Default starting point

   Double_t sdatanorm = TMath::Sqrt(fDnorm);
   Double_t a         = sdatanorm;
   Double_t b         = sdatanorm;

   iterate->InteriorPoint(a,b);
   resid->CalcResids(prob,iterate);
   resid->Set_r3_xz_alpha(iterate,0.0);

   fSys->Factor(prob,iterate);
   fSys->Solve(prob,iterate,resid,step);
   step->Negate();

   // Take the full affine scaling step
   iterate->Saxpy(step,1.0);
   // resid.CalcResids(prob,iterate); // Calc the resids if debugging.
   Double_t shift = 1.e3+2*iterate->Violation();
   iterate->ShiftBoundVariables(shift,shift);
}


//______________________________________________________________________________
void TQpSolverBase::SteveStart(TQpProbBase * /* formulation */,
                               TQpVar *iterate,TQpDataBase *prob,
                               TQpResidual *resid,TQpVar *step)
{
// Starting point algoritm according to Stephen Wright

   Double_t sdatanorm = TMath::Sqrt(fDnorm);
   Double_t a = 0.0;
   Double_t b = 0.0;

   iterate->InteriorPoint(a,b);

   // set the r3 component of the rhs to -(norm of data), and calculate
   // the residuals that are obtained when all values are zero.

   resid->Set_r3_xz_alpha(iterate,-sdatanorm);
   resid->CalcResids(prob,iterate);

   // next, assign 1 to all the complementary variables, so that there
   // are identities in the coefficient matrix when we do the solve.

   a = 1.0; b = 1.0;
   iterate->InteriorPoint(a,b);
   fSys->Factor(prob,iterate);
   fSys->Solve (prob,iterate,resid,step);
   step->Negate();

   // copy the "step" into the current vector

   iterate = step;

   // calculate the maximum violation of the complementarity
   // conditions, and shift these variables to restore positivity.
   Double_t shift = 1.5*iterate->Violation();
   iterate->ShiftBoundVariables(shift,shift);

   // do Mehrotra-type adjustment

   const Double_t mutemp = iterate->GetMu();
   const Double_t xsnorm = iterate->Norm1();
   const Double_t delta = 0.5*iterate->fNComplementaryVariables*mutemp/xsnorm;
   iterate->ShiftBoundVariables(delta,delta);
}


//______________________________________________________________________________
void TQpSolverBase::DumbStart(TQpProbBase * /* formulation */,
                              TQpVar *iterate,TQpDataBase * /* prob */,
                              TQpResidual * /* resid */,TQpVar * /* step */)
{
// Alternative starting point heuristic: sets the "complementary" variables to a large
// positive value (based on the norm of the problem data) and the remaining variables
// to zero .

   const Double_t sdatanorm = fDnorm;
   const Double_t a = 1.e3;
   const Double_t b = 1.e5;
   const Double_t bigstart = a*sdatanorm+b;
   iterate->InteriorPoint(bigstart,bigstart);
}


//______________________________________________________________________________
Double_t TQpSolverBase::FinalStepLength(TQpVar *iterate,TQpVar *step)
{
// Implements a version of Mehrotra starting point heuristic,
//  modified to ensure identical steps in the primal and dual variables.

   Int_t firstOrSecond;
   Double_t primalValue; Double_t primalStep; Double_t dualValue; Double_t dualStep;
   const Double_t maxAlpha = iterate->FindBlocking(step,primalValue,primalStep,
      dualValue,dualStep,firstOrSecond);
   Double_t mufull = iterate->MuStep(step,maxAlpha);

   mufull /= fGamma_a;

   Double_t alpha = 1.0;
   switch (firstOrSecond) {
      case 0:
         alpha = 1;              // No constraints were blocking
         break;
      case 1:
         alpha = (-primalValue+mufull/(dualValue+maxAlpha*dualStep))/primalStep;
         break;
      case 2:
         alpha = (-dualValue+mufull/(primalValue+maxAlpha*primalStep))/dualStep;
         break;
      default:
         R__ASSERT(0 && "Can't get here");
         break;
   }
   // make it at least fGamma_f * maxStep
   if (alpha < fGamma_f*maxAlpha) alpha = fGamma_f*maxAlpha;
   // back off just a touch
   alpha *= .99999999;

   return alpha;
}


//______________________________________________________________________________
void TQpSolverBase::DoMonitor(TQpDataBase *data,TQpVar *vars,TQpResidual *resids,
                              Double_t alpha,Double_t sigma,Int_t i,Double_t mu,
                              Int_t stop_code,Int_t level)
{
// Monitor progress / convergence aat each interior-point iteration

   this->DefMonitor(data,vars,resids,alpha,sigma,i,mu,stop_code,level);
}


//______________________________________________________________________________
Int_t TQpSolverBase::DoStatus(TQpDataBase *data,TQpVar *vars,TQpResidual *resids,
                              Int_t i,Double_t mu,Int_t level)
{
// Tests for termination. Unless the user supplies a specific termination 
// routine, this method calls another method defaultStatus, which returns 
// a code indicating the current convergence status.

   return this->DefStatus(data,vars,resids,i,mu,level);
}


//______________________________________________________________________________
Int_t TQpSolverBase::DefStatus(TQpDataBase * /* data */,TQpVar * /* vars */,
                               TQpResidual *resids,Int_t iterate,Double_t mu,Int_t /* level */)
{
// Default status method

   Int_t stop_code = kNOT_FINISHED;

   const Double_t gap   = TMath::Abs(resids->GetDualityGap());
   const Double_t rnorm = resids->GetResidualNorm();

   Int_t idx = iterate-1;
   if (idx <  0     ) idx = 0;
   if (idx >= fMaxit) idx = fMaxit-1;

   // store the historical record
   fMu_history[idx] = mu;
   fRnorm_history[idx] = rnorm;
   fPhi = (rnorm+gap)/fDnorm;
   fPhi_history[idx] = fPhi;

   if (idx > 0) {
      fPhi_min_history[idx] = fPhi_min_history[idx-1];
      if (fPhi < fPhi_min_history[idx]) fPhi_min_history[idx] = fPhi;
   } else
   fPhi_min_history[idx] = fPhi;

   if (iterate >= fMaxit) {
      stop_code = kMAX_ITS_EXCEEDED;
   }
   else if (mu <= fMutol && rnorm <= fArtol*fDnorm) {
      stop_code = kSUCCESSFUL_TERMINATION;
   }
   if (stop_code != kNOT_FINISHED)  return stop_code;

   // check infeasibility condition
   if (idx >= 10 && fPhi >= 1.e-8 && fPhi >= 1.e4*fPhi_min_history[idx])
      stop_code = kINFEASIBLE;
   if (stop_code != kNOT_FINISHED)  return stop_code;

   // check for unknown status: slow convergence first
   if (idx >= 30 && fPhi_min_history[idx] >= .5*fPhi_min_history[idx-30])
      stop_code = kUNKNOWN;

   if (rnorm/fDnorm > fArtol &&
      (fRnorm_history[idx]/fMu_history[idx])/(fRnorm_history[0]/fMu_history[0]) >= 1.e8)
      stop_code = kUNKNOWN;

   return stop_code;
}


//______________________________________________________________________________
TQpSolverBase &TQpSolverBase::operator=(const TQpSolverBase &source)
{
// Assignment operator

   if (this != &source) {
      TObject::operator=(source);

      fSys     = source.fSys;
      fDnorm   = source.fDnorm;
      fMutol   = source.fMutol;
      fArtol   = source.fArtol;
      fGamma_f = source.fGamma_f;
      fGamma_a = source.fGamma_a;
      fPhi     = source.fPhi;
      fIter    = source.fIter;

      if (fMaxit != source.fMaxit) {
         if (fMu_history) delete [] fMu_history;
         fMu_history = new Double_t[fMaxit];
         if (fRnorm_history) delete [] fRnorm_history;
         fRnorm_history = new Double_t[fMaxit];
         if (fPhi_history) delete [] fPhi_history;
         fPhi_history = new Double_t[fMaxit];
         if (fPhi_min_history) delete [] fPhi_min_history;
         fPhi_min_history = new Double_t[fMaxit];
      }

      fMaxit = source.fMaxit;
      memcpy(fMu_history,source.fMu_history,fMaxit*sizeof(Double_t));
      memcpy(fRnorm_history,source.fRnorm_history,fMaxit*sizeof(Double_t));
      memcpy(fPhi_history,source.fPhi_history,fMaxit*sizeof(Double_t));
      memcpy(fPhi_min_history,source.fPhi_min_history,fMaxit*sizeof(Double_t));
   }
   return *this;
}
