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

#ifndef ROOT_TQpSolverBase
#define ROOT_TQpSolverBase

#ifndef ROOT_TError
#include "TError.h"
#endif

#ifndef ROOT_TQpVar
#include "TQpVar.h"
#endif
#ifndef ROOT_TQpDataBase
#include "TQpDataBase.h"
#endif
#ifndef ROOT_TQpResidual
#include "TQpResidual.h"
#endif
#ifndef ROOT_TQpProbBase
#include "TQpProbBase.h"
#endif
#ifndef ROOT_TQpLinSolverBase
#include "TQpLinSolverBase.h"
#endif

#ifndef ROOT_TMatrixD
#include "TMatrixD.h"
#endif

///////////////////////////////////////////////////////////////////////////
//                                                                       //
// Abstract base class for QP solver using interior-point                //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

enum ETerminationCode
{
   kSUCCESSFUL_TERMINATION = 0,
   kNOT_FINISHED,
   kMAX_ITS_EXCEEDED,
   kINFEASIBLE,
   kUNKNOWN
};

class TQpSolverBase : public TObject
{

protected:
   TQpLinSolverBase *fSys;

   Double_t          fDnorm;                   // norm of problem data

   Double_t          fMutol;                   // termination parameters
   Double_t          fArtol;

   Double_t          fGamma_f;                 // parameters associated with the step length heuristic
   Double_t          fGamma_a;
   Double_t          fPhi;                     // merit function, defined as the sum of the complementarity gap
                                               // the residual norms, divided by (1+norm of problem data)
   Int_t             fMaxit;                   // maximum number of  iterations allowed

   Double_t         *fMu_history;              //[fMaxit] history of values of mu obtained on all iterations to date
   Double_t         *fRnorm_history;           //[fMaxit] history of values of residual norm obtained on all iterations to date
   Double_t         *fPhi_history;             //[fMaxit] history of values of phi obtained on all iterations to date

   Double_t         *fPhi_min_history;         //[fMaxit] the i-th entry of this array contains the minimum value of phi
                                               //          encountered by the algorithm on or before iteration i

public:
   Int_t             fIter;                    // iteration counter

   TQpSolverBase();
   TQpSolverBase(const TQpSolverBase &another);

   virtual ~TQpSolverBase();

   virtual void     Start       (TQpProbBase *formulation,
                                 TQpVar *iterate,TQpDataBase *prob,
                                 TQpResidual *resid,TQpVar *step);
                                               // starting point heuristic
   virtual void     DefStart    (TQpProbBase *formulation,
                                 TQpVar *iterate,TQpDataBase *prob,
                                 TQpResidual *resid,TQpVar *step);
                                               // default starting point heuristic
   virtual void     SteveStart  (TQpProbBase *formulation,
                                 TQpVar *iterate,TQpDataBase *prob,
                                 TQpResidual *resid,TQpVar *step);
                                               // alternative starting point heuristic
   virtual void     DumbStart   (TQpProbBase *formulation,
                                 TQpVar *iterate,TQpDataBase *prob,
                                 TQpResidual *resid,TQpVar *step);
                                               // alternative starting point heuristic: sets the
                                               // "complementary" variables to a large positive value
                                                // (based on the norm of the problem data) and the
                                               // remaining variables to zero

   virtual Int_t    Solve       (TQpDataBase *prob,TQpVar *iterate,
                                 TQpResidual *resids) = 0;
                                               // implements the interior-point method for solving the QP

   virtual Double_t FinalStepLength
                                (TQpVar *iterate,TQpVar *step);
                                               // Mehrotra's heuristic to calculate the final step

   virtual void     DoMonitor   (TQpDataBase *data,TQpVar *vars,
                                 TQpResidual *resids,Double_t alpha,
                                 Double_t sigma,Int_t i,Double_t mu,
                                 Int_t stop_code,Int_t level);
                                               // perform monitor operation at each interior-point iteration
   virtual void     DefMonitor  (TQpDataBase *data,TQpVar *vars,
                                 TQpResidual *resids,Double_t alpha,
                                 Double_t sigma,Int_t i,Double_t mu,
                                 Int_t stop_code,Int_t level) = 0;
                                               // default monitor: prints out one line of information
                                               // on each interior-point iteration
   virtual Int_t    DoStatus    (TQpDataBase *data,TQpVar *vars,
                                 TQpResidual *resids,Int_t i,Double_t mu,
                                 Int_t level);
                                               // this method called to test for convergence status at
                                               // at the end of each interior-point iteration
   virtual Int_t    DefStatus   (TQpDataBase *data,TQpVar *vars,
                                 TQpResidual *resids,Int_t i,Double_t mu,
                                 Int_t level);
                                               // default method for checking status. May be replaced
                                               // by a user-defined method

   TQpLinSolverBase *GetLinearSystem()            { return fSys; }
   void              SetMuTol       (Double_t m)  { fMutol = m; }
   Double_t          GetMuTol       ()            { return fMutol; }

   void              SetArTol       (Double_t ar) { fArtol = ar; }
   Double_t          GetArTol       ()            { return fArtol; }
   Double_t          DataNorm       ()            { return fDnorm; }

   TQpSolverBase &operator= (const TQpSolverBase &source);

   ClassDef(TQpSolverBase,1)                   // Qp Solver class
};
#endif
