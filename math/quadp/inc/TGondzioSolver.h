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

#ifndef ROOT_TGondzioSolver
#define ROOT_TGondzioSolver

#include "Rtypes.h"
#include "TQpSolverBase.h"

///////////////////////////////////////////////////////////////////////////
//                                                                       //
// Derived class of TQpSolverBase implementing Gondzio-correction        //
// version of Mehrotra's original predictor-corrector algorithm.         //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

class TGondzioSolver : public TQpSolverBase
{

protected:

   Int_t        fPrintlevel;                   // parameter in range [0,100] determines verbosity. (Higher value
                                               //  => more verbose.)
   Double_t     fTsig;                         // exponent in Mehrotra's centering parameter, which is usually
                                               // chosen to me (muaff/mu)^tsig, where muaff is the predicted
                                               // complementarity gap obtained from an affine-scaling step, while
                                               // mu is the current complementarity gap

   Int_t        fMaximum_correctors;           // maximum number of Gondzio corrector steps

   Int_t        fNumberGondzioCorrections;     // actual number of Gondzio corrections needed

   Double_t     fStepFactor0;                  // various parameters associated with Gondzio correction
   Double_t     fStepFactor1;
   Double_t     fAcceptTol;
   Double_t     fBeta_min;
   Double_t     fBeta_max;

   TQpVar      *fCorrector_step;               // storage for step vectors
   TQpVar      *fStep;

   TQpResidual *fCorrector_resid;              // storage for residual vectors

   TQpProbBase *fFactory;

public:

   TGondzioSolver();
   TGondzioSolver(TQpProbBase *of,TQpDataBase *prob,Int_t verbose=0);
   TGondzioSolver(const TGondzioSolver &another);

   ~TGondzioSolver() override;

   Int_t Solve           (TQpDataBase *prob,TQpVar *iterate,TQpResidual *resid) override;

   virtual void  Reset_parameters() {}         // reset parameters to their default values

   void  DefMonitor      (TQpDataBase *data,TQpVar *vars,TQpResidual *resids,
                                  Double_t alpha,Double_t sigma,Int_t i,Double_t mu,
                                  Int_t status_code,Int_t level) override;

   TGondzioSolver &operator=(const TGondzioSolver &source);

   ClassDefOverride(TGondzioSolver,1)                  // Gondzio Qp Solver class
};
#endif
