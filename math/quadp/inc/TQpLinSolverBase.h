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

#ifndef ROOT_TQpLinSolverBase
#define ROOT_TQpLinSolverBase

#include "TError.h"

#include "TQpVar.h"
#include "TQpDataBase.h"
#include "TQpResidual.h"
#include "TQpProbBase.h"

#include "TMatrixD.h"

///////////////////////////////////////////////////////////////////////////
//                                                                       //
// Implements the main solver for linear systems that arise in           //
// primal-dual interior-point methods for QP . This class  contains      //
// definitions of methods and data common to the sparse and dense        //
// special cases of the general formulation. The derived classes contain //
// the aspects that are specific to the sparse and dense forms.          //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

class TQpProbBase;
class TQpLinSolverBase : public TObject
{

protected:

   TVectorD     fNomegaInv;                    // stores a critical diagonal matrix as a vector
   TVectorD     fRhs;                          // right-hand side of the system

   Int_t        fNx;                           // dimensions of the vectors in the general QP formulation
   Int_t        fMy;
   Int_t        fMz;

   TVectorD     fDd;                           // temporary storage vectors
   TVectorD     fDq;

   TVectorD     fXupIndex;                     // index matrices for the upper and lower bounds on x and Cx
   TVectorD     fCupIndex;
   TVectorD     fXloIndex;
   TVectorD     fCloIndex;

   Int_t        fNxup;                         // dimensions of the upper and lower bound vectors
   Int_t        fNxlo;
   Int_t        fMcup;
   Int_t        fMclo;

   TQpProbBase *fFactory;

public:
   TQpLinSolverBase();
   TQpLinSolverBase(TQpProbBase *factory,TQpDataBase *data);
   TQpLinSolverBase(const TQpLinSolverBase &another);

   ~TQpLinSolverBase() override {}

   virtual void Factor          (TQpDataBase *prob,TQpVar *vars);
                                               // sets up the matrix for the main linear system in
                                               // "augmented system" form. The actual factorization is
                                               // performed by a routine specific to either the sparse
                                               // or dense case
   virtual void Solve           (TQpDataBase *prob,TQpVar *vars,TQpResidual *resids,TQpVar *step);
                                               // solves the system for a given set of residuals.
                                               // Assembles the right-hand side appropriate to the
                                               // matrix factored in factor, solves the system using
                                               // the factorization produced there, partitions the
                                               // solution vector into step components, then recovers
                                               // the step components eliminated during the block
                                               // elimination that produced the augmented system form
   virtual void JoinRHS         (TVectorD &rhs, TVectorD &rhs1,TVectorD &rhs2,TVectorD &rhs3);
                                               // assembles a single vector object from three given vectors
                                               //  rhs (output) final joined vector
                                               //  rhs1 (input) first part of rhs
                                               //  rhs2 (input) middle part of rhs
                                               //  rhs3 (input) last part of rhs
   virtual void SeparateVars    (TVectorD &vars1,TVectorD &vars2,TVectorD &vars3,TVectorD &vars);
                                               // extracts three component vectors from a given aggregated
                                               // vector.
                                               //  vars (input) aggregated vector
                                               //  vars1 (output) first part of vars
                                               //  vars2 (output) middle part of vars
                                               //  vars3 (output) last part of vars

   virtual void SolveXYZS       (TVectorD &stepx,TVectorD &stepy,TVectorD &stepz,TVectorD &steps,
                                 TVectorD &ztemp,TQpDataBase *data);
                                               // assemble right-hand side of augmented system and call
                                               // SolveCompressed to solve it

   virtual void SolveCompressed (TVectorD &rhs) = 0;
                                               // perform the actual solve using the factors produced in
                                               // factor.
                                               //  rhs on input contains the aggregated right-hand side of
                                               //  the augmented system; on output contains the solution in
                                               //  aggregated form

   virtual void PutXDiagonal    (TVectorD &xdiag) = 0;
                                               // places the diagonal resulting from the bounds on x into
                                               // the augmented system matrix
   virtual void PutZDiagonal    (TVectorD& zdiag) = 0;
                                               // places the diagonal resulting from the bounds on Cx into
                                               // the augmented system matrix
   virtual void ComputeDiagonals(TVectorD &dd,TVectorD &omega,TVectorD &t, TVectorD &lambda,
                                 TVectorD &u, TVectorD &pi,TVectorD &v, TVectorD &gamma,
                                 TVectorD &w, TVectorD &phi);
                                               // computes the diagonal matrices in the augmented system
                                               // from the current set of variables

   TQpLinSolverBase &operator= (const TQpLinSolverBase &source);

   ClassDefOverride(TQpLinSolverBase,1)                // Qp linear solver base class
};
#endif
