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

#ifndef ROOT_TQpVar
#define ROOT_TQpVar

#include "TError.h"

#include "TMatrixD.h"
#include "TVectorD.h"

///////////////////////////////////////////////////////////////////////////
//                                                                       //
// Class containing the variables for the general QP formulation         //
// In terms of in our abstract problem formulation, these variables are  //
// the vectors x, y, z and s.                                            //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

class TQpVar : public TObject
{

protected:
   Int_t fNx;
   Int_t fMy;
   Int_t fMz;
   Int_t fNxup;
   Int_t fNxlo;
   Int_t fMcup;
   Int_t fMclo;

   // these variables will be "Used" not copied
   TVectorD fXloIndex;
   TVectorD fXupIndex;
   TVectorD fCupIndex;
   TVectorD fCloIndex;

   static Double_t StepBound      (TVectorD &v,TVectorD &dir,Double_t maxStep);
   static Double_t FindBlocking   (TVectorD &w,TVectorD &wstep,TVectorD &u,TVectorD &ustep,
                                   Double_t maxStep,Double_t &w_elt,Double_t &wstep_elt,Double_t &u_elt,
                                   Double_t &ustep_elt,int& first_or_second);
   static Double_t FindBlockingSub(Int_t n,Double_t *w,Int_t incw,Double_t *wstep,Int_t incwstep,
                                   Double_t *u,Int_t incu,Double_t *ustep,Int_t incustep,
                                   Double_t maxStep,Double_t &w_elt,Double_t &wstep_elt,
                                   Double_t &u_elt,Double_t &ustep_elt,Int_t &first_or_second);

public:

   Int_t fNComplementaryVariables;             // number of complementary primal-dual variables.

   // these variables will be "Used" not copied
   TVectorD fX;
   TVectorD fS;
   TVectorD fY;
   TVectorD fZ;

   TVectorD fV;
   TVectorD fPhi;

   TVectorD fW;
   TVectorD fGamma;

   TVectorD fT;
   TVectorD fLambda;

   TVectorD fU;
   TVectorD fPi;

   TQpVar();
   // constructor in which the data and variable pointers are set to point to the given arguments
   TQpVar(TVectorD &x_in,TVectorD &s_in,TVectorD &y_in,TVectorD &z_in,
          TVectorD &v_in,TVectorD &gamma_in,TVectorD &w_in,TVectorD &phi_in,
          TVectorD &t_in,TVectorD &lambda_in,TVectorD &u_in,TVectorD &pi_in,
          TVectorD &ixlow_in,TVectorD &ixupp_in,TVectorD &iclow_in,TVectorD &icupp_in);

   // constructor that creates variables objects of specified dimensions.
   TQpVar(Int_t nx,Int_t my,Int_t mz,
      TVectorD &ixlow,TVectorD &ixupp,TVectorD &iclow,TVectorD &icupp);
   TQpVar(const TQpVar &another);

   ~TQpVar() override {}

   // Indicates what type is the blocking variable in the step length determination. If kt_block,
   // then the blocking variable is one of the slack variables t for a general lower bound,
   // and so on. Special value kno_block is for the case in which a step length of 1 can be
   // taken without hitting the bound.

   enum EVarBlock { kno_block,kt_block,klambda_block,ku_block,kpi_block,
                    kv_block,kgamma_block,kw_block,kphi_block };

   virtual Double_t GetMu       ();            // compute complementarity gap, obtained by taking the
                                               // inner product of the complementary vectors and dividing
                                               // by the total number of components
                                               // computes mu = (t'lambda +u'pi + v'gamma + w'phi)/
                                               //                    (mclow+mcupp+nxlow+nxupp)
   virtual Double_t MuStep      (TQpVar *step,Double_t alpha);
                                               // compute the complementarity gap resulting from a step
                                               // of length "alpha" along direction "step"
   virtual void     Saxpy       (TQpVar *b,Double_t alpha);
                                               // given variables b, compute a <- a + alpha b,
                                               // where a are the variables in this class

   virtual void     Negate      ();            // negate the value of all the variables in this structure

   virtual Double_t StepBound   (TQpVar *b);   // calculate the largest alpha in (0,1] such that the
                                               // nonnegative variables stay nonnegative in the given
                                               // search direction. In the general QP problem formulation
                                               // this is the largest value of alpha such that
                                               // (t,u,v,w,lambda,pi,phi,gamma) + alpha * (b->t,b->u,
                                               //   b->v,b->w,b->lambda,b->pi,b->phi,b->gamma) >= 0.

   virtual Double_t FindBlocking(TQpVar *step,Double_t &primalValue,Double_t &primalStep,Double_t &dualValue,
                                 Double_t &dualStep,Int_t &firstOrSecond);
                                               // Performs the same function as StepBound, and supplies
                                               // additional information about which component of the
                                               // nonnegative variables is responsible for restricting
                                               // alpha. In terms of the abstract formulation, the
                                               // components have the following meanings.
                                               //
                                               // primalValue: the value of the blocking component of the
                                               // primal variables (u,t,v,w).
                                               // primalStep: the corresponding value of the blocking
                                               // component of the primal step variables (b->u,b->t,
                                               // b->v,b->w)
                                               // dualValue: the value of the blocking component of the
                                               // dual variables (lambda,pi,phi,gamma).
                                               // dualStep: the corresponding value of the blocking
                                               // component of the dual step variables (b->lambda,b->pi,
                                               // b->phi,b->gamma)
                                               // firstOrSecond:  1 if the primal step is blocking, 2
                                               // if the dual step is block, 0 if no step is blocking.

   virtual void     InteriorPoint(Double_t alpha,Double_t beta);
                                               // sets components of (u,t,v,w) to alpha and of
                                               // (lambda,pi,phi,gamma) to beta
   virtual void     ShiftBoundVariables
                                 (Double_t alpha,Double_t beta);
                                               // add alpha to components of (u,t,v,w) and beta to
                                               // components of (lambda,pi,phi,gamma)
   virtual Bool_t   IsInteriorPoint();         // check whether this is an interior point. Useful as a
                                               // sanity check.
   virtual Double_t Violation    ();           // The amount by which the current variables violate the
                                               //  non-negativity constraints.
   void     Print        (Option_t *option="") const override;
   virtual Double_t Norm1        ();           // compute the 1-norm of the variables
   virtual Double_t NormInf      ();           // compute the inf-norm of the variables
   virtual Bool_t   ValidNonZeroPattern();

   TQpVar &operator= (const TQpVar &source);

   ClassDefOverride(TQpVar,1)                          // Qp Variables class
};
#endif
