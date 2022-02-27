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

#ifndef ROOT_TQpResidual
#define ROOT_TQpResidual

#include "TError.h"

#include "TQpVar.h"
#include "TQpDataDens.h"

#include "TMatrixD.h"

///////////////////////////////////////////////////////////////////////////
//                                                                       //
// Class containing the residuals of a QP when solved by an interior     //
// point QP solver. In terms of our abstract QP formulation, these       //
// residuals are rQ, rA, rC and r3.                                      //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

class TQpResidual : public TObject
{

protected:
   Double_t fResidualNorm;                     // The norm of the residuals, ommiting the complementariy conditions
   Double_t fDualityGap;                       // A quantity that measures progress toward feasibility. In terms
                                               //  of the abstract problem formulation, this quantity is defined as
                                               //   x' * Q * x - b' * y + c' * x - d' * z

   Int_t    fNx;
   Int_t    fMy;
   Int_t    fMz;

   Double_t fNxup;
   Double_t fNxlo;
   Double_t fMcup;
   Double_t fMclo;

   // these variables will be "Used" not copied
   TVectorD fXupIndex;
   TVectorD fXloIndex;
   TVectorD fCupIndex;
   TVectorD fCloIndex;

   static void GondzioProjection(TVectorD &v,Double_t rmin,Double_t rmax);

public:
   TVectorD fRQ;
   TVectorD fRA;
   TVectorD fRC;
   TVectorD fRz;
   TVectorD fRv;
   TVectorD fRw;
   TVectorD fRt;
   TVectorD fRu;
   TVectorD fRgamma;
   TVectorD fRphi;
   TVectorD fRlambda;
   TVectorD fRpi;

   TQpResidual();
   TQpResidual(Int_t nx,Int_t my,Int_t mz,
               TVectorD &ixlow,TVectorD &ixupp,TVectorD &iclow,TVectorD &icupp);
   TQpResidual(const TQpResidual &another);

   ~TQpResidual() override {}

   Double_t GetResidualNorm() { return fResidualNorm; }
   Double_t GetDualityGap  () { return fDualityGap; };

   void   CalcResids         (TQpDataBase *problem,TQpVar *vars);
                                               // calculate residuals, their norms, and duality/
                                               // complementarity gap, given a problem and variable set.
   void   Add_r3_xz_alpha    (TQpVar *vars,Double_t alpha);
                                               // Modify the "complementarity" component of the
                                               // residuals, by adding the pairwise products of the
                                               // complementary variables plus a constant alpha to this
                                               // term.
   void   Set_r3_xz_alpha    (TQpVar *vars,Double_t alpha);
                                               // Set the "complementarity" component of the residuals
                                               // to the pairwise products of the complementary variables
                                               // plus a constant alpha
   void   Clear_r3           ();               // set the complementarity component of the residuals
                                               // to 0.
   void   Clear_r1r2         ();               // set the noncomplementarity components of the residual
                                               // (the terms arising from the linear equalities in the
                                               // KKT conditions) to 0.
   void   Project_r3         (Double_t rmin,Double_t rmax);
                                               // perform the projection operation required by Gondzio
                                               // algorithm: replace each component r3_i of the
                                               // complementarity component of the residuals by
                                               // r3p_i-r3_i, where r3p_i is the projection of r3_i onto
                                               // the box [rmin, rmax]. Then if the resulting value is
                                               // less than -rmax, replace it by -rmax.
   Bool_t ValidNonZeroPattern();

   TQpResidual &operator= (const TQpResidual &source);

   ClassDefOverride(TQpResidual,1)                     // Qp Residual class
};
#endif
