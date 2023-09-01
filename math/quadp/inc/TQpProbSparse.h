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

#ifndef ROOT_TQpProbSparse
#define ROOT_TQpProbSparse

#include "TQpProbBase.h"
#include "TQpDataSparse.h"
#include "TQpVar.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TQpProbSparse                                                        //
//                                                                      //
// sparse matrix problem formulation                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TQpProbSparse : public TQpProbBase
{

public:

   TQpProbSparse() {}
   TQpProbSparse(Int_t nx,Int_t my,Int_t mz);
   TQpProbSparse(const TQpProbSparse &another);

   ~TQpProbSparse() override {}

#ifndef __CINT__
   virtual TQpDataBase      *MakeData      (Double_t *c,
                                            Int_t nnzQ,Int_t *irowQ,Int_t *icolQ,Double_t *Q,
                                            Double_t *xlo,Bool_t *ixlo,
                                            Double_t *xup,Bool_t *ixup,
                                            Int_t nnzA,Int_t *irowA,Int_t *icolA,Double_t *A,
                                            Double_t *bA,
                                            Int_t nnzC,Int_t *irowC,Int_t *icolC,Double_t *C,
                                            Double_t *clo,Bool_t *iclo,
                                            Double_t *cup,Bool_t *icup);
#endif
   TQpDataBase      *MakeData     (TVectorD     &c,
                                            TMatrixDBase &Q_in,
                                            TVectorD     &xlo, TVectorD &ixlo,
                                            TVectorD     &xup, TVectorD &ixup,
                                            TMatrixDBase &A_in,TVectorD &bA,
                                            TMatrixDBase &C_in,
                                            TVectorD     &clo, TVectorD &iclo,
                                            TVectorD     &cup, TVectorD &icup) override;
   TQpResidual      *MakeResiduals(const TQpDataBase *data) override;
   TQpVar           *MakeVariables(const TQpDataBase *data) override;
   TQpLinSolverBase *MakeLinSys   (const TQpDataBase *data) override;

   void JoinRHS       (TVectorD &rhs_in,TVectorD &rhs1_in,TVectorD &rhs2_in,TVectorD &rhs3_in) override;
   void SeparateVars  (TVectorD &x_in,TVectorD &y_in,TVectorD &z_in,TVectorD &vars_in) override;
           void MakeRandomData(TQpDataSparse *&data,TQpVar *&soln,Int_t nnzQ,Int_t nnzA,Int_t nnzC);

   TQpProbSparse &operator=(const TQpProbSparse &source);

   ClassDefOverride(TQpProbSparse,1)                   // Qp sparse problem formulation class
};
#endif
