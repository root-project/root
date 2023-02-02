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

#ifndef ROOT_TQpDataDens
#define ROOT_TQpDataDens

#include "TError.h"
#include "TQpDataBase.h"

#include "TQpVar.h"

#include "TMatrixD.h"
#include "TMatrixDSym.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TQpDataDens                                                          //
//                                                                      //
// Data for the dense QP formulation                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TQpDataDens : public TQpDataBase
{

protected:

   // these variables will be "Used" not copied
   TMatrixDSym fQ;                             // Quadratic part of Objective function
   TMatrixD    fA;                             // Equality constraints
   TMatrixD    fC;                             // Inequality constraints

public:

   TQpDataDens() {}
   // data objects of the specified dimensions
   TQpDataDens(Int_t nx,Int_t my,Int_t mz);

   // sets up pointers to the data objects that are passed as arguments
   TQpDataDens(TVectorD &c,TMatrixDSym &Q,TVectorD &xlow,TVectorD &ixlow,TVectorD &xupp,
               TVectorD &ixupp,TMatrixD &A,TVectorD &bA,TMatrixD &C,TVectorD &clow,
               TVectorD &iclow,TVectorD &cupp,TVectorD &icupp);
   TQpDataDens(const TQpDataDens &another);

   ~TQpDataDens() override {}

   void PutQIntoAt(TMatrixDBase &M,Int_t row,Int_t col) override;
                                               // insert the Hessian Q into the matrix M for the fundamental
                                               // linear system, where M is stored as a TMatrixD
   void PutAIntoAt(TMatrixDBase &M,Int_t row,Int_t col) override;
                                               // insert the constraint matrix A into the matrix M for the
                                               // fundamental linear system, where M is stored as a TMatrixD
   void PutCIntoAt(TMatrixDBase &M,Int_t row,Int_t col) override;
                                               // insert the constraint matrix C into the matrix M for the
                                               // fundamental linear system, where M is stored as a TMatrixD

   void Qmult     (Double_t beta,TVectorD& y,Double_t alpha,const TVectorD& x) override;
                                               // y = beta * y + alpha * Q * x
   void Amult     (Double_t beta,TVectorD& y,Double_t alpha,const TVectorD& x) override;
                                               // y = beta * y + alpha * A * x
   void Cmult     (Double_t beta,TVectorD& y,Double_t alpha,const TVectorD& x) override;
                                               // y = beta * y + alpha * C * x
   void ATransmult(Double_t beta,TVectorD& y,Double_t alpha,const TVectorD& x) override;
                                               // y = beta * y + alpha * A^T * x
   void CTransmult(Double_t beta,TVectorD& y,Double_t alpha,const TVectorD& x) override;
                                               // y = beta * y + alpha * C^T * x

   void GetDiagonalOfQ(TVectorD &dQ) override;  // extract the diagonal of Q and put it in the vector dQ

   Double_t DataNorm() override;
   void DataRandom(TVectorD &x,TVectorD &y,TVectorD &z,TVectorD &s) override;
                                               // Create a random problem (x,y,z,s)
                                               // the solution to the random problem
   void Print(Option_t *opt="") const override;

   Double_t ObjectiveValue(TQpVar *vars) override;

   TQpDataDens &operator= (const TQpDataDens &source);

   ClassDefOverride(TQpDataDens,1)                     // Qp Data class for Dens formulation
};
#endif
