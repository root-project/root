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

#ifndef ROOT_TQpDataSparse
#define ROOT_TQpDataSparse

#include "TQpDataBase.h"
#include "TQpVar.h"

#include "TMatrixDSparse.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TQpDataSparse                                                        //
//                                                                      //
// Data for the dense QP formulation                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TQpDataSparse : public TQpDataBase
{

protected:

   // these variables will be "Used" not copied
   TMatrixDSparse fQ;                          // quadratic part of Objective function
   TMatrixDSparse fA;                          // Equality constraints
   TMatrixDSparse fC;                          // Inequality constraints

public:

   TQpDataSparse() {}
   // data objects of the specified dimensions
   TQpDataSparse(Int_t nx,Int_t my,Int_t mz);

   // sets up pointers to the data objects that are passed as arguments
   TQpDataSparse(TVectorD &c,TMatrixDSparse &Q,TVectorD &xlow,TVectorD &ixlow,TVectorD &xupp,
                 TVectorD &ixupp,TMatrixDSparse &A,TVectorD &bA,TMatrixDSparse &C,TVectorD &clow,
                 TVectorD &iclow,TVectorD &cupp,TVectorD &icupp);
   TQpDataSparse(const TQpDataSparse &another);

   ~TQpDataSparse() override {}

   void SetNonZeros(Int_t nnzQ,Int_t nnzA,Int_t nnzC);

   void PutQIntoAt(TMatrixDBase &M,Int_t row,Int_t col) override;
                                               // insert the Hessian Q into the matrix M for the fundamental
                                               // linear system, where M is stored as a TMatrixDSparse
   void PutAIntoAt(TMatrixDBase &M,Int_t row,Int_t col) override;
                                               // insert the constraint matrix A into the matrix M for the
                                               // fundamental linear system, where M is stored as a TMatrixDSparse
   void PutCIntoAt(TMatrixDBase &M,Int_t row,Int_t col) override;
                                               // insert the constraint matrix C into the matrix M for the
                                               // fundamental linear system, where M is stored as a
                                               // TMatrixDSparse

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

   TQpDataSparse &operator= (const TQpDataSparse &source);

   ClassDefOverride(TQpDataSparse,1)                   // Qp Data class for Sparse formulation
};
#endif
