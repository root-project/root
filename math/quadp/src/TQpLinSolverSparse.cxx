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
// TQpLinSolverSparse                                                   //
//                                                                      //
// Implements the aspects of the solvers for dense general QP           //
// formulation that are specific to the dense case.                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Riostream.h"
#include "TQpLinSolverSparse.h"

ClassImp(TQpLinSolverSparse)

//______________________________________________________________________________
TQpLinSolverSparse::TQpLinSolverSparse(TQpProbSparse *factory,TQpDataSparse *data) :
                    TQpLinSolverBase(factory,data)
{
// Constructor

   const Int_t n = factory->fNx+factory->fMy+factory->fMz;
   fKkt.ResizeTo(n,n);

   if (fMy > 0) data->PutAIntoAt(fKkt,fNx,    0);
   if (fMz > 0) data->PutCIntoAt(fKkt,fNx+fMy,0);

   // trick to make sure that A and C are inserted symmetrically
   if (fMy > 0 || fMz > 0) {
      TMatrixDSparse tmp(TMatrixDSparse::kTransposed,fKkt);
      fKkt += tmp;
   }

   data->PutQIntoAt(fKkt,0,0);
}


//______________________________________________________________________________
TQpLinSolverSparse::TQpLinSolverSparse(const TQpLinSolverSparse &another) :
TQpLinSolverBase(another)
{
// Copy constructor

   *this = another;
}


//______________________________________________________________________________
void TQpLinSolverSparse::Factor(TQpDataBase *prob,TQpVar *vars)
{
// Sets up the matrix for the main linear system in "augmented system" form.

   TQpLinSolverBase::Factor(prob,vars);
   fSolveSparse.SetMatrix(fKkt);
}


//______________________________________________________________________________
void TQpLinSolverSparse::PutXDiagonal(TVectorD &xdiag)
{
// Places the diagonal resulting from the bounds on x into the augmented system matrix

   TMatrixDSparseDiag diag(fKkt);
   for (Int_t i = 0; i < xdiag.GetNrows(); i++)
      diag[i] = xdiag[i];
}


//______________________________________________________________________________
void TQpLinSolverSparse::PutZDiagonal(TVectorD &zdiag)
{
// Places the diagonal resulting from the bounds on Cx into the augmented system matrix

   TMatrixDSparseDiag diag(fKkt);
   for (Int_t i = 0; i < zdiag.GetNrows(); i++)
      diag[i+fNx+fMy] = zdiag[i];
}


//______________________________________________________________________________
void TQpLinSolverSparse::SolveCompressed(TVectorD &compressedRhs)
{
// Perform the actual solve using the factors produced in factor.
// rhs on input contains the aggregated right-hand side of the augmented system;
//  on output contains the solution in aggregated form .

   fSolveSparse.Solve(compressedRhs);
}


//______________________________________________________________________________
TQpLinSolverSparse &TQpLinSolverSparse::operator=(const TQpLinSolverSparse &source)
{
// Assignment operator

   if (this != &source) {
      TQpLinSolverBase::operator=(source);
      fKkt.ResizeTo(source.fKkt); fKkt = source.fKkt;
      fSolveSparse = source.fSolveSparse;
   }
   return *this;
}
