// @(#)root/matrix:$Id$
// Authors: Fons Rademakers, Eddy Offermann   Mar 2004

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

#include "TQpProbDens.h"
#include "TMatrixD.h"
#include "TQpLinSolverDens.h"

////////////////////////////////////////////////////////////////////////////////
///
/// \class TQpProbDens
///
/// dense matrix problem formulation
///
////////////////////////////////////////////////////////////////////////////////

ClassImp(TQpProbDens);

////////////////////////////////////////////////////////////////////////////////
/// Constructor

TQpProbDens::TQpProbDens(Int_t nx,Int_t my,Int_t mz) :
             TQpProbBase(nx,my,mz)
{
   // We do not want more constrains than variables
   R__ASSERT(nx-my-mz > 0);
}


////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

TQpProbDens::TQpProbDens(const TQpProbDens &another) : TQpProbBase(another)
{
   *this = another;
}


////////////////////////////////////////////////////////////////////////////////
/// Setup the data

TQpDataBase *TQpProbDens::MakeData(Double_t *c,
                                   Double_t *Q,
                                   Double_t *xlo,Bool_t   *ixlo,
                                   Double_t *xup,Bool_t   *ixup,
                                   Double_t *A,  Double_t *bA,
                                   Double_t *C,
                                   Double_t *clo,Bool_t   *iclo,
                                   Double_t *cup,Bool_t   *icup)
{
   TVectorD    vc  ; vc  .Use(fNx,c);
   TMatrixDSym mQ  ; mQ  .Use(fNx,Q);
   TVectorD    vxlo; vxlo.Use(fNx,xlo);
   TVectorD    vxup; vxup.Use(fNx,xup);
   TMatrixD    mA  ;
   TVectorD    vbA ;
   if (fMy > 0) {
      mA  .Use(fMy,fNx,A);
      vbA .Use(fMy,bA);
   }
   TMatrixD    mC  ;
   TVectorD    vclo;
   TVectorD    vcup;
   if (fMz > 0) {
      mC  .Use(fMz,fNx,C);
      vclo.Use(fMz,clo);
      vcup.Use(fMz,cup);
   }

   TVectorD vixlo(fNx);
   TVectorD vixup(fNx);
   for (Int_t ix = 0; ix < fNx; ix++) {
      vixlo[ix] = (ixlo[ix]) ? 1.0 : 0.0;
      vixup[ix] = (ixup[ix]) ? 1.0 : 0.0;
   }

   TVectorD viclo(fMz);
   TVectorD vicup(fMz);
   for (Int_t ic = 0; ic < fMz; ic++) {
      viclo[ic] = (iclo[ic]) ? 1.0 : 0.0;
      vicup[ic] = (icup[ic]) ? 1.0 : 0.0;
   }

   TQpDataDens *data = new TQpDataDens(vc,mQ,vxlo,vixlo,vxup,vixup,mA,vbA,mC,vclo,
      viclo,vcup,vicup);

   return data;
}


////////////////////////////////////////////////////////////////////////////////
/// Setup the data

TQpDataBase *TQpProbDens::MakeData(TVectorD     &c,
                                   TMatrixDBase &Q_in,
                                   TVectorD     &xlo, TVectorD &ixlo,
                                   TVectorD     &xup, TVectorD &ixup,
                                   TMatrixDBase &A_in,TVectorD &bA,
                                   TMatrixDBase &C_in,
                                   TVectorD     &clo, TVectorD &iclo,
                                   TVectorD     &cup, TVectorD &icup)
{
   TMatrixDSym &mQ = (TMatrixDSym &) Q_in;
   TMatrixD    &mA = (TMatrixD    &) A_in;
   TMatrixD    &mC = (TMatrixD    &) C_in;

   R__ASSERT(mQ.GetNrows() == fNx && mQ.GetNcols() == fNx);
   if (fMy > 0) R__ASSERT(mA.GetNrows() == fMy && mA.GetNcols() == fNx);
   else         R__ASSERT(mA.GetNrows() == fMy);
   if (fMz > 0) R__ASSERT(mC.GetNrows() == fMz && mC.GetNcols() == fNx);
   else         R__ASSERT(mC.GetNrows() == fMz);

   R__ASSERT(c.GetNrows()    == fNx);
   R__ASSERT(xlo.GetNrows()  == fNx);
   R__ASSERT(ixlo.GetNrows() == fNx);
   R__ASSERT(xup.GetNrows()  == fNx);
   R__ASSERT(ixup.GetNrows() == fNx);

   R__ASSERT(bA.GetNrows()   == fMy);
   R__ASSERT(clo.GetNrows()  == fMz);
   R__ASSERT(iclo.GetNrows() == fMz);
   R__ASSERT(cup.GetNrows()  == fMz);
   R__ASSERT(icup.GetNrows() == fMz);

   TQpDataDens *data = new TQpDataDens(c,mQ,xlo,ixlo,xup,ixup,mA,bA,mC,clo,iclo,cup,icup);

   return data;
}


////////////////////////////////////////////////////////////////////////////////
/// Setup the residuals

TQpResidual* TQpProbDens::MakeResiduals(const TQpDataBase *data_in)
{
   TQpDataDens *data = (TQpDataDens *) data_in;
   return new TQpResidual(fNx,fMy,fMz,data->fXloIndex,data->fXupIndex,data->fCloIndex,data->fCupIndex);
}


////////////////////////////////////////////////////////////////////////////////
/// Setup the variables

TQpVar* TQpProbDens::MakeVariables(const TQpDataBase *data_in)
{
   TQpDataDens *data = (TQpDataDens *) data_in;

   return new TQpVar(fNx,fMy,fMz,data->fXloIndex,data->fXupIndex,data->fCloIndex,data->fCupIndex);
}


////////////////////////////////////////////////////////////////////////////////
/// Setup the linear solver

TQpLinSolverBase* TQpProbDens::MakeLinSys(const TQpDataBase *data_in)
{
   TQpDataDens *data = (TQpDataDens *) data_in;
   return new TQpLinSolverDens(this,data);
}


////////////////////////////////////////////////////////////////////////////////
/// Assembles a single vector object from three given vectors .
///     rhs_out (output) final joined vector
///     rhs1_in (input) first part of rhs
///     rhs2_in (input) middle part of rhs
///     rhs3_in (input) last part of rhs .

void TQpProbDens::JoinRHS(TVectorD &rhs,TVectorD &rhs1_in,TVectorD &rhs2_in,TVectorD &rhs3_in)
{
   rhs.SetSub(0,rhs1_in);
   if (fMy > 0) rhs.SetSub(fNx,    rhs2_in);
   if (fMz > 0) rhs.SetSub(fNx+fMy,rhs3_in);
}


////////////////////////////////////////////////////////////////////////////////
/// Extracts three component vectors from a given aggregated vector.
///     vars_in  (input) aggregated vector
///     x_in (output) first part of vars
///     y_in (output) middle part of vars
///     z_in (output) last part of vars

void TQpProbDens::SeparateVars(TVectorD &x_in,TVectorD &y_in,TVectorD &z_in,TVectorD &vars_in)
{
   x_in = vars_in.GetSub(0,fNx-1);
   if (fMy > 0) y_in = vars_in.GetSub(fNx,    fNx+fMy-1);
   if (fMz > 0) z_in = vars_in.GetSub(fNx+fMy,fNx+fMy+fMz-1);
}


////////////////////////////////////////////////////////////////////////////////
/// Create a random QP problem

void TQpProbDens::MakeRandomData(TQpDataDens *&data,TQpVar *&soln,Int_t /*nnzQ*/,Int_t /*nnzA*/,Int_t /*nnzC*/)
{
   data = new TQpDataDens(fNx,fMy,fMz);
   soln = this->MakeVariables(data);
   data->DataRandom(soln->fX,soln->fY,soln->fZ,soln->fS);
}


////////////////////////////////////////////////////////////////////////////////
/// Assignment operator

TQpProbDens &TQpProbDens::operator=(const TQpProbDens &source)
{
   if (this != &source) {
      TQpProbBase::operator=(source);
   }
   return *this;
}
