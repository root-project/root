// @(#)root/matrix:$Name:  $:$Id: TQpProbSparse.cxx,v 1.56 2004/02/12 13:03:00 brun Exp $
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

#include "TQpProbSparse.h"
#include "TMatrixD.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TQpProbSparse                                                        //
//                                                                      //
// dense matrix problem formulation                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TQpProbSparse)

//______________________________________________________________________________
TQpProbSparse::TQpProbSparse(Int_t nx,Int_t my,Int_t mz)
  : TQpProbBase(nx,my,mz)
{
}

//______________________________________________________________________________
TQpProbSparse::TQpProbSparse(const TQpProbSparse &another) : TQpProbBase(another)
{                        
  *this = another;       
}                        

//______________________________________________________________________________
void TQpProbSparse::MakeData(Double_t *c,
                             Int_t nnzQ,Int_t *irowQ,Int_t *icolQ,Double_t *Q,
                             Double_t *xlow,Bool_t *ixlow,
                             Double_t *xupp,Bool_t *ixupp,
                             Int_t nnzA,Int_t *irowA,Int_t *icolA,Double_t *A,
                             Double_t *bA,
                             Int_t nnzC,Int_t *irowC,Int_t *icolC,Double_t *C,
                             Double_t *clow,Bool_t *iclow,             
                             Double_t *cupp,Bool_t *icupp,             
                             TQpDataBase *&data_in)
{ 
  TVectorD       vc   ; vc   .Use(fNx,c);
  TMatrixDSparse mQ   ; mQ   .Use(fNx,fNx,nnzQ,irowQ,icolQ,Q);
  TVectorD       vxlow; vxlow.Use(fNx,xlow);
  TVectorD       vxupp; vxupp.Use(fNx,xupp);
  TMatrixDSparse mA   ; mA   .Use(fMy,fNx,nnzA,irowA,icolA,A);
  TVectorD       vbA  ; vbA  .Use(fMy,bA);
  TMatrixDSparse mC   ; mC   .Use(fMz,fNx,nnzC,irowC,icolC,C);
  TVectorD       vclow; vclow.Use(fMz,clow);
  TVectorD       vcupp; vcupp.Use(fMz,cupp);

  TVectorD vixlow(fNx);
  TVectorD vixupp(fNx);
  for (Int_t ix = 0; ix < fNx; ix++) {
    vixlow[ix] = (ixlow[ix]) ? 1.0 : 0.0;
    vixupp[ix] = (ixupp[ix]) ? 1.0 : 0.0;
  }

  TVectorD viclow(fMz);
  TVectorD vicupp(fMz);
  for (Int_t ic = 0; ic < fMz; ic++) {
    viclow[ic] = (iclow[ic]) ? 1.0 : 0.0;
    vicupp[ic] = (icupp[ic]) ? 1.0 : 0.0;
  }

  TQpDataSparse *&data = (TQpDataSparse *&) data_in;
  data = new TQpDataSparse(vc,mQ,vxlow,vixlow,vxupp,vixupp,mA,vbA,mC,vclow,
                           viclow,vcupp,vicupp);
}

//______________________________________________________________________________
void TQpProbSparse::MakeData(TQpDataBase *&data_in)
{ 
  TQpDataSparse *&data = (TQpDataSparse *&) data_in;
  data = new TQpDataSparse(fNx,fMy,fMz);
}

//______________________________________________________________________________
TQpResidual* TQpProbSparse::MakeResiduals(const TQpDataBase *data_in)
{
  TQpDataSparse *data = (TQpDataSparse *) data_in;
  return new TQpResidual(fNx,fMy,fMz,data->fXloIndex,data->fXupIndex,data->fCloIndex,data->fCupIndex);
}

//______________________________________________________________________________
TQpVar* TQpProbSparse::MakeVariables(const TQpDataBase *data_in)
{
  TQpDataSparse *data = (TQpDataSparse *) data_in;

  return new TQpVar(fNx,fMy,fMz,data->fXloIndex,data->fXupIndex,data->fCloIndex,data->fCupIndex);
}

//______________________________________________________________________________
TQpLinSolverBase* TQpProbSparse::MakeLinSys(const TQpDataBase *data_in)
{ 
  TQpDataSparse *data = (TQpDataSparse *) data_in;
  return new TQpLinSolverSparse(this,data);
}

//______________________________________________________________________________
void TQpProbSparse::JoinRHS(TVectorD &rhs,TVectorD &rhs1_in,TVectorD &rhs2_in,TVectorD &rhs3_in)
{
  rhs.SetSub(0,rhs1_in);
  if (fMy > 0) rhs.SetSub(fNx,    rhs2_in);
  if (fMz > 0) rhs.SetSub(fNx+fMy,rhs3_in);
}

//______________________________________________________________________________
void TQpProbSparse::SeparateVars(TVectorD &x_in,TVectorD &y_in,TVectorD &z_in,TVectorD &vars_in)
{
  x_in = vars_in.GetSub(0,fNx-1);
  if (fMy > 0) y_in = vars_in.GetSub(fNx,    fNx+fMy-1);
  if (fMz > 0) z_in = vars_in.GetSub(fNx+fMy,fNx+fMy+fMz-1);
}

//______________________________________________________________________________
void TQpProbSparse::MakeRandomData(TQpDataSparse *&data,TQpVar *&soln,Int_t nnzQ,Int_t nnzA,Int_t nnzC)
{
  data = new TQpDataSparse(fNx,fMy,fMz);
  soln = this->MakeVariables(data);
  data->SetNonZeros(nnzQ,nnzA,nnzC);
  data->DataRandom(soln->fX,soln->fY,soln->fZ,soln->fS);
}

//______________________________________________________________________________
TQpProbSparse &TQpProbSparse::operator=(const TQpProbSparse &source)
{
  if (this != &source) {
    TQpProbBase::operator=(source);
  }
  return *this;
}
