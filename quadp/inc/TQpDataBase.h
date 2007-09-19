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

#ifndef ROOT_TQpDataBase
#define ROOT_TQpDataBase

#ifndef ROOT_TError
#include "TError.h"
#endif

#ifndef ROOT_TQpVar
#include "TQpVar.h"
#endif

#ifndef ROOT_TMatrixD
#include "TMatrixD.h"
#endif

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TQpDataBase                                                          //
//                                                                      //
// Data for the general QP formulation                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TQpDataBase : public TObject
{

protected:

   // as part of setting up a random test problem, generate a random
   //  set of upper, lower, and two-sided bounds
   static void RandomlyChooseBoundedVariables(TVectorD &x,TVectorD &dualx,TVectorD &blx,TVectorD &ixlow,
                                              TVectorD &bux,TVectorD &ixupp,Double_t &ix,Double_t percentLowerOnly,
                                              Double_t percentUpperOnly,Double_t percentBound);

public:

   Int_t    fNx;
   Int_t    fMy;
   Int_t    fMz;

   TVectorD fG;                                // linear part of Objective function
   TVectorD fBa;                               // vector of equality constraint
   TVectorD fXupBound;                         // Bounds on variables
   TVectorD fXupIndex;
   TVectorD fXloBound;
   TVectorD fXloIndex;
   TVectorD fCupBound;                         // Inequality constraints
   TVectorD fCupIndex;
   TVectorD fCloBound;
   TVectorD fCloIndex;

   TQpDataBase();
   TQpDataBase(Int_t nx,Int_t my,Int_t mz);
   TQpDataBase(const TQpDataBase &another);
   virtual ~TQpDataBase() {}

   virtual void PutQIntoAt(TMatrixDBase &M,Int_t row,Int_t col) = 0;
   virtual void PutAIntoAt(TMatrixDBase &M,Int_t row,Int_t col) = 0;
   virtual void PutCIntoAt(TMatrixDBase &M,Int_t row,Int_t col) = 0;

   virtual void Qmult     (Double_t beta,TVectorD& y,Double_t alpha,const TVectorD& x) = 0;
   virtual void Amult     (Double_t beta,TVectorD& y,Double_t alpha,const TVectorD& x) = 0;
   virtual void Cmult     (Double_t beta,TVectorD& y,Double_t alpha,const TVectorD& x) = 0;
   virtual void ATransmult(Double_t beta,TVectorD& y,Double_t alpha,const TVectorD& x) = 0;
   virtual void CTransmult(Double_t beta,TVectorD& y,Double_t alpha,const TVectorD& x) = 0;

   virtual void GetDiagonalOfQ(TVectorD &dQ) = 0;

   virtual TVectorD &GetG           () { return fG; }
   virtual TVectorD &GetBa          () { return fBa; }

   virtual TVectorD &GetXupperBound () { return fXupBound; }
   virtual TVectorD &GetiXupperBound() { return fXupIndex; }
   virtual TVectorD &GetXlowerBound () { return fXloBound; }
   virtual TVectorD &GetiXlowerBound() { return fXloIndex; }
   virtual TVectorD &GetSupperBound () { return fCupBound;  }
   virtual TVectorD &GetiSupperBound() { return fCupIndex; }
   virtual TVectorD &GetSlowerBound () { return fCloBound;  }
   virtual TVectorD &GetiSlowerBound() { return fCloIndex; }

   virtual Double_t  DataNorm      () = 0;
   virtual void      DataRandom    (TVectorD &x,TVectorD &y,TVectorD &z,TVectorD &s) = 0;
   virtual Double_t  ObjectiveValue(TQpVar *vars) = 0;

   TQpDataBase &operator= (const TQpDataBase &source);

   ClassDef(TQpDataBase,1)                     // Qp Base Data class
};
#endif
