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

#ifndef ROOT_TQpProbBase
#define ROOT_TQpProbBase

#ifndef ROOT_TError
#include "TError.h"
#endif

#ifndef ROOT_TQpVar
#include "TQpVar.h"
#endif
#ifndef ROOT_TQpDataBase
#include "TQpDataBase.h"
#endif
#ifndef ROOT_TQpLinSolverBase
#include "TQpLinSolverBase.h"
#endif
#ifndef ROOT_TQpResidual
#include "TQpResidual.h"
#endif

#ifndef ROOT_TMatrixD
#include "TMatrixD.h"
#endif

///////////////////////////////////////////////////////////////////////////
//                                                                       //
// default general problem formulation:                                  //
//                                                                       //
//  minimize    c' x + ( 1/2 ) x' * Q x        ;                         //
//  subject to                      A x  = b   ;                         //
//                          clo <=  C x <= cup ;                         //
//                          xlo <=    x <= xup ;                         //
//                                                                       //
//  The general linear equality constraints must have either an upper    //
//  or lower bound, but need not have both bounds. The variables may have//
//  no bounds; an upper bound; a lower bound or both an upper and lower  //
//  bound.                                                               //
//                                                                       //
//  However, for many (possibly most) QP's, the matrices in the          //
//  formulation have structure that may be exploited to solve the        //
//  problem more efficiently. This abstract problem formulation contains //
//  a setup such that one can derive and add special formulations .      //
//  The optimality conditions of the simple QP defined above are         //
//  follows:                                                             //
//                                                                       //
//  rQ  = c + Q * x - A' * y - C' * z = 0                                //
//  rA  = A * x - b                   = 0                                //
//  rC  = C * x - s - d               = 0                                //
//  r3  = S * z                       = 0                                //
//  s, z >= 0                                                            //
//                                                                       //
//  Where rQ, rA, rC and r3 newly defined quantities known as residual   //
//  vectors and x, y, z and s are variables of used in solution of the   //
//  QPs.                                                                 //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

class TQpLinSolverBase;
class TQpProbBase : public TObject
{

public:
   Int_t fNx;                                  // number of elements in x
   Int_t fMy;                                  // number of rows in A and b
   Int_t fMz;                                  // number of rows in C

   TQpProbBase();
   TQpProbBase(Int_t nx,Int_t my,Int_t mz);
   TQpProbBase(const TQpProbBase &another);

   virtual ~TQpProbBase() {}

   virtual TQpDataBase      *MakeData     (TVectorD     &c,
                                           TMatrixDBase &Q_in,
                                           TVectorD     &xlo, TVectorD &ixlo,
                                           TVectorD     &xup, TVectorD &ixup,
                                           TMatrixDBase &A_in,TVectorD &bA,
                                           TMatrixDBase &C_in,
                                           TVectorD     &clo, TVectorD &iclo,
                                           TVectorD     &cup, TVectorD &icup) = 0;
   virtual TQpResidual      *MakeResiduals(const TQpDataBase *data) = 0;
   virtual TQpVar           *MakeVariables(const TQpDataBase *data) = 0;
   virtual TQpLinSolverBase *MakeLinSys   (const TQpDataBase *data) = 0;

   virtual void JoinRHS     (TVectorD &rhs_in,TVectorD &rhs1_in,TVectorD &rhs2_in,TVectorD &rhs3_in) = 0;
   virtual void SeparateVars(TVectorD &x_in,TVectorD &y_in,TVectorD &z_in,TVectorD &vars_in)         = 0;

   TQpProbBase &operator= (const TQpProbBase &source);

   ClassDef(TQpProbBase,1)                     // Qp problem formulation base class
};
#endif
