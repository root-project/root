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

#ifndef ROOT_TQpLinSolverSparse
#define ROOT_TQpLinSolverSparse

#include "TQpLinSolverBase.h"
#ifndef ROOT_TQpProbSparse
#include "TQpProbSparse.h"
#endif
#ifndef ROOT_TQpDataSparse
#include "TQpDataSparse.h"
#endif

#include "TDecompSparse.h"

///////////////////////////////////////////////////////////////////////////
//                                                                       //
// Implements the aspects of the solvers for dense general QP            //
// formulation that are specific to the dense case.                      //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

class TQpProbSparse;
class TQpLinSolverSparse : public TQpLinSolverBase
{

protected:

   TMatrixDSparse fKkt;
   TDecompSparse  fSolveSparse;

public:
   TQpLinSolverSparse() {}
   TQpLinSolverSparse(TQpProbSparse *factory,TQpDataSparse *data);
   TQpLinSolverSparse(const TQpLinSolverSparse &another);

   virtual ~TQpLinSolverSparse() {}

   virtual void Factor         (TQpDataBase *prob,TQpVar *vars);
   virtual void SolveCompressed(TVectorD &rhs);
   virtual void PutXDiagonal   (TVectorD &xdiag);
   virtual void PutZDiagonal   (TVectorD &zdiag);

   TQpLinSolverSparse &operator= (const TQpLinSolverSparse &source);

   ClassDef(TQpLinSolverSparse,1)              // Qp linear solver class for Sparse formulation
};
#endif
