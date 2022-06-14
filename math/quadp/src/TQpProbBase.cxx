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

#include "TQpProbBase.h"

////////////////////////////////////////////////////////////////////////////////
///
/// \class TQpProbBase
///
/// default general problem formulation:
///
///              minimize    c' x + ( 1/2 ) x' * Q x        ;
///              subject to                      A x  = b   ;
///                                      clo <=  C x <= cup ;
///                                      xlo <=    x <= xup ;
///
////////////////////////////////////////////////////////////////////////////////

ClassImp(TQpProbBase);

////////////////////////////////////////////////////////////////////////////////
/// Default constructor

TQpProbBase::TQpProbBase()
{
   fNx = 0;
   fMy = 0;
   fMz = 0;
}


////////////////////////////////////////////////////////////////////////////////
/// Constructor

TQpProbBase::TQpProbBase(Int_t nx,Int_t my,Int_t mz)
{
   fNx = nx;
   fMy = my;
   fMz = mz;
}


////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

TQpProbBase::TQpProbBase(const TQpProbBase &another) : TObject(another)
{
   *this = another;
}


////////////////////////////////////////////////////////////////////////////////
/// Assignment operator

TQpProbBase &TQpProbBase::operator=(const TQpProbBase &source)
{
   if (this != &source) {
      TObject::operator=(source);
      fNx = source.fNx;
      fMy = source.fMy;
      fMz = source.fMz;
   }
   return *this;
}
