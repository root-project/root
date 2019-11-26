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

#include "TQpDataBase.h"

////////////////////////////////////////////////////////////////////////////////
///
/// \class TQpDataBase
///
/// Data for the general QP formulation
///
/// The Data class stores the data defining the problem and provides
/// methods for performing the operations with this data required by
/// the interior-point algorithms. These operations include assembling
/// the linear systems (5) or (7), performing matrix-vector operations
/// with the data, calculating norms of the data, reading input into the
/// data structure from various sources, generating random problem
/// instances, and printing the data.
///
////////////////////////////////////////////////////////////////////////////////

ClassImp(TQpDataBase);

////////////////////////////////////////////////////////////////////////////////
/// Default constructor

TQpDataBase::TQpDataBase()
{
   fNx = 0;
   fMy = 0;
   fMz = 0;
}


////////////////////////////////////////////////////////////////////////////////
/// Constructor

TQpDataBase::TQpDataBase(Int_t nx,Int_t my,Int_t mz)
{
   fNx = nx;
   fMy = my;
   fMz = mz;

   fG    .ResizeTo(fNx);

   fBa   .ResizeTo(fMy);

   fXupBound.ResizeTo(fNx);
   fXupIndex.ResizeTo(fNx);
   fXloBound.ResizeTo(fNx);
   fXloIndex.ResizeTo(fNx);

   fCupBound.ResizeTo(fMz);
   fCupIndex.ResizeTo(fMz);
   fCloBound.ResizeTo(fMz);
   fCloIndex.ResizeTo(fMz);
}


////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

TQpDataBase::TQpDataBase(const TQpDataBase &another) : TObject(another)
{
   *this = another;
}


////////////////////////////////////////////////////////////////////////////////
/// Randomly choose  x and its boundaries

void TQpDataBase::RandomlyChooseBoundedVariables(
                           TVectorD &x,TVectorD &dualx,TVectorD &xlow,TVectorD &ixlow,
                           TVectorD &xupp,TVectorD &ixupp,Double_t &ix,Double_t percentLowerOnly,
                           Double_t percentUpperOnly,Double_t percentBound)
{
   const Int_t n = x.GetNrows();

   // Initialize the upper and lower bounds on x

   Int_t i;
   for (i = 0; i < n; i++) {
      const Double_t r = Drand(ix);

      if (r < percentLowerOnly) {
         ixlow[i] = 1.0;
         xlow [i] = (Drand(ix)-0.5)*3.0;
         ixupp[i] = 0.0;
         xupp [i] = 0.0;
      }
      else if (r < percentLowerOnly+percentUpperOnly) {
         ixlow[i] = 0.0;
         xlow [i] = 0.0;
         ixupp[i] = 1.0;
         xupp [i] = (Drand(ix)-0.5)*3.0;
      }
      else if (r < percentLowerOnly+percentUpperOnly+percentBound) {
         ixlow[i] = 1.0;
         xlow [i] = (Drand(ix)-0.5)*3.0;
         ixupp[i] = 1.0;
         xupp [i] = xlow[i]+Drand(ix)*10.0;
      }
      else {
         // it is free
         ixlow[i] = 0.0;
         xlow [i] = 0.0;
         ixupp[i] = 0.0;
         xupp [i] = 0.0;
      }
   }

   for (i = 0; i < n; i++) {
      if (ixlow[i] == 0.0 && ixupp[i] == 0.0 ) {
         // x[i] not bounded
         x    [i] = 20.0*Drand(ix)-10.0;
         dualx[i] = 0.0;
      }
      else if (ixlow[i] != 0.0 && ixupp[i] != 0.0) {
         // x[i] is bounded above and below
         const Double_t r = Drand(ix);
         if (r < 0.33 ) {
            // x[i] is on its lower bound
            x    [i] = xlow[i];
            dualx[i] = 10.0*Drand(ix);
         }
         else if ( r > .66 ) {
            // x[i] is on its upper bound
            x    [i] =  xupp[i];
            dualx[i] = -10.0*Drand(ix);
         }
         else {
            // x[i] is somewhere in between
            const Double_t theta = .99*Drand(ix)+.005;
            x    [i] = (1-theta)*xlow[i]+theta*xupp[i];
            dualx[i] = 0.0;
         }
      }
      else if (ixlow[i] != 0.0) {
         // x[i] is only bounded below
         if (Drand(ix) < .33 ) {
            // x[i] is on its lower bound
            x    [i] = xlow[i];
            dualx[i] = 10.0*Drand(ix);
         }
         else {
            // x[i] is somewhere above its lower bound
            x    [i] = xlow[i]+0.005+10.0*Drand(ix);
            dualx[i] = 0.0;
         }
      }                          // x[i] only has an upper bound
      else {
         if (Drand(ix) > .66 ) {
            // x[i] is on its upper bound
            x    [i] = xupp[i];
            dualx[i] = -10.0*Drand(ix);
         }
         else {
            // x[i] is somewhere below its upper bound
            x    [i] = xupp[i]-0.005-10.0*Drand(ix);
            dualx[i] = 0.0;
         }
      }
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Assignment operator

TQpDataBase &TQpDataBase::operator=(const TQpDataBase &source)
{
   if (this != &source) {
      TObject::operator=(source);
      fNx = source.fNx;
      fMy = source.fMy;
      fMz = source.fMz;

      fG       .ResizeTo(source.fG)       ; fG        = source.fG       ;
      fBa      .ResizeTo(source.fBa)      ; fBa       = source.fBa      ;
      fXupBound.ResizeTo(source.fXupBound); fXupBound = source.fXupBound;
      fXupIndex.ResizeTo(source.fXupIndex); fXupIndex = source.fXupIndex;
      fXloBound.ResizeTo(source.fXloBound); fXloBound = source.fXloBound;
      fXloIndex.ResizeTo(source.fXloIndex); fXloIndex = source.fXloIndex;
      fCupBound.ResizeTo(source.fCupBound); fCupBound = source.fCupBound;
      fCupIndex.ResizeTo(source.fCupIndex); fCupIndex = source.fCupIndex;
      fCloBound.ResizeTo(source.fCloBound); fCloBound = source.fCloBound;
      fCloIndex.ResizeTo(source.fCloIndex); fCloIndex = source.fCloIndex;
   }
   return *this;
}
