// @(#)root/net:$Name:  $:$Id: TGridProof.cxx,v 1.0 2003/09/05 10:00:00 peters Exp $
// Author: Andreas Peters   05/09/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGridProof                                                           //
//                                                                      //
// Abstract base class defining interface to a GRID PROOF service.      //
// Objects of this class are created by TGrid methods.                  //
//                                                                      //
// Related classes are TGrid.                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGridProof.h"

ClassImp(TGridProof)

//______________________________________________________________________________
TGridProof::~TGridProof()
{
   // Clean up Grid PROOF environment.

   if (fProofSession) {
      //    delete fProofSession;
   }

   if (fProof)
      delete fProof;
}
