// @(#)root/tree:$Name:  $:$Id: TVirtualIndex.cxx,v 1.11 2004/07/02 21:51:29 brun Exp $
// Author: Rene Brun   05/07/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Abstract interface for Tree Index                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TVirtualIndex.h"
#include "TTree.h"

ClassImp(TVirtualIndex)

//______________________________________________________________________________
TVirtualIndex::TVirtualIndex(): TNamed()
{
// Default constructor for TVirtualIndex

   fTree         = 0;
}

//______________________________________________________________________________
TVirtualIndex::~TVirtualIndex()
{
}
