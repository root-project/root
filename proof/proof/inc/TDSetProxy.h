// @(#)root/proof:$Id$
// Author: Maarten Ballintijn  12/03/02

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDSetProxy
#define ROOT_TDSetProxy


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TDSetProxy                                                           //
//                                                                      //
// TDSet proxy for use on slaves.                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TDSet.h"

class TProofServ;


class TDSetProxy : public TDSet {

private:
   TProofServ     *fServ;     //!

public:
   TDSetProxy();
   TDSetProxy(const char *type, const char *objname = "*", const char *dir = "/");

   void           Reset();
   TDSetElement  *Next(Long64_t totalEntries = -1);

   void  SetProofServ(TProofServ *serv);

   ClassDef(TDSetProxy,1)  // TDSet proxy for use on slaves
};

#endif
