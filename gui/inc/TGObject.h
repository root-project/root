// @(#)root/gui:$Name$:$Id$
// Author: Fons Rademakers   27/12/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGObject
#define ROOT_TGObject


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGObject                                                             //
//                                                                      //
// This class is the baseclass for all ROOT GUI widgets.                //
// The ROOT GUI components emulate the Win95 look and feel and the code //
// is based on the XClass'95 code (see Copyleft in source).             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_GuiTypes
#include "GuiTypes.h"
#endif

class TGClient;


class TGObject : public TObject {

protected:
   Handle_t    fId;                  // X11/Win32 Window identifier
   TGClient   *fClient;              // Connection to display server

public:
   TGObject() { fId = 0; fClient = 0; }
   virtual ~TGObject() { }

   Handle_t GetId() const { return fId; }
   ULong_t  Hash() { return (ULong_t) fId >> 0; }
   Bool_t   IsEqual(TObject *obj) { return fId == ((TGObject *) obj)->fId; }

   ClassDef(TGObject,0)  //ROOT GUI base class
};

#endif
