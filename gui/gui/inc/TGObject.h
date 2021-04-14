// @(#)root/gui:$Id$
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


#include "TObject.h"
#include "GuiTypes.h"

class TGClient;


class TGObject : public TObject {


protected:
   Handle_t    fId;                  ///< X11/Win32 Window identifier
   TGClient   *fClient;              ///< Connection to display server

   TGObject& operator=(const TGObject& tgo)
     {if(this!=&tgo) { TObject::operator=(tgo); fId=tgo.fId;
     fClient=tgo.fClient; } return *this; }

public:
   TGObject(): fId(0), fClient(0) { }
   TGObject(const TGObject& tgo): TObject(tgo), fId(tgo.fId), fClient(tgo.fClient) { }
   virtual ~TGObject();
   Handle_t  GetId() const { return fId; }
   TGClient *GetClient() const { return fClient; }
   ULong_t   Hash() const { return (ULong_t) fId >> 0; }
   Bool_t    IsEqual(const TObject *obj) const;
   virtual void SaveAs(const char* filename = "", Option_t* option = "") const;

   ClassDef(TGObject,0)  //ROOT GUI base class
};

#endif
