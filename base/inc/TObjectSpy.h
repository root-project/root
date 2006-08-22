// @(#)root/base:$Name:  $:$Id: TObjectSpy.h,v 1.1 2006/08/18 17:34:46 rdm Exp $
// Author: Matevz Tadel   16/08/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TObjectSpy
#define ROOT_TObjectSpy

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TObjectSpy, TObjectRefSpy                                            //
//                                                                      //
// Monitors objects for deletion and reflects the deletion by reverting //
// the internal pointer to zero. When this pointer is zero we know the  //
// object has been deleted. This avoids the unsafe TestBit(kNotDeleted) //
// hack. The spied object must have the kMustCleanup bit set otherwise  //
// you will get an error.                                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif


class TObjectSpy : public TObject {

protected:
   TObject  *fObj;   // object being spied

public:
   TObjectSpy(TObject *obj = 0);
   TObjectSpy(const TObjectSpy& s) : TObject(), fObj(s.fObj) { }
   virtual ~TObjectSpy();

   TObjectSpy& operator=(const TObjectSpy& s) { fObj = s.fObj; return *this; }

   virtual void  RecursiveRemove(TObject *obj);
   TObject      *GetObject() const { return fObj; }
   void          SetObject(TObject *obj);

   ClassDef(TObjectSpy, 0);  // Spy object pointer for deletion
};


class TObjectRefSpy : public TObject {

protected:
   TObject  *&fObj;   // object being spied

public:
   TObjectRefSpy(TObject *&obj);
   TObjectRefSpy(const TObjectRefSpy& s) : TObject(), fObj(s.fObj) { }
   virtual ~TObjectRefSpy();

   TObjectRefSpy& operator=(const TObjectRefSpy& s) { fObj = s.fObj; return *this; }

   virtual void  RecursiveRemove(TObject *obj);
   TObject      *GetObject() const { return fObj; }

   ClassDef(TObjectRefSpy, 0);  // Spy object reference for deletion
};

#endif
