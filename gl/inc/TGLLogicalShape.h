// @(#)root/gl:$Name:  $:$Id: TGLLogicalShape.h,v 1.3 2005/05/26 12:29:50 rdm Exp $
// Author:  Richard Maunder  25/05/2005

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLLogicalShape
#define ROOT_TGLLogicalShape

#ifndef ROOT_TGLDrawable
#include "TGLDrawable.h"
#endif

class TContextMenu;

/*************************************************************************
 * TGLLogicalShape - TODO
 *
 *
 *
 *************************************************************************/
class TGLLogicalShape : public TGLDrawable { // Rename TGLLogicalObject?
private:
   // Fields
   mutable UInt_t fRef; //! physical instance ref counting

public:
   TGLLogicalShape(ULong_t ID);
   virtual ~TGLLogicalShape();

   virtual void Purge();

   virtual void InvokeContextMenu(TContextMenu & menu, UInt_t x, UInt_t y) const = 0;

   // Physical shape ref counting
   void   AddRef() const { ++fRef; }
   Bool_t SubRef() const { if (--fRef == 0) return kTRUE; return kFALSE; }
   UInt_t Ref()    const { return fRef; }

   ClassDef(TGLLogicalShape,0) // a logical (non-placed, local frame) drawable object
};

#endif // ROOT_TGLLogicalShape
