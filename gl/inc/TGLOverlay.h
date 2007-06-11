// @(#)root/gl:$Name$:$Id$
// Author:  Matevz Tadel, Feb 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLOverlay_H
#define ROOT_TGLOverlay_H

#include <GuiTypes.h>

class TGLRnrCtx;
class TGLSelectRecord;

#include <list>

class TGLOverlayElement
{
private:
   TGLOverlayElement(const TGLOverlayElement&);            // Not implemented
   TGLOverlayElement& operator=(const TGLOverlayElement&); // Not implemented

public:
   TGLOverlayElement() {}
   virtual ~TGLOverlayElement() {}

   virtual Bool_t MouseEnter(UInt_t* record);
   virtual Bool_t MouseStillInside(UInt_t* record);
   virtual Bool_t Handle(TGLRnrCtx& rnrCtx, Event_t* event, UInt_t* record);
   virtual void   MouseLeave();

   virtual void Render(TGLRnrCtx& rnrCtx) = 0;

   ClassDef(TGLOverlayElement, 0) // Base class for GL overlay elements.
};


class TGLOverlayList
{
private:
   TGLOverlayList(const TGLOverlayList&);            // Not implemented
   TGLOverlayList& operator=(const TGLOverlayList&); // Not implemented

protected:
   std::list<TGLOverlayElement*> fElements;

public:
   TGLOverlayList() {}
   virtual ~TGLOverlayList() {}

   // void AddElement(TGLOverlayElement* element);
   // void RemoveElement(TGLOverlayElement* element);

   // TGLOverlayElement* SelectElement(TGLSelectRecord& selRec, Int_t nameOff);

   ClassDef(TGLOverlayList, 0) // Collection of overlay elements to draw/select together.
};


#endif
