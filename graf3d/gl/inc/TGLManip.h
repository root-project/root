// @(#)root/gl:$Id$
// Author:  Richard Maunder  16/09/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLManip
#define ROOT_TGLManip

#include "TVirtualGL.h"
#include "TPoint.h"
#include "GuiTypes.h"
#include "TGLUtil.h"

class TGLPhysicalShape;
class TGLVertex3;
class TGLVector3;
class TGLCamera;
class TGLRect;
class TGLBoundingBox;


class TGLManip : public TVirtualGLManip
{
protected:
   TGLPhysicalShape  *fShape;             //! manipulated shape
   UInt_t             fSelectedWidget;    //! active width (axis) component
   Bool_t             fActive;            //! manipulator is active?

   // Mouse tracking - in WINDOW coords
   TPoint             fFirstMouse;        //! first (start) mouse position (in WINDOW coords)
   TPoint             fLastMouse;         //! last (latest) mouse position (in WINDOW coords)

   TGLManip(const TGLManip&);
   TGLManip& operator=(const TGLManip&);

   void CalcDrawScale(const TGLBoundingBox& box, const TGLCamera& camera,
                      Double_t& base, TGLVector3 axis[3]) const;

   const UChar_t* ColorFor(UInt_t widget) const;

public:
   TGLManip();
   TGLManip(TGLPhysicalShape* shape);
   virtual ~TGLManip();

   UInt_t GetSelectedWidget()   const { return fSelectedWidget; }
   void   SetSelectedWidget(UInt_t s) { fSelectedWidget = s; }

   Bool_t GetActive()   const { return fActive; }
   void   SetActive(Bool_t a) { fActive = a; }

   void               Attach(TGLPhysicalShape* shape) { fShape = shape; }
   TGLPhysicalShape * GetAttached() const { return fShape; }

   virtual void   Draw(const TGLCamera& camera) const = 0;
   // CRAPPY TVirtualGLManip TTTT, just override it here
   virtual Bool_t Select(const TGLCamera&, const TGLRect&, const TGLBoundingBox&) { return kFALSE; }

   virtual Bool_t HandleButton(const Event_t& event, const TGLCamera& camera);
   virtual Bool_t HandleMotion(const Event_t& event, const TGLCamera& camera);

   ClassDef(TGLManip, 0); // abstract base GL manipulator widget
};

#endif
