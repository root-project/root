// @(#)root/gl:$Name:  $:$Id: TGLManip.h,v 1.11 2006/05/15 07:43:33 brun Exp $
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

#ifndef ROOT_TVirtualGL
#include "TVirtualGL.h"
#endif
#ifndef ROOT_TPoint
#include "TPoint.h"
#endif
#ifndef ROOT_GuiTypes
#include "GuiTypes.h"
#endif
#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

class TGLPhysicalShape;
class TGLVertex3;
class TGLVector3;
class TGLCamera;
class TGLRect;
class TGLBoundingBox;
class TGLViewer;

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLManip                                                             //
//                                                                      //
// Abstract base class for viewer manipulators, which allow direct in   //
// viewer manipulation of a TGLPhysicalShape derv. object - currently   //
// translation, scaling and rotation along/round objects local axes.    //
// See derived classes for these implementations.                       //
//                                                                      //
// This class provides binding to the zero or one manipulated physical, //
// hit testing (selection) for manipulator sub component (widget), and  //
// some common mouse action handling/tracking.                          //
//////////////////////////////////////////////////////////////////////////

class TGLManip : public TVirtualGLManip {
protected:
   TGLPhysicalShape * fShape;             //! manipulated shape
   UInt_t             fSelectedWidget;    //! active width (axis) component
   Bool_t             fActive;            //! manipulator is active?

   // Mouse tracking - in WINDOW coords
   TPoint             fFirstMouse;        //! first (start) mouse position (in WINDOW coords)
   TPoint             fLastMouse;         //! last (latest) mouse position (in WINDOW coords)

   static Float_t     fgRed[4];
   static Float_t     fgGreen[4];
   static Float_t     fgBlue[4];
   static Float_t     fgYellow[4];
   static Float_t     fgWhite[4];
   static Float_t     fgGrey[4];

   TGLManip(const TGLManip&);
   TGLManip& operator=(const TGLManip&);

   void CalcDrawScale(const TGLBoundingBox & box, const TGLCamera & camera,
                      Double_t & base, TGLVector3 axis[3]) const;

public:
   TGLManip();
   TGLManip(TGLPhysicalShape * shape);
   virtual ~TGLManip();

   void               Attach(TGLPhysicalShape * shape) { fShape = shape; }
   TGLPhysicalShape * GetAttached() const { return fShape; }

   virtual void   Draw(const TGLCamera & camera) const = 0;
   virtual Bool_t Select(const TGLCamera & camera, const TGLRect & rect, const TGLBoundingBox & sceneBox);
   virtual Bool_t HandleButton(const Event_t & event, const TGLCamera & camera);
   virtual Bool_t HandleMotion(const Event_t & event, const TGLCamera & camera, const TGLBoundingBox & sceneBox);

   ClassDef(TGLManip,0) // abstract base GL manipulator widget
};

#endif
