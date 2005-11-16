// @(#)root/gl:$Name:  $:$Id: TGLManip.h
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
class TGLBoundingBox;
class TGLViewer;

class TGLManip
{
protected:
   // Nasty - this only needs to be here to make a external cross-thread select call 
   // on us - gVirutalGL issues
   TGLViewer        & fViewer; 
   TGLPhysicalShape * fShape;
   UInt_t             fSelectedWidget;
   Bool_t             fActive;

   // Mouse tracking - in WINDOW coords
   TPoint             fFirstMouse;
   TPoint             fLastMouse;

   static Float_t     fgRed[4];
   static Float_t     fgGreen[4];
   static Float_t     fgBlue[4];
   static Float_t     fgYellow[4];
   static Float_t     fgWhite[4];
   static Float_t     fgGrey[4];

   Double_t CalcDrawScale(const TGLBoundingBox & box, const TGLCamera & camera) const;

public:
   TGLManip(TGLViewer & viewer);
   TGLManip(TGLViewer & viewer, TGLPhysicalShape * shape);
   virtual ~TGLManip();
   
   void               Attach(TGLPhysicalShape * shape) { fShape = shape; }
   TGLPhysicalShape * GetAttached() const { return fShape; }

   virtual void   Draw(const TGLCamera & camera) const = 0; 
   virtual void   Select(const TGLCamera & camera);
   virtual Bool_t HandleButton(const Event_t * event, const TGLCamera & camera);
   virtual Bool_t HandleMotion(const Event_t * event, const TGLCamera & camera);

   ClassDef(TGLManip,0) // abstract base GL manipulator widget
};

#endif
