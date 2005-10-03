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

#ifndef ROOT_TGLQuadric
#include "TGLQuadric.h"
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
class TGLViewer;

class TGLManip
{
protected:
   enum EHeadShape { kArrow, kBox };

   // Nasty - this only needs to be here to make a external cross-thread select call 
   // on us - gVirutalGL issues
   TGLViewer        & fViewer; 
   TGLPhysicalShape * fShape;
   UInt_t             fSelectedWidget;
   Bool_t             fActive;
   Int_t              fFirstMouseX;
   Int_t              fFirstMouseY;
   Int_t              fLastMouseX;
   Int_t              fLastMouseY;

   //void TestHit() {}; // Draw out with gl names hit stack - process hit in overload
   static TGLQuadric  fgQuad;
   static UInt_t      fgQuality;

   static Float_t     fgRed[4];
   static Float_t     fgGreen[4];
   static Float_t     fgBlue[4];
   static Float_t     fgYellow[4];
   static Float_t     fgWhite[4];

   void DrawAxisWidgets(EHeadShape head) const; 
   void DrawAxisWidget(EHeadShape head, const TGLVertex3 & origin, const TGLVector3 & vector, 
                       Double_t size, Float_t rgba[4]) const;
   void DrawOrigin(const TGLVertex3 & origin, Double_t size, Float_t rgba[4]) const;
   void SetDrawColors(Float_t rgba[4]) const;

public:
   TGLManip(TGLViewer & viewer);
   TGLManip(TGLViewer & viewer, TGLPhysicalShape * shape);
   virtual ~TGLManip();
   
   void               Attach(TGLPhysicalShape * shape) { fShape = shape; }
   TGLPhysicalShape * GetAttached() const { return fShape; }

   virtual void   Draw() const = 0; 
   virtual void   Select();
   virtual Bool_t HandleButton(Event_t * event);
   virtual Bool_t HandleMotion(Event_t * event, const TGLCamera & camera);

   ClassDef(TGLManip,0) // abstract base GL manipulator widget
};

#endif
