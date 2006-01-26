// @(#)root/gl:$Name:  $:$Id: TGLRotateManip.h
// Author:  Richard Maunder  04/10/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLRotateManip
#define ROOT_TGLRotateManip

#ifndef ROOT_TGLManip
#include "TGLManip.h"
#endif
#ifndef ROOT_TGLUtil
#include "TGLUtil.h"
#endif

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLRotateManip                                                       //
//                                                                      //
// Rotate manipulator - attaches to physical shape and draws local axes //
// widgets - rings drawn from attached physical center, in plane defined// 
// by axis. User can mouse over (turns yellow) and L click/drag to      //
// rotate attached physical round the ring center.                      // 
// Widgets use standard 3D package axes colours: X red, Y green, Z blue.//
//////////////////////////////////////////////////////////////////////////

class TGLRotateManip : public TGLManip
{
private:
   // Active ring interaction - set on mouse down
   Bool_t     fShallowRing;
   Bool_t     fShallowFront;
   TGLPlane   fActiveRingPlane;
   TGLVertex3 fActiveRingCenter;

   // Normal interaction tracking (non-shallow)
   TGLLine3 fRingLine;
   TGLLine3 fRingLineOld;

   void DrawAxisRing(const TGLVertex3 & origin, const TGLVector3 & axis, 
                     Double_t radius, Float_t rgba[4]) const;
   Double_t CalculateAngleDelta(const TPoint & mouse, const TGLCamera & camera);
   TGLLine3 CalculateRingLine(const TPoint & mouse, const TGLCamera & camera) const;

public:
   TGLRotateManip();
   TGLRotateManip(TGLPhysicalShape * shape);
   virtual ~TGLRotateManip();
   
   virtual void   Draw(const TGLCamera & camera) const; 
   virtual Bool_t HandleButton(const Event_t & event, const TGLCamera & camera);
   virtual Bool_t HandleMotion(const Event_t & event, const TGLCamera & camera, const TGLBoundingBox & sceneBox);

   ClassDef(TGLRotateManip,0) // GL rotation manipulator widget
};

#endif
