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

class TGLRotateManip : public TGLManip
{
private:
   TGLLine3 fRingLine;
   TGLLine3 fRingLineOld;
   mutable TGLLine3 fDebugProj;

   void DrawAxisRing(const TGLVertex3 & origin, const TGLVector3 & axis, 
                     Double_t radius, Float_t rgba[4]) const;
   TGLLine3 CalculateRingLine(const TPoint & mouse, const TGLCamera & camera) const;

public:
   TGLRotateManip(TGLViewer & viewer);
   TGLRotateManip(TGLViewer & viewer, TGLPhysicalShape * shape);
   virtual ~TGLRotateManip();
   
   virtual void   Draw(const TGLCamera & camera) const; 
   virtual Bool_t HandleButton(const Event_t * event, const TGLCamera & camera);
   virtual Bool_t HandleMotion(const Event_t * event, const TGLCamera & camera);

   ClassDef(TGLRotateManip,0) // GL rotation manipulator widget
};

#endif
