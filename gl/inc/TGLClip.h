// @(#)root/gl:$Name:  $:$Id: TGLClip.h
// Author:  Richard Maunder  16/09/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLClip
#define ROOT_TGLClip

#ifndef ROOT_TGLPhysicalShape
#include "TGLPhysicalShape.h"
#endif

// Do we actually need this ? Could just use TGLPhysical + external
// mode flag in viewer...? TGLPhysical could just impl. PlaneSet
// get from BB and overload for other weird shapes (in TGLLogical)

// Or maybe embedded TGLPhysical so can be passed - just expose Draw() + BB()

class TGLClip
{
public:
   enum EMode { kInside, kOutside };
private:
   EMode fMode;
public:
   TGLClip();
   virtual ~TGLClip();

   EMode Mode() const         { return fMode; }
   void  SetMode(EMode mode)  { fMode = mode; }
 
   virtual void Draw(UInt_t LOD) const = 0;
   virtual void PlaneSet(TGLPlaneSet_t & set) const = 0;

   ClassDef(TGLClip,0); // abstract clipping object   
};

class TGLClipPlane : public TGLClip
{
private:
   TGLPlane fPlane;
public:
   TGLClipPlane(const TGLPlane &);
   virtual ~TGLClipPlane();

   void Set(const TGLPlane & plane);

   virtual void Draw(UInt_t LOD) const;
   virtual void PlaneSet(TGLPlaneSet_t & set) const;

   ClassDef(TGLClipPlane,0); // clipping plane
};

class TGLClipShape : public TGLClip, public TGLPhysicalShape
{
private:
   static float fgColor[4];

public:   
   TGLClipShape(const TGLLogicalShape & logicalShape, const TGLMatrix & transform);
   virtual ~TGLClipShape();   

   virtual void Draw(UInt_t LOD) const;
   virtual void PlaneSet(TGLPlaneSet_t & set) const { return BoundingBox().PlaneSet(set); }
   
   ClassDef(TGLClipShape,0); // clipping shape
};

#endif
