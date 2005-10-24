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

class TGLClip : public TGLPhysicalShape
{
public:
   enum EMode { kInside, kOutside };
private:
   EMode fMode;
public:
   TGLClip(const TGLLogicalShape & logical, const TGLMatrix & transform, const float color[4]);
   virtual ~TGLClip();

   EMode Mode() const         { return fMode; }
   void  SetMode(EMode mode)  { fMode = mode; }
 
   virtual void Draw(UInt_t LOD) const;
   virtual void PlaneSet(TGLPlaneSet_t & set) const = 0;

   ClassDef(TGLClip,0); // abstract clipping object   
};

class TGLClipPlane : public TGLClip
{
private:
   static const float fgColor[4];

public:
   TGLClipPlane(const TGLPlane &  plane, const TGLVertex3 & center, Double_t extents);
   virtual ~TGLClipPlane();

   void Set(const TGLPlane & plane);

   virtual void PlaneSet(TGLPlaneSet_t & set) const;

   ClassDef(TGLClipPlane,0); // clipping plane
};

class TGLClipBox : public TGLClip 
{
private:
   static const float fgColor[4];

public:   
   TGLClipBox(const TGLVector3 & halfLengths, const TGLVertex3 & center);
   virtual ~TGLClipBox();   

   virtual void PlaneSet(TGLPlaneSet_t & set) const;
   
   ClassDef(TGLClipBox,0); // clipping box
};

#endif
