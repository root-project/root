// @(#)root/gl:$Id$
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

#include "TGLPhysicalShape.h"
#include "TGLOverlay.h"

class TGLRnrCtx;
class TGLManipSet;

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLClip                                                              //
//                                                                      //
// Abstract clipping shape - derives from TGLPhysicalShape              //
// Adds clip mode (inside/outside) and pure virtual method to           //
// approximate shape as set of planes. This plane set is used to perform//
// interactive clipping using OpenGL clip planes.                       //
//////////////////////////////////////////////////////////////////////////

class TGLClip : public TGLPhysicalShape
{
public:
   enum EMode
   {
      kOutside, // Clip away what's outside
      kInside   // Clip away what's inside
   };
   enum EType
   {
      kClipNone = 0,
      kClipPlane,
      kClipBox
   };

protected:
   EMode  fMode;
   UInt_t fTimeStamp;
   Bool_t fValid;

public:
   TGLClip(const TGLLogicalShape & logical, const TGLMatrix & transform, const float color[4]);
   virtual ~TGLClip();

   virtual void Modified() { TGLPhysicalShape::Modified(); IncTimeStamp(); }

   virtual void Setup(const TGLBoundingBox & bbox) = 0;
   virtual void Setup(const TGLVector3&, const TGLVector3&);

   EMode GetMode() const      { return fMode; }
   void  SetMode(EMode mode)  { if (mode != fMode) { fMode = mode; ++fTimeStamp; } }

   UInt_t TimeStamp() const { return fTimeStamp; }
   void   IncTimeStamp()    { ++fTimeStamp; }

   Bool_t IsValid() const { return fValid;   }
   void   Invalidate()    { fValid = kFALSE; }

   virtual void Draw(TGLRnrCtx & rnrCtx) const;
   virtual void PlaneSet(TGLPlaneSet_t & set) const = 0;

   ClassDef(TGLClip,0); // abstract clipping object
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLClipPlane                                                         //
//                                                                      //
// Concrete clip plane object. This can be translated in all directions //
// rotated about the Y/Z local axes (the in-plane axes). It cannot be   //
// scaled.                                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGLClipPlane : public TGLClip
{
private:
   static const float fgColor[4];   //! Fixed color of clip plane

public:
   TGLClipPlane();
   virtual ~TGLClipPlane();

   virtual void Setup(const TGLBoundingBox & bbox);
   virtual void Setup(const TGLVector3& point, const TGLVector3& normal);

   void Set(const TGLPlane & plane);

   virtual void PlaneSet(TGLPlaneSet_t & set) const;

   ClassDef(TGLClipPlane, 0); // clipping plane
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLClipBox                                                           //
//                                                                      //
// Concrete clip box object. Can be translated, rotated and scaled in   //
// all (xyz) axes.                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGLClipBox : public TGLClip
{
private:
   static const float fgColor[4];   //! Fixed color of clip box

public:
   TGLClipBox();
   virtual ~TGLClipBox();

   virtual void Setup(const TGLBoundingBox & bbox);
   virtual void Setup(const TGLVector3& min_point, const TGLVector3& max_point);

   virtual void PlaneSet(TGLPlaneSet_t & set) const;

   ClassDef(TGLClipBox, 0); // clipping box
};

//////////////////////////////////////////////////////////////////////////
//
// TGLClipSet
//
// A collection of all available clipping objects, to be used by higher
// level objects. For the time being by TGLViewer/Scene.
//
//////////////////////////////////////////////////////////////////////////

class TGLClipSet : public TGLOverlayElement
{
private:
   TGLClipSet(const TGLClipSet&);            // Not implemented
   TGLClipSet& operator=(const TGLClipSet&); // Not implemented

protected:
   TGLClipPlane          *fClipPlane;
   TGLClipBox            *fClipBox;
   TGLClip               *fCurrentClip;  //! the current clipping shape

   Bool_t                 fAutoUpdate;
   Bool_t                 fShowClip;
   Bool_t                 fShowManip;
   TGLManipSet           *fManip;

   TGLBoundingBox         fLastBBox;

public:
   TGLClipSet();
   virtual ~TGLClipSet();

   virtual Bool_t MouseEnter(TGLOvlSelectRecord& selRec);
   virtual Bool_t MouseStillInside(TGLOvlSelectRecord& selRec);
   virtual Bool_t Handle(TGLRnrCtx& rnrCtx, TGLOvlSelectRecord& selRec,
                         Event_t* event);
   virtual void   MouseLeave();

   virtual void Render(TGLRnrCtx& rnrCtx);

   Bool_t    IsClipping()     const { return fCurrentClip != 0; }
   TGLClip*  GetCurrentClip() const { return fCurrentClip; }
   void      FillPlaneSet(TGLPlaneSet_t& set) const;

   // Clipping
   void  SetupClips(const TGLBoundingBox& sceneBBox);
   void  SetupCurrentClip(const TGLBoundingBox& sceneBBox);
   void  SetupCurrentClipIfInvalid(const TGLBoundingBox& sceneBBox);

   void  InvalidateClips();
   void  InvalidateCurrentClip();

   void  GetClipState(TGLClip::EType type, Double_t data[6]) const;
   void  SetClipState(TGLClip::EType type, const Double_t data[6]);

   TGLClip::EType GetClipType() const;
   void           SetClipType(TGLClip::EType type);

   // Clip control flags
   Bool_t GetAutoUpdate()     const { return fAutoUpdate; }
   void   SetAutoUpdate(Bool_t aup) { fAutoUpdate = aup;  }
   Bool_t GetShowManip()      const { return fShowManip; }
   void   SetShowManip(Bool_t show) { fShowManip = show; }
   Bool_t GetShowClip()       const { return fShowClip; }
   void   SetShowClip(Bool_t show)  { fShowClip = show; }

   ClassDef(TGLClipSet, 0); // A collection of supported clip-objects
};

#endif
