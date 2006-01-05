// @(#)root/gl:$Name:  $:$Id: TGLDrawable.h,v 1.8 2005/11/22 18:05:46 brun Exp $
// Author:  Richard Maunder  25/05/2005

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLDrawable
#define ROOT_TGLDrawable

#ifndef ROOT_TGLBoundingBox
#include "TGLBoundingBox.h"
#endif

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLDrawable                                                          //      
//                                                                      //
// Abstract base class for all GL drawable objects - TGLPhysicalShape & //
// TGLLogicalShape hierarchy. Provides hooks for using the display list //
// cache in TGLDrawable::Draw() - the external draw method for all      //
// shapes.                                                              //
//                                                                      //
// Defines pure virtual TGLDrawable::DirectDraw() which derived classes //
// must implement with actual GL drawing.                               //
//                                                                      //
// Display list caching can occur at either the physical or logical     //
// level (with or without translation). Currently we cache only certain //
// derived logical shapes as not all logicals can respect the LOD draw  //
// flag which is used in caching.                                       //
//////////////////////////////////////////////////////////////////////////

class TGLDrawable
{
private:
   // Fields
   ULong_t fID;        //! unique drawable ID

   // Methods
   // Non-copyable class
   TGLDrawable(const TGLDrawable &);
   const TGLDrawable & operator=(const TGLDrawable &);

protected:
   // Fields
   Bool_t            fDLCache;     //! potentially DL cached
   TGLBoundingBox    fBoundingBox; //! the drawables bounding box

   // TODO: Split to AABB for logical, and OBB for physical - moved out of here
   // can keep requirement that all drawables support returning a base BB class.

   // Methods
   virtual void DirectDraw(UInt_t LOD) const = 0; // Actual draw method (non DL cached)

public:
   TGLDrawable(ULong_t ID, bool DLCache);
   virtual ~TGLDrawable();

         ULong_t          ID()          const { return fID; }
   const TGLBoundingBox & BoundingBox() const { return fBoundingBox; }

   virtual Bool_t SupportsLOD() const = 0;
   virtual void   Draw(UInt_t LOD) const;

   virtual void DrawWireFrame(UInt_t lod) const
   {
      DirectDraw(lod);
   }
   virtual void DrawOutline(UInt_t lod) const
   {
      DirectDraw(lod);   
   }


   // Caching
   bool SetDLCache(bool DLCache);
   virtual bool UseDLCache(UInt_t LOD) const;
   virtual void Purge();

   ClassDef(TGLDrawable,0) // abstract GL drawable object
};

#endif // ROOT_TGLDrawable
