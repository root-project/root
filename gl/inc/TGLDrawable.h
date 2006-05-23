// @(#)root/gl:$Name: v5-11-02 $:$Id: TGLDrawable.h,v 1.13 2006/04/07 08:43:59 brun Exp $
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
#ifndef ROOT_TGLQuadric
#include "TGLQuadric.h"
#endif


class TGLDrawFlags;

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
protected:
   // Fields
   ULong_t fID;        //! unique drawable ID

protected:
   // Fields
   Bool_t            fCached;      //! use display list cached
   TGLBoundingBox    fBoundingBox; //! the drawables bounding box

   static TGLQuadric fgQuad;        //! Single common quadric drawing object

   // Non-copyable class
   TGLDrawable(const TGLDrawable &);
   const TGLDrawable & operator=(const TGLDrawable &);

   // TODO: Split to AABB for logical, and OBB for physical - moved out of here
   // can keep requirement that all drawables support returning a base BB class.

   // Methods
   virtual void DirectDraw(const TGLDrawFlags & flags) const = 0; // Actual draw method (non DL cached)

public:
   enum ELODAxes  { kLODAxesNone = 0,  // Implies draw/DL caching done at kLODUnsupported
                    kLODAxesX    = 1 << 0,
                    kLODAxesY    = 1 << 1,
                    kLODAxesZ    = 1 << 2,
                    kLODAxesAll  = kLODAxesX | kLODAxesY | kLODAxesZ
                  };

   TGLDrawable(ULong_t ID, Bool_t cached);
   virtual ~TGLDrawable();

         ULong_t          ID()          const { return fID; }
   const TGLBoundingBox & BoundingBox() const { return fBoundingBox; }

   virtual ELODAxes SupportedLODAxes() const = 0;
   virtual void     Draw(const TGLDrawFlags & flags) const;

   // Display List Caching
           Bool_t SetCached(Bool_t cached);
   virtual Bool_t ShouldCache(const TGLDrawFlags & flags) const;
   virtual void   Purge();

   ClassDef(TGLDrawable,0) // abstract GL drawable object
};

#endif // ROOT_TGLDrawable
