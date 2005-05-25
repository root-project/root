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

/*************************************************************************
 * TGLDrawable - TODO
 *
 *
 *
 *************************************************************************/
class TGLDrawable 
{
private:
   // Fields
   const UInt_t fID;        //! unique drawable ID

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
   TGLDrawable(UInt_t ID, bool DLCache);
   virtual ~TGLDrawable();

         UInt_t           ID()          const { return fID; }
   const TGLBoundingBox & BoundingBox() const { return fBoundingBox; }

   virtual void Draw(UInt_t LOD) const;

   // Caching
           bool SetDLCache(bool DLCache);
   virtual bool UseDLCache(UInt_t LOD) const;
   virtual void Purge();

   ClassDef(TGLDrawable,0) // abstract GL drawable object
};

#endif // ROOT_TGLDrawable
