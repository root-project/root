// @(#)root/gl:$Name:  $:$Id: TGLPhysicalShape.h,v 1.8 2005/10/03 15:19:35 brun Exp $
// Author:  Richard Maunder  25/05/2005
// Parts taken from original TGLSceneObject Timur Pocheptsov

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLPhysicalShape
#define ROOT_TGLPhysicalShape

#ifndef ROOT_TGLDrawable
#include "TGLDrawable.h"
#endif
#ifndef ROOT_TGLLogicalShape
#include "TGLLogicalShape.h"
#endif
#ifndef ROOT_TGLUtil
#include "TGLUtil.h" // For TGLMatrix
#endif

class TContextMenu;

/*************************************************************************
 * TGLPhysicalShape - TODO
 *
 *
 *
 *************************************************************************/
class TGLPhysicalShape : public TGLDrawable
{
private:
   // Fields
   const TGLLogicalShape & fLogicalShape; //! the associate logical shape
   TGLMatrix               fTransform;    //! transform (placement) of physical instance
   Float_t                 fColor[17];    //! GL color array
   Bool_t                  fSelected;     //! selected state
   Bool_t                  fInvertedWind; //! face winding TODO: can get directly from fTransform?
   Bool_t                  fModified;     //! has been modified - retain across scene rebuilds
   // TODO: Common UInt_t flags section (in TGLDrawable?) to avoid multiple bools

   // Methods
   void            UpdateBoundingBox(); 
   void            InitColor(const Float_t rgba[4]);

protected:
   // Methods
   virtual void DirectDraw(UInt_t LOD) const;

public:
   TGLPhysicalShape(ULong_t ID, const TGLLogicalShape & logicalShape,
                    const TGLMatrix & transform, Bool_t invertedWind,
                    const Float_t rgba[4]);
   TGLPhysicalShape(ULong_t ID, const TGLLogicalShape & logicalShape,
                    const double * transform, Bool_t invertedWind,
                    const Float_t rgba[4]);
   virtual ~TGLPhysicalShape();

   // Associated logical
   const TGLLogicalShape & GetLogical() const { return fLogicalShape; }

   virtual void Draw(UInt_t LOD) const;
   virtual void DrawWireFrame(UInt_t lod) const;
   virtual void DrawOutline(UInt_t lod) const;
   void         InvokeContextMenu(TContextMenu & menu, UInt_t x, UInt_t y) const;

   // Modified - selected treated as temporary modification
   Bool_t          IsModified() const                 { return fModified || IsSelected(); }

   // Selection
   Bool_t          IsSelected() const                 { return fSelected; }
   void            Select(Bool_t select)              { fSelected = select; }

   // Color
   const Float_t * Color() const                      { return fColor; }
   Bool_t          IsTransparent() const              { return fColor[3] < 1.f; }
   void            SetColor(const Float_t rgba[17]);

   // Geometry
   TGLVertex3      Translation() const;
   void            SetTranslation(const TGLVertex3 & trans);
   void            Shift(const TGLVector3 & shift);
   TGLVector3      Scale() const;
   void            SetScale(const TGLVector3 & scale);

   ClassDef(TGLPhysicalShape,0) // a physical (placed, global frame) drawable object
};

//______________________________________________________________________________
inline TGLVertex3 TGLPhysicalShape::Translation() const
{ 
   return fTransform.Translation(); 
}

//______________________________________________________________________________
inline void TGLPhysicalShape::SetTranslation(const TGLVertex3 & trans) 
{ 
   fTransform.Set(trans);
   UpdateBoundingBox();
}

//______________________________________________________________________________
inline void TGLPhysicalShape::Shift(const TGLVector3 & shift)
{
   fTransform.Shift(shift);
   UpdateBoundingBox();
   fModified = kTRUE;
}

//______________________________________________________________________________
inline TGLVector3 TGLPhysicalShape::Scale() const
{ 
   return fTransform.Scale(); 
}

//______________________________________________________________________________
inline void TGLPhysicalShape::SetScale(const TGLVector3 & scale) 
{ 
   TGLVertex3 origCenter = fBoundingBox.Center();
   fTransform.SetScale(scale); 
   UpdateBoundingBox();
   TGLVector3 shift = fBoundingBox.Center() - origCenter;
   Shift(-shift);
   UpdateBoundingBox();
   fModified = kTRUE;
}

#endif // ROOT_TGLPhysicalShape
