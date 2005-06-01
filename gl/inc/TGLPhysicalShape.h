// @(#)root/gl:$Name:  $:$Id: TGLPhysicalShape.h,v 1.3 2005/05/26 12:29:50 rdm Exp $
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

   // TODO: Common UInt_t flags section (in TGLDrawable?) to avoid multiple bools
protected:
   // Methods
   virtual void DirectDraw(UInt_t LOD) const;

public:
   TGLPhysicalShape(ULong_t ID, const TGLLogicalShape & logicalShape,
                    const TGLMatrix & transform, Bool_t invertedWind);
   TGLPhysicalShape(ULong_t ID, const TGLLogicalShape & logicalShape,
                    const double * transform, Bool_t invertedWind);
   virtual ~TGLPhysicalShape();

   // Associated logical
   const TGLLogicalShape & GetLogical() const { return fLogicalShape; }

   virtual void Draw(UInt_t LOD) const;
   void         InvokeContextMenu(TContextMenu & menu, UInt_t x, UInt_t y) const;

   // Selection
   Bool_t          IsSelected() const                 { return fSelected; }
   void            Select(Bool_t select)              { fSelected = select; }

   // Color
   const Float_t * GetColor() const                   { return fColor; }
   Bool_t          IsTransparent() const              { return fColor[3] < 1.f; }
   void            SetColor(const Float_t rgba[4]);

   // Geometry
   void            UpdateBoundingBox(); 
   TGLVertex3      GetTranslation() const;
   void            SetTranslation(const TGLVertex3 & trans);
   void            Shift(const TGLVector3 & shift);
   TGLVector3      GetScale() const;
   void            SetScale(const TGLVector3 & scale);

   ClassDef(TGLPhysicalShape,0) // a physical (placed, global frame) drawable object
};

//______________________________________________________________________________
inline TGLVertex3 TGLPhysicalShape::GetTranslation() const
{ 
   return fTransform.GetTranslation(); 
}

//______________________________________________________________________________
inline void TGLPhysicalShape::SetTranslation(const TGLVertex3 & trans) 
{ 
   fTransform.SetTranslation(trans);
   UpdateBoundingBox();
}

//______________________________________________________________________________
inline void TGLPhysicalShape::Shift(const TGLVector3 & shift)
{
   fTransform.Shift(shift);
   UpdateBoundingBox();
}

//______________________________________________________________________________
inline TGLVector3 TGLPhysicalShape::GetScale() const
{ 
   return fTransform.GetScale(); 
}

//______________________________________________________________________________
inline void TGLPhysicalShape::SetScale(const TGLVector3 & scale) 
{ 
   fTransform.SetScale(scale); 
   UpdateBoundingBox();
}

#endif // ROOT_TGLPhysicalShape
