// @(#)root/gl:$Name:$:$Id:$
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
   TGLPhysicalShape(UInt_t ID, const TGLLogicalShape & logicalShape,
                    const TGLMatrix & transform, Bool_t invertedWind);
   TGLPhysicalShape(UInt_t ID, const TGLLogicalShape & logicalShape,
                    const double * transform, Bool_t invertedWind);
   virtual ~TGLPhysicalShape();

   virtual void Draw(UInt_t LOD) const;

   void InvokeContextMenu(TContextMenu & menu, UInt_t x, UInt_t y) const;

   void Select(Bool_t select)  { fSelected = select; }

   const Float_t * GetColor() const { return fColor; }
   void SetColor(const Float_t rgba[4]);

   Bool_t IsSelected() const     { return fSelected; }
   Bool_t IsTransparent() const  { return fColor[3] < 1.f; }

   ClassDef(TGLPhysicalShape,0) // a physical (placed, global frame) drawable object
};

#endif // ROOT_TGLPhysicalShape
