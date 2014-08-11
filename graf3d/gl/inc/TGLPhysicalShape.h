// @(#)root/gl:$Id$
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

//#ifndef ROOT_TGLLogicalShape
//#include "TGLLogicalShape.h"
//#endif
#ifndef ROOT_TGLBoundingBox
#include "TGLBoundingBox.h"
#endif
#ifndef ROOT_TGLUtil
#include "TGLUtil.h" // For TGLMatrix
#endif

class TGLPShapeRef;
class TGLLogicalShape;
class TGLRnrCtx;

class TContextMenu;


class TGLPhysicalShape
{
   friend class TGLLogicalShape; // for replica-list management

private:
   TGLPhysicalShape(const TGLPhysicalShape&);            // Not implemented
   TGLPhysicalShape& operator=(const TGLPhysicalShape&); // Not implemented

public:
   // Flags for permitted manipulation of object
   enum EManip  { kTranslateX   = 1 << 0,
                  kTranslateY   = 1 << 1,
                  kTranslateZ   = 1 << 2,
                  kTranslateAll = kTranslateX | kTranslateY | kTranslateZ,
                  kScaleX       = 1 << 3,
                  kScaleY       = 1 << 4,
                  kScaleZ       = 1 << 5,
                  kScaleAll     = kScaleX | kScaleY | kScaleZ,
                  kRotateX      = 1 << 6,
                  kRotateY      = 1 << 7,
                  kRotateZ      = 1 << 8,
                  kRotateAll    = kRotateX | kRotateY | kRotateZ,
                  kManipAll     = kTranslateAll | kScaleAll | kRotateAll
                };
private:
   // Fields
   const TGLLogicalShape * fLogicalShape; //! the associated logical shape
         TGLPhysicalShape* fNextPhysical; //! pointer to next replica
         TGLPShapeRef    * fFirstPSRef;   //! pointer to first reference

   UInt_t                  fID;           //! unique physical ID within containing scene
   TGLMatrix               fTransform;    //! transform (placement) of physical instance
   TGLBoundingBox          fBoundingBox;  //! bounding box of the physical (transformed)
   Float_t                 fColor[17];    //! GL color array
   EManip                  fManip;        //! permitted manipulation bitflags - see EManip
   UChar_t                 fSelected;     //! selected state
   Bool_t                  fInvertedWind; //! face winding TODO: can get directly from fTransform?
   Bool_t                  fModified;     //! has been modified - retain across scene rebuilds
   Bool_t                  fIsScaleForRnr;//! cache

   // Methods
   void            UpdateBoundingBox();
   void            InitColor(const Float_t rgba[4]);

public:
   TGLPhysicalShape(UInt_t ID, const TGLLogicalShape & logicalShape,
                    const TGLMatrix & transform, Bool_t invertedWind,
                    const Float_t rgba[4]);
   TGLPhysicalShape(UInt_t ID, const TGLLogicalShape & logicalShape,
                    const double * transform, Bool_t invertedWind,
                    const Float_t rgba[4]);
   virtual ~TGLPhysicalShape();

   void   AddReference   (TGLPShapeRef* ref);
   void   RemoveReference(TGLPShapeRef* ref);

   UInt_t                 ID()          const { return fID; }
   const TGLBoundingBox & BoundingBox() const { return fBoundingBox; }

   virtual void CalculateShapeLOD(TGLRnrCtx & rnrCtx, Float_t& pixSize, Short_t& shapeLOD) const;
   virtual void QuantizeShapeLOD (Short_t shapeLOD, Short_t combiLOD, Short_t& quantLOD) const;

   void SetupGLColors(TGLRnrCtx & rnrCtx, const Float_t* color=0) const;
   virtual void Draw(TGLRnrCtx & rnrCtx) const;

   const TGLLogicalShape  * GetLogical()      const { return fLogicalShape; }
   const TGLPhysicalShape * GetNextPhysical() const { return fNextPhysical; }

   // Modification and manipulation
   EManip  GetManip()   const      { return fManip;  }
   void    SetManip(EManip manip)  { fManip = manip; }

   // Modified - treated as temporary modification
   void    Modified();
   Bool_t  IsModified() const      { return fModified; }

   // Selection
   Bool_t  IsSelected()  const    { return fSelected != 0; }
   UChar_t GetSelected() const    { return fSelected; }
   void    Select(UChar_t select) { fSelected = select; }

   // Color
   const Float_t  * Color() const                      { return fColor; }
   Bool_t           IsTransparent() const              { return fColor[3] < 1.f; }
   Bool_t           IsInvisible() const                { return fColor[3] == 0.f; }
   void             SetColor(const Float_t rgba[17]);
   void             SetColorOnFamily(const Float_t rgba[17]);
   void             SetDiffuseColor(const Float_t rgba[4]);
   void             SetDiffuseColor(const UChar_t rgba[4]);
   void             SetDiffuseColor(Color_t ci, UChar_t transparency);

   // Geometry
   TGLVector3       GetScale() const;
   TGLVertex3       GetTranslation() const;

   void             SetTransform(const TGLMatrix & transform);
   void             SetTransform(const Double_t vals[16]);
   void             SetTranslation(const TGLVertex3 & translation);
   void             Translate(const TGLVector3 & vect);
   void             Scale(const TGLVector3 & scale);
   void             Rotate(const TGLVertex3 & pivot, const TGLVector3 & axis, Double_t angle);

   // Context menu
   void             InvokeContextMenu(TContextMenu & menu, UInt_t x, UInt_t y) const;

   ClassDef(TGLPhysicalShape,0) // a physical (placed, global frame) drawable object
};


//______________________________________________________________________________
inline TGLVector3 TGLPhysicalShape::GetScale() const
{
   return fTransform.GetScale();
}

//______________________________________________________________________________
inline TGLVertex3 TGLPhysicalShape::GetTranslation() const
{
   return fTransform.GetTranslation();
}

//______________________________________________________________________________
inline void TGLPhysicalShape::SetTransform(const TGLMatrix & transform)
{
   fTransform = transform;
   UpdateBoundingBox();
   Modified();
}

//______________________________________________________________________________
inline void TGLPhysicalShape::SetTransform(const Double_t vals[16])
{
   fTransform.Set(vals);
   UpdateBoundingBox();
   Modified();
}

//______________________________________________________________________________
inline void TGLPhysicalShape::SetTranslation(const TGLVertex3 & translation)
{
   fTransform.SetTranslation(translation);
   UpdateBoundingBox();
   Modified();
}

//______________________________________________________________________________
inline void TGLPhysicalShape::Translate(const TGLVector3 & vect)
{
   fTransform.Translate(vect);
   UpdateBoundingBox();
   Modified();
}

//______________________________________________________________________________
inline void TGLPhysicalShape::Scale(const TGLVector3 & scale)
{
   TGLVertex3 origCenter = fBoundingBox.Center();
   fTransform.Scale(scale);
   UpdateBoundingBox();
   TGLVector3 shift = fBoundingBox.Center() - origCenter;
   Translate(-shift);
   UpdateBoundingBox();
   Modified();
}

//______________________________________________________________________________
inline void TGLPhysicalShape::Rotate(const TGLVertex3 & pivot, const TGLVector3 & axis, Double_t angle)
{
   TGLVertex3 c = BoundingBox().Center();
   fTransform.Rotate(pivot, axis, angle);
   UpdateBoundingBox();
   Modified();
}

#endif // ROOT_TGLPhysicalShape
