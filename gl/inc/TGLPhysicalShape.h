// @(#)root/gl:$Name:  $:$Id: TGLPhysicalShape.h,v 1.14 2006/01/18 16:57:58 brun Exp $
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

class TGLCamera;
class TContextMenu;

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLPhysicalShape                                                     //
//                                                                      //
// Concrete physical shape - a GL drawable. Physical shapes are the     //
// objects the user can actually see, select, move in the viewer. They  //
// placements of their associated local frame TGLLogicalShape into the  //
// world frame. The draw process is:                                    //
//                                                                      // 
// Load attributes - material colors etc                                // 
// Load translation matrix - placement                                  //
// Load gl name (for selection)                                         //
// Call our associated logical shape Draw() to draw placed shape        //
//                                                                      //
// The physical shape supports translation, scaling and rotation,       //
// selection, color changes, and permitted modification flags etc.      //
// A physical shape is always bound to a single, fixed logical shape    //
// - hence const & handle. It can perform mutable reference counting on //
// the logical to enable purging.                                       //
//                                                                      //
// See base/src/TVirtualViewer3D for description of common external 3D  //
// viewer architecture and how external viewer clients use it.          //
//////////////////////////////////////////////////////////////////////////

class TGLPhysicalShape : public TGLDrawable
{
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
   const TGLLogicalShape & fLogicalShape; //! the associated logical shape
   TGLMatrix               fTransform;    //! transform (placement) of physical instance
   Float_t                 fColor[17];    //! GL color array
   Bool_t                  fSelected;     //! selected state
   Bool_t                  fInvertedWind; //! face winding TODO: can get directly from fTransform?
   Bool_t                  fModified;     //! has been modified - retain across scene rebuilds
   EManip                  fManip;        //! permitted manipulation bitflags - see EManip
   // TODO: Common UInt_t flags section (in TGLDrawable?) to avoid multiple bools

   // Methods
   void            UpdateBoundingBox(); 
   void            InitColor(const Float_t rgba[4]);

protected:
   // Methods
   virtual void DirectDraw(const TGLDrawFlags & flags) const;

public:
   TGLPhysicalShape(ULong_t ID, const TGLLogicalShape & logicalShape,
                    const TGLMatrix & transform, Bool_t invertedWind,
                    const Float_t rgba[4]);
   TGLPhysicalShape(ULong_t ID, const TGLLogicalShape & logicalShape,
                    const double * transform, Bool_t invertedWind,
                    const Float_t rgba[4]);
   virtual ~TGLPhysicalShape();

   TGLDrawFlags CalcDrawFlags(const TGLCamera & camera, const TGLDrawFlags & sceneFlags) const;

   // TGLDrawable overloads
   virtual ELODAxes SupportedLODAxes() const;
   virtual void     Draw(const TGLDrawFlags & flags) const;

   const TGLLogicalShape & GetLogical() const { return fLogicalShape; }
   void             InvokeContextMenu(TContextMenu & menu, UInt_t x, UInt_t y) const;

   // Modification and manipulation
   EManip           GetManip() const         { return fManip; }
   void             SetManip(EManip manip)   { fManip = manip; }
   // Selected treated as temporary modification
   Bool_t           IsModified() const       { return fModified || IsSelected(); }

   // Selection
   Bool_t           IsSelected() const       { return fSelected; }
   void             Select(Bool_t select)    { fSelected = select; }

   // Color
   const Float_t  * Color() const                      { return fColor; }
   Bool_t           IsTransparent() const              { return fColor[3] < 1.f; }
   Bool_t           IsInvisible() const                { return fColor[3] == 0.f; }
   void             SetColor(const Float_t rgba[17]);

   // Geometry
   TGLVector3       GetScale() const;
   TGLVertex3       GetTranslation() const;

   void             SetTransform(const TGLMatrix & transform);
   void             SetTranslation(const TGLVertex3 & translation);
   void             Translate(const TGLVector3 & vect);
   void             Scale(const TGLVector3 & scale);
   void             Rotate(const TGLVertex3 & pivot, const TGLVector3 & axis, Double_t angle);

   ClassDef(TGLPhysicalShape,0) // a physical (placed, global frame) drawable object
};

//______________________________________________________________________________
inline TGLPhysicalShape::ELODAxes TGLPhysicalShape::SupportedLODAxes() const
{
   // Defined by our logical shape
   return fLogicalShape.SupportedLODAxes();
}

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
   fModified = kTRUE;

   // Any DL cache would be invalidate by this - NOTE does not work currently
   Purge();
}

//______________________________________________________________________________
inline void TGLPhysicalShape::SetTranslation(const TGLVertex3 & translation) 
{ 
   fTransform.SetTranslation(translation);
   UpdateBoundingBox();
   fModified = kTRUE;

   // Any DL cache would be invalidate by this - NOTE does not work currently
   Purge();
}

//______________________________________________________________________________
inline void TGLPhysicalShape::Translate(const TGLVector3 & vect)
{
   fTransform.Translate(vect);
   UpdateBoundingBox();
   fModified = kTRUE;

   // Any DL cache would be invalidate by this - NOTE does not work currently
   Purge();
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
   fModified = kTRUE;

   // Any DL cache would be invalidate by this - NOTE does not work currently
   Purge();
}

//______________________________________________________________________________
inline void TGLPhysicalShape::Rotate(const TGLVertex3 & pivot, const TGLVector3 & axis, Double_t angle)
{ 
   TGLVertex3 c = BoundingBox().Center();
   fTransform.Rotate(pivot, axis, angle);
   UpdateBoundingBox();
   fModified = kTRUE;

   // Any DL cache would be invalidate by this - NOTE does not work currently
   Purge();
}

#endif // ROOT_TGLPhysicalShape
