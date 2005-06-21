// @(#)root/gl:$Name:  $:$Id: TGLBoundingBox.h,v 1.4 2005/06/15 10:22:57 brun Exp $
// Author:  Richard Maunder  25/05/2005

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLBoundingBox
#define ROOT_TGLBoundingBox

#ifndef ROOT_TGLUtil
#include "TGLUtil.h"
#endif

/*************************************************************************
 * TGLBoundingBox - TODO
 *
 *
 *
 *************************************************************************/

// TODO: Create more compact version + axis aligned version, both with lazy
// sphere testing.
class TGLBoundingBox
{
private:
   // Fields

   // Box vertices are indexed thus:
   //
   //   7-------6
   //  /|      /|
   // 3-------2 |
   // | 4-----|-5
   // |/      |/
   // 0-------1
   //
   // 0123 near face
   // 4567 far face
   //
   // This could be more compact: 3 vertices which form plane cutting
   // box diagonally (e.g. 0,5,6 or 1,3,6 etc) would fix it in space - rest
   // could be calculated on demand - but not worth effort.....
   TGLVertex3 fVertex[8]; //! the 8 bounding box vertices
   Double_t   fVolume;    //! box volume - cached for speed

   // Methods
   void     UpdateVolume();
   Bool_t   ValidIndex(UInt_t index) const { return (index < 8); }
   Double_t Min(UInt_t index) const;
   Double_t Max(UInt_t index) const;

public:
   TGLBoundingBox();
   TGLBoundingBox(const TGLVertex3 vertex[8]);
   TGLBoundingBox(const Double_t vertex[8][3]);
   TGLBoundingBox(const TGLVertex3 & lowVertex, const TGLVertex3 & highVertex);
   TGLBoundingBox(const TGLBoundingBox & other);
   virtual ~TGLBoundingBox(); // ClassDef introduces virtual fns

   // Set orientated BB
   TGLBoundingBox & operator=(const TGLBoundingBox & other) { Set(other); return *this; }
   void Set(const TGLVertex3 vertex[8]);
   void Set(const Double_t vertex[8][3]);
   void Set(const TGLBoundingBox & other);
   void SetEmpty();

   // Set orientated AA
   void SetAligned(const TGLVertex3 & lowVertex, const TGLVertex3 & highVertex); // axis aligned
   void SetAligned(UInt_t nbPnts, const Double_t * pnts); // axis aligned

   void Transform(const TGLMatrix & matrix);
   void Scale(Double_t val);
   void Translate(const TGLVector3 & offset);

   inline const TGLVertex3 & operator [] (UInt_t index) const;
   Double_t XMin() const { return Min(0); }
   Double_t XMax() const { return Max(0); }
   Double_t YMin() const { return Min(1); }
   Double_t YMax() const { return Max(1); }
   Double_t ZMin() const { return Min(2); }
   Double_t ZMax() const { return Max(2); }

   inline TGLVertex3 Center() const;
   inline TGLVector3 Extents() const;
   inline TGLVector3 Axis(UInt_t i, Bool_t normalised = true) const;
   inline Double_t   Volume() const { return fVolume; }
   inline Bool_t     IsEmpty() const;

          EOverlap Overlap(const TGLPlane & plane) const;
          Bool_t   AlignedContains(const TGLBoundingBox & other) const; // we MUST be axis aligned
   static Bool_t   Intersect(const TGLBoundingBox & a, const TGLBoundingBox & b);

   void Draw() const;
   void Dump() const;

   ClassDef(TGLBoundingBox,0); // a 3D orientated bounding box
};

//______________________________________________________________________________
inline const TGLVertex3 & TGLBoundingBox::operator [] (UInt_t index) const
{
   if (ValidIndex(index)) {
      return fVertex[index];
   } else {
      assert(kFALSE);
      return fVertex[0];
   }
}

//______________________________________________________________________________
inline TGLVector3 TGLBoundingBox::Extents() const
{
   return TGLVector3(Axis(0,kFALSE).Mag(),
                     Axis(1,kFALSE).Mag(),
                     Axis(2,kFALSE).Mag());
}

//______________________________________________________________________________
inline TGLVertex3 TGLBoundingBox::Center() const
{
   TGLVector3 v = fVertex[6] - fVertex[0];
   return fVertex[0] + v/2.0;
}

//______________________________________________________________________________
inline TGLVector3 TGLBoundingBox::Axis(UInt_t i, Bool_t normalised) const
{
   //   7-------6
   //  /|      /|
   // 3-------2 |
   // | 4-----|-5
   // |/      |/
   // 0-------1

   TGLVector3 axis;
   if (i == 0) {
      axis.Set(fVertex[1] - fVertex[0]);
   } else if (i == 1) {
      axis.Set(fVertex[3] - fVertex[0]);
   } else if (i == 2) {
      axis.Set(fVertex[4] - fVertex[0]);
   } else {
      assert(kFALSE);
   }
   if (normalised) {
      axis.Normalise();
   }
   return axis;
}

//______________________________________________________________________________
inline void TGLBoundingBox::UpdateVolume()
{
   TGLVector3 extents = Extents();
   fVolume = fabs(extents.X() * extents.Y() * extents.Z());
}

inline Bool_t TGLBoundingBox::IsEmpty() const
{
   // TODO: Round errors - should have epsilon test
   return (Volume() == 0.0);
}

#endif // ROOT_TGLBoundingBox
