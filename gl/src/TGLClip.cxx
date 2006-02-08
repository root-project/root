// @(#)root/gl:$Name:  $:$Id: TGLClip.cxx,v 1.5 2006/01/30 17:42:06 rdm Exp $
// Author:  Richard Maunder  16/09/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLClip.h"
#include "TGLIncludes.h"

#include "TGLSceneObject.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"

TGLLogicalShape * CreateLogicalBox(TGLVector3 halfLengths)
{
   // Helper function to construct a TGLLogicalShape box based on
   // supplied half lengths

   //    y
   //    |
   //    |
   //    |________x
   //   /  3-------2
   //  /  /|      /|
   // z  7-------6 |
   //    | 0-----|-1
   //    |/      |/
   //    4-------5
   //
   // Construct box of points / segments / polys

   TBuffer3D buff(TBuffer3DTypes::kGeneric, 8, 3*8, 12, 3*12, 6, 6*6);

   buff.fPnts[ 0] = -halfLengths.X(); buff.fPnts[ 1] = -halfLengths.Y(); buff.fPnts[ 2] = -halfLengths.Z(); // 0
   buff.fPnts[ 3] =  halfLengths.X(); buff.fPnts[ 4] = -halfLengths.Y(); buff.fPnts[ 5] = -halfLengths.Z(); // 1
   buff.fPnts[ 6] =  halfLengths.X(); buff.fPnts[ 7] =  halfLengths.Y(); buff.fPnts[ 8] = -halfLengths.Z(); // 2
   buff.fPnts[ 9] = -halfLengths.X(); buff.fPnts[10] =  halfLengths.Y(); buff.fPnts[11] = -halfLengths.Z(); // 3

   buff.fPnts[12] = -halfLengths.X(); buff.fPnts[13] = -halfLengths.Y(); buff.fPnts[14] =  halfLengths.Z(); // 4
   buff.fPnts[15] =  halfLengths.X(); buff.fPnts[16] = -halfLengths.Y(); buff.fPnts[17] =  halfLengths.Z(); // 5
   buff.fPnts[18] =  halfLengths.X(); buff.fPnts[19] =  halfLengths.Y(); buff.fPnts[20] =  halfLengths.Z(); // 6
   buff.fPnts[21] = -halfLengths.X(); buff.fPnts[22] =  halfLengths.Y(); buff.fPnts[23] =  halfLengths.Z(); // 7

   buff.fSegs[ 0] = 1   ; buff.fSegs[ 1] = 0   ; buff.fSegs[ 2] = 1   ; // 0
   buff.fSegs[ 3] = 1   ; buff.fSegs[ 4] = 1   ; buff.fSegs[ 5] = 2   ; // 1
   buff.fSegs[ 6] = 1   ; buff.fSegs[ 7] = 2   ; buff.fSegs[ 8] = 3   ; // 2
   buff.fSegs[ 9] = 1   ; buff.fSegs[10] = 3   ; buff.fSegs[11] = 0   ; // 3
   buff.fSegs[12] = 1   ; buff.fSegs[13] = 4   ; buff.fSegs[14] = 5   ; // 4
   buff.fSegs[15] = 1   ; buff.fSegs[16] = 5   ; buff.fSegs[17] = 6   ; // 5
   buff.fSegs[18] = 1   ; buff.fSegs[19] = 6   ; buff.fSegs[20] = 7   ; // 6
   buff.fSegs[21] = 1   ; buff.fSegs[22] = 7   ; buff.fSegs[23] = 4   ; // 7
   buff.fSegs[24] = 1   ; buff.fSegs[25] = 0   ; buff.fSegs[26] = 4   ; // 8
   buff.fSegs[27] = 1   ; buff.fSegs[28] = 1   ; buff.fSegs[29] = 5   ; // 9
   buff.fSegs[30] = 1   ; buff.fSegs[31] = 2   ; buff.fSegs[32] = 6   ; // 10
   buff.fSegs[33] = 1   ; buff.fSegs[34] = 3   ; buff.fSegs[35] = 7   ; // 11

   buff.fPols[ 0] = 1   ; buff.fPols[ 1] = 4   ;  buff.fPols[ 2] = 0  ; // 0
   buff.fPols[ 3] = 9   ; buff.fPols[ 4] = 4   ;  buff.fPols[ 5] = 8  ;
   buff.fPols[ 6] = 1   ; buff.fPols[ 7] = 4   ;  buff.fPols[ 8] = 1  ; // 1
   buff.fPols[ 9] = 10  ; buff.fPols[10] = 5   ;  buff.fPols[11] = 9  ;
   buff.fPols[12] = 1   ; buff.fPols[13] = 4   ;  buff.fPols[14] = 2  ; // 2
   buff.fPols[15] = 11  ; buff.fPols[16] = 6   ;  buff.fPols[17] = 10 ;
   buff.fPols[18] = 1   ; buff.fPols[19] = 4   ;  buff.fPols[20] = 3  ; // 3
   buff.fPols[21] = 8   ; buff.fPols[22] = 7   ;  buff.fPols[23] = 11 ;
   buff.fPols[24] = 1   ; buff.fPols[25] = 4   ;  buff.fPols[26] = 0  ; // 4
   buff.fPols[27] = 3   ; buff.fPols[28] = 2   ;  buff.fPols[29] = 1  ;
   buff.fPols[30] = 1   ; buff.fPols[31] = 4   ;  buff.fPols[32] = 4  ; // 5
   buff.fPols[33] = 5   ; buff.fPols[34] = 6   ;  buff.fPols[35] = 7  ;

   buff.SetSectionsValid(TBuffer3D::kRawSizes | TBuffer3D::kRaw);
   return new TGLFaceSet(buff, 0);
}

TGLLogicalShape * CreateLogicalFace(Double_t width, Double_t depth)
{
   // Helper function to construct a TGLLogicalShape face (retangle)
   // based on supplied width/depth

   TBuffer3D buff(TBuffer3DTypes::kGeneric, 4, 3*4, 4, 3*4, 1, 6);

   buff.fPnts[ 0] = -width; buff.fPnts[ 1] = -depth; buff.fPnts[ 2] = 0.0; // 0
   buff.fPnts[ 3] =  width; buff.fPnts[ 4] = -depth; buff.fPnts[ 5] = 0.0; // 1
   buff.fPnts[ 6] =  width; buff.fPnts[ 7] =  depth; buff.fPnts[ 8] = 0.0; // 2
   buff.fPnts[ 9] = -width; buff.fPnts[10] =  depth; buff.fPnts[11] = 0.0; // 3

   buff.fSegs[ 0] = 1   ; buff.fSegs[ 1] = 0   ; buff.fSegs[ 2] = 1   ; // 0
   buff.fSegs[ 3] = 1   ; buff.fSegs[ 4] = 1   ; buff.fSegs[ 5] = 2   ; // 1
   buff.fSegs[ 6] = 1   ; buff.fSegs[ 7] = 2   ; buff.fSegs[ 8] = 3   ; // 2
   buff.fSegs[ 9] = 1   ; buff.fSegs[10] = 3   ; buff.fSegs[11] = 0   ; // 3

   buff.fPols[ 0] = 1   ; buff.fPols[ 1] = 4   ;  buff.fPols[ 2] = 0  ; // 0
   buff.fPols[ 3] = 1   ; buff.fPols[ 4] = 2   ;  buff.fPols[ 5] = 3  ;

   buff.SetSectionsValid(TBuffer3D::kRawSizes | TBuffer3D::kRaw);
   return new TGLFaceSet(buff, 0);
}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLClip                                                              //
//                                                                      //
// Abstract clipping shape - derives from TGLPhysicalShape              //
// Adds clip mode (inside/outside) and pure virtual method to           //
// approximate shape as set of planes. This plane set is used to perform//
// interactive clipping using OpenGL clip planes.                       //
//////////////////////////////////////////////////////////////////////////

ClassImp(TGLClip)

//______________________________________________________________________________
TGLClip::TGLClip(const TGLLogicalShape & logical, const TGLMatrix & transform, const float color[4]) :
   TGLPhysicalShape(0, logical, transform, kTRUE, color),
   fMode(kInside)
{
   // Construct a physical clipping object, taking the 'logical' shape, a
   // 'transform' matrix for placing this, and 'color'.
   //
   // Takes 'ownership' of logical and it will be destroyed when this is.

   // Take strong reference - destroy logical when ref released in
   // TGLPhysical::~TGLPhysical()
   logical.StrongRef(kTRUE);
}

//______________________________________________________________________________
TGLClip::~TGLClip()
{
   // Destroy clip object
}

//______________________________________________________________________________
void TGLClip::Draw(const TGLDrawFlags & flags) const
{
   // Draw out clipping object with blending and back + front filling.
   // Some clip objects are single face which we want to see both sides of.
   glDepthMask(GL_FALSE);
   glEnable(GL_BLEND);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   glDisable(GL_CULL_FACE);
   glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
   TGLPhysicalShape::Draw(flags);
   glPolygonMode(GL_FRONT, GL_FILL);
   glEnable(GL_CULL_FACE);
   glDisable(GL_BLEND);
   glDepthMask(GL_TRUE);
}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLClipPlane                                                         //
//                                                                      //
// Concrete clip plane object. This can be translated in all directions //
// rotated about the Y/Z local axes (the in-plane axes). It cannot be   //
// scaled.                                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TGLClipPlane)

const float TGLClipPlane::fgColor[4] = { 1.0, 0.6, 0.2, 0.5 };

//______________________________________________________________________________
TGLClipPlane::TGLClipPlane(const TGLVector3 & norm, const TGLVertex3 & center, Double_t extents) :
   TGLClip(*CreateLogicalFace(extents, extents), TGLMatrix(center), fgColor)
{
   // Construct a clip plane object, based on supplied 'plane', with initial manipulation
   // pivot at 'center', with drawn extents (in local x/y axes) of 'extents'
   //
   // Plane can have center pivot translated in all directions, and rotated round
   // center in X/Y axes , the in-plane axes. It cannot be scaled
   //
   // Note theorectically a plane is of course infinite - however we want to draw
   // the object in viewer - so we fake it with a single GL face (polygon) - extents
   // defines the width/depth of this - should be several times scene extents.
   //
   SetManip(EManip(kTranslateAll | kRotateX | kRotateY));

   TGLPlane plane(norm, center);
   Set(plane);
}

//______________________________________________________________________________
TGLClipPlane::~TGLClipPlane()
{
   // Destroy clip plane object
}

//______________________________________________________________________________
void TGLClipPlane::Set(const TGLPlane & plane)
{
   // Update clip plane object to follow passed 'plane' equation. Center pivot
   // is shifted to nearest point on new plane.
   TGLVertex3 oldCenter = BoundingBox().Center();
   TGLVertex3 newCenter = plane.NearestOn(oldCenter);
   SetTransform(TGLMatrix(newCenter, plane.Norm()));
}

//______________________________________________________________________________
void TGLClipPlane::PlaneSet(TGLPlaneSet_t & set) const
{
   // Return set of planes (actually a single) describing this clip plane

   // Get complete set from bounding box and discard all except first (top)
   BoundingBox().PlaneSet(set);
   set.resize(1);
}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLClipBox                                                           //
//                                                                      //
// Concrete clip box object. Can be translated, rotated and scaled in   //
// all (xyz) axes.                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TGLClipBox)

const float TGLClipBox::fgColor[4] = { 1.0, 0.6, 0.2, 0.3 };

//______________________________________________________________________________
TGLClipBox::TGLClipBox(const TGLVector3 & halfLengths, const TGLVertex3 & center) :
   TGLClip(*CreateLogicalBox(halfLengths), TGLMatrix(center), fgColor)
{
   // Construct an (initially) axis aligned clip pbox object, extents 'halfLengths',
   // centered on 'center' vertex.
   // Box can be translated, rotated and scaled in all (xyz) local axes.
}
//______________________________________________________________________________
TGLClipBox::~TGLClipBox()
{
   // Destroy clip box object
}

//______________________________________________________________________________
void TGLClipBox::PlaneSet(TGLPlaneSet_t & set) const
{
   // Return set of 6 planes describing faces of the box
   BoundingBox().PlaneSet(set);
}
