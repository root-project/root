// @(#)root/gl:$Name:  $:$Id: TGLBoundingBox.cxx,v 1.9 2005/07/08 15:39:29 brun Exp $
// Author:  Richard Maunder  25/05/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// TODO: Function descriptions
// TODO: Class def - same as header

#include "TGLBoundingBox.h"
#include "TGLIncludes.h"
#include "Riostream.h"

ClassImp(TGLBoundingBox)

//______________________________________________________________________________
TGLBoundingBox::TGLBoundingBox()
{
   // Construct an empty bounding box
   SetEmpty();
}

//______________________________________________________________________________
TGLBoundingBox::TGLBoundingBox(const TGLVertex3 vertex[8])
{
   // Construct a bounding box from provided 8 vertices
   Set(vertex);
}

//______________________________________________________________________________
TGLBoundingBox::TGLBoundingBox(const Double_t vertex[8][3])
{
   // Construct a bounding box from provided 8 vertices
   Set(vertex);
}

//______________________________________________________________________________
TGLBoundingBox::TGLBoundingBox(const TGLVertex3 & lowVertex, const TGLVertex3 & highVertex)
{
   // Construct an global axis ALIGNED bounding box from provided low/high vertex pair
   SetAligned(lowVertex, highVertex);
}

//______________________________________________________________________________
TGLBoundingBox::TGLBoundingBox(const TGLBoundingBox & other)
{
  // Construct a bounding box as copy of existing one
  Set(other);
}

//______________________________________________________________________________
TGLBoundingBox::~TGLBoundingBox()
{
  // Destroy bounding box
}

//______________________________________________________________________________
void TGLBoundingBox::UpdateCache()
{
   // Update the internal cached volume and axes vectors

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

   // Do axes first so Extents() is correct
   fAxes[0].Set(fVertex[1] - fVertex[0]);
   fAxes[1].Set(fVertex[3] - fVertex[0]);
   fAxes[2].Set(fVertex[4] - fVertex[0]);

   // Sometimes have zero volume BB due to single zero magnitude 
   // axis record and try to fix below
   Bool_t fixZeroMagAxis = kFALSE;
   Int_t zeroMagAxisInd = -1; 
   for (UInt_t i = 0; i<3; i++) {
      fAxesNorm[i] = fAxes[i];
      Double_t mag = fAxesNorm[i].Mag();
      if (mag > 0.0) {
         fAxesNorm[i] /= mag;
      } else {
         if (!fixZeroMagAxis && zeroMagAxisInd == -1) {
            zeroMagAxisInd = i;
            fixZeroMagAxis = kTRUE;
         } else if (fixZeroMagAxis) {
            fixZeroMagAxis = kFALSE;
         }
      }
   }

   // Try to cope with a zero volume bounding box where one 
   // axis is zero by using cross product of other two
   if (fixZeroMagAxis) {
      fAxesNorm[zeroMagAxisInd] = Cross(fAxesNorm[(zeroMagAxisInd+1)%3], 
                                        fAxesNorm[(zeroMagAxisInd+2)%3]);
   }

   TGLVector3 extents = Extents();
   fVolume = fabs(extents.X() * extents.Y() * extents.Z());
}

//______________________________________________________________________________
void TGLBoundingBox::Set(const TGLVertex3 vertex[8])
{
  // Set a bounding box from provided 8 verticies
   for (UInt_t v = 0; v < 8; v++) {
      fVertex[v] = vertex[v];
   }
   // Could change cached volume/axes
   UpdateCache();
}

//______________________________________________________________________________
void TGLBoundingBox::Set(const Double_t vertex[8][3])
{
  // Set a bounding box from provided 8 verticies
   for (UInt_t v = 0; v < 8; v++) {
      for (UInt_t a = 0; a < 3; a++) {
         fVertex[v][a] = vertex[v][a];
      }
   }
   // Could change cached volume/axes
   UpdateCache();
}

//______________________________________________________________________________
void TGLBoundingBox::Set(const TGLBoundingBox & other)
{
  // Set a bounding box from verticies of other
   for (UInt_t v = 0; v < 8; v++) {
      fVertex[v].Set(other.fVertex[v]);
   }
   // Could change cached volume/axes
   UpdateCache();
}

//______________________________________________________________________________
void TGLBoundingBox::SetEmpty()
{
  // Set bounding box empty - all vertexes at (0,0,0)
   for (UInt_t v = 0; v < 8; v++) {
      fVertex[v].Fill(0.0);
   }
   // Could change cached volume/axes
   UpdateCache();
}

//______________________________________________________________________________
void TGLBoundingBox::SetAligned(const TGLVertex3 & lowVertex, const TGLVertex3 & highVertex)
{
   // Set ALIGNED box from two low/high vertices. Box axes are aligned with 
   // global frame axes that verticies are specified in.

   // lowVertex = vertex[0]
   // highVertex = vertex[6]
   //
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

   TGLVector3 diff = highVertex - lowVertex;
   fVertex[0] = lowVertex;
   fVertex[1] = lowVertex; fVertex[1].X() += diff.X();
   fVertex[2] = lowVertex; fVertex[2].X() += diff.X(); fVertex[2].Y() += diff.Y();
   fVertex[3] = lowVertex; fVertex[3].Y() += diff.Y();
   fVertex[4] = highVertex; fVertex[4].X() -= diff.X(); fVertex[4].Y() -= diff.Y();
   fVertex[5] = highVertex; fVertex[5].Y() -= diff.Y();
   fVertex[6] = highVertex;
   fVertex[7] = highVertex; fVertex[7].X() -= diff.X();
   // Could change cached volume/axes
   UpdateCache();
}

//______________________________________________________________________________
void TGLBoundingBox::SetAligned(UInt_t nbPnts, const Double_t * pnts)
{
   // Set ALIGNED box from one or more points. Box axes are aligned with 
   // global frame axes that points are specified in.
   if (nbPnts < 1 || !pnts) {
      assert(false);
      return;
   }

   // Single point gives a zero volume BB
   TGLVertex3 low(pnts[0], pnts[1], pnts[2]);
   TGLVertex3 high(pnts[0], pnts[1], pnts[2]);

   for (UInt_t p = 1; p < nbPnts; p++) {
      for (UInt_t i = 0; i < 3; i++) {
         if (pnts[3*p + i] < low[i]) {
            low[i] = pnts[3*p + i] ;
         }
         if (pnts[3*p + i] > high[i]) {
            high[i] = pnts[3*p + i] ;
         }
      }
   }

   SetAligned(low, high);
}

//______________________________________________________________________________
void TGLBoundingBox::Scale(Double_t factor)
{
   // Isotropically scale bounding box along it's LOCAL axes, preserving center
   Scale(factor, factor, factor);
   // Could change cached volume/axes
   UpdateCache();
}

//______________________________________________________________________________
void TGLBoundingBox::Scale(Double_t xFactor, Double_t yFactor, Double_t zFactor)
{
   // Asymetrically scale box along it's LOCAL x,y,z axes, preserving center

   // Get x,y,z edges (non-normalised axis) and scale
   // them by factors
   const TGLVector3 xOffset = Axis(0, kFALSE)*(xFactor - 1.0) / 2.0;
   const TGLVector3 yOffset = Axis(1, kFALSE)*(yFactor - 1.0) / 2.0;
   const TGLVector3 zOffset = Axis(2, kFALSE)*(zFactor - 1.0) / 2.0;

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
   fVertex[0] += -xOffset - yOffset - zOffset; 
   fVertex[1] +=  xOffset - yOffset - zOffset; 
   fVertex[2] +=  xOffset + yOffset - zOffset; 
   fVertex[3] += -xOffset + yOffset - zOffset; 

   fVertex[4] += -xOffset - yOffset + zOffset; 
   fVertex[5] +=  xOffset - yOffset + zOffset; 
   fVertex[6] +=  xOffset + yOffset + zOffset; 
   fVertex[7] += -xOffset + yOffset + zOffset; 

   // Could change cached volume/axes
   UpdateCache();
}

//______________________________________________________________________________
void TGLBoundingBox::Translate(const TGLVector3 & offset)
{
   // Translate all vertexes by offset
   for (UInt_t v = 0; v < 8; v++) {
      fVertex[v] = fVertex[v] + offset;
   }

   // No cache change - volume and axes vectors remain same
}

//______________________________________________________________________________
void TGLBoundingBox::Transform(const TGLMatrix & matrix)
{
   // Transform all vertexes with matrix
   for (UInt_t v = 0; v < 8; v++) {
      matrix.TransformVertex(fVertex[v]);
   }

   // Could change cached volume/axes
   UpdateCache();
}

//______________________________________________________________________________
EOverlap TGLBoundingBox::Overlap(const TGLPlane & plane) const
{
   // Find overlap (Inside, Outside, Partial) of plane c.f. bounding box

   // TODO: Cheaper sphere test
   // Test all 8 box vertices against plane
   Int_t VerticesInsidePlane = 8;
   for (UInt_t v = 0; v < 8; v++) {
      if (plane.DistanceTo(fVertex[v]) < 0.0) {
         VerticesInsidePlane--;
      }
   }

   if ( VerticesInsidePlane == 0 ) {
      return kOutside;
   } else if ( VerticesInsidePlane == 8 ) {
      return kInside;
   } else {
      return kPartial;
   }
}

//______________________________________________________________________________
EOverlap TGLBoundingBox::Overlap(const TGLBoundingBox & other) const
{
   // Find overlap (Inside, Outside, Partial) of other bounding box c.f. us

   // Simplify code with refs
   const TGLBoundingBox & a = *this;
   const TGLBoundingBox & b = other;

   TGLVector3 aHL = a.Extents() / 2.0; // Half length extents
   TGLVector3 bHL = b.Extents() / 2.0; // Half length extents

   // Perform separating axes search - test is greatly simplified
   // if we convert into our local frame

   // Find translation in parent frame
   TGLVector3 parentT = b.Center() - a.Center();

   // Find translation in A's frame
   TGLVector3 aT(Dot(parentT, a.Axis(0)), Dot(parentT, a.Axis(1)), Dot(parentT, a.Axis(2)));

   // Find B's basis with respect to A's local frame
   // Get rotation matrix
   Double_t   roaT[3][3];
   UInt_t     i, k;
   for (i=0 ; i<3 ; i++) {
      for (k=0; k<3; k++) {
         roaT[i][k] = Dot(a.Axis(i), b.Axis(k));
         // Force very small components to zero to avoid rounding errors
         if (fabs(roaT[i][k]) < 1e-14) {
            roaT[i][k] = 0.0;
         }
      }
      // Normalise columns to avoid rounding errors
      Double_t norm = sqrt(roaT[i][0]*roaT[i][0] + roaT[i][1]*roaT[i][1] + roaT[i][2]*roaT[i][2]);
      roaT[i][0] /= norm; roaT[i][1] /= norm; roaT[i][2] /= norm; 
   }

   // Perform separating axis test for all 15 potential
   // axes. If no separating axes found, the two boxes overlap.
   Double_t ra, rb, t;

   // A's 3 basis vectors
   for (i=0; i<3; i++) {
      ra = aHL[i];
      rb = bHL[0]*fabs(roaT[i][0]) + bHL[1]*fabs(roaT[i][1]) + bHL[2]*fabs(roaT[i][2]);
      t = fabs(aT[i]);
      if (t > ra + rb)
         return kOutside;
      else if (ra < t + rb)
         return kPartial;
   }

   // B's 3 basis vectors
   for (k=0; k<3; k++) {
      ra = aHL[0]*fabs(roaT[0][k]) + aHL[1]*fabs(roaT[1][k]) + aHL[2]*fabs(roaT[2][k]);
      rb = bHL[k];
      t = fabs(aT[0]*roaT[0][k] + aT[1]*roaT[1][k] + aT[2]*roaT[2][k]);
      if (t > ra + rb)
         return kOutside;
      else if (ra < t + rb)
         return kPartial;
   }

   // Now the 9 cross products

   // A0 x B0
   ra = aHL[1]*fabs(roaT[2][0]) + aHL[2]*fabs(roaT[1][0]);
   rb = bHL[1]*fabs(roaT[0][2]) + bHL[2]*fabs(roaT[0][1]);
   t = fabs(aT[2]*roaT[1][0] - aT[1]*roaT[2][0]);
   if (t > ra + rb)
      return kOutside;
   else if (ra < t + rb)
      return kPartial;

   // A0 x B1
   ra = aHL[1]*fabs(roaT[2][1]) + aHL[2]*fabs(roaT[1][1]);
   rb = bHL[0]*fabs(roaT[0][2]) + bHL[2]*fabs(roaT[0][0]);
   t = fabs(aT[2]*roaT[1][1] - aT[1]*roaT[2][1]);
   if (t > ra + rb)
      return kOutside;
   else if (ra < t + rb)
      return kPartial;

   // A0 x B2
   ra = aHL[1]*fabs(roaT[2][2]) + aHL[2]*fabs(roaT[1][2]);
   rb = bHL[0]*fabs(roaT[0][1]) + bHL[1]*fabs(roaT[0][0]);
   t = fabs(aT[2]*roaT[1][2] - aT[1]*roaT[2][2]);
   if (t > ra + rb)
      return kOutside;
   else if (ra < t + rb)
      return kPartial;

   // A1 x B0
   ra = aHL[0]*fabs(roaT[2][0]) + aHL[2]*fabs(roaT[0][0]);
   rb = bHL[1]*fabs(roaT[1][2]) + bHL[2]*fabs(roaT[1][1]);
   t = fabs(aT[0]*roaT[2][0] - aT[2]*roaT[0][0]);
   if (t > ra + rb)
      return kOutside;
   else if (ra < t + rb)
      return kPartial;

   // A1 x B1
   ra = aHL[0]*fabs(roaT[2][1]) + aHL[2]*fabs(roaT[0][1]);
   rb = bHL[0]*fabs(roaT[1][2]) + bHL[2]*fabs(roaT[1][0]);
   t = fabs(aT[0]*roaT[2][1] - aT[2]*roaT[0][1]);
   if (t > ra + rb)
      return kOutside;
   else if (ra < t + rb)
      return kPartial;

   // A1 x B2
   ra = aHL[0]*fabs(roaT[2][2]) + aHL[2]*fabs(roaT[0][2]);
   rb = bHL[0]*fabs(roaT[1][1]) + bHL[1]*fabs(roaT[1][0]);
   t = fabs(aT[0]*roaT[2][2] - aT[2]*roaT[0][2]);
   if (t > ra + rb)
      return kOutside;
   else if (ra < t + rb)
      return kPartial;

   // A2 x B0
   ra = aHL[0]*fabs(roaT[1][0]) + aHL[1]*fabs(roaT[0][0]);
   rb = bHL[1]*fabs(roaT[2][2]) + bHL[2]*fabs(roaT[2][1]);
   t = fabs(aT[1]*roaT[0][0] - aT[0]*roaT[1][0]);
   if (t > ra + rb)
      return kOutside;
   else if (ra < t + rb)
      return kPartial;

   // A2 x B1
   ra = aHL[0]*fabs(roaT[1][1]) + aHL[1]*fabs(roaT[0][1]);
   rb = bHL[0]*fabs(roaT[2][2]) + bHL[2]*fabs(roaT[2][0]);
   t = fabs(aT[1]*roaT[0][1] - aT[0]*roaT[1][1]);
   if (t > ra + rb)
      return kOutside;
   else if (ra < t + rb)
      return kPartial;

   // A2 x B2
   ra = aHL[0]*fabs(roaT[1][2]) + aHL[1]*fabs(roaT[0][2]);
   rb = bHL[0]*fabs(roaT[2][1]) + bHL[1]*fabs(roaT[2][0]);
   t = fabs(aT[1]*roaT[0][2] - aT[0]*roaT[1][2]);
   if (t > ra + rb)
      return kOutside;
   else if (ra < t + rb)
      return kPartial;

   // No separating axis - b is inside a
   return kInside;
}

//______________________________________________________________________________
void TGLBoundingBox::Draw() const
{
   // Draw the bounding box out using current GL color
   glBegin(GL_LINE_LOOP);
   glVertex3dv(fVertex[0].CArr());
   glVertex3dv(fVertex[1].CArr());
   glVertex3dv(fVertex[2].CArr());
   glVertex3dv(fVertex[3].CArr());
   glVertex3dv(fVertex[7].CArr());
   glVertex3dv(fVertex[6].CArr());
   glVertex3dv(fVertex[5].CArr());
   glVertex3dv(fVertex[4].CArr());
   glEnd();
   glBegin(GL_LINES);
   glVertex3dv(fVertex[1].CArr());
   glVertex3dv(fVertex[5].CArr());
   glVertex3dv(fVertex[2].CArr());
   glVertex3dv(fVertex[6].CArr());
   glVertex3dv(fVertex[0].CArr());
   glVertex3dv(fVertex[3].CArr());
   glVertex3dv(fVertex[4].CArr());
   glVertex3dv(fVertex[7].CArr());
   glEnd();
}

//______________________________________________________________________________
Double_t TGLBoundingBox::Min(UInt_t index) const
{
   // Find minimum vertex value for axis of index X(0), Y(1), Z(2)
   Double_t min = fVertex[0][index];
   for (UInt_t v = 1; v < 8; v++) {
      if (fVertex[v][index] < min) {
         min = fVertex[v][index];
      }
   }
   return min;
}

//______________________________________________________________________________
Double_t TGLBoundingBox::Max(UInt_t index) const
{
   // Find maximum vertex value for axis of index X(0), Y(1), Z(2)
   Double_t max = fVertex[0][index];
   for (UInt_t v = 1; v < 8; v++) {
      if (fVertex[v][index] > max) {
         max = fVertex[v][index];
      }
   }
   return max;
}

//______________________________________________________________________________
void TGLBoundingBox::Dump() const
{
   // Output to std::cout the vertexes, center and volume of box
   for (UInt_t i = 0; i<8; i++) {
      std::cout << "[" << i << "] (" << fVertex[i].X() << "," << fVertex[i].Y() << "," << fVertex[i].Z() << ")" << std::endl;
   }
   std::cout << "Center ";
   Center().Dump();
   std::cout << " Volume " << Volume() << std::endl;
}

