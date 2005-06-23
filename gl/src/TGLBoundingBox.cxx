// @(#)root/gl:$Name:  $:$Id: TGLBoundingBox.cxx,v 1.6 2005/06/21 16:54:17 brun Exp $
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
   SetEmpty();
}

//______________________________________________________________________________
TGLBoundingBox::TGLBoundingBox(const TGLVertex3 vertex[8])
{
   Set(vertex);
}

//______________________________________________________________________________
TGLBoundingBox::TGLBoundingBox(const Double_t vertex[8][3])
{
   Set(vertex);
}

//______________________________________________________________________________
TGLBoundingBox::TGLBoundingBox(const TGLVertex3 & lowVertex, const TGLVertex3 & highVertex)
{
   SetAligned(lowVertex, highVertex);
}

//______________________________________________________________________________
TGLBoundingBox::TGLBoundingBox(const TGLBoundingBox & other)
{
   Set(other);
}

//______________________________________________________________________________
TGLBoundingBox::~TGLBoundingBox()
{
}

//______________________________________________________________________________
void TGLBoundingBox::Set(const TGLVertex3 vertex[8])
{
   for (UInt_t v = 0; v < 8; v++) {
      fVertex[v] = vertex[v];
   }
   UpdateVolume();
}

//______________________________________________________________________________
void TGLBoundingBox::Set(const Double_t vertex[8][3])
{
   for (UInt_t v = 0; v < 8; v++) {
      for (UInt_t a = 0; a < 3; a++) {
         fVertex[v][a] = vertex[v][a];
      }
   }
   UpdateVolume();
}

//______________________________________________________________________________
void TGLBoundingBox::Set(const TGLBoundingBox & other)
{
   for (UInt_t v = 0; v < 8; v++) {
      fVertex[v].Set(other.fVertex[v]);
   }
   UpdateVolume();
}

//______________________________________________________________________________
void TGLBoundingBox::SetEmpty()
{
   for (UInt_t v = 0; v < 8; v++) {
      fVertex[v].Fill(0.0);
   }
   UpdateVolume();
}

//______________________________________________________________________________
void TGLBoundingBox::SetAligned(const TGLVertex3 & lowVertex, const TGLVertex3 & highVertex)
{
   // Set aligned box from two axis aligned low/high vertices
   //   7-------6
   //  /|      /|
   // 3-------2 |
   // | 4-----|-5
   // |/      |/
   // 0-------1

   TGLVector3 diff = highVertex - lowVertex;
   fVertex[0] = lowVertex;
   fVertex[1] = lowVertex; fVertex[1].X() += diff.X();
   fVertex[2] = lowVertex; fVertex[2].X() += diff.X(); fVertex[2].Y() += diff.Y();
   fVertex[3] = lowVertex; fVertex[3].Y() += diff.Y();
   fVertex[4] = highVertex; fVertex[4].X() -= diff.X(); fVertex[4].Y() -= diff.Y();
   fVertex[5] = highVertex; fVertex[5].Y() -= diff.Y();
   fVertex[6] = highVertex;
   fVertex[7] = highVertex; fVertex[7].X() -= diff.X();
   UpdateVolume();
}

//______________________________________________________________________________
void TGLBoundingBox::SetAligned(UInt_t nbPnts, const Double_t * pnts)
{
   // Set aligned box using a range of points
   if (nbPnts < 2 || !pnts) {
      assert(false);
      return;
   }

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
void TGLBoundingBox::Scale(Double_t scale)
{
   TGLVertex3 center = Center();
   TGLVector3 newVector;
   for (UInt_t v = 0; v < 8; v++) {
      newVector.Set(fVertex[v] - center);
      newVector *= scale;
      fVertex[v].Set(center + newVector);
   }
   UpdateVolume();
}

//______________________________________________________________________________
void TGLBoundingBox::Translate(const TGLVector3 & offset)
{
   for (UInt_t v = 0; v < 8; v++) {
      fVertex[v] = fVertex[v] + offset;
   }

   // No volume change
}

//______________________________________________________________________________
void TGLBoundingBox::Transform(const TGLMatrix & matrix)
{
   for (UInt_t v = 0; v < 8; v++) {
      matrix.TransformVertex(fVertex[v]);
   }

   // Could change volume
   UpdateVolume();
}

//______________________________________________________________________________
EOverlap TGLBoundingBox::Overlap(const TGLPlane & plane) const
{
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
Bool_t TGLBoundingBox::AlignedContains(const TGLBoundingBox & other) const
{
   // Test is passed box is contained by us. This will ONLY work if we are axis aligned

   //   7-------6
   //  /|      /|
   // 3-------2 |
   // | 4-----|-5
   // |/      |/
   // 0-------1
   // TODO: Rounding errors - need a better test anyway....
   // This can probably be found as part of Intersect() with three overlap
   // cases
   //assert(fVertex[2].Z() == fVertex[0].Z()); // Front
   //assert(fVertex[4].Z() == fVertex[6].Z()); // Back
   //assert(fVertex[5].Y() == fVertex[0].Y()); // Bottom
   //assert(fVertex[3].Y() == fVertex[6].Y()); // Top
   //assert(fVertex[7].X() == fVertex[0].X()); // Left
   //assert(fVertex[1].X() == fVertex[6].X()); // Right

   // Just test the all other's vertexes lie within our axis limits
   for (UInt_t v = 0; v < 8; v++) {
      for (UInt_t a = 0; a < 3; a++)
      {
         if ((other.fVertex[v][a] < fVertex[0][a]) ||
             (other.fVertex[v][a] > fVertex[6][a])) {
            return kFALSE;
         }
      }
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLBoundingBox::Intersect(const TGLBoundingBox & a, const TGLBoundingBox & b)
{
   // TODO: For some reason this intersection test gives incorrect result if first
   // BB is smaller than other - no idea why as should be symetric - need to investigate.
   //assert(Volume() > other.Volume());

   TGLVector3 aHL = a.Extents() / 2.0; // Half length extents
   TGLVector3 bHL = b.Extents() / 2.0; // Half length extents

   // Perform seperating axes search - test is greatly simplified
   // if we convert  into our local frame

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
       }
    }

    // Perform separating axis test for all 15 potential
    // axes. If no seperating axes found, the two boxes overlap.
    Double_t ra, rb, t;

    // A's 3 basis vectors
    for (i=0; i<3; i++) {
       ra = aHL[i];
       rb = bHL[0]*fabs(roaT[i][0]) + bHL[1]*fabs(roaT[i][1]) + bHL[2]*fabs(roaT[i][2]);
       t = fabs(aT[i]);
       if (t > ra + rb)
          return kFALSE;
    }

    // B's 3 basis vectors
    for (k=0; k<3; k++) {
       ra = aHL[0]*fabs(roaT[0][k]) + aHL[1]*fabs(roaT[1][k]) + aHL[2]*fabs(roaT[2][k]);
       rb = bHL[k];
       t = fabs(aT[0]*roaT[0][k] + aT[1]*roaT[1][k] + aT[2]*roaT[2][k]);
       if (t > ra + rb)
          return kFALSE;
    }

    // Now the 9 cross products

    // A0 x B0
    ra = aHL[1]*fabs(roaT[2][0]) + aHL[2]*fabs(roaT[1][0]);
    rb = bHL[1]*fabs(roaT[0][2]) + bHL[2]*fabs(roaT[0][1]);
    t = fabs(aT[2]*roaT[1][0] - aT[1]*roaT[2][0]);
    if (t > ra + rb)
       return kFALSE;

    // A0 x B1
    ra = aHL[1]*fabs(roaT[2][1]) + aHL[2]*fabs(roaT[1][1]);
    rb = bHL[0]*fabs(roaT[0][2]) + bHL[2]*fabs(roaT[0][0]);
    t = fabs(aT[2]*roaT[1][1] - aT[1]*roaT[2][1]);
    if (t > ra + rb)
       return kFALSE;

    // A0 x B2
    ra = aHL[1]*fabs(roaT[2][2]) + aHL[2]*fabs(roaT[1][2]);
    rb = bHL[0]*fabs(roaT[0][1]) + bHL[1]*fabs(roaT[0][0]);
    t = fabs(aT[2]*roaT[1][2] - aT[1]*roaT[2][2]);
    if (t > ra + rb)
       return kFALSE;

    // A1 x B0
    ra = aHL[0]*fabs(roaT[2][0]) + aHL[2]*fabs(roaT[0][0]);
    rb = bHL[1]*fabs(roaT[1][2]) + bHL[2]*fabs(roaT[1][1]);
    t = fabs(aT[0]*roaT[2][0] - aT[2]*roaT[0][0]);
    if (t > ra + rb)
       return kFALSE;

    // A1 x B1
    ra = aHL[0]*fabs(roaT[2][1]) + aHL[2]*fabs(roaT[0][1]);
    rb = bHL[0]*fabs(roaT[1][2]) + bHL[2]*fabs(roaT[1][0]);
    t = fabs(aT[0]*roaT[2][1] - aT[2]*roaT[0][1]);
    if (t > ra + rb)
       return kFALSE;

    // A1 x B2
    ra = aHL[0]*fabs(roaT[2][2]) + aHL[2]*fabs(roaT[0][2]);
    rb = bHL[0]*fabs(roaT[1][1]) + bHL[1]*fabs(roaT[1][0]);
    t = fabs(aT[0]*roaT[2][2] - aT[2]*roaT[0][2]);
    if (t > ra + rb)
       return kFALSE;

    // A2 x B0
    ra = aHL[0]*fabs(roaT[1][0]) + aHL[1]*fabs(roaT[0][0]);
    rb = bHL[1]*fabs(roaT[2][2]) + bHL[2]*fabs(roaT[2][1]);
    t = fabs(aT[1]*roaT[0][0] - aT[0]*roaT[1][0]);
    if (t > ra + rb)
       return kFALSE;

    // A2 x B1
    ra = aHL[0]*fabs(roaT[1][1]) + aHL[1]*fabs(roaT[0][1]);
    rb = bHL[0]*fabs(roaT[2][2]) + bHL[2]*fabs(roaT[2][0]);
    t = fabs(aT[1]*roaT[0][1] - aT[0]*roaT[1][1]);
    if (t > ra + rb)
       return kFALSE;

    // A2 x B2
    ra = aHL[0]*fabs(roaT[1][2]) + aHL[1]*fabs(roaT[0][2]);
    rb = bHL[0]*fabs(roaT[2][1]) + bHL[1]*fabs(roaT[2][0]);
    t = fabs(aT[1]*roaT[0][2] - aT[0]*roaT[1][2]);
    if (t > ra + rb)
       return kFALSE;

    // No separating axis - two boxes overlap
    return true;
}

//______________________________________________________________________________
void TGLBoundingBox::Draw() const
{
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
   for (UInt_t i = 0; i<8; i++) {
      std::cout << "[" << i << "] (" << fVertex[i].X() << "," << fVertex[i].Y() << "," << fVertex[i].Z() << ")" << std::endl;
   }
   std::cout << "Center ";
   Center().Dump();
   std::cout << " Volume " << Volume() << std::endl;
}

