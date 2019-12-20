// @(#)root/geom:$Id$// Author: Andrei Gheata   24/10/01

// Contains() and DistFromOutside/Out() implemented by Mihaela Gheata

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TGeoTessellated
\ingroup Geometry_classes

Tessellated solid class. It is composed by a set of planar faces having triangular or
quadrilateral shape. The class does not provide navigation functionality, it just wraps the data
for the composing faces.
*/

#include "Riostream.h"

#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TGeoVolume.h"
#include "TVirtualGeoPainter.h"
#include "TGeoTessellated.h"
#include "TVirtualPad.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"
#include "TMath.h"
#include "TRandom.h"

ClassImp(TGeoTessellated);


/*
bool TGeoFacet::Check()
{
   Int_t nvert = fNvert;
   for (Int_t i = 0; i < fNvert; ++i) {
      const TGeoVector3 vi(fVertices[(i + 1) % fNvert] - fVertices[i];
      if (vi.Mag2() < kTolerance) {
        nvert--;
      }
    }

    if (nvert < 3) {
      std::cout << "Tile degenerated: Length of sides of facet are too small." << std::endl;
      return false;
    }

    // Compute normal using non-zero segments

    bool degenerated = true;
    for (size_t i = 0; i < NVERT - 1; ++i) {
      Vector3D<T> e1 = fVertices[i + 1] - fVertices[i];
      if (e1.Mag2() < kTolerance) continue;
      for (size_t j = i + 1; j < NVERT; ++j) {
        Vector3D<T> e2 = fVertices[(j + 1) % NVERT] - fVertices[j];
        if (e2.Mag2() < kTolerance) continue;
        fNormal = e1.Cross(e2);
        // e1 and e2 may be colinear
        if (fNormal.Mag2() < kTolerance) continue;
        fNormal.Normalize();
        degenerated = false;
        break;
      }
      if (!degenerated) break;
    }

    if (degenerated) {
      std::cout << "Tile degenerated 2: Length of sides of facet are too small." << std::endl;
      return false;
    }

    // Compute side vectors
    for (size_t i = 0; i < NVERT; ++i) {
      Vector3D<T> e1 = fVertices[(i + 1) % NVERT] - fVertices[i];
      if (e1.Mag2() < kTolerance) continue;
      fSideVectors[i] = fNormal.Cross(e1).Normalized();
      fDistance       = -fNormal.Dot(fVertices[i]);
      for (size_t j = i + 1; j < i + NVERT; ++j) {
        Vector3D<T> e2 = fVertices[(j + 1) % NVERT] - fVertices[j % NVERT];
        if (e2.Mag2() < kTolerance)
          fSideVectors[j % NVERT] = fSideVectors[(j - 1) % NVERT];
        else
          fSideVectors[j % NVERT] = fNormal.Cross(e2).Normalized();
      }
      break;
    }

    // Compute surface area
    fSurfaceArea = 0.;
    for (size_t i = 1; i < NVERT - 1; ++i) {
      Vector3D<T> e1 = fVertices[i] - fVertices[0];
      Vector3D<T> e2 = fVertices[i + 1] - fVertices[0];
      fSurfaceArea += 0.5 * (e1.Cross(e2)).Mag();
    }
    assert(fSurfaceArea > kTolerance * kTolerance);

    // Center of the tile
    for (size_t i = 0; i < NVERT; ++i)
      fCenter += fVertices[i];
    fCenter /= NVERT;
    return true;
  }
*/
////////////////////////////////////////////////////////////////////////////////
/// Default constructor

TGeoTessellated::TGeoTessellated(const char *name, Int_t nfacets) : TGeoBBox(name, 0, 0, 0)
{
   fNfacets = nfacets;
   fFacets.reserve(nfacets);
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

TGeoTessellated::TGeoTessellated(const TGeoTessellated &other) : TGeoBBox(other)
{
   fNvert   = other.fNvert;
   fNfacets = other.fNfacets;
   fFacets  = other.fFacets;
}

////////////////////////////////////////////////////////////////////////////////
/// Assignment operator

TGeoTessellated &TGeoTessellated::operator=(const TGeoTessellated &other)
{
   if (&other != this) {
      TGeoBBox::operator=(other);
      fNvert   = other.fNvert;
      fNfacets = other.fNfacets;
      fFacets  = other.fFacets;
   }
   return *this;
}



////////////////////////////////////////////////////////////////////////////////
/// Adding a triangular facet from vertex positions in absolute coordinates

void TGeoTessellated::AddFacet(double x0, double y0, double z0,
                               double x1, double y1, double z1,
                               double x2, double y2, double z2)
{
   if (GetNfacets() == fNfacets) {
      Error("AddFacet", "Already defined %d facets, cannot add more", fNfacets);
      return;
   }
   fNvert += 3;
   fFacets.emplace_back(x0, y0, z0, x1, y1, z1, x2, y2, z2);
}

////////////////////////////////////////////////////////////////////////////////
/// Adding a quadrilateral facet from vertex positions in absolute coordinates

void TGeoTessellated::AddFacet(double x0, double y0, double z0,
                               double x1, double y1, double z1,
                               double x2, double y2, double z2,
                               double x3, double y3, double z3)
{
   if (GetNfacets() == fNfacets) {
      Error("AddFacet", "Already defined %d facets, cannot add more", fNfacets);
      return;
   }
   fNvert += 4;
   fFacets.emplace_back(x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3);
}

////////////////////////////////////////////////////////////////////////////////
/// Compute bounding box

void TGeoTessellated::ComputeBBox()
{
   const double kBig = TGeoShape::Big();
   double vmin[3] = { kBig, kBig, kBig };
   double vmax[3] = { -kBig, -kBig, -kBig };
   for (const auto &facet : fFacets) {
      for (int i = 0; i < facet.fNvert; ++i) {
         for (int j = 0; j < 3; ++j) {
            vmin[j] = TMath::Min(vmin[j], facet.fVertices[i].operator[](j));
            vmax[j] = TMath::Max(vmax[j], facet.fVertices[i].operator[](j));
         }
      }
   }
   fDX = 0.5 * (vmax[0] - vmin[0]);
   fDY = 0.5 * (vmax[1] - vmin[1]);
   fDZ = 0.5 * (vmax[2] - vmin[2]);
   for (int i = 0; i < 3; ++i)
      fOrigin[i] = 0.5 * (vmax[i] + vmin[i]);
}

////////////////////////////////////////////////////////////////////////////////
/// Returns numbers of vertices, segments and polygons composing the shape mesh.

void TGeoTessellated::GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const
{
   nvert = fNvert;
   nsegs = fNvert;
   npols = GetNfacets();
}

////////////////////////////////////////////////////////////////////////////////
/// Creates a TBuffer3D describing *this* shape.
/// Coordinates are in local reference frame.

TBuffer3D *TGeoTessellated::MakeBuffer3D() const
{
   TBuffer3D* buff = new TBuffer3D(TBuffer3DTypes::kGeneric, 8, 24, 12, 36, 6, 36);
   if (buff)
   {
      SetPoints(buff->fPnts);
      SetSegsAndPols(*buff);
   }

   return buff;
}

////////////////////////////////////////////////////////////////////////////////
/// Fills TBuffer3D structure for segments and polygons.

void TGeoTessellated::SetSegsAndPols(TBuffer3D &buff) const
{
   Int_t c = GetBasicColor();

   buff.fSegs[ 0] = c   ; buff.fSegs[ 1] = 0   ; buff.fSegs[ 2] = 1   ;
   buff.fSegs[ 3] = c+1 ; buff.fSegs[ 4] = 1   ; buff.fSegs[ 5] = 2   ;
   buff.fSegs[ 6] = c+1 ; buff.fSegs[ 7] = 2   ; buff.fSegs[ 8] = 3   ;
   buff.fSegs[ 9] = c   ; buff.fSegs[10] = 3   ; buff.fSegs[11] = 0   ;
   buff.fSegs[12] = c+2 ; buff.fSegs[13] = 4   ; buff.fSegs[14] = 5   ;
   buff.fSegs[15] = c+2 ; buff.fSegs[16] = 5   ; buff.fSegs[17] = 6   ;
   buff.fSegs[18] = c+3 ; buff.fSegs[19] = 6   ; buff.fSegs[20] = 7   ;
   buff.fSegs[21] = c+3 ; buff.fSegs[22] = 7   ; buff.fSegs[23] = 4   ;
   buff.fSegs[24] = c   ; buff.fSegs[25] = 0   ; buff.fSegs[26] = 4   ;
   buff.fSegs[27] = c+2 ; buff.fSegs[28] = 1   ; buff.fSegs[29] = 5   ;
   buff.fSegs[30] = c+1 ; buff.fSegs[31] = 2   ; buff.fSegs[32] = 6   ;
   buff.fSegs[33] = c+3 ; buff.fSegs[34] = 3   ; buff.fSegs[35] = 7   ;

   buff.fPols[ 0] = c   ; buff.fPols[ 1] = 4   ;  buff.fPols[ 2] = 0  ;
   buff.fPols[ 3] = 9   ; buff.fPols[ 4] = 4   ;  buff.fPols[ 5] = 8  ;
   buff.fPols[ 6] = c+1 ; buff.fPols[ 7] = 4   ;  buff.fPols[ 8] = 1  ;
   buff.fPols[ 9] = 10  ; buff.fPols[10] = 5   ;  buff.fPols[11] = 9  ;
   buff.fPols[12] = c   ; buff.fPols[13] = 4   ;  buff.fPols[14] = 2  ;
   buff.fPols[15] = 11  ; buff.fPols[16] = 6   ;  buff.fPols[17] = 10 ;
   buff.fPols[18] = c+1 ; buff.fPols[19] = 4   ;  buff.fPols[20] = 3  ;
   buff.fPols[21] = 8   ; buff.fPols[22] = 7   ;  buff.fPols[23] = 11 ;
   buff.fPols[24] = c+2 ; buff.fPols[25] = 4   ;  buff.fPols[26] = 0  ;
   buff.fPols[27] = 3   ; buff.fPols[28] = 2   ;  buff.fPols[29] = 1  ;
   buff.fPols[30] = c+3 ; buff.fPols[31] = 4   ;  buff.fPols[32] = 4  ;
   buff.fPols[33] = 5   ; buff.fPols[34] = 6   ;  buff.fPols[35] = 7  ;
}

////////////////////////////////////////////////////////////////////////////////
/// Fill tessellated points to an array.

void TGeoTessellated::SetPoints(Double_t *points) const
{
   if (!points) return;
   Double_t xmin,xmax,ymin,ymax,zmin,zmax;
   xmin = -fDX+fOrigin[0];
   xmax =  fDX+fOrigin[0];
   ymin = -fDY+fOrigin[1];
   ymax =  fDY+fOrigin[1];
   zmin = -fDZ+fOrigin[2];
   zmax =  fDZ+fOrigin[2];
   points[ 0] = xmin; points[ 1] = ymin; points[ 2] = zmin;
   points[ 3] = xmin; points[ 4] = ymax; points[ 5] = zmin;
   points[ 6] = xmax; points[ 7] = ymax; points[ 8] = zmin;
   points[ 9] = xmax; points[10] = ymin; points[11] = zmin;
   points[12] = xmin; points[13] = ymin; points[14] = zmax;
   points[15] = xmin; points[16] = ymax; points[17] = zmax;
   points[18] = xmax; points[19] = ymax; points[20] = zmax;
   points[21] = xmax; points[22] = ymin; points[23] = zmax;
}

////////////////////////////////////////////////////////////////////////////////
/// Fill tessellated points in float.

void TGeoTessellated::SetPoints(Float_t *points) const
{
   if (!points) return;
   Double_t xmin,xmax,ymin,ymax,zmin,zmax;
   xmin = -fDX+fOrigin[0];
   xmax =  fDX+fOrigin[0];
   ymin = -fDY+fOrigin[1];
   ymax =  fDY+fOrigin[1];
   zmin = -fDZ+fOrigin[2];
   zmax =  fDZ+fOrigin[2];
   points[ 0] = xmin; points[ 1] = ymin; points[ 2] = zmin;
   points[ 3] = xmin; points[ 4] = ymax; points[ 5] = zmin;
   points[ 6] = xmax; points[ 7] = ymax; points[ 8] = zmin;
   points[ 9] = xmax; points[10] = ymin; points[11] = zmin;
   points[12] = xmin; points[13] = ymin; points[14] = zmax;
   points[15] = xmin; points[16] = ymax; points[17] = zmax;
   points[18] = xmax; points[19] = ymax; points[20] = zmax;
   points[21] = xmax; points[22] = ymin; points[23] = zmax;
}

////////////////////////////////////////////////////////////////////////////////
/// Fills a static 3D buffer and returns a reference.

const TBuffer3D & TGeoTessellated::GetBuffer3D(Int_t reqSections, Bool_t localFrame) const
{
   static TBuffer3D buffer(TBuffer3DTypes::kGeneric);

   FillBuffer3D(buffer, reqSections, localFrame);

   // TODO: A box itself has has nothing more as already described
   // by bounding box. How will viewer interpret?
   if (reqSections & TBuffer3D::kRawSizes) {
      if (buffer.SetRawSizes(8, 3*8, 12, 3*12, 6, 6*6)) {
         buffer.SetSectionsValid(TBuffer3D::kRawSizes);
      }
   }
   if ((reqSections & TBuffer3D::kRaw) && buffer.SectionsValid(TBuffer3D::kRawSizes)) {
      SetPoints(buffer.fPnts);
      if (!buffer.fLocalFrame) {
         TransformPoints(buffer.fPnts, buffer.NbPnts());
      }

      SetSegsAndPols(buffer);
      buffer.SetSectionsValid(TBuffer3D::kRaw);
   }

   return buffer;
}

////////////////////////////////////////////////////////////////////////////////
/// Fills the supplied buffer, with sections in desired frame
/// See TBuffer3D.h for explanation of sections, frame etc.

void TGeoTessellated::FillBuffer3D(TBuffer3D & buffer, Int_t reqSections, Bool_t localFrame) const
{
   TGeoShape::FillBuffer3D(buffer, reqSections, localFrame);

   if (reqSections & TBuffer3D::kBoundingBox) {
      Double_t halfLengths[3] = { fDX, fDY, fDZ };
      buffer.SetAABoundingBox(fOrigin, halfLengths);

      if (!buffer.fLocalFrame) {
         TransformPoints(buffer.fBBVertex[0], 8);
      }
      buffer.SetSectionsValid(TBuffer3D::kBoundingBox);
   }
}
