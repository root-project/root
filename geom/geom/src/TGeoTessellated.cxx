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

ClassImp(TGeoTessellated)

std::ostream &operator<<(std::ostream &os, TGeoFacet const &facet)
{
   os << "{";
   for (Int_t i = 0; i < facet.GetNvert(); ++i) {
      os << facet.GetVertex(i);
      if (i != facet.GetNvert()-1) os << ", ";
   }
   os << "}";
   return os;
}

TGeoFacet::TGeoFacet(const TGeoFacet &other) : fVertices(other.fVertices), fNvert(other.fNvert), fShared(other.fShared)
{
   memcpy(fIvert, other.fIvert, 4 * sizeof(Int_t));
   if (!fShared)
      fVertices = new VertexVec_t(*other.fVertices);
}

const TGeoFacet &TGeoFacet::operator = (const TGeoFacet &other)
{
   if (&other != this) {
      fVertices = other.fVertices;
      fNvert    = other.fNvert;
      fShared   = other.fShared;
      if (!fShared)
         fVertices = new VertexVec_t(*other.fVertices);
   }
   return *this;
}

TGeoVector3 TGeoFacet::ComputeNormal(bool &degenerated) const
{
   // Compute normal using non-zero segments
   constexpr Double_t kTolerance = 1.e-10;
   degenerated = true;
   TGeoVector3 normal;
   for (Int_t i = 0; i < fNvert - 1; ++i) {
      TGeoVector3 e1 = GetVertex(i + 1) - GetVertex(i);
      if (e1.Mag2() < kTolerance) continue;
      for (Int_t j = i + 1; j < fNvert; ++j) {
         TGeoVector3 e2 = GetVertex((j + 1) % fNvert) - GetVertex(j);
         if (e2.Mag2() < kTolerance) continue;
         normal = TGeoVector3::Cross(e1, e2);
         // e1 and e2 may be colinear
         if (normal.Mag2() < kTolerance) continue;
         normal.Normalize();
         degenerated = false;
         break;
      }
      if (!degenerated) break;
   }
   return normal;
}

bool TGeoFacet::Check() const
{
   constexpr Double_t kTolerance = 1.e-10;
   Int_t nvert = fNvert;
   for (Int_t i = 0; i < fNvert; ++i) {
      const TGeoVector3 vi(GetVertex((i + 1) % fNvert) - GetVertex(i));
      if (vi.Mag2() < kTolerance) {
        nvert--;
      }
   }

   if (nvert < 3) {
      std::cout << "Tile degenerated: Length of sides of facet are too small." << std::endl;
      return false;
   }

   bool degenerated = true;
   ComputeNormal(degenerated);
   if (degenerated) {
      std::cout << "Facet: " << *this << " is degenerated\n";
      return false;
   }

   // Compute surface area
   Double_t surfaceArea = 0.;
   for (Int_t i = 1; i < fNvert - 1; ++i) {
      TGeoVector3 e1 = GetVertex(i) - GetVertex(0);
      TGeoVector3 e2 = GetVertex(i + 1) - GetVertex(0);
      surfaceArea += 0.5 * TGeoVector3::Cross(e1, e2).Mag();
   }
   if (surfaceArea < kTolerance) {
      std::cout << "Facet: " << *this << " has zero surface area\n";
      return kFALSE;
   }

   // Center of the tile
   /*
   TGeoVector3 center;
   for (Int_t i = 0; i < fNvert; ++i)
      center += GetVertex(i);
   center /= fNvert;
   */
   return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Check if a connected neighbour facet has compatible normal

bool TGeoFacet::CheckNeighbour(const TGeoFacet $other) const
{
/*
   // Find a connecting segment
   Int_t line1[2], line2[2];
   Int_t npoints = 0;
   for (Int_t i = 0; i < fNvert; ++i) {
      auto ivert = GetVertexIndex(i);
      // Skip if next vertex is the same (degenerated)
      if (ivert == GetVertexIndex((i+1) % fNvert) continue;
      // Check if the other facet has the same vertex
      for (Int_t j = 0; j < other.GetNvert(); ++j) {
         if (ivert == other.GetVertexIndex(j)) {
            line1[npoints] = i;
            line2[npoints] = j;
            if (++npoints == 2) {

            }
         }

*/
   return true;
}


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
   fNvert    = other.fNvert;
   fNfacets  = other.fNfacets;
   fVertices = other.fVertices;
   fFacets   = other.fFacets;
}

////////////////////////////////////////////////////////////////////////////////
/// Assignment operator

TGeoTessellated &TGeoTessellated::operator=(const TGeoTessellated &other)
{
   if (&other != this) {
      TGeoBBox::operator=(other);
      fNvert    = other.fNvert;
      fNfacets  = other.fNfacets;
      fVertices = other.fVertices;
      fFacets   = other.fFacets;
   }
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Function to be called after reading tessellated volumes from the geometry file

void TGeoTessellated::AfterStreamer()
{
   // The pointer to the array of vertices is not streamed so update it to facets
   for (auto facet : fFacets)
      facet.SetVertices(&fVertices);
}

////////////////////////////////////////////////////////////////////////////////
/// Adding a triangular facet from vertex positions in absolute coordinates

void TGeoTessellated::AddFacet(const TGeoVector3 &pt0, const TGeoVector3 &pt1, const TGeoVector3 &pt2)
{
   if (GetNfacets() == fNfacets) {
      Error("AddFacet", "Already defined %d facets, cannot add more", fNfacets);
      return;
   }
   fNvert += 3;
   fNseg  += 3;
   fFacets.emplace_back(pt0, pt1, pt2);
   if (GetNfacets() == fNfacets)
      Close();
}

////////////////////////////////////////////////////////////////////////////////
/// Adding a quadrilateral facet from vertex positions in absolute coordinates

void TGeoTessellated::AddFacet(const TGeoVector3 &pt0, const TGeoVector3 &pt1, const TGeoVector3 &pt2, const TGeoVector3 &pt3)
{
   if (GetNfacets() == fNfacets) {
      Error("AddFacet", "Already defined %d facets, cannot add more", fNfacets);
      return;
   }
   fNvert += 4;
   fNseg  += 4;
   fFacets.emplace_back(pt0, pt1, pt2, pt3);
   if (GetNfacets() == fNfacets)
      Close();
}

////////////////////////////////////////////////////////////////////////////////
/// Close the shape: calculate bounding box and compact vertices

void TGeoTessellated::Close()
{
   // Compact the array of vertices
   constexpr double tolerance = 1.e-10;
   constexpr size_t ngrid = 10;
   ComputeBBox();
   double minExtent[3];
   minExtent[0] = fOrigin[0] - fDX - tolerance;
   minExtent[1] = fOrigin[1] - fDY - tolerance;
   minExtent[2] = fOrigin[2] - fDZ - tolerance;

   double invExtent[3];
   invExtent[0] = 0.5 / (fDX + tolerance);
   invExtent[1] = 0.5 / (fDY + tolerance);
   invExtent[2] = 0.5 / (fDZ + tolerance);

   auto AddVertex = [this](const Vertex_t &vertex)
   {
      // Check if vertex exists
      Int_t ivert = 0;
      for (const auto &current_vert : fVertices) {
         if (current_vert == vertex) return ivert;
         ivert++;
      }
      // Vertex new, just add it
      fVertices.push_back(vertex);
      return ivert;
   };

   auto GetHashIndex = [&, this](const Vertex_t &vertex)
   {
      // Get the hash index for a vertex in a 10x10x10 grid in the bounding box
      Int_t index = 0;
      for (int i = 0; i < 3; ++i) {
         Int_t ind = ngrid * (vertex[i] - minExtent[i]) * invExtent[i]; // between [0, ngrid-1]
         assert(ind < (Int_t)ngrid);
         for (int j = i + 1; j < 3; ++j) ind *= ngrid;
         index += ind;
      }
      return index;
   };

   // In case the number of vertices is small, just compare with all others
   Int_t ind[4];
   if (fNvert < 1000) {
      for (auto &facet : fFacets) {
         ind[3] = -1; // not used for triangular facets
         for (int i = 0; i < facet.GetNvert(); ++i) {
            // Check if vertex exists already
            ind[i] = AddVertex(facet.GetVertex(i));
         }
         facet.SetVertices(&fVertices, ind[0], ind[1], ind[2], ind[3]);
      }
   } else {
      // Use hash index for each vertex
      using CellVec_t = std::vector<int>;
      std::array<CellVec_t, ngrid * ngrid * ngrid> grid;
      for (auto &facet : fFacets) {
         ind[3] = -1; // not used for triangular facets
         for (int i = 0; i < facet.GetNvert(); ++i) {
            // Check if vertex exists already
            const Vertex_t &vertex = facet.GetVertex(i);
            Int_t hashind = GetHashIndex(vertex);
            bool isAdded = false;
            for (auto ivert : grid[hashind]) {
               if (vertex == fVertices[ivert]) {
                  ind[i] = ivert;
                  isAdded = true;
                  break;
               }
            }
            if (!isAdded) {
              fVertices.push_back(vertex);
              ind[i] = fVertices.size() - 1;
              grid[hashind].push_back(ind[i]);
            }
         }
         facet.SetVertices(&fVertices, ind[0], ind[1], ind[2], ind[3]);
      }
   }
   fNvert = fVertices.size();
   // Check facets
   for (auto &facet : fFacets) {
      facet.Check();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Compute bounding box

void TGeoTessellated::ComputeBBox()
{
   const double kBig = TGeoShape::Big();
   if (fVertices.size()) return;
   double vmin[3] = { kBig, kBig, kBig };
   double vmax[3] = { -kBig, -kBig, -kBig };
   for (const auto &facet : fFacets) {
      for (int i = 0; i < facet.GetNvert(); ++i) {
         for (int j = 0; j < 3; ++j) {
            vmin[j] = TMath::Min(vmin[j], facet.GetVertex(i).operator[](j));
            vmax[j] = TMath::Max(vmax[j], facet.GetVertex(i).operator[](j));
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
   nsegs = fNseg;
   npols = GetNfacets();
}

////////////////////////////////////////////////////////////////////////////////
/// Creates a TBuffer3D describing *this* shape.
/// Coordinates are in local reference frame.

TBuffer3D *TGeoTessellated::MakeBuffer3D() const
{
   const Int_t nvert = fNvert;
   const Int_t nsegs = fNseg;
   const Int_t npols = GetNfacets();
   TBuffer3D* buff = new TBuffer3D(TBuffer3DTypes::kGeneric, nvert, 3*nvert, nsegs, 3*nsegs, npols, 6*npols);
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
   const Int_t c = GetBasicColor();
   Int_t *segs = buff.fSegs;
   Int_t *pols = buff.fPols;

   Int_t indseg = 0; // segment internal data index
   Int_t indpol = 0; // polygon internal data index
   Int_t sind   = 0; // segment index
   for (const auto &facet : fFacets) {
      auto nvert = facet.GetNvert();
      pols[indpol++] = c;
      pols[indpol++] = nvert;
      for (auto j = 0; j < nvert; ++j) {
         Int_t k = (j + 1) % nvert;
         // segment made by next consecutive points
         segs[indseg++] = c;
         segs[indseg++] = facet.GetVertexIndex(j);
         segs[indseg++] = facet.GetVertexIndex(k);
         // add segment to current polygon and increment segment index
         pols[indpol + nvert - j - 1] = sind++;
      }
      indpol += nvert;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Fill tessellated points to an array.

void TGeoTessellated::SetPoints(Double_t *points) const
{
   Int_t ind = 0;
   for (const auto &vertex : fVertices) {
      vertex.CopyTo(&points[ind]);
      ind += 3;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Fill tessellated points in float.

void TGeoTessellated::SetPoints(Float_t *points) const
{
   Int_t ind = 0;
   for (const auto &vertex : fVertices) {
      points[ind++] = vertex.x();
      points[ind++] = vertex.y();
      points[ind++] = vertex.z();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Fills a static 3D buffer and returns a reference.

const TBuffer3D & TGeoTessellated::GetBuffer3D(Int_t reqSections, Bool_t localFrame) const
{
   static TBuffer3D buffer(TBuffer3DTypes::kGeneric);

   FillBuffer3D(buffer, reqSections, localFrame);

   const Int_t nvert = fNvert;
   const Int_t nsegs = fNseg;
   const Int_t npols = GetNfacets();

   if (reqSections & TBuffer3D::kRawSizes) {
      if (buffer.SetRawSizes(nvert, 3*nvert, nsegs, 3*nsegs, npols, 6*npols)) {
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
