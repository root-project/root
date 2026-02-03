// @(#)root/geom:$Id$// Author: Andrei Gheata   24/10/01

// Contains() and DistFromOutside/Out() implemented by Mihaela Gheata
// 2026-01: Revision to use BVH for navigation functions by Sandro Wenzel

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

#include <iostream>
#include <sstream>

#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TGeoVolume.h"
#include "TVirtualGeoPainter.h"
#include "TGeoTessellated.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"
#include "TMath.h"
#include "TBuffer.h"

#include <array>
#include <vector>

#include <bvh/v2/bvh.h>
#include <bvh/v2/vec.h>
#include <bvh/v2/ray.h>
#include <bvh/v2/node.h>
#include <bvh/v2/stack.h>
#include <bvh/v2/default_builder.h>
#include <cmath>
#include <limits>

ClassImp(TGeoTessellated);

using Vertex_t = Tessellated::Vertex_t;

////////////////////////////////////////////////////////////////////////////////
/// Compact consecutive equal vertices

int TGeoFacet::CompactFacet(Vertex_t *vert, int nvertices)
{
   // Compact the common vertices and return new facet
   if (nvertices < 2)
      return nvertices;
   int nvert = nvertices;
   int i = 0;
   while (i < nvert) {
      if (vert[(i + 1) % nvert] == vert[i]) {
         // shift last vertices left by one element
         for (int j = i + 2; j < nvert; ++j)
            vert[j - 1] = vert[j];
         nvert--;
      }
      i++;
   }
   return nvert;
}

////////////////////////////////////////////////////////////////////////////////
/// Check if a connected neighbour facet has compatible normal

bool TGeoFacet::IsNeighbour(const TGeoFacet &other, bool &flip) const
{

   // Find a connecting segment
   bool neighbour = false;
   int line1[2], line2[2];
   int npoints = 0;
   for (int i = 0; i < fNvert; ++i) {
      auto ivert = fIvert[i];
      // Check if the other facet has the same vertex
      for (int j = 0; j < other.GetNvert(); ++j) {
         if (ivert == other[j]) {
            line1[npoints] = i;
            line2[npoints] = j;
            if (++npoints == 2) {
               neighbour = true;
               bool order1 = line1[1] == line1[0] + 1;
               bool order2 = line2[1] == (line2[0] + 1) % other.GetNvert();
               flip = (order1 == order2);
               return neighbour;
            }
         }
      }
   }
   return neighbour;
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor. In case nfacets is zero, it is user's responsibility to
/// call CloseShape once all faces are defined.

TGeoTessellated::TGeoTessellated(const char *name, int nfacets) : TGeoBBox(name, 0, 0, 0)
{
   fNfacets = nfacets;
   if (nfacets)
      fFacets.reserve(nfacets);
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor providing directly the array of vertices. Facets have to be added
/// providing vertex indices rather than coordinates.

TGeoTessellated::TGeoTessellated(const char *name, const std::vector<Vertex_t> &vertices) : TGeoBBox(name, 0, 0, 0)
{
   fVertices = vertices;
   fNvert = fVertices.size();
}

////////////////////////////////////////////////////////////////////////////////
/// Add a vertex checking for duplicates, returning the vertex index

int TGeoTessellated::AddVertex(Vertex_t const &vert)
{
   constexpr double tolerance = 1.e-10;
   auto vertexHash = [&](Vertex_t const &vertex) {
      // Compute hash for the vertex
      long hash = 0;
      // helper function to generate hash from integer numbers
      auto hash_combine = [](long seed, const long value) {
         return seed ^ (std::hash<long>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
      };
      for (int i = 0; i < 3; i++) {
         // use tolerance to generate int with the desired precision from a real number for hashing
         hash = hash_combine(hash, std::roundl(vertex[i] / tolerance));
      }
      return hash;
   };

   auto hash = vertexHash(vert);
   bool isAdded = false;
   int ivert = -1;
   // Get the compatible vertices
   auto range = fVerticesMap.equal_range(hash);
   for (auto it = range.first; it != range.second; ++it) {
      ivert = it->second;
      if (fVertices[ivert] == vert) {
         isAdded = true;
         break;
      }
   }
   if (!isAdded) {
      ivert = fVertices.size();
      fVertices.push_back(vert);
      fVerticesMap.insert(std::make_pair(hash, ivert));
   }
   return ivert;
}

////////////////////////////////////////////////////////////////////////////////
/// Adding a triangular facet from vertex positions in absolute coordinates

bool TGeoTessellated::AddFacet(const Vertex_t &pt0, const Vertex_t &pt1, const Vertex_t &pt2)
{
   if (fDefined) {
      Error("AddFacet", "Shape %s already fully defined. Not adding", GetName());
      return false;
   }

   Vertex_t vert[3];
   vert[0] = pt0;
   vert[1] = pt1;
   vert[2] = pt2;
   int nvert = TGeoFacet::CompactFacet(vert, 3);
   if (nvert < 3) {
      Error("AddFacet", "Triangular facet at index %d degenerated. Not adding.", GetNfacets());
      return false;
   }
   int ind[3];
   for (auto i = 0; i < 3; ++i)
      ind[i] = AddVertex(vert[i]);
   fNseg += 3;
   fFacets.emplace_back(ind[0], ind[1], ind[2]);

   // if (fNfacets > 0 && GetNfacets() == fNfacets)
   //    CloseShape();
   return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Adding a triangular facet from indices of vertices

bool TGeoTessellated::AddFacet(int i0, int i1, int i2)
{
   if (fDefined) {
      Error("AddFacet", "Shape %s already fully defined. Not adding", GetName());
      return false;
   }
   if (fVertices.empty()) {
      Error("AddFacet", "Shape %s Cannot add facets by indices without vertices. Not adding", GetName());
      return false;
   }

   fNseg += 3;
   fFacets.emplace_back(i0, i1, i2);
   return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Adding a quadrilateral facet from vertex positions in absolute coordinates

bool TGeoTessellated::AddFacet(const Vertex_t &pt0, const Vertex_t &pt1, const Vertex_t &pt2, const Vertex_t &pt3)
{
   if (fDefined) {
      Error("AddFacet", "Shape %s already fully defined. Not adding", GetName());
      return false;
   }
   Vertex_t vert[4];
   vert[0] = pt0;
   vert[1] = pt1;
   vert[2] = pt2;
   vert[3] = pt3;
   int nvert = TGeoFacet::CompactFacet(vert, 4);
   if (nvert < 3) {
      Error("AddFacet", "Quadrilateral facet at index %d degenerated. Not adding.", GetNfacets());
      return false;
   }

   int ind[4];
   for (auto i = 0; i < nvert; ++i)
      ind[i] = AddVertex(vert[i]);
   fNseg += nvert;
   if (nvert == 3)
      fFacets.emplace_back(ind[0], ind[1], ind[2]);
   else
      fFacets.emplace_back(ind[0], ind[1], ind[2], ind[3]);

   if (fNfacets > 0 && GetNfacets() == fNfacets)
      CloseShape(false);
   return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Adding a quadrilateral facet from indices of vertices

bool TGeoTessellated::AddFacet(int i0, int i1, int i2, int i3)
{
   if (fDefined) {
      Error("AddFacet", "Shape %s already fully defined. Not adding", GetName());
      return false;
   }
   if (fVertices.empty()) {
      Error("AddFacet", "Shape %s Cannot add facets by indices without vertices. Not adding", GetName());
      return false;
   }

   fNseg += 4;
   fFacets.emplace_back(i0, i1, i2, i3);
   return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Compute normal for a given facet

Vertex_t TGeoTessellated::FacetComputeNormal(int ifacet, bool &degenerated) const
{
   // Compute normal using non-zero segments
   constexpr double kTolerance = 1.e-20;
   auto const &facet = fFacets[ifacet];
   int nvert = facet.GetNvert();
   degenerated = true;
   Vertex_t normal;
   for (int i = 0; i < nvert - 1; ++i) {
      Vertex_t e1 = fVertices[facet[i + 1]] - fVertices[facet[i]];
      if (e1.Mag2() < kTolerance)
         continue;
      for (int j = i + 1; j < nvert; ++j) {
         Vertex_t e2 = fVertices[facet[(j + 1) % nvert]] - fVertices[facet[j]];
         if (e2.Mag2() < kTolerance)
            continue;
         normal = Vertex_t::Cross(e1, e2);
         // e1 and e2 may be colinear
         if (normal.Mag2() < kTolerance)
            continue;
         normal.Normalize();
         degenerated = false;
         break;
      }
      if (!degenerated)
         break;
   }
   return normal;
}

////////////////////////////////////////////////////////////////////////////////
/// Check validity of facet

bool TGeoTessellated::FacetCheck(int ifacet) const
{
   constexpr double kTolerance = 1.e-10;
   auto const &facet = fFacets[ifacet];
   int nvert = facet.GetNvert();
   bool degenerated = true;
   FacetComputeNormal(ifacet, degenerated);
   if (degenerated) {
      std::cout << "Facet: " << ifacet << " is degenerated\n";
      return false;
   }

   // Compute surface area
   double surfaceArea = 0.;
   for (int i = 1; i < nvert - 1; ++i) {
      Vertex_t e1 = fVertices[facet[i]] - fVertices[facet[0]];
      Vertex_t e2 = fVertices[facet[i + 1]] - fVertices[facet[0]];
      surfaceArea += 0.5 * Vertex_t::Cross(e1, e2).Mag();
   }
   if (surfaceArea < kTolerance) {
      std::cout << "Facet: " << ifacet << " has zero surface area\n";
      return false;
   }

   return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Close the shape: calculate bounding box and compact vertices

void TGeoTessellated::CloseShape(bool check, bool fixFlipped, bool verbose)
{
   if (fIsClosed && fBVH) {
      return;
   }
   // Compute bounding box
   fDefined = true;
   fNvert = fVertices.size();
   fNfacets = fFacets.size();
   ComputeBBox();
   // Cleanup the vertex map
   std::multimap<long, int>().swap(fVerticesMap);

   if (check && fVertices.size() > 0) {
      // Check facets
      for (auto i = 0; i < fNfacets; ++i)
         FacetCheck(i);

      fClosedBody = CheckClosure(fixFlipped, verbose);
   }
   BuildBVH();
   CalculateNormals();
   fIsClosed = true;
}

////////////////////////////////////////////////////////////////////////////////
/// Check closure of the solid and check/fix flipped normals

bool TGeoTessellated::CheckClosure(bool fixFlipped, bool verbose)
{
   int *nn = new int[fNfacets];
   bool *flipped = new bool[fNfacets];
   bool hasorphans = false;
   bool hasflipped = false;
   for (int i = 0; i < fNfacets; ++i) {
      nn[i] = 0;
      flipped[i] = false;
   }

   for (int icrt = 0; icrt < fNfacets; ++icrt) {
      // all neighbours checked?
      if (nn[icrt] >= fFacets[icrt].GetNvert())
         continue;
      for (int i = icrt + 1; i < fNfacets; ++i) {
         bool isneighbour = fFacets[icrt].IsNeighbour(fFacets[i], flipped[i]);
         if (isneighbour) {
            if (flipped[icrt])
               flipped[i] = !flipped[i];
            if (flipped[i])
               hasflipped = true;
            nn[icrt]++;
            nn[i]++;
            if (nn[icrt] == fFacets[icrt].GetNvert())
               break;
         }
      }
      if (nn[icrt] < fFacets[icrt].GetNvert())
         hasorphans = true;
   }

   if (hasorphans && verbose) {
      Error("Check", "Tessellated solid %s has following not fully connected facets:", GetName());
      for (int icrt = 0; icrt < fNfacets; ++icrt) {
         if (nn[icrt] < fFacets[icrt].GetNvert())
            std::cout << icrt << " (" << fFacets[icrt].GetNvert() << " edges, " << nn[icrt] << " neighbours)\n";
      }
   }
   fClosedBody = !hasorphans;
   int nfixed = 0;
   if (hasflipped) {
      if (verbose)
         Warning("Check", "Tessellated solid %s has following facets with flipped normals:", GetName());
      for (int icrt = 0; icrt < fNfacets; ++icrt) {
         if (flipped[icrt]) {
            if (verbose)
               std::cout << icrt << "\n";
            if (fixFlipped) {
               fFacets[icrt].Flip();
               nfixed++;
            }
         }
      }
      if (nfixed && verbose)
         Info("Check", "Automatically flipped %d facets to match first defined facet", nfixed);
   }
   delete[] nn;
   delete[] flipped;

   return !hasorphans;
}

////////////////////////////////////////////////////////////////////////////////
/// Compute bounding box

void TGeoTessellated::ComputeBBox()
{
   const double kBig = TGeoShape::Big();
   double vmin[3] = {kBig, kBig, kBig};
   double vmax[3] = {-kBig, -kBig, -kBig};
   for (const auto &facet : fFacets) {
      for (int i = 0; i < facet.GetNvert(); ++i) {
         for (int j = 0; j < 3; ++j) {
            vmin[j] = TMath::Min(vmin[j], fVertices[facet[i]].operator[](j));
            vmax[j] = TMath::Max(vmax[j], fVertices[facet[i]].operator[](j));
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

void TGeoTessellated::GetMeshNumbers(int &nvert, int &nsegs, int &npols) const
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
   const int nvert = fNvert;
   const int nsegs = fNseg;
   const int npols = GetNfacets();
   auto buff = new TBuffer3D(TBuffer3DTypes::kGeneric, nvert, 3 * nvert, nsegs, 3 * nsegs, npols, 6 * npols);
   if (buff) {
      SetPoints(buff->fPnts);
      SetSegsAndPols(*buff);
   }
   return buff;
}

////////////////////////////////////////////////////////////////////////////////
/// Prints basic info

void TGeoTessellated::Print(Option_t *) const
{
   std::cout << "=== Tessellated shape " << GetName() << " having " << GetNvertices() << " vertices and "
             << GetNfacets() << " facets\n";
}

////////////////////////////////////////////////////////////////////////////////
/// Fills TBuffer3D structure for segments and polygons.

void TGeoTessellated::SetSegsAndPols(TBuffer3D &buff) const
{
   const int c = GetBasicColor();
   int *segs = buff.fSegs;
   int *pols = buff.fPols;

   int indseg = 0; // segment internal data index
   int indpol = 0; // polygon internal data index
   int sind = 0;   // segment index
   for (const auto &facet : fFacets) {
      auto nvert = facet.GetNvert();
      pols[indpol++] = c;
      pols[indpol++] = nvert;
      for (auto j = 0; j < nvert; ++j) {
         int k = (j + 1) % nvert;
         // segment made by next consecutive points
         segs[indseg++] = c;
         segs[indseg++] = facet[j];
         segs[indseg++] = facet[k];
         // add segment to current polygon and increment segment index
         pols[indpol + nvert - j - 1] = sind++;
      }
      indpol += nvert;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Fill tessellated points to an array.

void TGeoTessellated::SetPoints(double *points) const
{
   int ind = 0;
   for (const auto &vertex : fVertices) {
      vertex.CopyTo(&points[ind]);
      ind += 3;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Fill tessellated points in float.

void TGeoTessellated::SetPoints(Float_t *points) const
{
   int ind = 0;
   for (const auto &vertex : fVertices) {
      points[ind++] = vertex.x();
      points[ind++] = vertex.y();
      points[ind++] = vertex.z();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Resize the shape by scaling vertices within maxsize and center to origin

void TGeoTessellated::ResizeCenter(double maxsize)
{
   using Vector3_t = Vertex_t;

   if (!fDefined) {
      Error("ResizeCenter", "Not all faces are defined");
      return;
   }
   Vector3_t origin(fOrigin[0], fOrigin[1], fOrigin[2]);
   double maxedge = TMath::Max(TMath::Max(fDX, fDY), fDZ);
   double scale = maxsize / maxedge;
   for (size_t i = 0; i < fVertices.size(); ++i) {
      fVertices[i] = scale * (fVertices[i] - origin);
   }
   fOrigin[0] = fOrigin[1] = fOrigin[2] = 0;
   fDX *= scale;
   fDY *= scale;
   fDZ *= scale;
}

////////////////////////////////////////////////////////////////////////////////
/// Fills a static 3D buffer and returns a reference.

const TBuffer3D &TGeoTessellated::GetBuffer3D(int reqSections, Bool_t localFrame) const
{
   static TBuffer3D buffer(TBuffer3DTypes::kGeneric);

   FillBuffer3D(buffer, reqSections, localFrame);

   const int nvert = fNvert;
   const int nsegs = fNseg;
   const int npols = GetNfacets();

   if (reqSections & TBuffer3D::kRawSizes) {
      if (buffer.SetRawSizes(nvert, 3 * nvert, nsegs, 3 * nsegs, npols, 6 * npols)) {
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
/// Reads a single tessellated solid from an .obj file.

TGeoTessellated *TGeoTessellated::ImportFromObjFormat(const char *objfile, bool check, bool verbose)
{
   using std::vector, std::string, std::ifstream, std::stringstream, std::endl;

   vector<Vertex_t> vertices;
   vector<string> sfacets;

   struct FacetInd_t {
      int i0 = -1;
      int i1 = -1;
      int i2 = -1;
      int i3 = -1;
      int nvert = 0;
      FacetInd_t(int a, int b, int c)
      {
         i0 = a;
         i1 = b;
         i2 = c;
         nvert = 3;
      };
      FacetInd_t(int a, int b, int c, int d)
      {
         i0 = a;
         i1 = b;
         i2 = c;
         i3 = d;
         nvert = 4;
      };
   };

   vector<FacetInd_t> facets;
   // List of geometric vertices, with (x, y, z [,w]) coordinates, w is optional and defaults to 1.0.
   // struct vtx_t { double x = 0; double y = 0; double z = 0; double w = 1; };

   // Texture coordinates in u, [,v ,w]) coordinates, these will vary between 0 and 1. v, w are optional and default to
   // 0.
   // struct tex_t { double u; double v; double w; };

   // List of vertex normals in (x,y,z) form; normals might not be unit vectors.
   // struct vn_t { double x; double y; double z; };

   // Parameter space vertices in ( u [,v] [,w] ) form; free form geometry statement
   // struct vp_t { double u; double v; double w; };

   // Faces are defined using lists of vertex, texture and normal indices which start at 1.
   // Polygons such as quadrilaterals can be defined by using more than three vertex/texture/normal indices.
   //     f v1//vn1 v2//vn2 v3//vn3 ...

   // Records starting with the letter "l" specify the order of the vertices which build a polyline.
   //     l v1 v2 v3 v4 v5 v6 ...

   string line;
   int ind[4] = {0};
   ifstream file(objfile);
   if (!file.is_open()) {
      ::Error("TGeoTessellated::ImportFromObjFormat", "Unable to open %s", objfile);
      return nullptr;
   }

   while (getline(file, line)) {
      stringstream ss(line);
      string tag;

      // We ignore everything which is not a vertex or a face
      if (line.rfind('v', 0) == 0 && line.rfind("vt", 0) != 0 && line.rfind("vn", 0) != 0 && line.rfind("vn", 0) != 0) {
         // Decode the vertex
         double pos[4] = {0, 0, 0, 1};
         ss >> tag >> pos[0] >> pos[1] >> pos[2] >> pos[3];
         vertices.emplace_back(pos[0] * pos[3], pos[1] * pos[3], pos[2] * pos[3]);
      }

      else if (line.rfind('f', 0) == 0) {
         // Decode the face
         ss >> tag;
         string word;
         sfacets.clear();
         while (ss >> word)
            sfacets.push_back(word);
         if (sfacets.size() > 4 || sfacets.size() < 3) {
            ::Error("TGeoTessellated::ImportFromObjFormat", "Detected face having unsupported %zu vertices",
                    sfacets.size());
            return nullptr;
         }
         int nvert = 0;
         for (auto &sword : sfacets) {
            stringstream ssword(sword);
            string token;
            getline(ssword, token, '/'); // just need the vertex index, which is the first token
            // Convert string token to integer

            ind[nvert++] = stoi(token) - 1;
            if (ind[nvert - 1] < 0) {
               ::Error("TGeoTessellated::ImportFromObjFormat", "Unsupported relative vertex index definition in %s",
                       objfile);
               return nullptr;
            }
         }
         if (nvert == 3)
            facets.emplace_back(ind[0], ind[1], ind[2]);
         else
            facets.emplace_back(ind[0], ind[1], ind[2], ind[3]);
      }
   }

   int nvertices = (int)vertices.size();
   int nfacets = (int)facets.size();
   if (nfacets < 3) {
      ::Error("TGeoTessellated::ImportFromObjFormat", "Not enough faces detected in %s", objfile);
      return nullptr;
   }

   string sobjfile(objfile);
   if (verbose)
      std::cout << "Read " << nvertices << " vertices and " << nfacets << " facets from " << sobjfile << endl;

   auto tsl = new TGeoTessellated(sobjfile.erase(sobjfile.find_last_of('.')).c_str(), vertices);

   for (int i = 0; i < nfacets; ++i) {
      auto facet = facets[i];
      if (facet.nvert == 3)
         tsl->AddFacet(facet.i0, facet.i1, facet.i2);
      else
         tsl->AddFacet(facet.i0, facet.i1, facet.i2, facet.i3);
   }
   tsl->CloseShape(check, true, verbose);
   tsl->Print();
   return tsl;
}

// implementation of some geometry helper functions in anonymous namespace
namespace {

inline Tessellated::Vertex_t cross(const Tessellated::Vertex_t &a, const Tessellated::Vertex_t &b)
{
   return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

inline double dot(const Tessellated::Vertex_t &a, const Tessellated::Vertex_t &b)
{
   return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

using Vertex_t = Tessellated::Vertex_t;
// The classic Moeller-Trumbore ray triangle-intersection kernel:
// - Compute triangle edges e1, e2
// - Compute determinant det
// - Reject parallel rays
// - Compute barycentric coordinates u, v
// - Compute ray parameter t
double rayTriangle(const Vertex_t &orig, const Vertex_t &dir, const Vertex_t &v0, const Vertex_t &v1,
                   const Vertex_t &v2, double rayEPS = 1e-8)
{
   constexpr double EPS = 1e-8;
   const double INF = std::numeric_limits<double>::infinity();
   Vertex_t e1{v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
   Vertex_t e2{v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
   auto p = cross(dir, e2);
   auto det = dot(e1, p);
   if (std::abs(det) <= EPS) {
      return INF;
   }

   Vertex_t tvec{orig[0] - v0[0], orig[1] - v0[1], orig[2] - v0[2]};
   auto invDet = 1.0 / det;
   auto u = dot(tvec, p) * invDet;
   if (u < 0.0 || u > 1.0) {
      return INF;
   }
   auto q = cross(tvec, e1);
   auto v = dot(dir, q) * invDet;
   if (v < 0.0 || u + v > 1.0) {
      return INF;
   }
   auto t = dot(e2, q) * invDet;
   return (t > rayEPS) ? t : INF;
}

template <typename T = float>
struct Vec3f {
   T x, y, z;
   template <typename Type>
   Vec3f(Type x_, Type y_, Type z_) : x{T(x_)}, y{T(y_)}, z{T(z_)}
   {
   }
};

template <typename T>
inline Vec3f<T> operator-(const Vec3f<T> &a, const Vec3f<T> &b)
{
   return {a.x - b.x, a.y - b.y, a.z - b.z};
}

template <typename T>
inline Vec3f<T> cross(const Vec3f<T> &a, const Vec3f<T> &b)
{
   return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

template <typename T>
inline T dot(const Vec3f<T> &a, const Vec3f<T> &b)
{
   return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Kernel to get closest/shortest distance between a point and a triangl (a,b,c).
// Performed by default in float since Safety is approximation in any case.
// Project point onto triangle plane
// If projection lies inside → distance to plane
// Otherwise compute min distance to the three edges
// Return squared distance
template <typename T = float>
T pointTriangleDistSq(const Vec3f<T> &p, const Vec3f<T> &a, const Vec3f<T> &b, const Vec3f<T> &c)
{
   // Edges
   Vec3f<T> ab = b - a;
   Vec3f<T> ac = c - a;
   Vec3f<T> ap = p - a;

   auto d1 = dot(ab, ap);
   auto d2 = dot(ac, ap);
   if (d1 <= T(0.0) && d2 <= T(0.0)) {
      return dot(ap, ap); // barycentric (1,0,0)
   }

   Vec3f<T> bp = p - b;
   auto d3 = dot(ab, bp);
   auto d4 = dot(ac, bp);
   if (d3 >= T(0.0) && d4 <= d3) {
      return dot(bp, bp); // (0,1,0)
   }

   T vc = d1 * d4 - d3 * d2;
   if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
      T v = d1 / (d1 - d3);
      Vec3f<T> proj = {a.x + v * ab.x, a.y + v * ab.y, a.z + v * ab.z};
      Vec3f<T> d = p - proj;
      return dot(d, d); // edge AB
   }

   Vec3f<T> cp = p - c;
   T d5 = dot(ab, cp);
   T d6 = dot(ac, cp);
   if (d6 >= T(0.0f) && d5 <= d6) {
      return dot(cp, cp); // (0,0,1)
   }

   T vb = d5 * d2 - d1 * d6;
   if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
      T w = d2 / (d2 - d6);
      Vec3f<T> proj = {a.x + w * ac.x, a.y + w * ac.y, a.z + w * ac.z};
      Vec3f<T> d = p - proj;
      return dot(d, d); // edge AC
   }

   T va = d3 * d6 - d5 * d4;
   if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
      T w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
      Vec3f<T> proj = {b.x + w * (c.x - b.x), b.y + w * (c.y - b.y), b.z + w * (c.z - b.z)};
      Vec3f<T> d = p - proj;
      return dot(d, d); // edge BC
   }

   // Inside face region
   T denom = T(1.0f) / (va + vb + vc);
   T v = vb * denom;
   T w = vc * denom;

   Vec3f<T> proj = {a.x + ab.x * v + ac.x * w, a.y + ab.y * v + ac.y * w, a.z + ab.z * v + ac.z * w};

   Vec3f<T> d = p - proj;
   return dot(d, d);
}

template <typename T>
inline Vec3f<T> normalize(const Vec3f<T> &v)
{
   T len2 = dot(v, v);
   if (len2 == T(0.0f)) {
      std::cerr << "Degnerate triangle. Cannot determine normal";
      return {0, 0, 0};
   }
   T invLen = T(1.0f) / std::sqrt(len2);
   return {v.x * invLen, v.y * invLen, v.z * invLen};
}

template <typename T>
inline Vec3f<T> triangleNormal(const Vec3f<T> &a, const Vec3f<T> &b, const Vec3f<T> &c)
{
   const Vec3f<T> e1 = b - a;
   const Vec3f<T> e2 = c - a;
   return normalize(cross(e1, e2));
}

} // end anonymous namespace

////////////////////////////////////////////////////////////////////////////////
/// DistFromOutside

Double_t TGeoTessellated::DistFromOutside(const Double_t *point, const Double_t *dir, Int_t /*iact*/, Double_t stepmax,
                                          Double_t * /*safe*/) const
{
   // use the BVH intersector in combination with leaf ray-triangle testing
   double local_step = Big(); // we need this otherwise the lambda get's confused

   using Scalar = float;
   using Vec3 = bvh::v2::Vec<Scalar, 3>;
   using Node = bvh::v2::Node<Scalar, 3>;
   using Bvh = bvh::v2::Bvh<Node>;
   using Ray = bvh::v2::Ray<Scalar, 3>;

   // let's fetch the bvh
   auto mybvh = (Bvh *)fBVH;
   if (!mybvh) {
      assert(false);
      return -1.;
   }

   auto truncate_roundup = [](double orig) {
      float epsilon = std::numeric_limits<float>::epsilon() * std::fabs(orig);
      // Add the bias to x before assigning it to y
      return static_cast<float>(orig + epsilon);
   };

   // let's do very quick checks against the top node
   const auto topnode_bbox = mybvh->get_root().get_bbox();
   if ((-point[0] + topnode_bbox.min[0]) > stepmax) {
      return Big();
   }
   if ((-point[1] + topnode_bbox.min[1]) > stepmax) {
      return Big();
   }
   if ((-point[2] + topnode_bbox.min[2]) > stepmax) {
      return Big();
   }
   if ((point[0] - topnode_bbox.max[0]) > stepmax) {
      return Big();
   }
   if ((point[1] - topnode_bbox.max[1]) > stepmax) {
      return Big();
   }
   if ((point[2] - topnode_bbox.max[2]) > stepmax) {
      return Big();
   }

   // the ray used for bvh interaction
   Ray ray(Vec3(point[0], point[1], point[2]), // origin
           Vec3(dir[0], dir[1], dir[2]),       // direction
           0.0f,                               // minimum distance (could give stepmax ?)
           truncate_roundup(local_step));

   static constexpr bool use_robust_traversal = true;

   // Traverse the BVH and apply concrete object intersection in BVH leafs
   bvh::v2::GrowingStack<Bvh::Index> stack;
   mybvh->intersect<false, use_robust_traversal>(ray, mybvh->get_root().index, stack, [&](size_t begin, size_t end) {
      for (size_t prim_id = begin; prim_id < end; ++prim_id) {
         auto objectid = mybvh->prim_ids[prim_id];
         const auto &facet = fFacets[objectid];
         const auto &n = fOutwardNormals[objectid];

         // quick normal test. Coming from outside, the dot product must be negative
         if (n[0] * dir[0] + n[1] * dir[1] + n[2] * dir[2] > 0.) {
            continue;
         }

         auto thisdist = rayTriangle(Vertex_t(point[0], point[1], point[2]), Vertex_t(dir[0], dir[1], dir[2]),
                                     fVertices[facet[0]], fVertices[facet[1]], fVertices[facet[2]], 0.);

         if (thisdist < local_step) {
            local_step = thisdist;
         }
      }
      return false; // go on after this
   });

   return local_step;
}

////////////////////////////////////////////////////////////////////////////////
/// DistFromOutside

Double_t TGeoTessellated::DistFromInside(const Double_t *point, const Double_t *dir, Int_t /*iact*/,
                                         Double_t /*stepmax*/, Double_t * /*safe*/) const
{
   // use the BVH intersector in combination with leaf ray-triangle testing
   double local_step = Big(); // we need this otherwise the lambda get's confused

   using Scalar = float;
   using Vec3 = bvh::v2::Vec<Scalar, 3>;
   using Node = bvh::v2::Node<Scalar, 3>;
   using Bvh = bvh::v2::Bvh<Node>;
   using Ray = bvh::v2::Ray<Scalar, 3>;

   // let's fetch the bvh
   auto mybvh = (Bvh *)fBVH;
   if (!mybvh) {
      assert(false);
      return -1.;
   }

   auto truncate_roundup = [](double orig) {
      float epsilon = std::numeric_limits<float>::epsilon() * std::fabs(orig);
      // Add the bias to x before assigning it to y
      return static_cast<float>(orig + epsilon);
   };

   // the ray used for bvh interaction
   Ray ray(Vec3(point[0], point[1], point[2]), // origin
           Vec3(dir[0], dir[1], dir[2]),       // direction
           0.,                                 // minimum distance (could give stepmax ?)
           truncate_roundup(local_step));

   static constexpr bool use_robust_traversal = true;

   // Traverse the BVH and apply concrete object intersection in BVH leafs
   bvh::v2::GrowingStack<Bvh::Index> stack;
   mybvh->intersect<false, use_robust_traversal>(ray, mybvh->get_root().index, stack, [&](size_t begin, size_t end) {
      for (size_t prim_id = begin; prim_id < end; ++prim_id) {
         auto objectid = mybvh->prim_ids[prim_id];
         auto facet = fFacets[objectid];
         const auto &n = fOutwardNormals[objectid];

         // Only exiting surfaces are relevant (from inside--> dot product must be positive)
         if (n[0] * dir[0] + n[1] * dir[1] + n[2] * dir[2] <= 0.0) {
            continue;
         }

         const auto &v0 = fVertices[facet[0]];
         const auto &v1 = fVertices[facet[1]];
         const auto &v2 = fVertices[facet[2]];

         const double t =
            rayTriangle(Vertex_t{point[0], point[1], point[2]}, Vertex_t{dir[0], dir[1], dir[2]}, v0, v1, v2, 0.);
         if (t < local_step) {
            local_step = t;
         }
      }
      return false; // go on after this
   });

   return local_step;
}

////////////////////////////////////////////////////////////////////////////////
/// Capacity

Double_t TGeoTessellated::Capacity() const
{
   // For explanation of the following algorithm see:
   // https://en.wikipedia.org/wiki/Polyhedron#Volume
   // http://wwwf.imperial.ac.uk/~rn/centroid.pdf

   double vol = 0.0;
   for (size_t i = 0; i < fFacets.size(); ++i) {
      auto &facet = fFacets[i];
      auto a = fVertices[facet[0]];
      auto b = fVertices[facet[1]];
      auto c = fVertices[facet[2]];
      vol +=
         a[0] * (b[1] * c[2] - b[2] * c[1]) + b[0] * (c[1] * a[2] - c[2] * a[1]) + c[0] * (a[1] * b[2] - a[2] * b[1]);
   }
   return vol / 6.0;
}

////////////////////////////////////////////////////////////////////////////////
/// BuildBVH

void TGeoTessellated::BuildBVH()
{
   using Scalar = float;
   using BBox = bvh::v2::BBox<Scalar, 3>;
   using Vec3 = bvh::v2::Vec<Scalar, 3>;
   using Node = bvh::v2::Node<Scalar, 3>;
   using Bvh = bvh::v2::Bvh<Node>;

   // helper determining axis aligned bounding box from a facet;
   auto GetBoundingBox = [this](TGeoFacet const &facet) {
      const auto nvertices = facet.GetNvert();
      if (nvertices != 3)
         Fatal("BuildBVH", "only facets with 3 vertices supported"); // for now only triangles
      const auto &v1 = fVertices[facet[0]];
      const auto &v2 = fVertices[facet[1]];
      const auto &v3 = fVertices[facet[2]];
      BBox bbox;
      bbox.min[0] = std::min(std::min(v1[0], v2[0]), v3[0]) - 0.001f;
      bbox.min[1] = std::min(std::min(v1[1], v2[1]), v3[1]) - 0.001f;
      bbox.min[2] = std::min(std::min(v1[2], v2[2]), v3[2]) - 0.001f;
      bbox.max[0] = std::max(std::max(v1[0], v2[0]), v3[0]) + 0.001f;
      bbox.max[1] = std::max(std::max(v1[1], v2[1]), v3[1]) + 0.001f;
      bbox.max[2] = std::max(std::max(v1[2], v2[2]), v3[2]) + 0.001f;
      return bbox;
   };

   // we need bounding boxes enclosing the primitives and centers of primitives
   // (replaced here by centers of bounding boxes) to build the bvh
   auto bboxes_ptr = new std::vector<BBox>();
   // fBoundingBoxes = (void *)bboxes_ptr;
   auto &bboxes = *bboxes_ptr;
   std::vector<Vec3> centers;

   // loop over all the triangles/Facets;
   int nd = fFacets.size();
   for (int i = 0; i < nd; ++i) {
      auto &facet = fFacets[i];

      // fetch the bounding box of this node and add to the vector of bounding boxes
      (bboxes).push_back(GetBoundingBox(facet));
      centers.emplace_back((bboxes).back().get_center());
   }

   // check if some previous object is registered and delete if necessary
   if (fBVH) {
      delete (Bvh *)fBVH;
      fBVH = nullptr;
   }

   // create the bvh
   typename bvh::v2::DefaultBuilder<Node>::Config config;
   config.quality = bvh::v2::DefaultBuilder<Node>::Quality::High;
   auto bvh = bvh::v2::DefaultBuilder<Node>::build(bboxes, centers, config);
   auto bvhptr = new Bvh;
   *bvhptr = std::move(bvh); // copy structure
   fBVH = (void *)(bvhptr);

   return;
}

////////////////////////////////////////////////////////////////////////////////
/// Contains

bool TGeoTessellated::Contains(Double_t const *point) const
{
   // we do the parity test
   using Scalar = float;
   using Vec3 = bvh::v2::Vec<Scalar, 3>;
   using Node = bvh::v2::Node<Scalar, 3>;
   using Bvh = bvh::v2::Bvh<Node>;
   using Ray = bvh::v2::Ray<Scalar, 3>;

   // let's fetch the bvh
   auto mybvh = (Bvh *)fBVH;
   if (!mybvh) {
      assert(false);
      return -1.;
   }

   auto truncate_roundup = [](double orig) {
      float epsilon = std::numeric_limits<float>::epsilon() * std::fabs(orig);
      // Add the bias to x before assigning it to y
      return static_cast<float>(orig + epsilon);
   };

   // let's do very quick checks against the top node
   // is this useful for inside?
   if (!TGeoBBox::Contains(point)) {
      return false;
   }

   // doesn't need to be normalized and probes all normals
   double test_dir[3] = {1.0, 1.41421356237, 1.73205080757};

   double local_step = Big();
   // the ray used for bvh interaction
   Ray ray(Vec3(point[0], point[1], point[2]),          // origin
           Vec3(test_dir[0], test_dir[1], test_dir[2]), // direction
           0.0f,                                        // minimum distance (could give stepmax ?)
           truncate_roundup(local_step));

   static constexpr bool use_robust_traversal = true;

   // Traverse the BVH and apply concrete object intersection in BVH leafs
   bvh::v2::GrowingStack<Bvh::Index> stack;
   size_t crossings = 0;
   mybvh->intersect<false, use_robust_traversal>(ray, mybvh->get_root().index, stack, [&](size_t begin, size_t end) {
      for (size_t prim_id = begin; prim_id < end; ++prim_id) {
         auto objectid = mybvh->prim_ids[prim_id];
         auto &facet = fFacets[objectid];

         // for the parity test, we probe all crossing surfaces
         const auto &v0 = fVertices[facet[0]];
         const auto &v1 = fVertices[facet[1]];
         const auto &v2 = fVertices[facet[2]];

         const double t = rayTriangle(Vertex_t(point[0], point[1], point[2]),
                                      Vertex_t(test_dir[0], test_dir[1], test_dir[2]), v0, v1, v2, 0.);

         if (t != std::numeric_limits<double>::infinity()) {
            ++crossings;
         }
      }
      return false;
   });

   return crossings & 1;
}

namespace {
// some helpers for point - axis aligned bounding box functions
// using bvh types

// determines if a point is inside the bounding box
template <typename T>
bool contains(bvh::v2::BBox<T, 3> const &box, bvh::v2::Vec<T, 3> const &p)
{
   auto min = box.min;
   auto max = box.max;
   return (p[0] >= min[0] && p[0] <= max[0]) && (p[1] >= min[1] && p[1] <= max[1]) &&
          (p[2] >= min[2] && p[2] <= max[2]);
}

// determines the largest squared distance of point to any of the bounding box corners
template <typename T>
auto RmaxSqToNode(bvh::v2::BBox<T, 3> const &box, bvh::v2::Vec<T, 3> const &p)
{
   // construct the 8 corners to get the maximal distance
   const auto minCorner = box.min;
   const auto maxCorner = box.max;
   using Vec3 = bvh::v2::Vec<T, 3>;
   // these are the corners of the bounding box
   const std::array<bvh::v2::Vec<T, 3>, 8> corners{
      Vec3{minCorner[0], minCorner[1], minCorner[2]}, Vec3{minCorner[0], minCorner[1], maxCorner[2]},
      Vec3{minCorner[0], maxCorner[1], minCorner[2]}, Vec3{minCorner[0], maxCorner[1], maxCorner[2]},
      Vec3{maxCorner[0], minCorner[1], minCorner[2]}, Vec3{maxCorner[0], minCorner[1], maxCorner[2]},
      Vec3{maxCorner[0], maxCorner[1], minCorner[2]}, Vec3{maxCorner[0], maxCorner[1], maxCorner[2]}};

   T Rmax_sq{0};
   for (const auto &corner : corners) {
      float R_sq = 0.;
      const auto dx = corner[0] - p[0];
      R_sq += dx * dx;
      const auto dy = corner[1] - p[1];
      R_sq += dy * dy;
      const auto dz = corner[2] - p[2];
      R_sq += dz * dz;
      Rmax_sq = std::max(Rmax_sq, R_sq);
   }
   return Rmax_sq;
};

// determines the minimum squared distance of point to a bounding box ("safey square")
template <typename T>
auto SafetySqToNode(bvh::v2::BBox<T, 3> const &box, bvh::v2::Vec<T, 3> const &p)
{
   T sqDist{0.0};
   for (int i = 0; i < 3; i++) {
      T v = p[i];
      if (v < box.min[i]) {
         sqDist += (box.min[i] - v) * (box.min[i] - v);
      } else if (v > box.max[i]) {
         sqDist += (v - box.max[i]) * (v - box.max[i]);
      }
   }
   return sqDist;
};

// Helper classes/structs used for priority queue - BVH traversal
// structure keeping cost (value) for a BVH index
struct BVHPrioElement {
   size_t bvh_node_id;
   float value;
};

// A priority queue for BVHPrioElement with an additional clear method
// for quick reset
template <typename Comparator>
class BVHPrioQueue : public std::priority_queue<BVHPrioElement, std::vector<BVHPrioElement>, Comparator> {
public:
   using std::priority_queue<BVHPrioElement, std::vector<BVHPrioElement>,
                             Comparator>::priority_queue; // constructor inclusion

   // convenience method to quickly clear/reset the queue (instead of having to pop one by one)
   void clear() { this->c.clear(); }
};

} // namespace

/// a reusable safety kernel, which optionally returns the closest face
template <bool returnFace>
inline Double_t TGeoTessellated::SafetyKernel(const Double_t *point, bool in, int *closest_facet_id) const
{
   float smallest_safety_sq = TGeoShape::Big();

   using Scalar = float;
   using Vec3 = bvh::v2::Vec<Scalar, 3>;
   using Node = bvh::v2::Node<Scalar, 3>;
   using Bvh = bvh::v2::Bvh<Node>;

   // let's fetch the bvh
   auto mybvh = (Bvh *)fBVH;

   // testpoint object in float for quick BVH interaction
   Vec3 testpoint(point[0], point[1], point[2]);

   auto currnode = mybvh->nodes[0]; // we start from the top BVH node
   // we do a quick check on the top node (in case we are outside shape)
   bool outside_top = false;
   if (!in) {
      outside_top = !::contains(currnode.get_bbox(), testpoint);
      if (outside_top) {
         const auto safety_sq_to_top = SafetySqToNode(currnode.get_bbox(), testpoint);
         // we simply return safety to the outer bounding box as an estimate
         return std::sqrt(safety_sq_to_top);
      }
   }

   // comparator bringing out "smallest" value on top
   auto cmp = [](BVHPrioElement a, BVHPrioElement b) { return a.value > b.value; };
   static thread_local BVHPrioQueue<decltype(cmp)> queue(cmp);
   queue.clear();

   // algorithm is based on standard iterative tree traversal with priority queues
   float current_safety_to_node_sq = 0.f;

   if (returnFace) {
      *closest_facet_id = -1;
   }

   do {
      if (currnode.is_leaf()) {
         // we are in a leaf node and actually talk to a face/triangular primitive
         const auto begin_prim_id = currnode.index.first_id();
         const auto end_prim_id = begin_prim_id + currnode.index.prim_count();

         for (auto p_id = begin_prim_id; p_id < end_prim_id; p_id++) {
            const auto object_id = mybvh->prim_ids[p_id];

            const auto &facet = fFacets[object_id];
            const auto &v1 = fVertices[facet[0]];
            const auto &v2 = fVertices[facet[1]];
            const auto &v3 = fVertices[facet[2]];

            auto thissafetySQ = pointTriangleDistSq(Vec3f{point[0], point[1], point[2]}, Vec3f{v1[0], v1[1], v1[2]},
                                                    Vec3f{v2[0], v2[1], v2[2]}, Vec3f{v3[0], v3[1], v3[2]});

            if (thissafetySQ < smallest_safety_sq) {
               smallest_safety_sq = thissafetySQ;
               if (returnFace) {
                  *closest_facet_id = object_id;
               }
            }
         }
      } else {
         // not a leave node ... for further traversal,
         // we inject the children into priority queue based on distance to it's bounding box
         const auto leftchild_id = currnode.index.first_id();
         const auto rightchild_id = leftchild_id + 1;

         for (size_t childid : {leftchild_id, rightchild_id}) {
            if (childid >= mybvh->nodes.size()) {
               continue;
            }

            const auto &node = mybvh->nodes[childid];
            const auto inside = contains(node.get_bbox(), testpoint);

            if (inside) {
               // this must be further considered because we are inside the bounding box
               queue.push(BVHPrioElement{childid, -1.});
            } else {
               auto safety_to_node_square = SafetySqToNode(node.get_bbox(), testpoint);
               if (safety_to_node_square <= smallest_safety_sq) {
                  // this should be further considered
                  queue.push(BVHPrioElement{childid, safety_to_node_square});
               }
            }
         }
      }

      if (queue.size() > 0) {
         auto currElement = queue.top();
         currnode = mybvh->nodes[currElement.bvh_node_id];
         current_safety_to_node_sq = currElement.value;
         queue.pop();
      } else {
         break;
      }
   } while (current_safety_to_node_sq <= smallest_safety_sq);

   return std::nextafter(std::sqrt(smallest_safety_sq), 0.0f);
}

////////////////////////////////////////////////////////////////////////////////
/// Safety

Double_t TGeoTessellated::Safety(const Double_t *point, Bool_t in) const
{
   // we could use some caching here (in future) since queries to the solid will likely
   // be made with some locality

   // fall-back to precise safety kernel
   return SafetyKernel<false>(point, in);
}

////////////////////////////////////////////////////////////////////////////////
/// ComputeNormal interface

void TGeoTessellated::ComputeNormal(const Double_t *point, const Double_t *dir, Double_t *norm) const
{
   // We take the approach to identify closest facet to the point via safety
   // and returning the normal from this face.

   // TODO: Before doing that we could check for cached points from other queries

   // use safety kernel
   int closest_face_id = -1;
   /*double saf = */ SafetyKernel<true>(point, true, &closest_face_id);

   if (closest_face_id < 0) {
      norm[0] = 1.;
      norm[1] = 0.;
      norm[2] = 0.;
      return;
   }

   const auto &n = fOutwardNormals[closest_face_id];
   norm[0] = n[0];
   norm[1] = n[1];
   norm[2] = n[2];

   // change sign depending on dir
   if (norm[0] * dir[0] + norm[1] * dir[1] + norm[2] * dir[2] < 0) {
      norm[0] = -norm[0];
      norm[1] = -norm[1];
      norm[2] = -norm[2];
   }
   return;
}

////////////////////////////////////////////////////////////////////////////////
/// trivial (non-BVH) DistFromInside function

Double_t TGeoTessellated::DistFromInside_Loop(const Double_t *point, const Double_t *dir) const
{
   // Bias the starting point slightly along the direction
   Vertex_t p(point[0], point[1], point[2]);
   Vertex_t d(dir[0], dir[1], dir[2]);

   double dist = Big();
   for (size_t i = 0; i < fFacets.size(); ++i) {
      const auto &facet = fFacets[i];
      const auto &n = fOutwardNormals[i];

      // Only exiting surfaces are relevant (from inside--> dot product must be positive)
      if (n[0] * dir[0] + n[1] * dir[1] + n[2] * dir[2] <= 0.0) {
         continue;
      }

      const auto &v0 = fVertices[facet[0]];
      const auto &v1 = fVertices[facet[1]];
      const auto &v2 = fVertices[facet[2]];

      const double t = rayTriangle(p, d, v0, v1, v2, 0.);

      if (t < dist) {
         dist = t;
      }
   }
   return dist;
}

////////////////////////////////////////////////////////////////////////////////
/// trivial (non-BVH) DistFromOutside function

Double_t TGeoTessellated::DistFromOutside_Loop(const Double_t *point, const Double_t *dir) const
{
   // Bias the starting point slightly along the direction
   Vertex_t p(point[0], point[1], point[2]);
   Vertex_t d(dir[0], dir[1], dir[2]);

   double dist = Big();
   for (size_t i = 0; i < fFacets.size(); ++i) {
      const auto &facet = fFacets[i];
      const auto &n = fOutwardNormals[i];

      // Only exiting surfaces are relevant (from outside, the dot product must be negative)
      if (n[0] * dir[0] + n[1] * dir[1] + n[2] * dir[2] > 0.0) {
         continue;
      }

      const auto &v0 = fVertices[facet[0]];
      const auto &v1 = fVertices[facet[1]];
      const auto &v2 = fVertices[facet[2]];

      const double t = rayTriangle(p, d, v0, v1, v2, 0.);

      if (t < dist) {
         dist = t;
      }
   }
   return dist;
}

////////////////////////////////////////////////////////////////////////////////
/// trivial (non-BVH) Contains

bool TGeoTessellated::Contains_Loop(const Double_t *point) const
{
   // Fixed ray direction
   const Vertex_t test_dir{1.0, 1.41421356237, 1.73205080757};

   // Bias point slightly along ray to avoid t == 0 hits
   Vertex_t p(point[0], point[1], point[2]);

   int crossings = 0;
   for (size_t i = 0; i < fFacets.size(); ++i) {
      const auto &facet = fFacets[i];

      const auto &v0 = fVertices[facet[0]];
      const auto &v1 = fVertices[facet[1]];
      const auto &v2 = fVertices[facet[2]];

      const double t = rayTriangle(p, test_dir, v0, v1, v2, 0.);
      if (t != std::numeric_limits<double>::infinity()) {
         ++crossings;
      }
   }
   return (crossings & 1);
}

////////////////////////////////////////////////////////////////////////////////
/// Custom streamer which performs Closing on read.
/// Recalculation of BVH and normals is fast

void TGeoTessellated::Streamer(TBuffer &b)
{
   if (b.IsReading()) {
      b.ReadClassBuffer(TGeoTessellated::Class(), this);
      CloseShape();
   } else {
      b.WriteClassBuffer(TGeoTessellated::Class(), this);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate the normals

void TGeoTessellated::CalculateNormals()
{
   fOutwardNormals.clear();
   for (auto &facet : fFacets) {
      auto &v1 = fVertices[facet[0]];
      auto &v2 = fVertices[facet[1]];
      auto &v3 = fVertices[facet[2]];
      using Vec3d = Vec3f<double>;
      auto norm = triangleNormal(Vec3d{v1[0], v1[1], v1[2]}, Vec3d{v2[0], v2[1], v2[2]}, Vec3d{v3[0], v3[1], v3[2]});
      fOutwardNormals.emplace_back(Vertex_t{norm.x, norm.y, norm.z});
   }
}
