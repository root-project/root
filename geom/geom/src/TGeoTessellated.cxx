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

#include <array>
#include <vector>

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

   if (fNfacets > 0 && GetNfacets() == fNfacets)
      CloseShape();
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
   // Compute bounding box
   fDefined = true;
   fNvert = fVertices.size();
   fNfacets = fFacets.size();
   ComputeBBox();
   // Cleanup the vertex map
   std::multimap<long, int>().swap(fVerticesMap);

   if (fVertices.size() > 0) {
      if (!check)
         return;

      // Check facets
      for (auto i = 0; i < fNfacets; ++i)
         FacetCheck(i);

      fClosedBody = CheckClosure(fixFlipped, verbose);
   }
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
