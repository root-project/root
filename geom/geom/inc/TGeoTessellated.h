// @(#)root/geom:$Id$
// Author: Andrei Gheata   20/12/19

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoTessellated
#define ROOT_TGeoTessellated

#include <map>
#include "TGeoVector3.h"
#include "TGeoTypedefs.h"
#include "TGeoBBox.h"

class TGeoFacet {
   using Vertex_t = Tessellated::Vertex_t;
   using VertexVec_t = Tessellated::VertexVec_t;

private:
   int fIvert[4] = {0, 0, 0, 0};     // Vertex indices in the array
   int fNvert = 0;                   // number of vertices (can be 3 or 4)

private:
   void SetVertices(int nvert = 0, int i0 = -1, int i1 = -1, int i2 = -1, int i3 = -1)
   {
      fNvert = nvert;
      fIvert[0] = i0;
      fIvert[1] = i1;
      fIvert[2] = i2;
      fIvert[3] = i3;
   }

public:
   TGeoFacet() {}
   TGeoFacet(int i0, int i1, int i2) { SetVertices(3, i0, i1, i2); }
   TGeoFacet(int i0, int i1, int i2, int i3) { SetVertices(4, i0, i1, i2, i3); }

   int operator[](int ivert) const { return fIvert[ivert]; }
   static int CompactFacet(Vertex_t *vert, int nvertices);
   Vertex_t ComputeNormal(bool &degenerated) const;
   int GetNvert() const { return fNvert; }

   void Flip()
   {
      int iv = fIvert[0];
      fIvert[0] = fIvert[2];
      fIvert[2] = iv;
   }
   bool IsNeighbour(const TGeoFacet &other, bool &flip) const;
};

class TGeoTessellated : public TGeoBBox {

public:
   using Vertex_t = Tessellated::Vertex_t;

private:
   int fNfacets = 0;                // Number of facets
   int fNvert = 0;                  // Number of vertices
   int fNseg = 0;                   // Number of segments
   bool fDefined = false;           //! Shape fully defined
   bool fClosedBody = false;        // The faces are making a closed body
   std::vector<Vertex_t> fVertices; // List of vertices
   std::vector<TGeoFacet> fFacets;  // List of facets
   std::multimap<long, int> fVerticesMap; //! Temporary map used to deduplicate vertices

   TGeoTessellated(const TGeoTessellated &) = delete;
   TGeoTessellated &operator=(const TGeoTessellated &) = delete;

public:
   // constructors
   TGeoTessellated() {}
   TGeoTessellated(const char *name, int nfacets = 0);
   TGeoTessellated(const char *name, const std::vector<Vertex_t> &vertices);
   // destructor
   ~TGeoTessellated() override {}

   void ComputeBBox() override;
   void CloseShape(bool check = true, bool fixFlipped = true, bool verbose = true);

   bool AddFacet(const Vertex_t &pt0, const Vertex_t &pt1, const Vertex_t &pt2);
   bool AddFacet(const Vertex_t &pt0, const Vertex_t &pt1, const Vertex_t &pt2, const Vertex_t &pt3);
   bool AddFacet(int i1, int i2, int i3);
   bool AddFacet(int i1, int i2, int i3, int i4);
   int AddVertex(const Vertex_t &vert);

   bool FacetCheck(int ifacet) const;
   Vertex_t FacetComputeNormal(int ifacet, bool &degenerated) const;

   int GetNfacets() const { return fFacets.size(); }
   int GetNsegments() const { return fNseg; }
   int GetNvertices() const { return fNvert; }
   bool IsClosedBody() const { return fClosedBody; }
   bool IsDefined() const { return fDefined; }

   const TGeoFacet &GetFacet(int i) const { return fFacets[i]; }
   const Vertex_t &GetVertex(int i) const { return fVertices[i]; }

   int DistancetoPrimitive(int, int) override { return 99999; }
   const TBuffer3D &GetBuffer3D(int reqSections, Bool_t localFrame) const override;
   void GetMeshNumbers(int &nvert, int &nsegs, int &npols) const override;
   int GetNmeshVertices() const override { return fNvert; }
   void InspectShape() const override {}
   TBuffer3D *MakeBuffer3D() const override;
   void Print(Option_t *option = "") const override;
   void SavePrimitive(std::ostream &, Option_t *) override {}
   void SetPoints(double *points) const override;
   void SetPoints(Float_t *points) const override;
   void SetSegsAndPols(TBuffer3D &buff) const override;
   void Sizeof3D() const override {}

   /// Resize and center the shape in a box of size maxsize
   void ResizeCenter(double maxsize);

   /// Flip all facets
   void FlipFacets()
   {
      for (auto facet : fFacets)
         facet.Flip();
   }

   bool CheckClosure(bool fixFlipped = true, bool verbose = true);

   /// Reader from .obj format
   static TGeoTessellated *ImportFromObjFormat(const char *objfile, bool check = false, bool verbose = false);

   ClassDefOverride(TGeoTessellated, 1) // tessellated shape class
};

#endif
