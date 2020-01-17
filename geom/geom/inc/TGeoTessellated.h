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

#include "TGeoVector3.h"
#include "TGeoBBox.h"

class TGeoFacet {
   using Vertex_t    = TGeoVector3;
   using VertexVec_t = std::vector<Vertex_t>;

private:
   Int_t     fIvert[4]    = {0};       // Vertex indices in the array
   VertexVec_t *fVertices = nullptr;   //! array of vertices
   int       fNvert       = 0;         // number of vertices (can be 3 or 4)
   bool      fShared      = false;     // Vector of vertices shared flag

public:
   TGeoFacet() {}
   TGeoFacet(const TGeoFacet &other);

   ~TGeoFacet() { if (!fShared) delete fVertices; }

   const TGeoFacet &operator = (const TGeoFacet &other);

   // Triangular facet
   TGeoFacet(const TGeoVector3 &pt0, const TGeoVector3 &pt1, const TGeoVector3 &pt2)
      : fIvert{0, 1, 2}
   {
      fVertices = new VertexVec_t;
      fVertices->push_back(pt0);
      fVertices->push_back(pt1);
      fVertices->push_back(pt2);
      fNvert = 3;
   }

   // Quadrilateral facet
   TGeoFacet(const TGeoVector3 &pt0, const TGeoVector3 &pt1, const TGeoVector3 &pt2, const TGeoVector3 &pt3)
      : fIvert{0, 1, 2, 3}
   {
      fVertices = new VertexVec_t;
      fVertices->push_back(pt0);
      fVertices->push_back(pt1);
      fVertices->push_back(pt2);
      fVertices->push_back(pt3);      
      fNvert = 4;
   }
   
   void SetVertices(VertexVec_t *vertices, int i0 = -1, int i1 = -1, int i2 = -1, int i3 = -1)
   {
      if (!fShared) delete fVertices;
      fVertices = vertices;
      if (i0 >= 0) {
         fIvert[0] = i0;
         fIvert[1] = i1;
         fIvert[2] = i2;
         fIvert[3] = i3;
      }
      fShared   = true;
   }
   
   TGeoVector3 ComputeNormal(bool &degenerated) const;
   Int_t GetNvert() const { return fNvert; }

   Vertex_t &GetVertex(int ivert) { return fVertices->operator[](fIvert[ivert]); }
   const Vertex_t &GetVertex(int ivert) const { return fVertices->operator[](fIvert[ivert]); }

   Int_t GetVertexIndex(int ivert) const { return fIvert[ivert]; }
   
   bool Check() const;
   bool CheckNeighbour(const TGeoFacet $other) const;
};

std::ostream &operator<<(std::ostream &os, TGeoFacet const &facet);

class TGeoTessellated : public TGeoBBox
{
  using Vertex_t = TGeoVector3;

private:
   int fNfacets = 0;
   int fNvert   = 0;
   int fNseg    = 0;
   std::vector<Vertex_t>  fVertices;        // List of vertices
   std::vector<TGeoFacet> fFacets;          // List of facets

   virtual void FillBuffer3D(TBuffer3D & buffer, Int_t reqSections, Bool_t localFrame) const;
public:
   // constructors
   TGeoTessellated() {}
   TGeoTessellated(const char *name, Int_t nfacets);
   // destructor
   virtual ~TGeoTessellated() {}

   TGeoTessellated(const TGeoTessellated&);
   TGeoTessellated& operator=(const TGeoTessellated&);
   
   void ComputeBBox();
   void Close();

   void AddFacet(const TGeoVector3 &pt0, const TGeoVector3 &pt1, const TGeoVector3 &pt2);
   void AddFacet(const TGeoVector3 &pt0, const TGeoVector3 &pt1, const TGeoVector3 &pt2, const TGeoVector3 &pt3);

   int  GetNfacets() const { return fFacets.size(); }
   int  GetNsegments() const { return fNseg; }
   int  GetNvertices() const { return fNvert; }
   const TGeoFacet &GetFacet(Int_t i) { return fFacets[i]; }

   virtual void          AfterStreamer();
   virtual Int_t         DistancetoPrimitive(Int_t, Int_t) { return 99999; }
   virtual const TBuffer3D &GetBuffer3D(Int_t reqSections, Bool_t localFrame) const;
   virtual void          GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const;
   virtual Int_t         GetNmeshVertices() const { return fNvert; }
   virtual void          InspectShape() const {}
   virtual TBuffer3D    *MakeBuffer3D() const;
   virtual void          SavePrimitive(std::ostream &, Option_t *) {}
   virtual void          SetPoints(Double_t *points) const;
   virtual void          SetPoints(Float_t *points) const;
   virtual void          SetSegsAndPols(TBuffer3D &buff) const;
   virtual void          Sizeof3D() const {}
   
   ClassDef(TGeoTessellated, 1)         // tessellated shape class
};

#endif
