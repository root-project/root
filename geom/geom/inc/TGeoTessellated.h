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

struct TGeoFacet {
   using Vertex_t = TGeoVector3;
   Vertex_t fVertices[4];   // array of vertices
   int      fNvert = 0;     // number of vertices (can be 3 or 4)

   TGeoFacet() {}
   // Triangular facet
   TGeoFacet(double x0, double y0, double z0,
             double x1, double y1, double z1,
             double x2, double y2, double z2)
   {
      fVertices[0].Set(x0, y0, z0);
      fVertices[1].Set(x1, y1, z1);
      fVertices[2].Set(x2, y2, z2);
      fNvert = 3;
   }

   // Quadrilateral facet
   TGeoFacet(double x0, double y0, double z0,
             double x1, double y1, double z1,
             double x2, double y2, double z2,
             double x3, double y3, double z3)
   {
      fVertices[0].Set(x0, y0, z0);
      fVertices[1].Set(x1, y1, z1);
      fVertices[2].Set(x2, y2, z2);
      fVertices[3].Set(x3, y3, z3);
      fNvert = 4;
   }

   //bool Check();
};

class TGeoTessellated : public TGeoBBox
{
  using Vector3_t = TGeoVector3;

private:
   int fNfacets = 0;
   int fNvert   = 0;
   std::vector<TGeoFacet> fFacets;

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
   void Close() { ComputeBBox(); }

   void AddFacet(double x0, double y0, double z0,
                 double x1, double y1, double z1,
                 double x2, double y2, double z2);
   void AddFacet(double x0, double y0, double z0,
                 double x1, double y1, double z1,
                 double x2, double y2, double z2,
                 double x3, double y3, double z3);
   int  GetNfacets() const { return fFacets.size(); }
   int  GetNvertices() const { return fNvert; }
   const TGeoFacet &GetFacet(Int_t i) { return fFacets[i]; }

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
