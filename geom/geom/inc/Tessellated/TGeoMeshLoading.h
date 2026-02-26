// @(#)root/geom:$Id$// Author: Ben Salisbury   21/11/24

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef TGEOMESHLOADING_HH
#define TGEOMESHLOADING_HH

#include <map>    // for map
#include <memory> // for unique_ptr
#include <string> // for string

#include "RtypesCore.h" // for Double_t, UInt_t, Bool_t

#include "TString.h"          // for TString
#include "TGeoTriangleMesh.h" // for TGeoTriangleMesh, TGeoTriangleMesh::LengthUnit
#include "Math/Vector3D.h"    // for ROOT::Math::XYZVector
#include "TGeoTypedefs.h"     // for Vertex_t

namespace Tessellated {

namespace ASCIIStl {

Double_t GetScaleFactor(const TGeoTriangleMesh::LengthUnit unit);
ROOT::Math::XYZVector Vector3FromString(const std::string &coordinatestring, const Double_t scale);
std::string VertexCoordinatesAsString(std::string line);
UInt_t LookupVertexIndex(const std::string &key, std::map<std::string, size_t> &pointmap, Bool_t &newpoint);

}; // namespace ASCIIStl

// common helper functions
std::string Vector3ToString(const ROOT::Math::XYZVector &vect);

// Helper function to import full meshes from files (.stl, .obj)
std::unique_ptr<TGeoTriangleMesh>
ImportMeshFromASCIIStl(const TString &t_filename, TGeoTriangleMesh::LengthUnit t_unit);
std::unique_ptr<TGeoTriangleMesh> ImportMeshFromObjFormat(const char *objfile, const TGeoTriangleMesh::LengthUnit unit);

/// Helper class to create triangular meshes by hand
class MeshBuilder {
private:
   std::map<std::string, UInt_t> fStringToIndex{};
   std::vector<ROOT::Math::XYZVector> fVertices;
   std::vector<TGeoTriangle> fTriangles;
   UInt_t fCounter{0};

   inline static ROOT::Math::XYZVector ToVector3D(const Vertex_t &vertex)
   {
      return ROOT::Math::XYZVector{vertex.x(), vertex.y(), vertex.z()};
   }

public:
   void Reset();
   UInt_t AddVertex(const ROOT::Math::XYZVector &vertex);
   UInt_t AddVertex(const Vertex_t &vertex);
   Int_t LookupVertex(const ROOT::Math::XYZVector &vertex) const;
   void AddFacet(const ROOT::Math::XYZVector &v0, const ROOT::Math::XYZVector &v1, const ROOT::Math::XYZVector &v2);
   void AddFacet(const Vertex_t &v0, const Vertex_t &v1, const Vertex_t &v2);
   void AddFacet(const UInt_t v0, const UInt_t v1, const UInt_t v2);
   void AddFacet(const ROOT::Math::XYZVector &v0, const ROOT::Math::XYZVector &v1, const ROOT::Math::XYZVector &v2,
                 const ROOT::Math::XYZVector &v3);
   void AddFacet(const Vertex_t &v0, const Vertex_t &v1, const Vertex_t &v2, const Vertex_t &v3);
   void AddFacet(const UInt_t v0, const UInt_t v1, const UInt_t v2, const UInt_t v3);

   std::unique_ptr<TGeoTriangleMesh> CreateMesh() const;
};

}; // namespace Tessellated

#endif /*TGEOMESHLOADING_HH*/
