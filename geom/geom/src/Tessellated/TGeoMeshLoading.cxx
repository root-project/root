// @(#)root/geom:$Id$// Author: Ben Salisbury   21/11/24

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \namespace TGeoMeshLoading
\ingroup Geometry_classes

Namespace to contain helper functionality to import mesh data into a TGeoTriangleMesh
object for use with TGeoTessellated.
*/

#include "Tessellated/TGeoMeshLoading.h"

#include <cstdlib>  // for exit
#include <array>    // for array
#include <fstream>  // for operator<<, basic_ifstream, basic_istream
#include <iostream> // for cerr, cout
#include <map>      // for map, _Rb_tree_iterator
#include <sstream>  // for stringstream
#include <string>   // for string, operator+, stod, basic_string, char_t...
#include <vector>   // for vector

#include "Tessellated/TGeoTriangle.h" // for TGeoTriangle
#include "TVector3.h"  // for TVector3

namespace Tessellated {
namespace ASCIIStl {

////////////////////////////////////////////////////////////////////////////////
/// Given a unit specifier, return the conversion factor from cm to required value

Double_t GetScaleFactor(const TGeoTriangleMesh::LengthUnit unit)
{
   switch (unit) {
   case TGeoTriangleMesh::LengthUnit::kMilliMeter: return 0.1;
   case TGeoTriangleMesh::LengthUnit::kCentiMeter: return 1;
   case TGeoTriangleMesh::LengthUnit::kMeter: return 100;
   }
   return 1;
}

////////////////////////////////////////////////////////////////////////////////
/// Test if a key (alias a string "x;y;z" with x,y,z being the vertex elements (doubles))
/// has already been entered in the pointmap. If so, return the index, else create
/// entry and return that index

UInt_t LookupVertexIndex(const std::string &key, std::map<std::string, size_t> &pointmap, Bool_t &newpoint)
{
   auto pos = pointmap.find(key);
   newpoint = false;

   if (pos == pointmap.end()) {
      size_t mapsize = pointmap.size();
      newpoint = true;
      pointmap[key] = mapsize;
   }
   return static_cast<unsigned int>(pointmap[key]);
}

////////////////////////////////////////////////////////////////////////////////
/// Convert a std::string containing 3 vertex elements into a "hash"

std::string VertexCoordinatesAsString(std::string line)
{
   auto found = line.find_first_not_of("abcdefghijklmnopqrstuvwxyz ");
   if (found != std::string::npos) {
      line.erase(0, found);
   }

   found = line.find_first_of(' ');
   auto xcoords = line.substr(0, found);
   line.erase(0, found);

   found = line.find_first_not_of(' ');
   line.erase(0, found);
   found = line.find_first_of(' ');
   auto ycoords = line.substr(0, found);
   line.erase(0, found);

   found = line.find_first_of(' ');
   line.erase(0, found);
   auto zcoords = line;

   return xcoords + ';' + ycoords + ';' + zcoords;
}

////////////////////////////////////////////////////////////////////////////////
/// Convert a std::string containing 3 vertex elements into a TVector3

TVector3 Vector3FromString(const std::string &coordinatestring, const Double_t scale)
{
   auto foundx = coordinatestring.find_first_of(';');
   auto xcoords = coordinatestring.substr(0, foundx);
   auto x = std::stod(xcoords);

   auto foundy = coordinatestring.find_last_of(';');
   auto ycoords = coordinatestring.substr(foundx + 1, foundy);
   auto y = std::stod(ycoords);

   auto zcoords = coordinatestring.substr(foundy + 1, std::string::npos);
   auto z = std::stod(zcoords);

   return TVector3{x * scale, y * scale, z * scale};
}

}; // namespace ASCIIStl

////////////////////////////////////////////////////////////////////////////////
/// Helper function to create string out of TVector3

std::string Vector3ToString(const TVector3 &vect)
{
   return std::to_string(vect.X()) + ";" + std::to_string(vect.Y()) + ";" + std::to_string(vect.Z());
}

////////////////////////////////////////////////////////////////////////////////
/// Import triangle mesh in form of a stl file
///
///\param[in] stlfile ASCII stl filename containing the mesh
///\param[in] unit    length unit to be used
///                                 TGeoTriangleMesh::LengthUnit::kMilliMeter
///                                 TGeoTriangleMesh::LengthUnit::kCentiMeter
///                                 TGeoTriangleMesh::LengthUnit::kMeter
///\return std::unique_ptr<TGeoTriangleMesh>
///

std::unique_ptr<TGeoTriangleMesh> ImportMeshFromASCIIStl(const TString &stlfilename, const TGeoTriangleMesh::LengthUnit unit)
{
   const auto scale = ASCIIStl::GetScaleFactor(unit);
   std::unique_ptr<TGeoTriangleMesh> mesh{new TGeoTriangleMesh(stlfilename)};

   auto stlfile = std::ifstream(stlfilename.Data());
   std::vector<TGeoTriangle> triangles{};
   std::vector<TVector3> points{};

   if (stlfile.is_open()) {
      std::map<std::string, size_t> pointmap{};
      UInt_t runningindex{0U};
      std::array<UInt_t, 3> indices{};
      std::string line{""};
      while (getline(stlfile, line)) {
         auto vertexline = line.find("vertex");
         if (vertexline != std::string::npos) {
            Bool_t newpoint{kFALSE};
            const std::string &key = ASCIIStl::VertexCoordinatesAsString(line);
            const UInt_t index = ASCIIStl::LookupVertexIndex(key, pointmap, newpoint);
            indices[runningindex % 3] = index;
            if (newpoint) {
               points.push_back(ASCIIStl::Vector3FromString(key, scale));
            }
            if (runningindex % 3 == 2) {

               triangles.emplace_back(indices);
            }
            ++runningindex;
         }
      }
      stlfile.close();
   } else {
      std::cerr << "File " << stlfilename << " does not exist. Exiting." << std::endl;
      std::exit(-1);
   }

   mesh->SetPoints(points);
   mesh->SetTriangles(triangles);
   mesh->SetupTriangles();
   return mesh;
}

////////////////////////////////////////////////////////////////////////////////
/// Reads a single tessellated solid from an .obj file.

std::unique_ptr<TGeoTriangleMesh> ImportMeshFromObjFormat(const char *objfile, const TGeoTriangleMesh::LengthUnit unit)
{
   const auto scale = ASCIIStl::GetScaleFactor(unit);
   std::unique_ptr<TGeoTriangleMesh> mesh{new TGeoTriangleMesh(objfile)};

   std::vector<std::string> sfacets;

   std::vector<TVector3> vertices;

   std::vector<TGeoTriangle> facets;

   std::map<std::string, UInt_t> stringToIndex{};
   std::map<UInt_t, UInt_t> indexToGlobalIndexMapping{};
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

   std::string line;
   int ind[4] = {0};
   std::ifstream file(objfile);
   if (!file.is_open()) {
      ::Error("Tessellated::ImportMeshFromObjFormat", "Unable to open %s", objfile);
      return nullptr;
   }
   Int_t currentIndex{0};
   while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string tag;

      // We ignore everything which is not a vertex or a face
      if (line.rfind('v', 0) == 0 && line.rfind("vt", 0) != 0 && line.rfind("vn", 0) != 0 && line.rfind("vn", 0) != 0) {
         // Decode the vertex
         double pos[4] = {0, 0, 0, 1};
         ss >> tag >> pos[0] >> pos[1] >> pos[2] >> pos[3];
         const std::string key = std::to_string(pos[0] * pos[3] * scale) + ';' + std::to_string(pos[1] * pos[3] * scale) + ';' +
                                 std::to_string(pos[2] * pos[3] * scale);
         if (stringToIndex.find(key) == stringToIndex.end()) {
            stringToIndex[key] = currentIndex;
            vertices.emplace_back(pos[0] * pos[3] * scale, pos[1] * pos[3] * scale, pos[2] * pos[3] * scale);
         }
         indexToGlobalIndexMapping[currentIndex] = stringToIndex[key];
         currentIndex++;
      }

      else if (line.rfind('f', 0) == 0) {
         // Decode the face
         ss >> tag;
         std::string word;
         sfacets.clear();
         while (ss >> word)
            sfacets.push_back(word);
         if (sfacets.size() > 4 || sfacets.size() < 3) {
            ::Error("Tessellated::ImportMeshFromObjFormat", "Detected face having unsupported %zu vertices",
                    sfacets.size());
            return nullptr;
         }
         int nvert = 0;
         for (auto &sword : sfacets) {
            std::stringstream ssword(sword);
            std::string token;
            std::getline(ssword, token, '/'); // just need the vertex index, which is the first token
            // Convert string token to integer

            ind[nvert++] = std::stoi(token) - 1;
            if (ind[nvert - 1] < 0) {
               ::Error("Tessellated::ImportMeshFromObjFormat", "Unsupported relative vertex index definition in %s",
                       objfile);
               return nullptr;
            }
         }
         if (nvert == 3) {
            facets.emplace_back(std::array<UInt_t, 3>{indexToGlobalIndexMapping[ind[0]],
                                                      indexToGlobalIndexMapping[ind[1]],
                                                      indexToGlobalIndexMapping[ind[2]]});
         } else {
            facets.emplace_back(std::array<UInt_t, 3>{indexToGlobalIndexMapping[ind[0]],
                                                      indexToGlobalIndexMapping[ind[1]],
                                                      indexToGlobalIndexMapping[ind[2]]});
            facets.emplace_back(std::array<UInt_t, 3>{indexToGlobalIndexMapping[ind[0]],
                                                      indexToGlobalIndexMapping[ind[2]],
                                                      indexToGlobalIndexMapping[ind[3]]});
         }
      }
   }

   int nfacets = (int)facets.size();
   if (nfacets < 3) {
      ::Error("Tessellated::ImportMeshFromObjFormat", "Not enough faces detected in %s", objfile);
      return nullptr;
   }
   mesh->SetPoints(vertices);
   mesh->SetTriangles(facets);
   mesh->SetupTriangles();
   return mesh;
}

////////////////////////////////////////////////////////////////////////////////
/// Reset the MeshBuilder internal state to define next triangle mesh

void MeshBuilder::Reset()
{
   fStringToIndex.clear();
   fVertices.clear();
   fTriangles.clear();
   fCounter = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Add a vertex to the mesh. Create a string to look up if vertex already known
/// \param[in] vertex to be added
/// \return UInt_t indicating vertex index

UInt_t MeshBuilder::AddVertex(const TVector3 &vertex)
{
   const std::string &strOfVertex = Vector3ToString(vertex);
   if (fStringToIndex.find(strOfVertex) == fStringToIndex.end()) {
      fVertices.push_back(vertex);
      fStringToIndex[strOfVertex] = fCounter++;
   }
   return fStringToIndex[strOfVertex];
}

////////////////////////////////////////////////////////////////////////////////
/// Add a Vertex_t vertex to the mesh. 
/// \param[in] vertex to be added
/// \return UInt_t indicating vertex index

UInt_t MeshBuilder::AddVertex(const Vertex_t &vertex)
{
   return AddVertex(TVector3FromVertex_t(vertex));
}


////////////////////////////////////////////////////////////////////////////////
/// Check if a vertex is existent. Create a string to look up if vertex is
/// already known
/// \param[in] vertex to be looked up
/// \return Int_t -1 for when vertex is not included, else the vertex index

Int_t MeshBuilder::LookupVertex(const TVector3 &vertex) const
{
   const std::string &strOfVertex = Vector3ToString(vertex);
   if (fStringToIndex.find(strOfVertex) == fStringToIndex.end()) {
      return -1;
   }
   return fStringToIndex.at(strOfVertex);
}

////////////////////////////////////////////////////////////////////////////////
/// Add a triangular face providing the vertices directly

void MeshBuilder::AddFacet(const TVector3 &v0, const TVector3 &v1, const TVector3 &v2)
{
   std::array<UInt_t, 3> indices;
   indices[0] = AddVertex(v0);
   indices[1] = AddVertex(v1);
   indices[2] = AddVertex(v2);
   fTriangles.emplace_back(indices);
}

////////////////////////////////////////////////////////////////////////////////
/// Add a triangular face providing the vertices directly

void MeshBuilder::AddFacet(const Vertex_t &v0, const Vertex_t &v1, const Vertex_t &v2)
{
   AddFacet(TVector3FromVertex_t(v0), TVector3FromVertex_t(v1), TVector3FromVertex_t(v2));
}


////////////////////////////////////////////////////////////////////////////////
/// Add a triangular face providing the indices of the vertices (as returned from
/// MeshBuilder::AddVertex)

void MeshBuilder::AddFacet(const UInt_t v0, const UInt_t v1, const UInt_t v2)
{
   std::array<UInt_t, 3> indices{v0, v1, v2};
   fTriangles.emplace_back(indices);
}

////////////////////////////////////////////////////////////////////////////////
/// Add a quadrilateral face providing the vertices directly
/// Note, that 2 triangles are made out of the single quadrilateral

void MeshBuilder::AddFacet(const TVector3 &v0, const TVector3 &v1, const TVector3 &v2, const TVector3 &v3)
{
   AddFacet(v0, v1, v2);
   AddFacet(v0, v2, v3);
}

////////////////////////////////////////////////////////////////////////////////
/// Add a quadrilateral face providing the vertices directly
/// Note, that 2 triangles are made out of the single quadrilateral


void MeshBuilder::AddFacet(const Vertex_t &v0, const Vertex_t &v1, const Vertex_t &v2, const Vertex_t &v3)
{
   AddFacet(TVector3FromVertex_t(v0), TVector3FromVertex_t(v1), TVector3FromVertex_t(v2), TVector3FromVertex_t(v3));
}


////////////////////////////////////////////////////////////////////////////////
/// Add a quadrilateral face providing the indices of the vertices (as returned from
/// MeshBuilder::AddVertex).
/// Note, that 2 triangles are made out of the single quadrilateral

void MeshBuilder::AddFacet(const UInt_t v0, const UInt_t v1, const UInt_t v2, const UInt_t v3)
{
   AddFacet(v0, v1, v2);
   AddFacet(v0, v2, v3);
}

////////////////////////////////////////////////////////////////////////////////
/// Return the Mesh as defined by the AddVertex/AddFace calls since last
/// MeshBuilder::Reset() call

std::unique_ptr<TGeoTriangleMesh> MeshBuilder::CreateMesh() const
{
   std::unique_ptr<TGeoTriangleMesh> mesh{new TGeoTriangleMesh("Created manually using the Tessellated::MeshBuilder")};
   mesh->SetPoints(fVertices);
   mesh->SetTriangles(fTriangles);
   mesh->SetupTriangles();
   return mesh;
}

////////////////////////////////////////////////////////////////////////////////
/// Helper function to create a triangular mesh for a box with halflengths
/// dx,dy,dz centered at (0,0,0)

// std::unique_ptr<TGeoTriangleMesh> CreateBoxMesh(const Double_t dx, const Double_t dy, const Double_t dz)
// {
//    MeshBuilder builder;
//    builder.AddFacet(
//       TVector3{},
//       TVector3{},
//       TVector3{},
//    )
// }

}; // namespace Tessellated