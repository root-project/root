/// \file
/// \ingroup tutorial_geom
/// Macro allowing to vizualize tessellations from Wavefront's .obj format.
///
/// \image html geom_visualizeWavefrontObj.png width=500px
/// \macro_code
///
/// \author Andrei Gheata

#include <TROOT.h>
#include <TColor.h>
#include <TDatime.h>
#include <TRandom3.h>
#include <TGeoManager.h>
#include <TGeoTessellated.h>
#include <Tessellated/TGeoMeshLoading.h>
#include <TVirtualGeoConverter.h>
#include <TView.h>

//______________________________________________________________________________
int randomColor()
{
   gRandom = new TRandom3();
   TDatime dt;
   gRandom->SetSeed(dt.GetTime());
   int ci = TColor::GetFreeColorIndex();
   TColor *color = new TColor(ci, gRandom->Rndm(), gRandom->Rndm(), gRandom->Rndm());
   return ci;
}

//______________________________________________________________________________
void tessellatedNav(const char *dot_obj_file = "", bool check = true, int mode = 1)
{
   // Input a file in .obj format (https://en.wikipedia.org/wiki/Wavefront_.obj_file)
   // The file should have a single object inside, only vertex and faces information is used
   TString name = dot_obj_file;
   TString sfile = dot_obj_file;
   if (sfile.IsNull()) {
      sfile = gROOT->GetTutorialsDir();
      sfile += "/visualisation/geom/teddy.obj";
   }
   name.ReplaceAll(".obj", "");
   gROOT->GetListOfCanvases()->Delete();
   if (gGeoManager)
      delete gGeoManager;
   auto geom = new TGeoManager(name, "Imported from .obj file");
   TGeoMaterial *mat = new TGeoMaterial("Al", 26.98, 13, 2.7);
   TGeoMedium *med = new TGeoMedium("MED", 1, mat);
   TGeoVolume *top = geom->MakeBox("TOP", med, 10, 10, 10);
   geom->SetTopVolume(top);

   //Creating mesh is now a little more verbose, as there are several options to create it from, which is why it had to be moved out from
   //TGeoTessellated
   auto mesh = Tessellated::ImportMeshFromObjFormat(sfile.Data(), Tessellated::TGeoTriangleMesh::LengthUnit::kCentiMeter);
   // auto mesh = Tessellated::ImportMeshFromASCIIStl(
      // "BoxKristall_12.stl", Tessellated::TGeoTriangleMesh::LengthUnit::kMilliMeter);
   if (!mesh) {
      return;
   }
   if (check) {
      bool fixTriangleOrientation = true;
      bool verbose = false;
      mesh->CheckClosure(fixTriangleOrientation, verbose);
   }

   // mesh->ResizeCenter(10.);

   auto tsl = new TGeoTessellated(); 
   tsl->SetMesh(std::move(mesh));
   tsl->InspectShape();

   tsl->ResizeCenter(5.);

   if (mode == 1) {
      std::unique_ptr<Tessellated::TPartitioningI>octree{Tessellated::TOctree::CreateOctree(tsl, 5, 1, true)};
      tsl->SetPartitioningStruct(octree);
   } else if (mode == 2) {
      std::unique_ptr<Tessellated::TPartitioningI> bvh{new Tessellated::TBVH()};
      bvh->SetTriangleMesh(tsl->GetTriangleMesh());
      tsl->SetPartitioningStruct(bvh);
   } else {
      // You run without a partitioning structure. Perfectly fine, but ssssslllllooooowwwww, besides for meshes with only a few triangles < 100, than it can be
      // faster than with a paritioning structure.
   }
   tsl->InspectShape();

   TGeoVolume *vol = new TGeoVolume(name, tsl, med);
   vol->SetLineColor(randomColor());
   vol->SetLineWidth(2);
   top->AddNode(vol, 1);
   geom->CloseGeometry();

   if (gROOT->IsBatch())
      return;
   // Set the view
   top->Draw();
   TView *view = gPad->GetView();
   if (!view)
      return;
   view->Top();

   // Raytracing will call VecGeom navigation
   top->Raytrace();
}
