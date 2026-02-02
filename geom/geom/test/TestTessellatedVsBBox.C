// Script to compare the navigation outcome of TGeoTessellated representing a box and TGeoBBox 
// with a few randomly sampled points and directions. Boxes may be simple, but we can compare them 
// 1:1 with TGeoBBox. Anything containing curvatures (cylinders, etc.) and the approximative nature
// of the mesh to TGeoTube will ruin the comparison. Ideally, a full geant3/4_vmc simulation should
// be used for this comparison purpose, but that may be a little ambitious here.  

Bool_t isSame(Double_t a, Double_t b, Double_t epsilon = 1e-6)
{
   return (std::abs(a - b) < epsilon);
}

struct TestResult_t {
   Bool_t contains{kFALSE};
   Double_t distance_in_dir{-1};
   Double_t safety{-1};
   Double_t norm[3] = {0, 0, 0};

   Bool_t operator==(const TestResult_t &other) const
   {
      return other.contains == contains && isSame(other.distance_in_dir, distance_in_dir) &&
             isSame(other.safety, safety) && isSame(norm[0], other.norm[0]) && isSame(norm[1], other.norm[1]) &&
             isSame(norm[2], other.norm[2]);
   }

   TString toString() const
   {
      return TString{TString{((contains) ? "Inside " : "Outside ")} + TString{" with distance in direction of "} +
                     TString{std::to_string(distance_in_dir)} + TString{" with a safety of "} +
                     TString{std::to_string(safety)} + TString{" with a normal of ("} +
                     TString{std::to_string(norm[0])} + TString{", "} + TString{std::to_string(norm[1])} +
                     TString{", "} + TString{std::to_string(norm[2])} + TString{")"}};
   }
};

TestResult_t TestPoint(TGeoShape *shape, Double_t x, Double_t y, Double_t z, Double_t dx, Double_t dy, Double_t dz)
{
   TestResult_t result;
   Double_t xyz[3] = {x, y, z};
   result.contains = shape->Contains(xyz);
   TVector3 dirVec{dx, dy, dz};
   dirVec = dirVec.Unit();
   Double_t dir[3] = {dirVec.X(), dirVec.Y(), dirVec.Z()};

   Double_t distance_in_dir = 0;
   
   if (result.contains) {
      result.distance_in_dir = shape->DistFromInside(xyz, dir, 3, 1e30, nullptr);
   } else {
      result.distance_in_dir = shape->DistFromOutside(xyz, dir, 3, 1e30, nullptr);
   }

   // shape->ComputeNormal(xyz, dir, result.norm); // ignore ComputeNormal for the time being, as it is not quite clear how to properly 
                                                   // compute it and it is of no relevance for simulation
   result.safety = shape->Safety(xyz, result.contains);

   return result;
}

void TestTessellatedVsBBox(Int_t numberOfPoints = 100000000)
{
   TString meshfile = gROOT->GetTutorialsDir();
   meshfile += "/visualisation/geom/Box.stl";

   TGeoTessellated *arbn = new TGeoTessellated();
   std::unique_ptr<Tessellated::TGeoTriangleMesh> mesh = Tessellated::ImportMeshFromASCIIStl(
      meshfile, Tessellated::TGeoTriangleMesh::LengthUnit::kMilliMeter);
   if (mesh->CheckClosure(kTRUE, 0)) {
      arbn->SetMesh(std::move(mesh));
   } else {
      std::cerr << mesh->GetMeshFile() << " does not contain valid mesh. Aborting TestTGeoTessellated!" ;
      exit(-1);
   }
   arbn->InspectShape();
   TGeoBBox *bbox = new TGeoBBox(2.54 / 2, 2.54 / 2, 22 / 2);
   bbox->InspectShape();

   // with octree
   TGeoTessellated *arbnWithOctree = new TGeoTessellated();
   arbnWithOctree->SetMesh(Tessellated::ImportMeshFromASCIIStl(meshfile,
                                                               Tessellated::TGeoTriangleMesh::LengthUnit::kMilliMeter));
   std::unique_ptr<Tessellated::TPartitioningI> octree{Tessellated::TOctree::CreateOctree(arbnWithOctree, 5, 1, kTRUE)};
   arbnWithOctree->SetPartitioningStruct(octree);

   // with bvh tree
   TGeoTessellated *arbnWithBVH = new TGeoTessellated();
   arbnWithBVH->SetMesh(Tessellated::ImportMeshFromASCIIStl(meshfile,
                                                            Tessellated::TGeoTriangleMesh::LengthUnit::kMilliMeter));

   std::unique_ptr<Tessellated::TPartitioningI> bvh{new Tessellated::TBVH()};
   bvh->SetTriangleMesh(arbnWithBVH->GetTriangleMesh());
   arbnWithBVH->SetPartitioningStruct(bvh);

   TRandom *eventGenerator = new TRandom();
   for (Int_t point = 0; point < numberOfPoints; ++point) {
      if (point % (numberOfPoints/100) == 0) {
        std::cout << (Double_t)point/(Double_t)numberOfPoints*100.  << " % "<< std::endl;
      }
      Double_t x = eventGenerator->Uniform(-2, 3);
      Double_t y = eventGenerator->Uniform(-2, 3);
      Double_t z = eventGenerator->Uniform(-2, 24);
      Double_t dx = eventGenerator->Uniform(-1, 1);
      Double_t dy = eventGenerator->Uniform(-1, 1);
      Double_t dz = eventGenerator->Uniform(-1, 1);
      TestResult_t a = TestPoint(arbn, x, y, z, dx, dy, dz);
      TestResult_t b = TestPoint(arbnWithOctree, x, y, z, dx, dy, dz);
      TestResult_t c = TestPoint(arbnWithBVH, x, y, z, dx, dy, dz);
      // Require an offset, as origin of box mesh is (1.27, 1.27, 11) but TGeoBBox at (0,0,0)
      TestResult_t d = TestPoint(bbox, x - 1.27, y - 1.27, z - 11, dx, dy, dz);
      if (a == b && a == c && a == d) {

      } else {
         std::cerr << "Error found for " << point << "th testpoint "
                << "(" << x << ", " << y << ", " << z << ") shot at "
                << "(" << dx << ", " << dy << ", " << dz << ")\n"
                << " Arbn results in " << a.toString() << "\n"
                << " Arbn with Octree results in " << b.toString() << "\n"
                << " Arbn with BVH results in " << c.toString() << "\n"
                << " BBox  results in " << d.toString();
         ::Fatal("", "");
      }
   }
}

