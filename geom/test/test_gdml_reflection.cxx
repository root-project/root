// A reflected placement must survive a GDML round trip - also when the placement
// carries a rotation as well, and also when it sits inside an assembly.

#include <gtest/gtest.h>

#include <TGeoManager.h>
#include <TGeoMaterial.h>
#include <TGeoMatrix.h>
#include <TGeoMedium.h>
#include <TGeoNode.h>
#include <TGeoVolume.h>
#include <TString.h>

namespace {

// The global placement of the volume named "inner"
struct Placement {
   double translation[3] = {0., 0., 0.};
   double xaxis[3] = {0., 0., 0.};
   bool reflection = false;
   bool found = false;
};

void FindInner(TGeoNode *node, const TGeoHMatrix &upto, Placement &p)
{
   TGeoHMatrix here = upto;
   here.Multiply(node->GetMatrix());
   if (TString(node->GetVolume()->GetName()) == "inner") {
      const double *tr = here.GetTranslation();
      for (int k = 0; k < 3; ++k)
         p.translation[k] = tr[k];
      const double localx[3] = {1., 0., 0.};
      here.LocalToMasterVect(localx, p.xaxis);
      p.reflection = here.IsReflection();
      p.found = true;
      return;
   }
   for (int i = 0; i < node->GetNdaughters() && !p.found; ++i)
      FindInner(node->GetDaughter(i), here, p);
}

Placement InnerPlacement()
{
   Placement p;
   TGeoHMatrix identity;
   FindInner(gGeoManager->GetTopNode(), identity, p);
   return p;
}

// A world holding one reflected box with an off-centre daughter in it. The
// rotation passed to ReflectZ is what distinguishes the two cases, and
// inAssembly puts the reflected placement inside an assembly instead of
// directly in the world.
void BuildGeometry(double phi, double theta, double psi, bool inAssembly)
{
   delete gGeoManager;
   new TGeoManager("refl", "reflected placement through GDML");
   TGeoMedium *medium = new TGeoMedium("m", 1, new TGeoMaterial("m", 26.98, 13, 2.7));
   TGeoVolume *top = gGeoManager->MakeBox("world", medium, 200., 200., 200.);
   gGeoManager->SetTopVolume(top);

   TGeoVolume *outer = gGeoManager->MakeBox("outer", medium, 80., 80., 80.);
   outer->AddNode(gGeoManager->MakeBox("inner", medium, 5., 5., 5.), 1, new TGeoTranslation(10., 20., 30.));

   TGeoRotation *rotation = new TGeoRotation("r", phi, theta, psi);
   rotation->ReflectZ(kTRUE); // determinant -1
   TGeoCombiTrans *placement = new TGeoCombiTrans(0., 0., 0., rotation);
   if (inAssembly) {
      TGeoVolume *assembly = new TGeoVolumeAssembly("asm");
      assembly->AddNode(outer, 1, placement);
      top->AddNode(assembly, 1, new TGeoTranslation(0., 0., 0.));
   } else {
      top->AddNode(outer, 1, placement);
   }
   gGeoManager->CloseGeometry();
}

void CheckRoundTrip(double phi, double theta, double psi, bool inAssembly, const char *filename)
{
   BuildGeometry(phi, theta, psi, inAssembly);
   const Placement before = InnerPlacement();
   ASSERT_TRUE(before.found);
   EXPECT_TRUE(before.reflection);

   gGeoManager->Export(filename);
   delete gGeoManager;
   ASSERT_NE(TGeoManager::Import(filename), nullptr);

   const Placement after = InnerPlacement();
   ASSERT_TRUE(after.found);
   EXPECT_EQ(before.reflection, after.reflection);
   for (int k = 0; k < 3; ++k) {
      EXPECT_NEAR(before.translation[k], after.translation[k], 1e-9);
      EXPECT_NEAR(before.xaxis[k], after.xaxis[k], 1e-9);
   }
}

} // namespace

TEST(GDMLReflection, PureReflection)
{
   CheckRoundTrip(0., 0., 0., false, "gdml_reflection_pure.gdml");
}

TEST(GDMLReflection, ReflectionWithRotation)
{
   CheckRoundTrip(30., 40., 50., false, "gdml_reflection_rotated.gdml");
}

TEST(GDMLReflection, PureReflectionInAssembly)
{
   CheckRoundTrip(0., 0., 0., true, "gdml_reflection_pure_assembly.gdml");
}

TEST(GDMLReflection, ReflectionWithRotationInAssembly)
{
   CheckRoundTrip(30., 40., 50., true, "gdml_reflection_rotated_assembly.gdml");
}
