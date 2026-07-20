#include <TGeometry.h>
#include <TRotMatrix.h>
#include <TPolyMarker3D.h>
#include <TCanvas.h>
#include <TNode.h>
#include <TBRIK.h>

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>

// TGeometry transforms graphs, lines and markers of g3d if there is an active rotation.
TEST(TGeometry, TransformTPolyMarker3D)
{
   constexpr auto filename1 = "beforeTransform.svg";
   constexpr auto filename2 = "afterTransform.svg";
   TCanvas c1;
   TPolyMarker3D marker;
   marker.SetPoint(0, 0.5, 0.5, 0.);
   marker.SetMarkerSize(2);
   marker.SetMarkerStyle(20);

   // First draw without an active geometry
   marker.Draw("same");
   c1.SaveAs(filename1);
   c1.Clear();

   // Then prepare two geometry levels, so the graph is affected by the transformation
   TGeometry geo;
   [[maybe_unused]] TBRIK *brik = new TBRIK("BRIK", "BRIK", "void", 0.2, 0.3, 0.2);
   TRotMatrix rot("rot1", "rot1", 1, 2, 3, 1, 2, 3);
   TNode *node = new TNode("BRIKNode", "BRIKNode", "BRIK", 0, 0, -0.1, "rot1");
   node->cd();
   geo.PushLevel();
   TNode *node2 = new TNode("BRIKNode2", "BRIKNode2", "BRIK", 0, 0, -0.1, "rot1");
   geo.UpdateMatrix(node2);

   // The marker will move now
   marker.Draw("same");
   geo.Draw("same");
   c1.SaveAs(filename2);

   // Check the svg to ensure that it moved
   std::ifstream file1(filename1);
   std::ifstream file2(filename2);
   std::string line1, line2;
   while (std::getline(file1, line1)) {
      if (line1.find("circle cx=") != std::string::npos)
         break;
   }
   while (std::getline(file2, line2)) {
      if (line2.find("circle cx=") != std::string::npos)
         break;
   }
   EXPECT_TRUE(line1.find("<circle cx=") == 0) << line1;
   EXPECT_TRUE(line2.find("<circle cx=") == 0) << line2;
   EXPECT_NE(line1, line2);

   if (HasFailure()) {
      std::cout << std::ifstream(filename1).rdbuf() << "\n";
      std::cout << std::ifstream(filename2).rdbuf() << "\n";
   }

   std::remove(filename1);
   std::remove(filename2);
}