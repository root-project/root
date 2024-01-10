// test TH3::Project3D name

#include "gtest/gtest.h"
#include "gmock/gmock.h"
using ::testing::StrEq;

#include "TH2D.h"
#include "TH3D.h"

TEST(TH3, Project3D) {
   auto h3 = new TH3D("h3","Title",10,0.,1.,10,0.,1.,10,0.,1.);
   auto h2 = (TH2D*)h3->Project3D("Projection_xy");
   EXPECT_THAT(h2->GetName() , StrEq("h3_Projection_xy"));
}