// @(#)root/mathcore:$Id$
// Author: Lorenzo Moneta March 2023



#include "Math/Delaunay2D.h"

#include "gtest/gtest.h"

// test Delauney interpolation on edges of a triangle
// some of these tests failed when using the older version
// see issue #

TEST(Delaunay2D,interpolation_at_edges)
{
   // give point making three triangles
   // points are taken from

   //triangle1 is { P0,P1,P4}, tr2 is { P1,P2,P4}, tr3 is {P2,P3,P4}

   double x[] = { 750, 1000, 1000, 900, 750 };
   double y[] = { 400, 400, 500, 600, 500};
   double z[] = {100,100,500,600,250};

   // need to provide some min/max value to have a numerical error in
   // computation of  barycentric coordinates

   ROOT::Math::Delaunay2D d(5,x,y,z,170.,1000.,15.,600.);

   // interpolate on horizontal edge between tr2 and tr3 (line P2-P4)
   //This was giving an error
   // result should be linear interp betwen P4(x=750,z=250) and P2(x=1000,z=500)
   EXPECT_DOUBLE_EQ( d.Interpolate(760,500),  260.);
   EXPECT_DOUBLE_EQ( d.Interpolate(780,500),  280.);
   EXPECT_DOUBLE_EQ( d.Interpolate(800,500),  300.);
   EXPECT_DOUBLE_EQ( d.Interpolate(900,500),  400.);

   // interpolate on vertical edge (p0,p4), same x different y
   // DeltaY = 100 deltaZ = 150
   EXPECT_DOUBLE_EQ( d.Interpolate(750,420),  130.);
   EXPECT_DOUBLE_EQ( d.Interpolate(750,450),  175.);
   EXPECT_DOUBLE_EQ( d.Interpolate(750,490),  235);

   // interpolate now on diagonal edge (P1,P4)
   EXPECT_DOUBLE_EQ( d.Interpolate(800,480),  220.);
   EXPECT_DOUBLE_EQ( d.Interpolate(900,440),  160.);
   EXPECT_DOUBLE_EQ( d.Interpolate(950,420),  130.);

   // interpolate on vertices
   EXPECT_DOUBLE_EQ( d.Interpolate(750,400),  z[0]);
   EXPECT_DOUBLE_EQ( d.Interpolate(1000,400),  z[1]);
   EXPECT_DOUBLE_EQ( d.Interpolate(1000,500), z[2]);
   EXPECT_DOUBLE_EQ( d.Interpolate(900,600),  z[3]);
   EXPECT_DOUBLE_EQ( d.Interpolate(750,500),  z[4]);

   // interpolate outside triangles
   // small underflow in x and y
   EXPECT_DOUBLE_EQ( d.Interpolate(750-0.0001,450),  0.);
   EXPECT_DOUBLE_EQ( d.Interpolate(800,400-0.0001),  0.);
   EXPECT_DOUBLE_EQ( d.Interpolate(1001,500),  0.);

}


