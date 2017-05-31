// @(#)root/test:$Id$
// Author: Lorenzo Moneta       06/2005
//         Martin Storø Nyfløtt 05/2017

#include "gtest/gtest.h"
#include "StatFunction.h"
#include "Math/DistFuncMathCore.h"

using namespace ROOT::Math;

// test statistical functions
TEST(StressMathCore, BetaDistribution)
{
   CREATE_DIST(beta);
   dist.SetParameters(2, 2);
   dist.Test(0.01, 0.99, 0., 1., false);
   CREATE_DIST_C(beta);
   distc.SetParameters(2, 2);
   distc.Test(0.01, 0.99, 0., 1., true);
}

TEST(StressMathCore, GammaDistribution)
{
   CREATE_DIST_C(gamma);
   distc.SetParameters(2, 1);
   distc.Test(0.05, 5, 0., TMath::Infinity(), true);
}

TEST(StressMathCore, ChisquareDistribution)
{
   CREATE_DIST_C(chisquared);
   distc.SetParameters(10, 0);
   distc.ScaleTol2(10000000); // t.b.c.
   distc.Test(0.05, 30, 0., TMath::Infinity(), true);
}

TEST(StressMathCore, NormalDistribution)
{
   CREATE_DIST(gaussian);
   dist.SetParameters(2, 1);
   dist.ScaleTol2(100);
   dist.Test(-3, 5, false);
   CREATE_DIST_C(gaussian);
   distc.SetParameters(1, 0);
   distc.ScaleTol2(100);
   distc.Test(-3, 5, 1, 0, true);
}

TEST(StressMathCore, BreitWignerDistribution)
{
   CREATE_DIST(breitwigner);
   dist.SetParameters(1);
   dist.ScaleTol1(1E8);
   dist.ScaleTol2(10);
   dist.Test(-5, 5, false);
   CREATE_DIST_C(breitwigner);
   distc.SetParameters(1);
   distc.ScaleTol1(1E8);
   distc.ScaleTol2(10);
   distc.Test(-5, 5, 1, 0, true);
}

TEST(StressMathCore, FDistribution)
{
   CREATE_DIST(fdistribution);
   dist.SetParameters(5, 4);
   dist.ScaleTol1(1000000);
   dist.ScaleTol2(10);
   // if enlarge scale test fails
   dist.Test(0.05, 5, 0, 1, false);
   CREATE_DIST_C(fdistribution);
   distc.SetParameters(5, 4);
   distc.ScaleTol1(100000000);
   distc.ScaleTol2(10);
   // if enlarge scale test fails
   distc.Test(0.05, 5, 0, TMath::Infinity(), true);
}

TEST(StressMathCore, LognormalDistribution)
{
   CREATE_DIST(lognormal);
   dist.SetParameters(1, 1);
   dist.ScaleTol1(1000);
   dist.Test(0.01, 5, 0, 1, false);
   CREATE_DIST_C(lognormal);
   distc.SetParameters(1, 1);
   distc.ScaleTol1(1000);
   distc.ScaleTol2(1000000); // t.b.c.
   distc.Test(0.01, 5, 0, TMath::Infinity(), true);
}

TEST(StressMathCore, ExponentialDistribution)
{
   CREATE_DIST(exponential);
   dist.SetParameters(2);
   dist.ScaleTol2(100);
   dist.Test(0., 5., 0., 1., false);
   CREATE_DIST_C(exponential);
   distc.SetParameters(2);
   distc.ScaleTol2(100);
   distc.Test(0., 5., 0., 1., true);
}

TEST(StressMathCore, LandauDistribution)
{
   CREATE_DIST(landau);
   dist.SetParameters(2);
   // Landau is not very precise (put prec at 10-6)
   // as indicated in Landau paper (
   dist.ScaleTol1(10000);
   dist.ScaleTol2(1.E10);
   dist.Test(-1, 10, -TMath::Infinity(), TMath::Infinity(), false);
   CREATE_DIST_C(landau);
   distc.SetParameters(2);
   distc.ScaleTol1(1000);
   distc.ScaleTol2(1E10);
   distc.Test(-1, 10, -TMath::Infinity(), TMath::Infinity(), true);
}

TEST(StressMathCore, UniformDistribution)
{
   CREATE_DIST(uniform);
   dist.SetParameters(1, 2);
   dist.Test(1., 2., 1., 2., false);
   CREATE_DIST_C(uniform);
   distc.SetParameters(1, 2);
   distc.Test(1., 2., 1., 2., true);
}
