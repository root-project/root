// @(#)root/test:$Id$
// Authors: David Gonzalez Maline November 2008
//          Martin Storø Nyfløtt  June 2017

#include <sstream>

#include "TH2.h"
#include "TH3.h"
#include "THn.h"
#include "THnSparse.h"

#include "TProfile.h"

#include "TF1.h"

#include "HFitInterface.h"

#include "TFile.h"

#include "gtest/gtest.h"

#include "StressHistogramGlobal.h"

using namespace std;

TEST(StressHistogram, TestConversion1D)
{
   const int nbins[3] = {50, 11, 12};
   const double minRangeArray[3] = {2., 4., 4.};
   const double maxRangeArray[3] = {5., 8., 10.};

   const int nevents = 500;

   TF1 f("gaus1D", gaus1d, minRangeArray[0], maxRangeArray[0], 3);
   f.SetParameters(10., 3.5, .4);

   TH1C h1c("h1c", "h1-title", nbins[0], minRangeArray[0], maxRangeArray[0]);
   TH1S h1s("h1s", "h1-title", nbins[0], minRangeArray[0], maxRangeArray[0]);
   TH1I h1i("h1i", "h1-title", nbins[0], minRangeArray[0], maxRangeArray[0]);
   TH1F h1f("h1f", "h1-title", nbins[0], minRangeArray[0], maxRangeArray[0]);
   TH1D h1d("h1d", "h1-title", nbins[0], minRangeArray[0], maxRangeArray[0]);

   h1c.FillRandom("gaus1D", nevents);
   h1s.FillRandom("gaus1D", nevents);
   h1i.FillRandom("gaus1D", nevents);
   h1f.FillRandom("gaus1D", nevents);
   h1d.FillRandom("gaus1D", nevents);

   unique_ptr<THnSparse> s1c(THnSparse::CreateSparse("s1c", "s1cTitle", &h1c));
   unique_ptr<THnSparse> s1s(THnSparse::CreateSparse("s1s", "s1sTitle", &h1s));
   unique_ptr<THnSparse> s1i(THnSparse::CreateSparse("s1i", "s1iTitle", &h1i));
   unique_ptr<THnSparse> s1f(THnSparse::CreateSparse("s1f", "s1fTitle", &h1f));
   unique_ptr<THnSparse> s1d(THnSparse::CreateSparse("s1d", "s1dTitle", &h1d));

   unique_ptr<TH1> h1cn ((TH1 *)h1c.Clone("h1cn"));
   unique_ptr<TH1> h1sn ((TH1 *)h1s.Clone("h1sn"));
   unique_ptr<TH1> h1in ((TH1 *)h1i.Clone("h1in"));
   unique_ptr<TH1> h1fn ((TH1 *)h1f.Clone("h1fn"));
   unique_ptr<TH1> h1dn ((TH1 *)h1s.Clone("h1dn"));

   EXPECT_TRUE(HistogramsEquals(*s1c.get(), h1c));
   EXPECT_TRUE(HistogramsEquals(*s1s.get(), h1s));
   EXPECT_TRUE(HistogramsEquals(*s1i.get(), h1i));
   EXPECT_TRUE(HistogramsEquals(*s1f.get(), h1f));
   EXPECT_TRUE(HistogramsEquals(*s1d.get(), h1d));

   unique_ptr<THn> n1c(THn::CreateHn("n1c", "n1cTitle", h1cn.get()));
   unique_ptr<THn> n1s(THn::CreateHn("n1s", "n1sTitle", h1sn.get()));
   unique_ptr<THn> n1i(THn::CreateHn("n1i", "n1iTitle", h1in.get()));
   unique_ptr<THn> n1f(THn::CreateHn("n1f", "n1fTitle", h1fn.get()));
   unique_ptr<THn> n1d(THn::CreateHn("n1d", "n1dTitle", h1dn.get()));

   EXPECT_TRUE(HistogramsEquals(*n1c.get(), *h1cn.get()));
   EXPECT_TRUE(HistogramsEquals(*n1s.get(), *h1sn.get()));
   EXPECT_TRUE(HistogramsEquals(*n1i.get(), *h1in.get()));
   EXPECT_TRUE(HistogramsEquals(*n1f.get(), *h1fn.get()));
   EXPECT_TRUE(HistogramsEquals(*n1d.get(), *h1dn.get()));

}

TEST(StressHistogram, TestConversion2D)
{
   const int nbins[3] = {50, 11, 12};
   const double minRangeArray[3] = {2., 4., 4.};
   const double maxRangeArray[3] = {5., 8., 10.};

   const int nevents = 500;

   TF2 f("gaus2D", gaus2d, minRangeArray[0], maxRangeArray[0], minRangeArray[1], maxRangeArray[1], 5);
   f.SetParameters(10., 3.5, .4, 6, 1);

   TH2C h2c("h2c", "h2-title", nbins[0], minRangeArray[0], maxRangeArray[0], nbins[1], minRangeArray[1], maxRangeArray[1]);
   TH2S h2s("h2s", "h2-title", nbins[0], minRangeArray[0], maxRangeArray[0], nbins[1], minRangeArray[1], maxRangeArray[1]);
   TH2I h2i("h2i", "h2-title", nbins[0], minRangeArray[0], maxRangeArray[0], nbins[1], minRangeArray[1], maxRangeArray[1]);
   TH2F h2f("h2f", "h2-title", nbins[0], minRangeArray[0], maxRangeArray[0], nbins[1], minRangeArray[1], maxRangeArray[1]);
   TH2D h2d("h2d", "h2-title", nbins[0], minRangeArray[0], maxRangeArray[0], nbins[1], minRangeArray[1], maxRangeArray[1]);

   h2c.FillRandom("gaus2D", nevents);
   h2s.FillRandom("gaus2D", nevents);
   h2i.FillRandom("gaus2D", nevents);
   h2f.FillRandom("gaus2D", nevents);
   h2d.FillRandom("gaus2D", nevents);

   unique_ptr<THnSparse> s2c(THnSparse::CreateSparse("s2c", "s2cTitle", &h2c));
   unique_ptr<THnSparse> s2s(THnSparse::CreateSparse("s2s", "s2sTitle", &h2s));
   unique_ptr<THnSparse> s2i(THnSparse::CreateSparse("s2i", "s2iTitle", &h2i));
   unique_ptr<THnSparse> s2f(THnSparse::CreateSparse("s2f", "s2fTitle", &h2f));
   unique_ptr<THnSparse> s2d(THnSparse::CreateSparse("s2d", "s2dTitle", &h2d));

   unique_ptr<TH2> h2cn((TH2 *)h2c.Clone("h2cn"));
   unique_ptr<TH2> h2sn((TH2 *)h2s.Clone("h2sn"));
   unique_ptr<TH2> h2in((TH2 *)h2i.Clone("h2in"));
   unique_ptr<TH2> h2fn((TH2 *)h2f.Clone("h2fn"));
   unique_ptr<TH2> h2dn((TH2 *)h2d.Clone("h2dn"));

   EXPECT_TRUE(HistogramsEquals(*s2c.get(), h2c));
   EXPECT_TRUE(HistogramsEquals(*s2s.get(), h2s));
   EXPECT_TRUE(HistogramsEquals(*s2i.get(), h2i));
   EXPECT_TRUE(HistogramsEquals(*s2f.get(), h2f));
   EXPECT_TRUE(HistogramsEquals(*s2d.get(), h2d));

   unique_ptr<THn> n2c(THn::CreateHn("n2c", "n2cTitle", h2cn.get()));
   unique_ptr<THn> n2s(THn::CreateHn("n2s", "n2sTitle", h2sn.get()));
   unique_ptr<THn> n2i(THn::CreateHn("n2i", "n2iTitle", h2in.get()));
   unique_ptr<THn> n2f(THn::CreateHn("n2f", "n2fTitle", h2fn.get()));
   unique_ptr<THn> n2d(THn::CreateHn("n2d", "n2dTitle", h2dn.get()));

   EXPECT_TRUE(HistogramsEquals(*n2c.get(), *h2cn.get()));
   EXPECT_TRUE(HistogramsEquals(*n2s.get(), *h2sn.get()));
   EXPECT_TRUE(HistogramsEquals(*n2i.get(), *h2in.get()));
   EXPECT_TRUE(HistogramsEquals(*n2f.get(), *h2fn.get()));
   EXPECT_TRUE(HistogramsEquals(*n2d.get(), *h2dn.get()));
}

TEST(StressHistogram, TestConversion3D)
{
   const int nbins[3] = {50, 11, 12};
   const double minRangeArray[3] = {2., 4., 4.};
   const double maxRangeArray[3] = {5., 8., 10.};

   const int nevents = 500;

   TF3 f("gaus3D", gaus3d, minRangeArray[0], maxRangeArray[0], minRangeArray[1], maxRangeArray[1], minRangeArray[2], maxRangeArray[2], 7);
   f.SetParameters(10., 3.5, .4, 6, 1, 7, 2);

   TH3C h3c("h3c", "h3-title", nbins[0], minRangeArray[0], maxRangeArray[0], nbins[1], minRangeArray[1], maxRangeArray[1], nbins[2], minRangeArray[2], maxRangeArray[2]);
   TH3S h3s("h3s", "h3-title", nbins[0], minRangeArray[0], maxRangeArray[0], nbins[1], minRangeArray[1], maxRangeArray[1], nbins[2], minRangeArray[2], maxRangeArray[2]);
   TH3I h3i("h3i", "h3-title", nbins[0], minRangeArray[0], maxRangeArray[0], nbins[1], minRangeArray[1], maxRangeArray[1], nbins[2], minRangeArray[2], maxRangeArray[2]);
   TH3F h3f("h3f", "h3-title", nbins[0], minRangeArray[0], maxRangeArray[0], nbins[1], minRangeArray[1], maxRangeArray[1], nbins[2], minRangeArray[2], maxRangeArray[2]);
   TH3D h3d("h3d", "h3-title", nbins[0], minRangeArray[0], maxRangeArray[0], nbins[1], minRangeArray[1], maxRangeArray[1], nbins[2], minRangeArray[2], maxRangeArray[2]);

   h3c.FillRandom("gaus3D", nevents);
   h3s.FillRandom("gaus3D", nevents);
   h3i.FillRandom("gaus3D", nevents);
   h3f.FillRandom("gaus3D", nevents);
   h3d.FillRandom("gaus3D", nevents);

   unique_ptr<THnSparse> s3c(THnSparse::CreateSparse("s3c", "s3cTitle", &h3c));
   unique_ptr<THnSparse> s3s(THnSparse::CreateSparse("s3s", "s3sTitle", &h3s));
   unique_ptr<THnSparse> s3i(THnSparse::CreateSparse("s3i", "s3iTitle", &h3i));
   unique_ptr<THnSparse> s3f(THnSparse::CreateSparse("s3f", "s3fTitle", &h3f));
   unique_ptr<THnSparse> s3d(THnSparse::CreateSparse("s3d", "s3dTitle", &h3d));

   unique_ptr<TH3> h3cn((TH3 *)h3c.Clone("h3cn"));
   unique_ptr<TH3> h3sn((TH3 *)h3s.Clone("h3sn"));
   unique_ptr<TH3> h3in((TH3 *)h3i.Clone("h3in"));
   unique_ptr<TH3> h3fn((TH3 *)h3f.Clone("h3fn"));
   unique_ptr<TH3> h3dn((TH3 *)h3d.Clone("h3dn"));

   EXPECT_TRUE(HistogramsEquals(*s3c.get(), h3c));
   EXPECT_TRUE(HistogramsEquals(*s3s.get(), h3s));
   EXPECT_TRUE(HistogramsEquals(*s3i.get(), h3i));
   EXPECT_TRUE(HistogramsEquals(*s3f.get(), h3f));
   EXPECT_TRUE(HistogramsEquals(*s3d.get(), h3d));

   unique_ptr<THn> n3c(THn::CreateHn("n3c", "n3cTitle", h3cn.get()));
   unique_ptr<THn> n3s(THn::CreateHn("n3s", "n3sTitle", h3sn.get()));
   unique_ptr<THn> n3i(THn::CreateHn("n3i", "n3iTitle", h3in.get()));
   unique_ptr<THn> n3f(THn::CreateHn("n3f", "n3fTitle", h3fn.get()));
   unique_ptr<THn> n3d(THn::CreateHn("n3d", "n3dTitle", h3dn.get()));

   EXPECT_TRUE(HistogramsEquals(*n3c.get(), *h3cn.get()));
   EXPECT_TRUE(HistogramsEquals(*n3s.get(), *h3sn.get()));
   EXPECT_TRUE(HistogramsEquals(*n3i.get(), *h3in.get()));
   EXPECT_TRUE(HistogramsEquals(*n3f.get(), *h3fn.get()));
   EXPECT_TRUE(HistogramsEquals(*n3d.get(), *h3dn.get()));
}
