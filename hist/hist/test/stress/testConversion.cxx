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

TEST(StressHistorgram, TestConversion1D)
{
   const int nbins[3] = {50, 11, 12};
   const double minRangeArray[3] = {2., 4., 4.};
   const double maxRangeArray[3] = {5., 8., 10.};

   const int nevents = 500;

   TF1 *f = new TF1("gaus1D", gaus1d, minRangeArray[0], maxRangeArray[0], 3);
   f->SetParameters(10., 3.5, .4);

   TH1 *h1c = new TH1C("h1c", "h1-title", nbins[0], minRangeArray[0], maxRangeArray[0]);
   TH1 *h1s = new TH1S("h1s", "h1-title", nbins[0], minRangeArray[0], maxRangeArray[0]);
   TH1 *h1i = new TH1I("h1i", "h1-title", nbins[0], minRangeArray[0], maxRangeArray[0]);
   TH1 *h1f = new TH1F("h1f", "h1-title", nbins[0], minRangeArray[0], maxRangeArray[0]);
   TH1 *h1d = new TH1D("h1d", "h1-title", nbins[0], minRangeArray[0], maxRangeArray[0]);

   h1c->FillRandom("gaus1D", nevents);
   h1s->FillRandom("gaus1D", nevents);
   h1i->FillRandom("gaus1D", nevents);
   h1f->FillRandom("gaus1D", nevents);
   h1d->FillRandom("gaus1D", nevents);

   THnSparse *s1c = THnSparse::CreateSparse("s1c", "s1cTitle", h1c);
   THnSparse *s1s = THnSparse::CreateSparse("s1s", "s1sTitle", h1s);
   THnSparse *s1i = THnSparse::CreateSparse("s1i", "s1iTitle", h1i);
   THnSparse *s1f = THnSparse::CreateSparse("s1f", "s1fTitle", h1f);
   THnSparse *s1d = THnSparse::CreateSparse("s1d", "s1dTitle", h1d);

   TH1 *h1cn = (TH1 *)h1c->Clone("h1cn");
   TH1 *h1sn = (TH1 *)h1s->Clone("h1sn");
   TH1 *h1in = (TH1 *)h1i->Clone("h1in");
   TH1 *h1fn = (TH1 *)h1f->Clone("h1fn");
   TH1 *h1dn = (TH1 *)h1s->Clone("h1dn");

   EXPECT_TRUE(HistogramsEquals(s1c, h1c));
   EXPECT_TRUE(HistogramsEquals(s1s, h1s));
   EXPECT_TRUE(HistogramsEquals(s1i, h1i));
   EXPECT_TRUE(HistogramsEquals(s1f, h1f));
   EXPECT_TRUE(HistogramsEquals(s1d, h1d));

   delete s1c;
   delete s1s;
   delete s1i;
   delete s1f;
   delete s1d;

   THn *n1c = THn::CreateHn("n1c", "n1cTitle", h1cn);
   THn *n1s = THn::CreateHn("n1s", "n1sTitle", h1sn);
   THn *n1i = THn::CreateHn("n1i", "n1iTitle", h1in);
   THn *n1f = THn::CreateHn("n1f", "n1fTitle", h1fn);
   THn *n1d = THn::CreateHn("n1d", "n1dTitle", h1dn);

   EXPECT_TRUE(HistogramsEquals(n1c, h1cn));
   EXPECT_TRUE(HistogramsEquals(n1s, h1sn));
   EXPECT_TRUE(HistogramsEquals(n1i, h1in));
   EXPECT_TRUE(HistogramsEquals(n1f, h1fn));
   EXPECT_TRUE(HistogramsEquals(n1d, h1dn));

   delete n1c;
   delete n1s;
   delete n1i;
   delete n1f;
   delete n1d;
}

TEST(StressHistorgram, TestConversion2D)
{
   const int nbins[3] = {50, 11, 12};
   const double minRangeArray[3] = {2., 4., 4.};
   const double maxRangeArray[3] = {5., 8., 10.};

   const int nevents = 500;

   TF2 *f = new TF2("gaus2D", gaus2d, minRangeArray[0], maxRangeArray[0], minRangeArray[1], maxRangeArray[1], 5);
   f->SetParameters(10., 3.5, .4, 6, 1);

   TH2 *h2c = new TH2C("h2c", "h2-title", nbins[0], minRangeArray[0], maxRangeArray[0], nbins[1], minRangeArray[1],
                       maxRangeArray[1]);

   TH2 *h2s = new TH2S("h2s", "h2-title", nbins[0], minRangeArray[0], maxRangeArray[0], nbins[1], minRangeArray[1],
                       maxRangeArray[1]);
   TH2 *h2i = new TH2I("h2i", "h2-title", nbins[0], minRangeArray[0], maxRangeArray[0], nbins[1], minRangeArray[1],
                       maxRangeArray[1]);
   TH2 *h2f = new TH2F("h2f", "h2-title", nbins[0], minRangeArray[0], maxRangeArray[0], nbins[1], minRangeArray[1],
                       maxRangeArray[1]);
   TH2 *h2d = new TH2D("h2d", "h2-title", nbins[0], minRangeArray[0], maxRangeArray[0], nbins[1], minRangeArray[1],
                       maxRangeArray[1]);

   h2c->FillRandom("gaus2D", nevents);
   h2s->FillRandom("gaus2D", nevents);
   h2i->FillRandom("gaus2D", nevents);
   h2f->FillRandom("gaus2D", nevents);
   h2d->FillRandom("gaus2D", nevents);

   THnSparse *s2c = THnSparse::CreateSparse("s2c", "s2cTitle", h2c);
   THnSparse *s2s = THnSparse::CreateSparse("s2s", "s2sTitle", h2s);
   THnSparse *s2i = THnSparse::CreateSparse("s2i", "s2iTitle", h2i);
   THnSparse *s2f = THnSparse::CreateSparse("s2f", "s2fTitle", h2f);
   THnSparse *s2d = THnSparse::CreateSparse("s2d", "s2dTitle", h2d);

   TH2 *h2cn = (TH2 *)h2c->Clone("h2cn");
   TH2 *h2sn = (TH2 *)h2s->Clone("h2sn");
   TH2 *h2in = (TH2 *)h2i->Clone("h2in");
   TH2 *h2fn = (TH2 *)h2f->Clone("h2fn");
   TH2 *h2dn = (TH2 *)h2d->Clone("h2dn");

   EXPECT_TRUE(HistogramsEquals(s2c, h2c));
   EXPECT_TRUE(HistogramsEquals(s2s, h2s));
   EXPECT_TRUE(HistogramsEquals(s2i, h2i));
   EXPECT_TRUE(HistogramsEquals(s2f, h2f));
   EXPECT_TRUE(HistogramsEquals(s2d, h2d));

   delete s2c;
   delete s2s;
   delete s2i;
   delete s2f;
   delete s2d;

   THn *n2c = THn::CreateHn("n2c", "n2cTitle", h2cn);
   THn *n2s = THn::CreateHn("n2s", "n2sTitle", h2sn);
   THn *n2i = THn::CreateHn("n2i", "n2iTitle", h2in);
   THn *n2f = THn::CreateHn("n2f", "n2fTitle", h2fn);
   THn *n2d = THn::CreateHn("n2d", "n2dTitle", h2dn);

   EXPECT_TRUE(HistogramsEquals(n2c, h2cn));
   EXPECT_TRUE(HistogramsEquals(n2s, h2sn));
   EXPECT_TRUE(HistogramsEquals(n2i, h2in));
   EXPECT_TRUE(HistogramsEquals(n2f, h2fn));
   EXPECT_TRUE(HistogramsEquals(n2d, h2dn));

   delete n2c;
   delete n2s;
   delete n2i;
   delete n2f;
   delete n2d;
}

TEST(StressHistorgram, TestConversion3D)
{
   const int nbins[3] = {50, 11, 12};
   const double minRangeArray[3] = {2., 4., 4.};
   const double maxRangeArray[3] = {5., 8., 10.};

   const int nevents = 500;

   TF3 *f = new TF3("gaus3D", gaus3d, minRangeArray[0], maxRangeArray[0], minRangeArray[1], maxRangeArray[1],
                    minRangeArray[2], maxRangeArray[2], 7);
   f->SetParameters(10., 3.5, .4, 6, 1, 7, 2);

   TH3 *h3c = new TH3C("h3c", "h3-title", nbins[0], minRangeArray[0], maxRangeArray[0], nbins[1], minRangeArray[1],
                       maxRangeArray[1], nbins[2], minRangeArray[2], maxRangeArray[2]);

   TH3 *h3s = new TH3S("h3s", "h3-title", nbins[0], minRangeArray[0], maxRangeArray[0], nbins[1], minRangeArray[1],
                       maxRangeArray[1], nbins[2], minRangeArray[2], maxRangeArray[2]);
   TH3 *h3i = new TH3I("h3i", "h3-title", nbins[0], minRangeArray[0], maxRangeArray[0], nbins[1], minRangeArray[1],
                       maxRangeArray[1], nbins[2], minRangeArray[2], maxRangeArray[2]);
   TH3 *h3f = new TH3F("h3f", "h3-title", nbins[0], minRangeArray[0], maxRangeArray[0], nbins[1], minRangeArray[1],
                       maxRangeArray[1], nbins[2], minRangeArray[2], maxRangeArray[2]);
   TH3 *h3d = new TH3D("h3d", "h3-title", nbins[0], minRangeArray[0], maxRangeArray[0], nbins[1], minRangeArray[1],
                       maxRangeArray[1], nbins[2], minRangeArray[2], maxRangeArray[2]);

   h3c->FillRandom("gaus3D", nevents);
   h3s->FillRandom("gaus3D", nevents);
   h3i->FillRandom("gaus3D", nevents);
   h3f->FillRandom("gaus3D", nevents);
   h3d->FillRandom("gaus3D", nevents);

   THnSparse *s3c = THnSparse::CreateSparse("s3c", "s3cTitle", h3c);
   THnSparse *s3s = THnSparse::CreateSparse("s3s", "s3sTitle", h3s);
   THnSparse *s3i = THnSparse::CreateSparse("s3i", "s3iTitle", h3i);
   THnSparse *s3f = THnSparse::CreateSparse("s3f", "s3fTitle", h3f);
   THnSparse *s3d = THnSparse::CreateSparse("s3d", "s3dTitle", h3d);

   TH3 *h3cn = (TH3 *)h3c->Clone("h3cn");
   TH3 *h3sn = (TH3 *)h3s->Clone("h3sn");
   TH3 *h3in = (TH3 *)h3i->Clone("h3in");
   TH3 *h3fn = (TH3 *)h3f->Clone("h3fn");
   TH3 *h3dn = (TH3 *)h3d->Clone("h3dn");

   EXPECT_TRUE(HistogramsEquals(s3c, h3c));
   EXPECT_TRUE(HistogramsEquals(s3s, h3s));
   EXPECT_TRUE(HistogramsEquals(s3i, h3i));
   EXPECT_TRUE(HistogramsEquals(s3f, h3f));
   EXPECT_TRUE(HistogramsEquals(s3d, h3d));

   delete s3c;
   delete s3s;
   delete s3i;
   delete s3f;
   delete s3d;

   THn *n3c = THn::CreateHn("n3c", "n3cTitle", h3cn);
   THn *n3s = THn::CreateHn("n3s", "n3sTitle", h3sn);
   THn *n3i = THn::CreateHn("n3i", "n3iTitle", h3in);
   THn *n3f = THn::CreateHn("n3f", "n3fTitle", h3fn);
   THn *n3d = THn::CreateHn("n3d", "n3dTitle", h3dn);

   EXPECT_TRUE(HistogramsEquals(n3c, h3cn));
   EXPECT_TRUE(HistogramsEquals(n3s, h3sn));
   EXPECT_TRUE(HistogramsEquals(n3i, h3in));
   EXPECT_TRUE(HistogramsEquals(n3f, h3fn));
   EXPECT_TRUE(HistogramsEquals(n3d, h3dn));

   delete n3c;
   delete n3s;
   delete n3i;
   delete n3f;
   delete n3d;
}
