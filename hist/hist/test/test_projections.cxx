#include "TH2F.h"
#include "TH3F.h"
#include "TProfile.h"   // ProjectionX
#include "TProfile2D.h" // ProjectionX
#include "THashList.h"  // GetLabels
#include <limits>
#include "TMath.h"

#include "gtest/gtest.h"

template <typename V1, typename V2>
void expect_list_eq_names(const V1 &v1, const V2 &v2)
{
   ASSERT_EQ(v1.GetEntries(), v2.GetEntries());
   for (decltype(v1.GetEntries()) i = 0; i < v1.GetEntries(); ++i) {
      EXPECT_STREQ(v1.At(i)->GetName(), v2.At(i)->GetName());
   }
}

// Test projection from 2D hist for labels/nbins
TEST(Projections, Issue_6658_2D)
{
   TH2F hist2d("hist", "", 2, 0, 2, 2, 0, 2);
   auto xaxis_2d = hist2d.GetXaxis();
   xaxis_2d->SetBinLabel(1, "A");
   xaxis_2d->SetBinLabel(2, "B");
   auto xaxis_2d_nbins = xaxis_2d->GetNbins();
   auto *labels_2d = xaxis_2d->GetLabels();

   auto *hist_px = hist2d.ProjectionX();
   auto xaxis_px = hist_px->GetXaxis();
   auto xaxis_px_nbins = xaxis_px->GetNbins();
   auto *labels_px = xaxis_px->GetLabels();

   EXPECT_EQ(xaxis_2d_nbins, xaxis_px_nbins);
   hist_px->LabelsDeflate();
   EXPECT_EQ(xaxis_2d_nbins, xaxis_px_nbins);
   expect_list_eq_names(*labels_2d, *labels_px);

   auto prof_px = hist2d.ProfileX();
   auto prof_px_xaxis = prof_px->GetXaxis();
   auto prof_px_nbins = prof_px_xaxis->GetNbins();
   auto *prof_px_labels = prof_px_xaxis->GetLabels();
   EXPECT_EQ(xaxis_2d_nbins, prof_px_nbins);
   expect_list_eq_names(*labels_2d, *prof_px_labels);
}

// Test projection from 3D hist for labels/nbins
TEST(Projections, Issue_6658_3D)
{
   TH3F hist3d("hist3", "", 2, 0, 2, 2, 0, 3, 2, 0, 4);
   auto *xaxis_3d = hist3d.GetXaxis();
   xaxis_3d->SetBinLabel(1, "A");
   xaxis_3d->SetBinLabel(2, "B");
   auto xaxis_3d_nbins = xaxis_3d->GetNbins();
   auto *labels_3d = xaxis_3d->GetLabels();

   auto *hist_px = hist3d.ProjectionX("x");
   auto *xaxis_px = hist_px->GetXaxis();
   auto xaxis_px_nbins = xaxis_px->GetNbins();
   auto *labels_px = xaxis_px->GetLabels();
   EXPECT_EQ(xaxis_3d_nbins, xaxis_px_nbins);
   expect_list_eq_names(*labels_3d, *labels_px);

   auto *prof2_px = hist3d.Project3DProfile("yx");
   auto *xaxis_prof2_px = prof2_px->GetXaxis();
   auto prof2_px_nbins = xaxis_prof2_px->GetNbins();
   auto *labels_prof2_px = xaxis_prof2_px->GetLabels();
   EXPECT_EQ(xaxis_3d_nbins, prof2_px_nbins);
   expect_list_eq_names(*labels_3d, *labels_prof2_px);
}

// Test projection from Profile2D hist for labels/nbins
TEST(Projections, Issue_6658_Profile2D)
{
   TProfile2D prof2d("prof2d", "", 2, 0, 2, 2, 0, 3);
   auto *xaxis_2d = prof2d.GetXaxis();
   xaxis_2d->SetBinLabel(1, "A");
   xaxis_2d->SetBinLabel(2, "B");
   auto xaxis_2d_nbins = xaxis_2d->GetNbins();
   auto *labels_2d = xaxis_2d->GetLabels();

   auto *hist_pxy = prof2d.ProjectionXY("xy");
   auto *xaxis_pxy = hist_pxy->GetXaxis();
   auto xaxis_pxy_nbins = xaxis_pxy->GetNbins();
   auto *labels_pxy = xaxis_pxy->GetLabels();
   EXPECT_EQ(xaxis_2d_nbins, xaxis_pxy_nbins);
   expect_list_eq_names(*labels_2d, *labels_pxy);
}

// Test projection from TH3D for correct output on user range on projected axis
TEST(Projections, RangesAndOptionO)
{
   TH3D h("h", "h", 3, 0., 3., 3, 0., 3., 3, 0., 3.);

   for (int ix = 1; ix <= 3; ++ix) {
      for (int iy = 1; iy <= 3; ++iy) {
         for (int iz = 1; iz <= 3; ++iz) {
            auto bin = h.GetBin(ix, iy, iz);
            h.SetBinContent(bin, 100 * ix + 10 * iy + iz);
         }
      }
   }

   h.GetXaxis()->SetRange(2, 3);
   auto expectedForX = [](int ix) {
      double s = 0.;
      for (int iy = 1; iy <= 3; ++iy)
         for (int iz = 1; iz <= 3; ++iz)
            s += 100 * ix + 10 * iy + iz;
      return s;
   };
   auto x2 = expectedForX(2);
   auto x3 = expectedForX(3);

   {
      auto px = h.Project3D("x");
      EXPECT_EQ(px->GetNbinsX(), 2); // selected length
      EXPECT_DOUBLE_EQ(px->GetBinContent(1), x2);
      EXPECT_DOUBLE_EQ(px->GetBinContent(2), x3);
   }

   {
      auto pxo = h.Project3D("xo");
      ASSERT_NE(pxo, nullptr);
      EXPECT_EQ(pxo->GetNbinsX(), 3);               // original length
      EXPECT_DOUBLE_EQ(pxo->GetBinContent(1), 0.0); // outside selection
      EXPECT_DOUBLE_EQ(pxo->GetBinContent(2), x2);
      EXPECT_DOUBLE_EQ(pxo->GetBinContent(3), x3);
   }
}

// Test projection from TH3D for correct output on user range on integrated axis
TEST(Projections, SelectionAcrossNonTargetAxis)
{
   TH3D h("h", "h", 3, 0., 3., 3, 0., 3., 3, 0., 3.);

   for (int ix = 1; ix <= 3; ++ix) {
      for (int iy = 1; iy <= 3; ++iy) {
         for (int iz = 1; iz <= 3; ++iz) {
            auto bin = h.GetBin(ix, iy, iz);
            h.SetBinContent(bin, 100 * ix + 10 * iy + iz);
         }
      }
   }

   h.GetYaxis()->SetRange(2, 3);

   auto px = h.Project3D("x");

   auto expectedForX = [](int ix) {
      double s = 0.;
      for (int iy = 2; iy <= 3; ++iy)
         for (int iz = 1; iz <= 3; ++iz)
            s += 100 * ix + 10 * iy + iz;
      return s;
   };

   EXPECT_DOUBLE_EQ(px->GetBinContent(1), expectedForX(1));
   EXPECT_DOUBLE_EQ(px->GetBinContent(2), expectedForX(2));
   EXPECT_DOUBLE_EQ(px->GetBinContent(3), expectedForX(3));
}

// Test TH2D projection for correctness for user ranges on both axes
TEST(Projections, ProjectionYRange)
{
   Double_t xedges[] = {0, 1, 2};
   Double_t yedges[] = {-2, -1, 0, 1, 2};
   TH2D h("h", "h;X;Y", 2, xedges, 4, yedges);

   for (int ix = 1; ix <= 3; ++ix) {
      for (int iy = 1; iy <= 4; ++iy) {
         auto bin = h.GetBin(ix, iy);
         h.SetBinContent(bin, 10 * ix + iy);
      }
   }

   h.GetXaxis()->SetRange(1, 2);
   h.GetYaxis()->SetRange(2, 4);

   auto py = h.ProjectionY();

   auto expectedForY = [](int iy) {
      double s = 0.;
      for (int ix = 1; ix <= 2; ++ix)
         s += 10 * ix + iy;
      return s;
   };

   EXPECT_DOUBLE_EQ(py->GetBinContent(0), 0.0);
   EXPECT_DOUBLE_EQ(py->GetBinContent(1), expectedForY(2));
   EXPECT_DOUBLE_EQ(py->GetBinContent(2), expectedForY(3));
   EXPECT_DOUBLE_EQ(py->GetBinContent(3), expectedForY(4));
   EXPECT_DOUBLE_EQ(py->GetBinContent(4), 0.0);
}
// Test TH2D projection for correct flow inclusion for default options
TEST(Projections, UFOF)
{
   TH2D h2("h2", "", 3, 0, 3, 4, 0, 4);
   h2.Sumw2();
   for (int bx = 0; bx <= 4; ++bx) {
      for (int by = 0; by <= 5; ++by) {
         h2.SetBinContent(bx, by, 10 * bx + by);
         h2.SetBinError(bx, by, 1.0);
      }
   }

   auto hpx = h2.ProjectionX();
   for (int bx = 0; bx <= 4; ++bx) {
      const double exp = 60 * bx + 15;
      double got = hpx->GetBinContent(bx);
      EXPECT_DOUBLE_EQ(got, exp);
      double e = hpx->GetBinError(bx);
      EXPECT_DOUBLE_EQ(e, TMath::Sqrt(6.0));
   }
}
// Test TH2D projection for correct flow exclusion for specified user range
TEST(Projections, UFOFWithRange)
{
   TH2D h2("h2", "", 5, 0, 5, 4, 0, 4);
   h2.Sumw2();
   for (int bx = 0; bx <= 6; ++bx)
      for (int by = 0; by <= 5; ++by) {
         h2.SetBinContent(bx, by, 100 * bx + by);
         h2.SetBinError(bx, by, 2.0);
      }

   h2.GetXaxis()->SetRange(2, 4);

   auto hpx = h2.ProjectionX("hpx_ranged", 1, 4, "");
   EXPECT_EQ(hpx->GetXaxis()->GetNbins(), 3);
   for (int i = 0; i < 3; ++i) {
      int outbin = 2 + i;
      double exp = 0;
      for (int by = 1; by <= 4; ++by)
         exp += 100 * outbin + by;
      double got = hpx->GetBinContent(i + 1);
      EXPECT_DOUBLE_EQ(got, exp);
      EXPECT_DOUBLE_EQ(hpx->GetBinError(i + 1), 4.0);
   }
   EXPECT_DOUBLE_EQ(hpx->GetBinContent(0), 0);
   EXPECT_DOUBLE_EQ(hpx->GetBinContent(4), 0);
}

// Test TH2D projection correctness with option "o"
TEST(Projections, OriginalRange)
{
   TH2D h2("h2", "", 6, 0, 6, 3, 0, 3);
   for (int bx = 0; bx <= 7; ++bx)
      for (int by = 0; by <= 4; ++by)
         h2.SetBinContent(bx, by, bx + 10 * by);

   h2.GetXaxis()->SetRange(2, 5);

   auto hpx = h2.ProjectionX("h_o", 1, 3, "o");
   EXPECT_EQ(hpx->GetXaxis()->GetNbins(), 6);
   for (int bx = 1; bx <= 6; ++bx) {
      double got = hpx->GetBinContent(bx);
      if (bx < 2 || bx > 5) {
         EXPECT_EQ(got, 0.0);
         continue;
      }
      double exp = 0;
      for (int by = 1; by <= 3; ++by)
         exp += bx + 10 * by;
      EXPECT_DOUBLE_EQ(got, exp);
   }
}

// Test TH2D projection with variable bins and user range along projected axis
TEST(Projections, VarBinsRange)
{
   double edgesX[] = {0, 0.5, 1.5, 3.0, 5.0};
   TH2D h("h", "", 4, edgesX, 2, 0, 2);

   for (int bx = 0; bx <= 5; ++bx)
      for (int by = 0; by <= 3; ++by)
         h.SetBinContent(bx, by, 100 * bx + by);

   h.GetXaxis()->SetRange(2, 3);

   auto hpx = h.ProjectionX("hpx_var", 1, 2, "");
   EXPECT_EQ(hpx->GetXaxis()->GetNbins(), 2);
   EXPECT_DOUBLE_EQ(hpx->GetXaxis()->GetBinLowEdge(1), edgesX[1]);
   EXPECT_DOUBLE_EQ(hpx->GetXaxis()->GetBinUpEdge(2), edgesX[3]);

   for (int i = 0; i < 2; ++i) {
      int outbin = 2 + i;
      double exp = 0;
      for (int by = 1; by <= 2; ++by)
         exp += 100 * outbin + by;
      double got = hpx->GetBinContent(i + 1);
      EXPECT_DOUBLE_EQ(got, exp);
   }
}

// https://github.com/root-project/issues/20174
TEST(Projections, ProjectionYInfiniteUpperEdge)
{
   Double_t xedges[] = {0., 1.};
   Double_t yedges[] = {1., std::numeric_limits<double>::infinity()};
   TH2D h("h_inf", "h_inf;X;Y", 1, xedges, 1, yedges);
   h.SetBinContent(1, 1, 11.);
   auto projY = h.ProjectionY();
   EXPECT_EQ(projY->GetBinContent(1), h.Integral(1, 1, 1, 1));
}
TEST(Projections, ProjectionYInfiniteLowerEdge)
{
   Double_t xedges[] = {0, 1.};
   Double_t yedges[] = {-std::numeric_limits<double>::infinity(), 1};
   TH2D h("h_inf", "h_inf;X;Y", 1, xedges, 1, yedges);
   h.SetBinContent(1, 1, 11.);
   auto projY = h.ProjectionY();
   EXPECT_EQ(projY->GetBinContent(1), h.Integral(1, 1, 1, 1));
}

TEST(Projections, Projection3DInfiniteEdges)
{
   Double_t xedges[] = {0, 1.};
   Double_t yedges[] = {-std::numeric_limits<double>::infinity(), 1};
   Double_t zedges[] = {0, std::numeric_limits<double>::infinity()};
   TH3D h("h_inf", "h_inf;X;Y;Z", 1, xedges, 1, yedges, 1, zedges);
   h.SetBinContent(1, 1, 1, 11.);
   auto projY = h.Project3D("zy");
   EXPECT_EQ(projY->GetBinContent(projY->GetBin(1, 1)), h.Integral(1, 1, 1, 1, 1, 1));

   auto projx = h.Project3D("x");
   EXPECT_EQ(projx->GetBinContent(projx->GetBin(1)), h.Integral(1, 1, 1, 1, 1, 1));
}
