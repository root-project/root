#include "TH2F.h"
#include "TH3F.h"
#include "TProfile.h"   // ProjectionX
#include "TProfile2D.h" // ProjectionX
#include "THashList.h"  // GetLabels

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
