// Unit tests for TScatter and TScatter2D improvements (Issue #16376)

#include "gtest/gtest.h"

#include "TCanvas.h"
#include "TH2.h"
#include "TROOT.h"
#include "TScatter.h"
#include "TScatter2D.h"
#include "TView.h"

#include <string>

TEST(TScatterTest, GetObjectInfoAndDistance) {
  gROOT->SetBatch(true);
  TCanvas c("c_scatter", "c_scatter", 800, 600);

  const int n = 3;
  double x[n] = {10.0, 20.0, 30.0};
  double y[n] = {100.0, 200.0, 300.0};
  double col[n] = {1.5, 2.5, 3.5};
  double size[n] = {10.0, 20.0, 30.0};

  TScatter scatter(n, x, y, col, size);
  scatter.GetZaxis()->SetRangeUser(1.0, 5.0);
  scatter.Draw("A");
  c.Update();

  // Test GetObjectInfo on an exact marker position (point 1)
  Int_t px1 = c.XtoAbsPixel(c.XtoPad(x[1]));
  Int_t py1 = c.YtoAbsPixel(c.YtoPad(y[1]));
  char *info1 = scatter.GetObjectInfo(px1, py1);
  ASSERT_NE(info1, nullptr);
  std::string s1(info1);
  EXPECT_NE(s1.find("point=1"), std::string::npos);
  EXPECT_NE(s1.find("xval=20"), std::string::npos);
  EXPECT_NE(s1.find("yval=200"), std::string::npos);
  EXPECT_NE(s1.find("col=2.5"), std::string::npos);
  EXPECT_NE(s1.find("size=20"), std::string::npos);

  // Test GetObjectInfo away from any marker
  Int_t px_empty = c.XtoAbsPixel(c.XtoPad(5.0));
  Int_t py_empty = c.YtoAbsPixel(c.YtoPad(50.0));
  char *infoEmpty = scatter.GetObjectInfo(px_empty, py_empty);
  ASSERT_NE(infoEmpty, nullptr);
  std::string sEmpty(infoEmpty);
  EXPECT_EQ(sEmpty.find("point="), std::string::npos);
  EXPECT_NE(sEmpty.find("x="), std::string::npos);
  EXPECT_NE(sEmpty.find("y="), std::string::npos);

  // Test DistancetoPrimitive for exact position and positions away
  int dist_near = scatter.DistancetoPrimitive(px1, py1);
  EXPECT_EQ(dist_near, 0);

  int dist_close = scatter.DistancetoPrimitive(px1 + 2, py1 + 2);
  EXPECT_LE(dist_close, 3);

  int dist_far = scatter.DistancetoPrimitive(px_empty, py_empty);
  EXPECT_GT(dist_far, 10);
}

TEST(TScatter2DTest, ObjectInfoAndDistance) {
  gROOT->SetBatch(true);
  TCanvas c("c_scatter2d", "c_scatter2d", 800, 600);

  const int n = 3;
  double x[n] = {10.0, 20.0, 30.0};
  double y[n] = {100.0, 200.0, 300.0};
  double z[n] = {1.0, 2.0, 3.0};
  double col[n] = {1.5, 2.5, 3.5};
  double size[n] = {10.0, 20.0, 30.0};

  TScatter2D scatter2d(n, x, y, z, col, size);
  scatter2d.Draw();
  c.Update();

  TView *view = c.GetView();
  ASSERT_NE(view, nullptr);

  // Test GetObjectInfo on projected position of point 1 (20, 200, 2)
  Double_t p3[3] = {x[1], y[1], z[1]};
  Double_t pndc[3];
  view->WCtoNDC(p3, pndc);
  Int_t px1 = c.XtoAbsPixel(c.XtoPad(pndc[0]));
  Int_t py1 = c.YtoAbsPixel(c.YtoPad(pndc[1]));

  char *info1 = scatter2d.GetObjectInfo(px1, py1);
  ASSERT_NE(info1, nullptr);
  std::string s1(info1);
  EXPECT_NE(s1.find("point=1"), std::string::npos);
  EXPECT_NE(s1.find("xval=20"), std::string::npos);
  EXPECT_NE(s1.find("yval=200"), std::string::npos);
  EXPECT_NE(s1.find("zval=2"), std::string::npos);
  EXPECT_NE(s1.find("col=2.5"), std::string::npos);
  EXPECT_NE(s1.find("size=20"), std::string::npos);

  // Test GetObjectInfo away from any marker
  Int_t px_empty = c.XtoAbsPixel(0.1);
  Int_t py_empty = c.YtoAbsPixel(0.1);
  char *infoEmpty = scatter2d.GetObjectInfo(px_empty, py_empty);
  ASSERT_NE(infoEmpty, nullptr);
  std::string sEmpty(infoEmpty);
  EXPECT_EQ(sEmpty.find("point="), std::string::npos);
  EXPECT_NE(sEmpty.find("x="), std::string::npos);
  EXPECT_NE(sEmpty.find("y="), std::string::npos);

  // Test DistancetoPrimitive for several positions: exact, close, intermediate,
  // and far
  int dist_exact = scatter2d.DistancetoPrimitive(px1, py1);
  EXPECT_EQ(dist_exact, 0);

  int dist_close = scatter2d.DistancetoPrimitive(px1 + 2, py1 + 2);
  EXPECT_LE(dist_close, 3);

  int dist_mid = scatter2d.DistancetoPrimitive(px1 + 10, py1 + 10);
  EXPECT_GT(dist_mid, dist_close);

  int dist_far = scatter2d.DistancetoPrimitive(px_empty, py_empty);
  EXPECT_GT(dist_far, 50);
}
