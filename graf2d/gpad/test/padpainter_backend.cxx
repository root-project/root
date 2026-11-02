#include "gtest/gtest.h"

#include "TROOT.h"
#include "TSystem.h"
#include "TVirtualPadPainter.h"

#include <memory>
#include <string>

TEST(TPadPainterBackend, BatchQueriesDoNotLoadLibraries)
{
   gROOT->SetBatch(kTRUE);
   std::unique_ptr<TVirtualPadPainter> painter(TVirtualPadPainter::PadPainter("native"));
   ASSERT_NE(painter, nullptr);

   const std::string libraries = gSystem->GetLibraries();
   EXPECT_FALSE(painter->IsX11());
   EXPECT_EQ(std::string(gSystem->GetLibraries()), libraries);
   EXPECT_FALSE(painter->IsCocoa());
   EXPECT_EQ(std::string(gSystem->GetLibraries()), libraries);
   EXPECT_FALSE(painter->IsSupportAlpha());
   EXPECT_EQ(std::string(gSystem->GetLibraries()), libraries);
}
