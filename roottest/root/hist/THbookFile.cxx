#include <THbookFile.h>
#include <TH1F.h>

#include "gtest/gtest.h"

TEST(THbookFile, ReadTH1)
{
   THbookFile file("mb4i1.hbook");
   EXPECT_STREQ(file.GetCurDir(), "//lun10");
   EXPECT_EQ(file.GetListOfKeys()->size(), 9);

   TObject *histo = file.Get(1);
   auto th1 = dynamic_cast<TH1F *>(histo);
   ASSERT_NE(th1, nullptr);
   EXPECT_EQ(th1->GetEntries(), 500);
}
