#include "gtest/gtest.h"

#include "TNamed.h"

TEST(TNamed, Sanity)
{
   TNamed n("Name", "Title");
   EXPECT_STREQ("Name", n.GetName());
   EXPECT_STREQ("Title", n.GetTitle());
}
