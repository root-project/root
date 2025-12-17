#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <RZip.h>

TEST(RZip, HeaderBasics)
{
   unsigned char header[9] = {'Z', 'S', '\1', 1, 0, 0, 2, 1, 0};
   int srcsize = 0;
   int tgtsize = 0;

   EXPECT_EQ(0, R__unzip_header(&srcsize, header, &tgtsize));
   EXPECT_EQ(10, srcsize);
   EXPECT_EQ(258, tgtsize);
}
