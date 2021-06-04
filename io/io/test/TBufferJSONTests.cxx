#include "TBufferJSON.h"
#include "TNamed.h"
#include <string>

#include "gtest/gtest.h"

// check utf8 coding with two bytes - most frequent usecase
TEST(TBufferJSON, utf8_2)
{
   std::string str0 = u8"test \u0444"; // this should be cyrillic letter f

   auto len0 = str0.length();

   EXPECT_EQ(len0, 7);

   EXPECT_EQ((unsigned char) str0[len0-2], 0xd1); // utf8 coding for \u0444
   EXPECT_EQ((unsigned char) str0[len0-1], 0x84); // utf8 coding for \u0444

   TNamed named0("name", str0.c_str());

   auto json = TBufferJSON::ToJSON(&named0);

   auto named1 = TBufferJSON::FromJSON<TNamed>(json.Data());

   EXPECT_EQ(str0, named1->GetTitle());
}

// check utf8 coding with three bytes
TEST(TBufferJSON, utf8_3)
{
   std::string str0 = u8"test \u7546"; // no idea that

   auto len0 = str0.length();

   EXPECT_EQ(len0, 8);

   EXPECT_EQ((unsigned char) str0[len0-3], 0xe7);
   EXPECT_EQ((unsigned char) str0[len0-2], 0x95);
   EXPECT_EQ((unsigned char) str0[len0-1], 0x86);

   TNamed named0("name", str0.c_str());

   auto json = TBufferJSON::ToJSON(&named0);

   auto named1 = TBufferJSON::FromJSON<TNamed>(json.Data());

   EXPECT_EQ(str0, named1->GetTitle());
}

