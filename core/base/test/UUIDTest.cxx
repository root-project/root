#include "gtest/gtest.h"

#include "TUUID.h"

#include <set>
#include <string>

TEST(TUUID, UUIDv4)
{
   std::set<TUUID> uuids;
   for (int i = 0; i < 10000; ++i) {
      uuids.insert(TUUID::UUIDv4());
   }
   EXPECT_EQ(10000u, uuids.size());

   TUUID u;
   EXPECT_EQ('1', u.AsString()[14]);
   u = TUUID::UUIDv4();
   std::string str = u.AsString();
   EXPECT_EQ('4', str[14]);
   EXPECT_TRUE(str[19] == '8' || str[19] == '9' || str[19] == 'a' || str[19] == 'b');
}
