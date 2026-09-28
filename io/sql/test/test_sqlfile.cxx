#include <gtest/gtest.h>
#include "TSQLFile.h"
#include "TH1I.h"
#include "TObjString.h"
#include "ROOT/TestSupport.hxx"
#include <string>

TEST(TSQLFileTest, WriteAndReadHistogram)
{
   ROOT::TestSupport::FileRaii dbFile{"test_sqlfile1.db"};
   std::string uri = std::string("sqlite://") + dbFile.GetPath();

   {
      TSQLFile f(uri.c_str(), "RECREATE");
      ASSERT_FALSE(f.IsZombie());

      TH1I h("h1", "Test Histogram", 10, 0, 10);
      h.Fill(3);
      h.Write("myhist");

      TObjString str("Hello SQL!");
      str.Write("mystr");
   }

   {
      TSQLFile f(uri.c_str(), "READ");
      ASSERT_FALSE(f.IsZombie());

      auto h2 = dynamic_cast<TH1I *>(f.Get("myhist"));
      ASSERT_NE(h2, nullptr);
      EXPECT_EQ(h2->GetEntries(), 1);
      EXPECT_STREQ(h2->GetTitle(), "Test Histogram");
      delete h2;

      auto str2 = dynamic_cast<TObjString *>(f.Get("mystr"));
      ASSERT_NE(str2, nullptr);
      EXPECT_STREQ(str2->GetString().Data(), "Hello SQL!");
      delete str2;
   }
}
