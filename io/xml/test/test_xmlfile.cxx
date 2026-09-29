#include <gtest/gtest.h>
#include "TXMLFile.h"
#include "TH1I.h"
#include "TObjString.h"
#include "ROOT/TestSupport.hxx"

TEST(TXMLFileTest, WriteAndReadHistogram)
{
   ROOT::TestSupport::FileRaii xmlFile{"test_xmlfile1.xml"};

   {
      TXMLFile f(xmlFile.GetPath().c_str(), "RECREATE");
      ASSERT_FALSE(f.IsZombie());

      auto h = new TH1I("h1", "Test Histogram", 10, 0, 10);
      h->Fill(3);
      h->Write("myhist");
      delete h;

      auto str = new TObjString("Hello XML!");
      str->Write("mystr");
      delete str;
   }

   {
      TXMLFile f(xmlFile.GetPath().c_str(), "READ");
      ASSERT_FALSE(f.IsZombie());

      auto h2 = dynamic_cast<TH1I *>(f.Get("myhist"));
      ASSERT_NE(h2, nullptr);
      EXPECT_EQ(h2->GetEntries(), 1);
      EXPECT_STREQ(h2->GetTitle(), "Test Histogram");
      delete h2;

      auto str2 = dynamic_cast<TObjString *>(f.Get("mystr"));
      ASSERT_NE(str2, nullptr);
      EXPECT_STREQ(str2->GetString().Data(), "Hello XML!");
      delete str2;
   }
}
