#include "TBufferJSON.h"
#include "TNamed.h"
#include "TList.h"
#include "TInterpreter.h"
#include <cmath>
#include <limits>
#include <string>

#include "gtest/gtest.h"

// check utf8 coding with two bytes - most frequent usecase
TEST(TBufferJSON, utf8_2)
{
   std::string str0 = reinterpret_cast<const char *>(u8"test \u0444"); // this should be cyrillic letter f

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
   std::string str0 = reinterpret_cast<const char *>(u8"test \u7546"); // no idea that

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

TEST(TBufferJSON, TList_Add_Without_Option)
{
   TList lst0;
   lst0.Add(new TNamed("name0", "title0"));  // without option
   lst0.Add(new TNamed("name1", "title1"), ""); // with empty option
   lst0.Add(new TNamed("name2", "title2"), "option2"); // with non-empty option
   auto json = TBufferJSON::ToJSON(&lst0);

   // auto f = new TMemFile("test.root", "RECREATE");
   // f->WriteTObject(&lst0, "list");

   lst0.Delete();

   auto lst1 = TBufferJSON::FromJSON<TList>(json.Data());

   // auto lst1 = f->Get<TList>("list");
   // delete f;

   EXPECT_NE(lst1, nullptr);

   auto link = lst1->FirstLink();

   EXPECT_STREQ("name0", link->GetObject()->GetName());
   EXPECT_STREQ("title0", link->GetObject()->GetTitle());
   EXPECT_STREQ("", link->GetAddOption()); // by default empty string returned
   EXPECT_EQ(dynamic_cast<TObjOptLink *>(link), nullptr);

   link = link->Next();

   EXPECT_STREQ("name1", link->GetObject()->GetName());
   EXPECT_STREQ("title1", link->GetObject()->GetTitle());
   EXPECT_STREQ("", link->GetAddOption());
   EXPECT_NE(dynamic_cast<TObjOptLink *>(link), nullptr); // this will fail in normal ROOT I/O

   link = link->Next();

   EXPECT_STREQ("name2", link->GetObject()->GetName());
   EXPECT_STREQ("title2", link->GetObject()->GetTitle());
   EXPECT_STREQ("option2", link->GetAddOption());
   EXPECT_NE(dynamic_cast<TObjOptLink *>(link), nullptr);

   link = link->Next();

   EXPECT_EQ(link, nullptr);
}

// https://github.com/root-project/root/issues/19768
struct ContentF {
   float v;
};
TEST(TBufferJSON, SpecialNumbersFloat)
{
   gInterpreter->Declare("struct ContentF { float v; };");
   ContentF *p = nullptr;
   TBufferJSON::FromJSON<ContentF>(p, "{ \"_typename\" : \"ContentF\", \"v\" : \"inff\" }");
   EXPECT_TRUE(p && std::isinf(p->v) && (p->v > 0));
   delete p;
   p = nullptr;
   TBufferJSON::FromJSON<ContentF>(p, "{ \"_typename\" : \"ContentF\", \"v\" : \"-inff\" }");
   EXPECT_TRUE(p && std::isinf(p->v) && (p->v < 0));
   delete p;
   p = nullptr;
   TBufferJSON::FromJSON<ContentF>(p, "{ \"_typename\" : \"ContentF\", \"v\" : \"nanf\" }");
   EXPECT_TRUE(p && std::isnan(p->v));
   delete p;
   p = nullptr;

   auto maxVal = std::numeric_limits<float>::max(); // 3.40282e+38f
   auto ovfVal = std::numeric_limits<float>::max() * static_cast<float>(1 + 1e-7);
   auto infVal = std::numeric_limits<float>::infinity();
   EXPECT_EQ(ovfVal, infVal);
   auto nanVal = std::numeric_limits<float>::quiet_NaN();
   p = new ContentF;
   p->v = maxVal;
   EXPECT_EQ(TBufferJSON::ToJSON(p, TBufferJSON::kStoreInfNaN),
             "{\n  \"_typename\" : \"ContentF\",\n  \"v\" : 3.402823e38\n}");
   p->v = -maxVal;
   EXPECT_EQ(TBufferJSON::ToJSON(p, TBufferJSON::kStoreInfNaN),
             "{\n  \"_typename\" : \"ContentF\",\n  \"v\" : -3.402823e38\n}");
   p->v = infVal;
   EXPECT_EQ(TBufferJSON::ToJSON(p, TBufferJSON::kStoreInfNaN),
             "{\n  \"_typename\" : \"ContentF\",\n  \"v\" : \"inff\"\n}");
   p->v = -infVal;
   EXPECT_EQ(TBufferJSON::ToJSON(p, TBufferJSON::kStoreInfNaN),
             "{\n  \"_typename\" : \"ContentF\",\n  \"v\" : \"-inff\"\n}");
   p->v = nanVal;
   EXPECT_EQ(TBufferJSON::ToJSON(p, TBufferJSON::kStoreInfNaN),
             "{\n  \"_typename\" : \"ContentF\",\n  \"v\" : \"nanf\"\n}");
   delete p;
   p = nullptr;
}

// https://github.com/root-project/root/issues/19768
struct ContentD {
   double v;
};
TEST(TBufferJSON, SpecialNumbersDouble)
{
   gInterpreter->Declare("struct ContentD { double v; };");
   ContentD *p = nullptr;
   TBufferJSON::FromJSON<ContentD>(p, "{ \"_typename\" : \"ContentD\", \"v\" : \"inf\" }");
   EXPECT_TRUE(p && std::isinf(p->v) && (p->v > 0));
   delete p;
   p = nullptr;
   TBufferJSON::FromJSON<ContentD>(p, "{ \"_typename\" : \"ContentD\", \"v\" : \"-inf\" }");
   EXPECT_TRUE(p && std::isinf(p->v) && (p->v < 0));
   delete p;
   p = nullptr;
   TBufferJSON::FromJSON<ContentD>(p, "{ \"_typename\" : \"ContentD\", \"v\" : \"nan\" }");
   EXPECT_TRUE(p && std::isnan(p->v));
   delete p;
   p = nullptr;

   auto maxVal = std::numeric_limits<double>::max(); // 1.7976931e+308
   auto ovfVal = std::numeric_limits<double>::max() * (1 + 1e-15);
   auto infVal = std::numeric_limits<double>::infinity();
   EXPECT_EQ(ovfVal, infVal);
   auto nanVal = std::numeric_limits<double>::quiet_NaN();
   p = new ContentD;
   p->v = maxVal;
   EXPECT_EQ(TBufferJSON::ToJSON(p, TBufferJSON::kStoreInfNaN),
             "{\n  \"_typename\" : \"ContentD\",\n  \"v\" : 1.79769313486232e308\n}");
   p->v = -maxVal;
   EXPECT_EQ(TBufferJSON::ToJSON(p, TBufferJSON::kStoreInfNaN),
             "{\n  \"_typename\" : \"ContentD\",\n  \"v\" : -1.79769313486232e308\n}");
   p->v = infVal;
   EXPECT_EQ(TBufferJSON::ToJSON(p, TBufferJSON::kStoreInfNaN),
             "{\n  \"_typename\" : \"ContentD\",\n  \"v\" : \"inf\"\n}");
   p->v = -infVal;
   EXPECT_EQ(TBufferJSON::ToJSON(p, TBufferJSON::kStoreInfNaN),
             "{\n  \"_typename\" : \"ContentD\",\n  \"v\" : \"-inf\"\n}");
   p->v = nanVal;
   EXPECT_EQ(TBufferJSON::ToJSON(p, TBufferJSON::kStoreInfNaN),
             "{\n  \"_typename\" : \"ContentD\",\n  \"v\" : \"nan\"\n}");
   delete p;
   p = nullptr;
}
