#include <gtest/gtest.h>
#include "TXMLFile.h"
#include "TXMLEngine.h"
#include "TH1I.h"
#include "TObjString.h"
#include "TList.h"
#include "TNamed.h"
#include "ROOT/TestSupport.hxx"

#include <memory>
#include <string>

// =============================================================================
// TXMLEngine Tests: DOM construction, attributes, parsing and serialization
// =============================================================================

TEST(TXMLEngineTest, ConstructAndParseString)
{
   TXMLEngine xml;

   // Build a DOM tree
   XMLNodePointer_t mainNode = xml.NewChild(nullptr, nullptr, "config");
   xml.NewAttr(mainNode, nullptr, "version", "1.0");

   XMLNodePointer_t child1 = xml.NewChild(mainNode, nullptr, "server", "localhost");
   xml.NewIntAttr(child1, "port", 8080);

   XMLNodePointer_t child2 = xml.NewChild(mainNode, nullptr, "timeout", "30");

   XMLDocPointer_t doc = xml.NewDoc();
   xml.DocSetRootElement(doc, mainNode);

   TString xmlStr;
   xml.SaveSingleNode(mainNode, &xmlStr);
   xml.FreeDoc(doc);

   EXPECT_FALSE(xmlStr.IsNull());

   // Parse back from string
   XMLDocPointer_t parsedDoc = xml.ParseString(xmlStr.Data());
   ASSERT_NE(parsedDoc, nullptr);

   XMLNodePointer_t root = xml.DocGetRootElement(parsedDoc);
   ASSERT_NE(root, nullptr);
   EXPECT_STREQ(xml.GetNodeName(root), "config");
   EXPECT_TRUE(xml.HasAttr(root, "version"));
   EXPECT_STREQ(xml.GetAttr(root, "version"), "1.0");

   XMLNodePointer_t n1 = xml.GetChild(root);
   ASSERT_NE(n1, nullptr);
   EXPECT_STREQ(xml.GetNodeName(n1), "server");
   EXPECT_STREQ(xml.GetNodeContent(n1), "localhost");
   EXPECT_EQ(xml.GetIntAttr(n1, "port"), 8080);

   XMLNodePointer_t n2 = xml.GetNext(n1);
   ASSERT_NE(n2, nullptr);
   EXPECT_STREQ(xml.GetNodeName(n2), "timeout");
   EXPECT_STREQ(xml.GetNodeContent(n2), "30");

   EXPECT_EQ(xml.GetNext(n2), nullptr);

   xml.FreeDoc(parsedDoc);
}

TEST(TXMLEngineTest, FileSaveAndParse)
{
   ROOT::TestSupport::FileRaii xmlFile{"test_engine_doc.xml"};

   {
      TXMLEngine xml;
      XMLNodePointer_t root = xml.NewChild(nullptr, nullptr, "dataset");
      xml.NewAttr(root, nullptr, "name", "Run2026");

      XMLNodePointer_t item = xml.NewChild(root, nullptr, "events", "10000");
      xml.NewAttr(item, nullptr, "type", "MC");

      XMLDocPointer_t doc = xml.NewDoc();
      xml.DocSetRootElement(doc, root);
      xml.SaveDoc(doc, xmlFile.GetPath().c_str());
      xml.FreeDoc(doc);
   }

   {
      TXMLEngine xml;
      XMLDocPointer_t doc = xml.ParseFile(xmlFile.GetPath().c_str());
      ASSERT_NE(doc, nullptr);

      XMLNodePointer_t root = xml.DocGetRootElement(doc);
      ASSERT_NE(root, nullptr);
      EXPECT_STREQ(xml.GetNodeName(root), "dataset");
      EXPECT_STREQ(xml.GetAttr(root, "name"), "Run2026");

      XMLNodePointer_t item = xml.GetChild(root);
      ASSERT_NE(item, nullptr);
      EXPECT_STREQ(xml.GetNodeName(item), "events");
      EXPECT_STREQ(xml.GetNodeContent(item), "10000");
      EXPECT_STREQ(xml.GetAttr(item, "type"), "MC");

      xml.FreeDoc(doc);
   }
}

// =============================================================================
// TXMLFile Tests: Object serialization, directories, keys, and retrieval
// =============================================================================

TEST(TXMLFileTest, WriteAndReadObjects)
{
   ROOT::TestSupport::FileRaii xmlFile{"test_xmlfile_objects.xml"};

   {
      TXMLFile f(xmlFile.GetPath().c_str(), "RECREATE");
      ASSERT_TRUE(f.IsOpen());
      ASSERT_FALSE(f.IsZombie());

      auto h1 = std::make_unique<TH1I>("h1", "Test Histogram", 10, 0, 10);
      h1->Fill(2);
      h1->Fill(5);
      h1->Fill(5);
      h1->Write("myhist");

      auto str = std::make_unique<TObjString>("Hello XML World");
      str->Write("mystr");

      auto named = std::make_unique<TNamed>("entryKey", "entryValue");
      named->Write("myentry");

      f.Close();
   }

   {
      TXMLFile f(xmlFile.GetPath().c_str(), "READ");
      ASSERT_TRUE(f.IsOpen());
      ASSERT_FALSE(f.IsZombie());

      std::unique_ptr<TH1I> h1(dynamic_cast<TH1I *>(f.Get("myhist")));
      ASSERT_NE(h1, nullptr);
      EXPECT_EQ(h1->GetEntries(), 3);
      EXPECT_EQ(h1->GetBinContent(h1->FindBin(2)), 1);
      EXPECT_EQ(h1->GetBinContent(h1->FindBin(5)), 2);
      EXPECT_STREQ(h1->GetTitle(), "Test Histogram");

      std::unique_ptr<TObjString> str(dynamic_cast<TObjString *>(f.Get("mystr")));
      ASSERT_NE(str, nullptr);
      EXPECT_STREQ(str->GetString().Data(), "Hello XML World");

      std::unique_ptr<TNamed> named(dynamic_cast<TNamed *>(f.Get("myentry")));
      ASSERT_NE(named, nullptr);
      EXPECT_STREQ(named->GetName(), "entryKey");
      EXPECT_STREQ(named->GetTitle(), "entryValue");

      f.Close();
   }
}

TEST(TXMLFileTest, SubdirectoriesAndKeyListing)
{
   ROOT::TestSupport::FileRaii xmlFile{"test_xmlfile_subdir.xml"};

   {
      TXMLFile f(xmlFile.GetPath().c_str(), "RECREATE");
      ASSERT_TRUE(f.IsOpen());

      TDirectory *subDir = f.mkdir("subfolder");
      ASSERT_NE(subDir, nullptr);
      subDir->cd();

      auto subStr = std::make_unique<TObjString>("Nested Content");
      subStr->Write("nestedStr");

      f.cd();
      auto rootStr = std::make_unique<TObjString>("Root Content");
      rootStr->Write("topStr");

      f.Close();
   }

   {
      TXMLFile f(xmlFile.GetPath().c_str(), "READ");
      ASSERT_TRUE(f.IsOpen());

      // Check root directory keys
      TList *keys = f.GetListOfKeys();
      ASSERT_NE(keys, nullptr);
      EXPECT_NE(keys->FindObject("subfolder"), nullptr);
      EXPECT_NE(keys->FindObject("topStr"), nullptr);

      std::unique_ptr<TObjString> topStr(dynamic_cast<TObjString *>(f.Get("topStr")));
      ASSERT_NE(topStr, nullptr);
      EXPECT_STREQ(topStr->GetString().Data(), "Root Content");

      // Check subdirectory keys and content
      TDirectory *subDir = nullptr;
      f.GetObject("subfolder", subDir);
      ASSERT_NE(subDir, nullptr);

      std::unique_ptr<TObjString> nestedStr(dynamic_cast<TObjString *>(subDir->Get("nestedStr")));
      ASSERT_NE(nestedStr, nullptr);
      EXPECT_STREQ(nestedStr->GetString().Data(), "Nested Content");

      f.Close();
   }
}
