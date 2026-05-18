#include "gtest/gtest.h"

#include <string>

#include "TNamed.h"
#include "TH1.h"
#include "TBufferJSON.h"
#include "THttpCallArg.h"
#include "TROOT.h"
#include "TRootSnifferFull.h"

#include "ROOT/TestSupport.hxx"


// simple class to access protected method

class TDecodeTest : public TRootSniffer {
   public:
      TDecodeTest() : TRootSniffer("sniffer") {}

      std::string Decode(const char *value, Bool_t remove_quotes = kTRUE)
      {
         TString res = DecodeUrlOptionValue(value, remove_quotes);
         return res.Data();
      }
};

// check basic URL parameters decoding
TEST(TRootSniffer, decode_url_options)
{
   TDecodeTest test;

   EXPECT_EQ(test.Decode(""), "");

   // single quote has to be escaped
   EXPECT_EQ(test.Decode("\""), "\\\"");

   // single backalsh has to be escaped
   EXPECT_EQ(test.Decode("\\"), "\\\\");

   // remove quotes
   EXPECT_EQ(test.Decode("\"\""), "");

   // remove quotes and escape quotes
   EXPECT_EQ(test.Decode("\"\"\""), "\\\"");

   // remove quotes and escape backslah
   EXPECT_EQ(test.Decode("\"\\\""), "\\\\");

   // remove quotes and remove special charsescape backslah
   EXPECT_EQ(test.Decode("\"abc\njkl\t\""), "abcjkl");

   // escape quotes in the middle
   EXPECT_EQ(test.Decode("someFunc(\"someArg\");someArray[3];"), "someFunc(\\\"someArg\\\");someArray[3];");

   // keep quotes
   EXPECT_EQ(test.Decode("\"\"", kFALSE), "\"\"");

   // keep quotes and escape inside quotes
   EXPECT_EQ(test.Decode("\"\"\"", kFALSE), "\"\\\"\"");

   // keep quotes and keep german letters - remove new line
   EXPECT_EQ(test.Decode("\"Gänse\nfüßchen\"", kFALSE), "\"Gänsefüßchen\"");
}

// check JSON representation for the objects
TEST(TRootSniffer, root_json)
{
   TNamed obj("obj", "title");

   TRootSnifferFull sniffer("sniffer");

   sniffer.RegisterObject("/", &obj);

   std::string res;
   sniffer.Produce("/obj", "root.json", "", res);
   EXPECT_EQ(res, "{\n"
                  "  \"_typename\" : \"TNamed\",\n"
                  "  \"fUniqueID\" : 0,\n"
                  "  \"fBits\" : 8,\n"
                  "  \"fName\" : \"obj\",\n"
                  "  \"fTitle\" : \"title\"\n"
                  "}");
}

// check XML representation for the objects
TEST(TRootSniffer, root_xml)
{
   TNamed obj("obj", "title");

   TRootSnifferFull sniffer("sniffer");

   sniffer.RegisterObject("/", &obj);

   std::string res;
   sniffer.Produce("/obj", "root.xml", "", res);
   EXPECT_EQ(res, "<Object class=\"TNamed\">\n"
                  "  <TNamed version=\"1\">\n"
                  "    <TObject fUniqueID=\"0\" fBits=\"8\"/>\n"
                  "    <fName str=\"obj\"/>\n"
                  "    <fTitle str=\"title\"/>\n"
                  "  </TNamed>\n"
                  "</Object>\n");
}

// check BINARY representation for the objects
TEST(TRootSniffer, root_bin)
{
   TNamed obj("obj", "title");

   TRootSnifferFull sniffer("sniffer");

   sniffer.RegisterObject("/", &obj);

   std::string res;
   sniffer.Produce("/obj", "root.bin", "", res);
   // keep minimal margin for binary format change
   EXPECT_NEAR(res.length(), 26, 4);
}

// check hierarchy request
TEST(TRootSniffer, item_json)
{
   TNamed obj("obj", "title");

   TRootSnifferFull sniffer("sniffer");

   sniffer.RegisterObject("/", &obj);

   std::string res;
   sniffer.Produce("/obj", "item.json", "", res);

   EXPECT_EQ(res, "{\n"
                  "  \"_name\" : \"obj\",\n"
                  "  \"_root_version\" : " + std::to_string(gROOT->GetVersionCode()) + ",\n"
                  "  \"_kind\" : \"ROOT.TNamed\",\n"
                  "  \"_title\" : \"title\"\n"
                  "}") << "return value of item.json";
}

// simple method execution
TEST(TRootSniffer, exe_json)
{
   TNamed obj("obj","title");

   TRootSnifferFull sniffer("sniffer");

   sniffer.RegisterObject("/", &obj);

   std::string res0;
   // by default methods execution is not allowed
   sniffer.Produce("/obj", "exe.json", "method=GetTitle", res0);
   EXPECT_EQ(res0, "") << "return value of exe.json in readonly";

   // disable readonly to get method executed
   sniffer.SetReadOnly(kFALSE);

   // only now one can execute method
   std::string res1;
   sniffer.Produce("/obj", "exe.json", "method=GetTitle", res1);
   EXPECT_EQ(res1, "\"title\"") << "return value of exe.json for GetTitle";
}


// execute method with post data - lot of gymnastic around
TEST(TRootSniffer, exe_post_json)
{
   TH1I hist("hist", "title", 10, 0, 10);
   hist.SetDirectory(nullptr);
   hist.SetBinContent(5, 10);

   std::string json;
   {
      // only temporary to create json
      TH1I hist2("hist", "title", 10, 0, 10);
      hist.SetDirectory(nullptr);
      hist2.SetBinContent(5, 20);
      json = TBufferJSON::ToJSON(&hist2).Data();
   }

   TRootSnifferFull sniffer("sniffer");

   sniffer.RegisterObject("/", &hist);

   // disable readonly to execute method
   sniffer.SetReadOnly(kFALSE);
   // allow use of POST data to decode object from JSON
   sniffer.SetAllowPostObject(kTRUE);

   THttpCallArg arg;
   arg.SetPostData(std::move(json));
   sniffer.SetCurrentCallArg(&arg);

   // before execution content is 10
   EXPECT_EQ(hist.GetBinContent(5), 10);

   std::string res;
   sniffer.Produce("/hist", "exe.json", "method=Add&prototype='const TH1*,Double_t'&h1=_post_object_json_&_destroy_post_", res);
   EXPECT_EQ(res, "1") << "return value of exe.json";

   // and now most important - bin content has to change
   EXPECT_EQ(hist.GetBinContent(5), 30) << "check of histogram content";
}

// changing object title
TEST(TRootSniffer, set_title)
{
   TNamed obj("obj", "title");

   TRootSnifferFull sniffer("sniffer");
   // disable readonly to get method executed
   sniffer.SetReadOnly(kFALSE);

   sniffer.RegisterObject("/", &obj);

   std::string res;

   sniffer.Produce("/obj", "exe.json", "method=SetTitle&title=NewTitle", res);
   EXPECT_EQ(res, "null") << "return value of exe.json when methout return void";
   EXPECT_EQ(std::string("NewTitle"), obj.GetTitle()) << "compare object title with applied value";

   res = "";
   sniffer.Produce("/obj", "exe.json", "method=SetTitle&title=\"QuotedTitle\"", res);
   EXPECT_EQ(res, "null");
   EXPECT_EQ(std::string("QuotedTitle"), obj.GetTitle()) << "compare object title with applied value";

   res = "";
   sniffer.Produce("/obj", "exe.json", "method=SetTitle&title=%22UrlStyleQuotedTitle%22", res);
   EXPECT_EQ(res, "null");
   EXPECT_EQ(std::string("UrlStyleQuotedTitle"), obj.GetTitle()) << "compare object title with applied value";

   res = "";
   sniffer.Produce("/obj", "exe.json", "method=SetTitle&title=Mail\"Formed\"Title", res);
   EXPECT_EQ(res, "null");
   EXPECT_EQ(std::string("Mail\"Formed\"Title"), obj.GetTitle()) << "compare object title with applied value";
}

// testing command execution with different signatures
TEST(TRootSniffer, cmd_json)
{
   TNamed obj("obj", "title");

   TRootSnifferFull sniffer("sniffer");
   sniffer.SetReadOnly(kFALSE);

   sniffer.RegisterObject("/", &obj);
   sniffer.RegisterCommand("/Print1", "/obj/->Print(%arg1%)", "");
   sniffer.RegisterCommand("/Print2", "/obj/->Print(\"%arg1%\")", "");
   sniffer.RegisterCommand("/GetSize", "/obj/->Sizeof()", "");

   std::string res;
   // quotes are in URL
   sniffer.Produce("/Print1", "cmd.json", "arg1=%22*%22", res);
   EXPECT_EQ(res, "0") << "return value of cmd.json";

   res = "";
   // skipping quotes from URL - when they are necessary
   // sniffer should have add them automatically
   sniffer.Produce("/Print1", "cmd.json", "arg1=*", res);
   EXPECT_EQ(res, "0") << "return value of cmd.json";

   res = "";
   // skipping quotes from URL - when they are necessary
   // while value looks like number, sniffer will not quote it
   // result of process line is not result is
   sniffer.Produce("/Print1", "cmd.json", "arg1=0", res);
   EXPECT_EQ(res, "0") << "return value of cmd.json";

   res = "";
   // quotes are in command definition
   sniffer.Produce("/Print2", "cmd.json", "arg1=*", res);
   EXPECT_EQ(res, "0") << "return value of cmd.json";

   res = "";
   // quotes are in command definition but we try to add our own
   // sniffer will remove them
   sniffer.Produce("/Print2", "cmd.json", "arg1=\"*\"", res);
   EXPECT_EQ(res, "0") << "return value of cmd.json";

   res = "";
   // Execute command which returns some value
   sniffer.Produce("/GetSize", "cmd.json", "", res);
   // returns only strings sizes
   EXPECT_EQ(res, "10") << "return value of cmd.json with object size";
}

// check JSON representation for the objects
TEST(TRootSniffer, multi_json)
{
   TNamed obj1("obj1", "title1");
   TNamed obj2("obj2", "title2");

   TRootSnifferFull sniffer("sniffer");

   sniffer.RegisterObject("/", &obj1);
   sniffer.RegisterObject("/", &obj2);

   std::string items = "/obj1/root.json\n/obj2/root.json\n";

   THttpCallArg arg;
   arg.SetPostData(std::move(items));
   sniffer.SetCurrentCallArg(&arg);

   std::string res;
   sniffer.Produce("", "multi.json", "number=2", res);
   EXPECT_EQ(res, "[{\n"
                  "  \"_typename\" : \"TNamed\",\n"
                  "  \"fUniqueID\" : 0,\n"
                  "  \"fBits\" : 8,\n"
                  "  \"fName\" : \"obj1\",\n"
                  "  \"fTitle\" : \"title1\"\n"
                  "}, {\n"
                  "  \"_typename\" : \"TNamed\",\n"
                  "  \"fUniqueID\" : 0,\n"
                  "  \"fBits\" : 8,\n"
                  "  \"fName\" : \"obj2\",\n"
                  "  \"fTitle\" : \"title2\"\n"
                  "}]") << "return value of multi.json";
}
