#include "gtest/gtest.h"

#include <string>
#include <fstream>

#include "THttpServer.h"
#include "TROOT.h"

#include "TSystem.h"
#include "TNamed.h"
#include "TRandom.h"

#include "ROOT/TestSupport.hxx"

Int_t httpport = 0;
TString server_url;

// main http server
std::string execute_request(const char *url, const char *post = nullptr)
{
   TString fname = TString::Format("http_server_%d.output", httpport),
           pname, exec;

   if (post) {
      pname = TString::Format("http_server_%d.post", httpport);
      std::ofstream f(pname.Data());
      f << post;
      exec = TString::Format("curl -sS -X POST '%s%s' --data-binary @%s -o %s", server_url.Data(), url, pname.Data(), fname.Data());
   } else {
      exec = TString::Format("curl -sS '%s%s' -o %s", server_url.Data(), url, fname.Data());
   }

   printf("Execute %s\n", exec.Data());

   std::string res;

   if (gSystem->Exec(exec) != 0)
      res = "<fail>";
   else
      res = THttpServer::ReadFileContent(fname.Data());

   gSystem->Unlink(fname);
   if (!pname.IsNull())
      gSystem->Unlink(pname);

   return res;
}

// main http server
TEST(THttpServer, main)
{
   THttpServer serv("");

   gRandom->SetSeed(0);

   for(int ntry = 0; ntry < 100; ++ntry) {
      Int_t port = (Int_t) (25000 + gRandom->Rndm() * 1000);
      // only two threads, bind to loopback address only
      TString arg = TString::Format("http:%d?loopback&thrds=2", port);
      if (serv.CreateEngine(arg)) {
         httpport = port;
         break;
      }
   }

   EXPECT_NE(httpport, 0);

   if (!httpport)
      return;

   server_url = TString::Format("http:/localhost:%d", httpport);

   TNamed obj1("obj1", "title1");
   TNamed obj2("obj2", "title2");

   serv.Register("/", &obj1);
   serv.Register("/", &obj2);

   // let process requests in separate thread
   serv.CreateServerThread();

   // check JSON representation for the object
   std::string res = execute_request("/obj1/root.json");
   EXPECT_EQ(res, "{\n"
                  "  \"_typename\" : \"TNamed\",\n"
                  "  \"fUniqueID\" : 0,\n"
                  "  \"fBits\" : 8,\n"
                  "  \"fName\" : \"obj1\",\n"
                  "  \"fTitle\" : \"title1\"\n"
                  "}");

   // check XML representation for the object
   res = execute_request("/obj1/root.xml");
   EXPECT_EQ(res, "<Object class=\"TNamed\">\n"
                  "  <TNamed version=\"1\">\n"
                  "    <TObject fUniqueID=\"0\" fBits=\"8\"/>\n"
                  "    <fName str=\"obj1\"/>\n"
                  "    <fTitle str=\"title1\"/>\n"
                  "  </TNamed>\n"
                  "</Object>\n");


   // check BINARY representation for the object
   res = execute_request("/obj1/root.bin");
   // keep minimal margin for binary format change
   EXPECT_NEAR(res.length(), 28, 4);

   // check ROOT file creation
   res = execute_request("/obj1/file.root");
   // TODO: anlyze why so big size for small object
   EXPECT_NEAR(res.length(), 1024, 100);

   // check item request hierarchy request
   res = execute_request("/obj1/item.json");

   EXPECT_EQ(res, "{\n"
                  "  \"_name\" : \"obj1\",\n"
                  "  \"_root_version\" : " + std::to_string(gROOT->GetVersionCode()) + ",\n"
                  "  \"_kind\" : \"ROOT.TNamed\",\n"
                  "  \"_title\" : \"title1\"\n"
                  "}");


   // check multi request to several objects
   res = execute_request("/multi.json?number=2", "/obj1/root.json\n/obj2/root.json\n");
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
                  "}]");


   // by default methods execution is not allowed
   res = execute_request("/obj1/exe.json?method=GetTitle");
   EXPECT_EQ(res, "");


   // disable readonly to get extra functionality
   serv.SetReadOnly(kFALSE);

   // only now one can execute method
   res = execute_request("/obj1/exe.json?method=GetTitle");
   EXPECT_EQ(res, "\"title1\"");

   res = execute_request("/obj1/exe.json?method=SetTitle&title=NewTitle");
   EXPECT_EQ(res, "null");
   EXPECT_EQ(std::string("NewTitle"), obj1.GetTitle());
   obj1.SetTitle("title1");
}
