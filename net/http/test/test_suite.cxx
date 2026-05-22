Int_t server_hash = 0;
TString server_url;
TString unix_socket;

// submit requests to server
std::string execute_request(const char *url, const char *post = nullptr)
{
   TString fname = TString::Format("http_server_%d.output", server_hash),
           pname, exec;

   if (post) {
      pname = TString::Format("http_server_%d.post", server_hash);
      std::ofstream f(pname.Data());
      f << post;
   }

   if (post)
      exec = TString::Format("curl -sS -X POST %s '%s%s' --data-binary @%s -o %s", unix_socket.Data(), server_url.Data(), url, pname.Data(), fname.Data());
   else
      exec = TString::Format("curl -sS %s '%s%s' -o %s", unix_socket.Data(), server_url.Data(), url, fname.Data());

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


void test_suite(THttpServer &serv)
{
   // let process requests in separate thread
   serv.CreateServerThread();

   TNamed obj1("obj1", "title1");
   TNamed obj2("obj2", "title2");

   serv.Register("/", &obj1);
   serv.Register("/", &obj2);

   // check JSON representation for the object
   std::string res = execute_request("/obj1/root.json");
   EXPECT_EQ(res, "{\n"
                  "  \"_typename\" : \"TNamed\",\n"
                  "  \"fUniqueID\" : 0,\n"
                  "  \"fBits\" : 8,\n"
                  "  \"fName\" : \"obj1\",\n"
                  "  \"fTitle\" : \"title1\"\n"
                  "}") << "result of root.json request";

   // check XML representation for the object
   res = execute_request("/obj1/root.xml");
   EXPECT_EQ(res, "<Object class=\"TNamed\">\n"
                  "  <TNamed version=\"1\">\n"
                  "    <TObject fUniqueID=\"0\" fBits=\"8\"/>\n"
                  "    <fName str=\"obj1\"/>\n"
                  "    <fTitle str=\"title1\"/>\n"
                  "  </TNamed>\n"
                  "</Object>\n") << "result of root.xml request";


   // check BINARY representation for the object
   res = execute_request("/obj1/root.bin");
   // keep minimal margin for binary format change
   EXPECT_NEAR(res.length(), 28, 4) << "size of root.bin request";

   // check ROOT file creation
   res = execute_request("/obj1/file.root");
   // TODO: anlyze why so big size for small object
   EXPECT_NEAR(res.length(), 1024, 100) << "size of file.root request";

   // check item request hierarchy request
   res = execute_request("/obj1/item.json");

   EXPECT_EQ(res, "{\n"
                  "  \"_name\" : \"obj1\",\n"
                  "  \"_root_version\" : " + std::to_string(gROOT->GetVersionCode()) + ",\n"
                  "  \"_kind\" : \"ROOT.TNamed\",\n"
                  "  \"_title\" : \"title1\"\n"
                  "}") << "result of item.json request";


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
                  "}]") << "result of multi.json request";


   // by default methods execution is not allowed
   res = execute_request("/obj1/exe.json?method=GetTitle");
   EXPECT_EQ(res, "") << "exe.json should be empty in readonly";


   // disable readonly to get extra functionality
   serv.SetReadOnly(kFALSE);

   // only now one can execute method
   res = execute_request("/obj1/exe.json?method=GetTitle");
   EXPECT_EQ(res, "\"title1\"") << "result of exe.json with object title";

   res = execute_request("/obj1/exe.json?method=SetTitle&title=NewTitle");
   EXPECT_EQ(res, "null") << "returns null when executes void method";
   EXPECT_EQ(std::string("NewTitle"), obj1.GetTitle()) << "title must match with submitted value";
   obj1.SetTitle("title1");
}
