#include "gtest/gtest.h"

#include <string>
#include <fstream>
#include <iostream>

#include "THttpServer.h"
#include "TROOT.h"

#include "TSystem.h"
#include "TNamed.h"
#include "TRandom.h"

#include "ROOT/TestSupport.hxx"

#include "./test_suite.cxx"

// main http server
TEST(THttpServer, ssl)
{
   struct DoFilesCleanup {
      ~DoFilesCleanup()
      {
         gSystem->Unlink("server.pem");
         gSystem->Unlink("server.crt");
         gSystem->Unlink("server.key");
      }
   } docleanup;

   int res = gSystem->Exec("openssl genrsa -out server.key 2048");
   EXPECT_EQ(res, 0) << "Generate new RSA key";
   if (res)
      return;

   if (gSystem->AccessPathName("server.key")) {
      std::cerr << "Fail to access server.key file";
      return;
   }

   res = gSystem->Exec("openssl req -x509 -new -key server.key"
                       " -out server.crt"
                       " -days 3650 -sha256"
                       " -subj \"/C=GE/ST=Hesse/L=Darmstadt/O=GSI/CN=localhost\""
                       " -addext \"subjectAltName=DNS:localhost,IP:127.0.0.1,IP:::1\""
                       " -addext \"basicConstraints=critical,CA:TRUE\""
                       " -addext \"keyUsage=critical,digitalSignature,keyEncipherment,keyCertSign\"");
   EXPECT_EQ(res, 0) << "Generate new server certificate";
   if (res)
      return;

   if (gSystem->AccessPathName("server.crt")) {
      std::cerr << "Fail to access server.crt file";
      return;
   }

   res = gSystem->Exec("cat server.crt server.key > server.pem");
   EXPECT_EQ(res, 0) << "Generate server.pcm file for THttpServer";
   if (res)
      return;

   if (gSystem->AccessPathName("server.pem")) {
      std::cerr << "Fail to access server.pem file";
      return;
   }

   THttpServer serv("");

   gRandom->SetSeed(0);

   Int_t httpport = 0;

   for(int ntry = 0; ntry < 100; ++ntry) {
      Int_t port = (Int_t) (25000 + gRandom->Rndm() * 1000);
      // only two threads, bind to loopback address only
      TString arg = TString::Format("https:%d?loopback&ssl_cert=server.pem&thrds=3", port);
      if (serv.CreateEngine(arg)) {
         httpport = port;
         break;
      }
   }

   EXPECT_NE(httpport, 0) << "Fail to allocate HTTP port for test";
   if (!httpport)
      return;

   server_hash = httpport;
   unix_socket = "--cacert server.crt"; // curl argument
   server_url = TString::Format("https://localhost:%d", httpport);

   test_suite(serv);

   (void) docleanup; // object used only for automatic files cleanup
}
