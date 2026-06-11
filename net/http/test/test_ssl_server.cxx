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

void cleanup_files()
{
   gSystem->Unlink("server.pem");
   gSystem->Unlink("server.crt");
   gSystem->Unlink("server.key");
   gSystem->Unlink("server.key.orig");
}

// main http server
TEST(THttpServer, ssl)
{
   cleanup_files();

   int res = gSystem->Exec("openssl genrsa -des3 -passout pass:aaaa -out server.key 2048");
   EXPECT_EQ(res, 0) << "Generate new RSA key";
   if (res) {
      cleanup_files();
      return;
   }

   res = gSystem->Exec("openssl req -new -passin pass:aaaa -key server.key -subj \"/C=GE/ST=Hesse/L=Darmstadt/O=GSI/CN=localhost\" -out server.csr");
   EXPECT_EQ(res, 0) << "Generate new server key";
   if (res) {
      cleanup_files();
      return;
   }

   gSystem->CopyFile("server.key", "server.key.orig");

   res = gSystem->Exec("openssl rsa -in server.key.orig -passin pass:aaaa -out server.key");
   EXPECT_EQ(res, 0) << "Convert key into RSA";
   if (res) {
      cleanup_files();
      return;
   }

   res = gSystem->Exec("openssl x509 -req -days 3650 -in server.csr -signkey server.key -out server.crt");
   EXPECT_EQ(res, 0) << "Generate server certificate";
   if (res) {
      cleanup_files();
      return;
   }

   res = gSystem->Exec("cat server.crt server.key > server.pem");
   EXPECT_EQ(res, 0) << "Generate server certificate";
   if (res) {
      cleanup_files();
      return;
   }

   if (gSystem->AccessPathName("server.pem")) {
      std::cerr << "Fail to access server.pem file";
      cleanup_files();
      return;
   }

   if (gSystem->AccessPathName("server.crt")) {
      std::cerr << "Fail to access server.crt file";
      cleanup_files();
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

   EXPECT_NE(httpport, 0);

   if (!httpport) {
      cleanup_files();
      return;
   }

   server_hash = httpport;
   unix_socket = "--cacert server.crt"; // curl argument
   server_url = TString::Format("https:/localhost:%d", httpport);

   test_suite(serv);

   cleanup_files();
}
