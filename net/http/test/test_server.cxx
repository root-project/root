#include "gtest/gtest.h"

#include <filesystem>
#include <string>
#include <fstream>

#include "THttpServer.h"
#include "TROOT.h"

#include "TSystem.h"
#include "TNamed.h"
#include "TRandom.h"

#include "ROOT/TestSupport.hxx"

#include "./test_suite.cxx"

// main http server
TEST(THttpServer, main)
{
   THttpServer serv("");

   gRandom->SetSeed(0);

   Int_t httpport = 0;

   for(int ntry = 0; ntry < 100; ++ntry) {
      Int_t port = (Int_t) (25000 + gRandom->Rndm() * 1000);
      // only two threads, bind to loopback address only
      TString arg = TString::Format("http:%d?loopback&thrds=3", port);
      if (serv.CreateEngine(arg)) {
         httpport = port;
         break;
      }
   }

   EXPECT_NE(httpport, 0);

   if (!httpport)
      return;

   server_hash = httpport;
   server_url = TString::Format("http://localhost:%d", httpport);

   test_suite(serv);
}

// a prefix with several directories serves a file from the first one having it
TEST(THttpServer, locations)
{
   THttpServer serv("");

   TString top = TString::Format("http_locations_%d", gSystem->GetPid());
   TString dir1 = top + "/dir1/", dir2 = top + "/dir2/";
   ASSERT_EQ(gSystem->mkdir(dir1, kTRUE), 0);
   ASSERT_EQ(gSystem->mkdir(dir2, kTRUE), 0);
   std::ofstream((dir1 + "both.txt").Data()) << "1";
   std::ofstream((dir2 + "both.txt").Data()) << "2";
   std::ofstream((dir2 + "second.txt").Data()) << "2";

   TString res;

   // one directory: the path in it, whether the file exists or not
   serv.AddLocation("loc/", dir1);
   EXPECT_TRUE(serv.IsFileRequested("/loc/second.txt", res));
   EXPECT_EQ(res, dir1 + "second.txt");

   serv.AppendLocation("loc/", dir2);
   EXPECT_TRUE(serv.IsFileRequested("/loc/both.txt", res));
   EXPECT_EQ(res, dir1 + "both.txt");
   EXPECT_TRUE(serv.IsFileRequested("/loc/second.txt", res));
   EXPECT_EQ(res, dir2 + "second.txt");
   EXPECT_TRUE(serv.IsFileRequested("/loc/missing.txt", res));
   EXPECT_EQ(res, dir1 + "missing.txt");
   EXPECT_FALSE(serv.IsFileRequested("/loc/../both.txt", res));

   // AppendLocation() creates the prefix, AddLocation() replaces all its directories
   serv.AppendLocation("loc2/", dir2);
   EXPECT_TRUE(serv.IsFileRequested("/loc2/both.txt", res));
   EXPECT_EQ(res, dir2 + "both.txt");
   serv.AddLocation("loc/", dir2);
   EXPECT_TRUE(serv.IsFileRequested("/loc/both.txt", res));
   EXPECT_EQ(res, dir2 + "both.txt");

   serv.AddLocation("loc/", nullptr);
   serv.AddLocation("loc2/", nullptr);
   serv.AddLocation("absent/", nullptr);
   EXPECT_FALSE(serv.IsFileRequested("/loc/both.txt", res));
   EXPECT_FALSE(serv.IsFileRequested("/loc2/both.txt", res));

   std::filesystem::remove_all(top.Data());
}
