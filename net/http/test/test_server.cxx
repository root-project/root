#include "gtest/gtest.h"

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
   server_url = TString::Format("http:/localhost:%d", httpport);

   test_suite(serv);
}
