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

   TString fname;
   Bool_t create = kFALSE;

   for (Int_t ntry = 0; ntry < 10; ++ntry) {
      fname = "root_test_";
      auto f = gSystem->TempFileName(fname, "/tmp", ".socket");
      if (!f)
         continue;
      fclose(f);

      EXPECT_EQ(fname[0], '/') << "first symbol in temporary file must be slash";
      create = serv.CreateEngine("socket:" + fname + "?socket_mode=0700&thrd=2");

      if (create)
         break;

      gSystem->Unlink(fname);
   }

   EXPECT_EQ(create, kTRUE) << "create unix socket";

   if (!create)
      return;

   server_hash = fname.Hash();  // dummy number for file suffix
   unix_socket = "--unix-socket " + fname; // curl argument
   server_url = "http://localhost"; // dummy host name for curl

   test_suite(serv);

   gSystem->Unlink(fname); // cleanup socket file
}
