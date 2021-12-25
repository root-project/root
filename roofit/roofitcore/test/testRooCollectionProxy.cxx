// Tests for RooAbsCollectionProxy
// Authors: Jonas Rembser, CERN  02/2022

#include <RooListProxy.h>
#include <RooSetProxy.h>
#include <RooWorkspace.h>
#include <RooRealVar.h>

#include <TFile.h>

#include <gtest/gtest.h>

// Test IO of RooSetProxy and RooListProxy if they were written to file before
// they became aliases for template specifications of RooCollectionProxy
TEST(RooCollectionProxy, BackwardsCompatibilityv626)
{

   // // To create reference file:

   // RooRealVar var1{"var1", "var1", 1.0};
   // RooRealVar var2{"var2", "var2", 1.0};
   // RooRealVar var3{"var3", "var3", 1.0};

   // RooListProxy listProxy("listProxy", "listProxy", &var1);
   // listProxy.add(var2);
   // RooSetProxy setProxy("setProxy", "setProxy", &var1);
   // setProxy.add(var3);

   // RooWorkspace w{"w"};
   // w.import(var1);
   // w.writeToFile("testRooCollectionProxy_v626.root");

   TFile f{"testRooCollectionProxy_v626.root"};

   auto w = f.Get<RooWorkspace>("w");

   EXPECT_TRUE(w->var("var1"));
   EXPECT_TRUE(w->var("var2"));
   EXPECT_TRUE(w->var("var3"));
}
