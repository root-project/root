// Tests for RooRealIntegral
// Authors: Jonas Rembser, CERN 10/2022

#include <RooConstVar.h>
#include <RooGaussian.h>
#include <RooGenericPdf.h>
#include <RooProduct.h>
#include <RooProjectedPdf.h>
#include <RooRealIntegral.h>
#include <RooRealVar.h>

#include <gtest/gtest.h>

#include <memory>

namespace {
RooArgList getSortedServers(RooAbsArg const &arg)
{
   // Sort alphabetically in case the two integrals didn't add the servers in
   // the same order:
   RooArgList servers{arg.servers().begin(), arg.servers().end()};
   servers.sort();
   return servers;
}
} // namespace

// Verify that the value servers of a RooRealIntegral are the direct
// mathematical value servers of the integral, and not the leaves in the
// computation graphs. For the Batch mode, it is important for the evaluation
// of the computation graph that no direct value servers are skipped. See also
// GitHub issue #11578.
TEST(RooRealIntegral, ClientServerInterface1)
{
   using namespace RooFit;

   RooRealVar x{"x", "", 0, 1};

   RooRealVar mu{"mu", "", -0.005, -5, 5};

   // This is the key in this test: the mathematically direct value server of
   // the integral is the derived "muMod", and not the leaf of the computation
   // graph "mu"
   RooProduct muMod{"mu_mod", "", {mu, RooConst(10)}};

   RooGaussian gauss{"gauss", "", x, muMod, RooConst(2.0)};
   RooGenericPdf pdf{"gaussWrapped", "gauss", gauss};

   std::unique_ptr<RooAbsReal> integ1{gauss.createIntegral(x, *pdf.getIntegratorConfig(), nullptr)};
   std::unique_ptr<RooAbsReal> integ2{pdf.createIntegral(x, *pdf.getIntegratorConfig(), nullptr)};

   RooArgList servers1{getSortedServers(*integ1)};
   RooArgList servers2{getSortedServers(*integ2)};

   // Check that the server structure of the Gaussian integral looks as
   // expected, which should be, if you use Print("v"):
   //
   //     (-S) RooRealVar::x ""
   //     (--) RooGenericPdf::pdf "gauss"
   //     (V-) RooProduct::mu_mod ""
   //     (V-) RooConstVar::2 "2"
   //
   // What is important is that the indirect value server "mu" doesn't appear
   // among the servers, and the direct value server "mu_mod" does.

   EXPECT_EQ(servers1.size(), 4);

   // Respect the alphabetical order here
   EXPECT_EQ(std::string(servers1[0].GetName()), "2");
   EXPECT_EQ(std::string(servers1[1].GetName()), "gauss");
   EXPECT_EQ(std::string(servers1[2].GetName()), "mu_mod");
   EXPECT_EQ(std::string(servers1[3].GetName()), "x");

   EXPECT_TRUE(servers1[0].isValueServer(*integ1));
   EXPECT_FALSE(servers1[1].isValueServer(*integ1));
   EXPECT_TRUE(servers1[2].isValueServer(*integ1));
   EXPECT_FALSE(servers1[3].isValueServer(*integ1));

   EXPECT_FALSE(servers1[0].isShapeServer(*integ1));
   EXPECT_FALSE(servers1[1].isShapeServer(*integ1));
   EXPECT_FALSE(servers1[2].isShapeServer(*integ1));
   EXPECT_TRUE(servers1[3].isShapeServer(*integ1));

   // The Gaussian PDF wrapped in a RooGenericPdf should have exactly the same
   // server structure, so let's check that:

   EXPECT_EQ(servers2.size(), servers1.size());

   for (std::size_t i = 0; i < servers1.size(); ++i) {
      RooAbsArg const &s1 = servers1[i];
      RooAbsArg const &s2 = servers2[i];

      // The 2nd server is the integrated function, which doesn't have the same
      // name (it's "gaussWrapped" for the second integral instead of "gauss")
      if (i != 1) {
         EXPECT_EQ(std::string(s1.GetName()), s2.GetName());
      }

      EXPECT_EQ(s1.isValueServer(*integ1), s2.isValueServer(*integ2));
      EXPECT_EQ(s1.isShapeServer(*integ1), s2.isShapeServer(*integ2));
   }
}

/// Here we are integrating a function that has shape servers to verify that
/// they are correctly propagated as shape servers to the integral.
TEST(RooRealIntegral, IntegrateFuncWithShapeServers)
{
   using namespace RooFit;

   RooRealVar x("x", "", 0, 1);

   RooRealVar mu("mu", "", -0.005, -5, 5);
   RooProduct muMod("mu_mod", "", RooArgSet(mu, RooConst(10)));
   RooRealVar sigma("sigma", "", 1, 0.5, 2);

   RooGaussian gauss("gauss", "", x, muMod, sigma);
   RooGenericPdf pdf("pdf", "gauss", gauss);

   // Project over sigma, meaning sigma should now become a shape server
   RooProjectedPdf gaussProj("gaussProj", "", gauss, sigma);

   EXPECT_TRUE(x.isValueServer(gaussProj));
   EXPECT_FALSE(x.isShapeServer(gaussProj));
   EXPECT_TRUE(muMod.isValueServer(gaussProj));
   EXPECT_FALSE(muMod.isShapeServer(gaussProj));
   EXPECT_FALSE(sigma.isValueServer(gaussProj));
   EXPECT_TRUE(sigma.isShapeServer(gaussProj));

   // Integrating also over x, so both x and sigma should now be shape servers of the integral
   std::unique_ptr<RooAbsReal> integ1{gaussProj.createIntegral(x, *pdf.getIntegratorConfig(), nullptr)};

   EXPECT_FALSE(x.isValueServer(*integ1)); // x is now not a value server anymore
   EXPECT_TRUE(x.isShapeServer(*integ1));
   EXPECT_TRUE(muMod.isValueServer(*integ1));
   EXPECT_FALSE(muMod.isShapeServer(*integ1));
   EXPECT_FALSE(sigma.isValueServer(*integ1));
   EXPECT_TRUE(sigma.isShapeServer(*integ1)); // sigma should still be shape server!

   // Also check that that the number of servers is right (should be 3 for x,
   // mu, and sigma, and 1 more for the underlying PDF)
   EXPECT_EQ(gaussProj.servers().size(), 4);
   EXPECT_EQ(integ1->servers().size(), 4);
}

// Verify that using observable clones -- i.e., variables with the same names
// as the ones in the computation graph -- does not change the client-server
// structure of a RooRealIntegral. Covers GitHub issue #11637.
TEST(RooRealIntegral, UseCloneAsIntegrationVariable)
{
   RooRealVar x1{"x", "x1", -10, 10};
   RooRealVar x2{"x", "x2", -10, 10};

   RooGenericPdf gauss{"gauss", "std::exp(-0.5 * (x*x))", x1};

   RooRealIntegral integ1{"integ1", "", gauss, x1};
   RooRealIntegral integ2{"integ2", "", gauss, x2};

   // Check that client-server structure is as expected.
   for (auto const &integ : {integ1, integ2}) {

      RooArgList servers{getSortedServers(integ)};

      EXPECT_EQ(std::string(servers[0].GetName()), "gauss");
      EXPECT_EQ(std::string(servers[1].GetName()), "x");

      EXPECT_FALSE(servers[0].isValueServer(integ));
      EXPECT_FALSE(servers[1].isValueServer(integ));

      EXPECT_FALSE(servers[0].isShapeServer(integ));
      EXPECT_TRUE(servers[1].isShapeServer(integ));

      EXPECT_EQ(servers.size(), 2);
   }
}
