// Extra unit test binary because we tamper with the (global) environment

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ROOT/RCurlConnection.hxx"
#include "ROOT/RError.hxx"
#include "ROOT/RRawFileCurl.hxx"
#include <ROOT/TestSupport.hxx>

#include "TCurlFile.h"
#include "TSystem.h"

#include <memory>
#include <utility>

TEST(RCurlConnection, CredFromEnv)
{
   ROOT::Internal::RCurlConnection conn("http://localhost");

   gSystem->Unsetenv("S3_ACCESS_KEY");
   conn.SetCredentialsFromEnvironment();
   EXPECT_EQ(ROOT::Internal::EHTTPCredentialsType::kNone, conn.GetCredentialsType());

   gSystem->Setenv("S3_ACCESS_KEY", "x");
   {
      ROOT::TestSupport::CheckDiagsRAII diagRAII;
      diagRAII.requiredDiag(kWarning, "[ROOT.HTTPClient]",
                            "found S3_ACCESS_KEY environment variable but S3_SECRET_KEY unset",
                            false /* matchFullMessage */);
      conn.SetCredentialsFromEnvironment();
      EXPECT_EQ(ROOT::Internal::EHTTPCredentialsType::kNone, conn.GetCredentialsType());
   }
   gSystem->Setenv("S3_SECRET_KEY", "");
   {
      ROOT::TestSupport::CheckDiagsRAII diagRAII;
      diagRAII.requiredDiag(kWarning, "[ROOT.HTTPClient]",
                            "found S3_ACCESS_KEY environment variable but S3_SECRET_KEY unset",
                            false /* matchFullMessage */);
      conn.SetCredentialsFromEnvironment();
      EXPECT_EQ(ROOT::Internal::EHTTPCredentialsType::kNone, conn.GetCredentialsType());
   }

   gSystem->Setenv("S3_SECRET_KEY", "y");
   conn.SetCredentialsFromEnvironment();
   EXPECT_EQ(ROOT::Internal::EHTTPCredentialsType::kS3, conn.GetCredentialsType());

   gSystem->Setenv("S3_ACCESS_KEY", "");
   conn.SetCredentialsFromEnvironment();
   EXPECT_EQ(ROOT::Internal::EHTTPCredentialsType::kNone, conn.GetCredentialsType());
}

TEST(CurlFile, S3Credentials)
{
   gSystem->Setenv("S3_ACCESS_KEY", "");
   const std::string url = "https://root-project-s3test.s3.cern.ch/hsimple.root";

   {
      ROOT::Internal::RRawFileCurl f(url, ROOT::Internal::RRawFile::ROptions());
      try {
         f.GetSize();
         FAIL() << "unauthenticated access should fail";
      } catch (const ROOT::RException &e) {
         EXPECT_THAT(e.what(), ::testing::HasSubstr("cannot determine file size"));
      }
   }

   {
      ROOT::TestSupport::CheckDiagsRAII diagRAII;
      diagRAII.requiredDiag(kError, "TCurlFile::TCurlFile", "can not read data", false /* matchFullMessage */);
      diagRAII.requiredDiag(kError, "TCurlFile::Init", "failed to read the file", false /* matchFullMessage */);
      auto f = std::make_unique<TCurlFile>(url.c_str());
      EXPECT_TRUE(f->IsZombie());
   }

   const auto testAccessKey = std::getenv("ROOT_TEST_S3_ACCESS_KEY");
   const auto testSecretKey = std::getenv("ROOT_TEST_S3_SECRET_KEY");
   if (!testAccessKey || testAccessKey[0] == '\0' || !testSecretKey || testSecretKey[0] == '\0') {
      GTEST_SKIP() << "Missing S3 test credentials <ROOT_TEST_S3_[ACCESS|SECRET]_KEY>, skipping";
   }
   if (ROOT::Internal::RCurlConnection::GetCurlVersion() <= 0x078100) {
      GTEST_SKIP() << "libcurl <= 7.81 is known to produce an AWSv4 signature incompatible with Ceph S3";
   }

   gSystem->Setenv("S3_ACCESS_KEY", testAccessKey);
   gSystem->Setenv("S3_SECRET_KEY", testSecretKey);

   ROOT::Internal::RRawFileCurl raw(url, ROOT::Internal::RRawFile::ROptions());
   EXPECT_LT(0, raw.GetSize());
   auto f = std::make_unique<TCurlFile>(url.c_str());
   EXPECT_LT(0, f->GetSize());

   gSystem->Unsetenv("S3_ACCESS_KEY");
   gSystem->Unsetenv("S3_SECRET_KEY");
}
