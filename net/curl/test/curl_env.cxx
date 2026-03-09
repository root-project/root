// Extra unit test binary because we tamper with the (global) environment

#include "gtest/gtest.h"

#include "ROOT/RCurlConnection.hxx"
#include <ROOT/TestSupport.hxx>

#include "TCurlFile.h"
#include "TSystem.h"

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
