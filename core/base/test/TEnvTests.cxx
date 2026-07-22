#include "gtest/gtest.h"

#include "ROOT/TestSupport.hxx"

#include "TEnv.h"
#include "TString.h"
#include "TSystem.h"
#include "TUUID.h"

#include <string>

class TEnvTest : public testing::Test {
protected:
   TString fRcName;
   TString fUserDir;
   TString fUserConfig;
   TString fLocalDir;
   TString fLocalConfig;

   void SetUp() override
   {
      fRcName = "testenv";

      // Create directories structure <UUID>/local with user-level config in <UUID> and local-level config
      // in <UUID>/local

      const auto uuid = TUUID::UUIDv4();

      fUserDir = uuid.AsString();
      fUserConfig = fRcName;
      gSystem->PrependPathName(fUserDir.Data(), fUserConfig);

      fLocalDir = "local";
      gSystem->PrependPathName(fUserDir, fLocalDir);
      fLocalConfig = fRcName;
      gSystem->PrependPathName(fLocalDir.Data(), fLocalConfig);

      fLocalDir.ReplaceAll("\\", "\\\\");
      ASSERT_EQ(0, gSystem->mkdir(fLocalDir.Data(), /*recursive=*/true));

      gSystem->Setenv("ROOTENV_USER_PATH", fUserDir.Data());
   }

   void TearDown() override
   {
      gSystem->Unsetenv("ROOTENV_USER_PATH");

      gSystem->Unlink((fLocalConfig + ".bak").Data());
      gSystem->Unlink(fLocalConfig.Data());
      gSystem->Unlink(fLocalDir.Data());
      gSystem->Unlink((fUserConfig + ".bak").Data());
      gSystem->Unlink(fUserConfig.Data());
      gSystem->Unlink(fUserDir.Data());
   }
};

TEST_F(TEnvTest, ROOTENV_USER_PATH)
{
   TString homeConfig = fRcName;
   gSystem->PrependPathName(gSystem->HomeDirectory(), homeConfig);

   // Initially, neither the local nor the user config file exist
   EXPECT_TRUE(gSystem->AccessPathName(fLocalConfig.Data()));
   EXPECT_TRUE(gSystem->AccessPathName(fUserConfig.Data()));

   TEnv env(fRcName);
   EXPECT_FALSE(env.IsLocalLevelDisabled());
   env.SetValue("EnvRec");
   EXPECT_EQ(1, env.GetValue("EnvRec", 0));
   env.SaveLevel(kEnvUser);

   // Now, the user config file should have been created in the custom directory
   EXPECT_TRUE(gSystem->AccessPathName(fLocalConfig.Data()));
   EXPECT_FALSE(gSystem->AccessPathName(fUserConfig.Data()));
   EXPECT_TRUE(gSystem->AccessPathName(homeConfig.Data()));

   TString pwd = gSystem->pwd();
   EXPECT_TRUE(gSystem->cd(fLocalDir.Data()));
   env.SaveLevel(kEnvLocal);
   EXPECT_TRUE(gSystem->cd(pwd.Data()));

   EXPECT_FALSE(gSystem->AccessPathName(fLocalConfig.Data()));
   EXPECT_FALSE(gSystem->AccessPathName(fUserConfig.Data()));
   EXPECT_TRUE(gSystem->AccessPathName(homeConfig.Data()));
}

TEST_F(TEnvTest, DisableLocalLevel)
{
   EXPECT_TRUE(gEnv->IsLocalLevelDisabled());

   gEnv->WriteFile(fLocalConfig.Data());
   EXPECT_FALSE(gSystem->AccessPathName(fLocalConfig.Data()));

   TEnv env(fRcName, /*disableLocalLevel=*/true);
   {
      ROOT::TestSupport::CheckDiagsRAII checkDiag;
      checkDiag.requiredDiag(kError, "TEnv::ReadFile", "local level disabled, won't read", /*matchFullMessage=*/false);
      EXPECT_EQ(-1, env.ReadFile(fLocalConfig.Data(), kEnvLocal));
   }

   {
      ROOT::TestSupport::CheckDiagsRAII checkDiag;
      checkDiag.requiredDiag(kError, "TEnv::SaveLevel", "local level disabled, won't save", /*matchFullMessage=*/false);
      env.SaveLevel(kEnvLocal);
      EXPECT_TRUE(gSystem->AccessPathName(fRcName));
   }

   {
      ROOT::TestSupport::CheckDiagsRAII checkDiag;
      checkDiag.requiredDiag(kError, "TEnv::SetValue", "local level disabled", /*matchFullMessage=*/false);
      env.SetValue("xyz", kEnvLocal);
   }
}
