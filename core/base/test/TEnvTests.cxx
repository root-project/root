#include "gtest/gtest.h"

#include "TEnv.h"
#include "TString.h"
#include "TSystem.h"
#include "TUUID.h"

#include <string>

TEST(TEnv, ROOTENV_USER_PATH)
{
   auto uuid = TUUID::UUIDv4();

   // Create directories structure <UUID>/local with user-level config in <UUID> and local-level config
   // in <UUID>/local
   const TString rcname = "testenv";
   const TString userDir = uuid.AsString();
   TString userConfig = rcname;
   gSystem->PrependPathName(userDir.Data(), userConfig);
   const TString userBak = userConfig + ".bak";
   TString localDir = "local";
   gSystem->PrependPathName(userDir, localDir);
   TString localConfig = rcname;
   gSystem->PrependPathName(localDir.Data(), localConfig);
   TString homeConfig = rcname;
   gSystem->PrependPathName(gSystem->HomeDirectory(), homeConfig);

   localDir.ReplaceAll("\\", "\\\\");
   ASSERT_EQ(0, gSystem->mkdir(localDir.Data(), /*recursive=*/true));

   gSystem->Setenv("ROOTENV_USER_PATH", userDir.Data());
   EXPECT_TRUE(gSystem->AccessPathName(localConfig.Data()));
   EXPECT_TRUE(gSystem->AccessPathName(userConfig.Data()));

   TEnv env("testenv");
   env.SetValue("EnvRec");
   EXPECT_EQ(1, env.GetValue("EnvRec", 0));
   env.SaveLevel(kEnvUser);

   EXPECT_TRUE(gSystem->AccessPathName(localConfig.Data()));
   EXPECT_FALSE(gSystem->AccessPathName(userConfig.Data()));
   EXPECT_TRUE(gSystem->AccessPathName(homeConfig.Data()));

   EXPECT_EQ(0, gSystem->Unlink(localDir.Data()));
   EXPECT_EQ(0, gSystem->Unlink(userConfig.Data()));
   EXPECT_EQ(0, gSystem->Unlink(userBak.Data()));
   EXPECT_EQ(0, gSystem->Unlink(userDir.Data()));
   gSystem->Unsetenv("ROOTENV_USER_PATH");
}
