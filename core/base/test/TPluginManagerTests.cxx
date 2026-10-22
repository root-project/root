#include "gtest/gtest.h"

#include "TPluginManager.h"
#include "TROOT.h"
#include "TEnv.h"
#include "TSystem.h"
#include "TUUID.h"

#include <fstream>

// Regression test for https://github.com/root-project/root/issues/23536:
// TPluginManager::FindHandler must correctly resolve plugins across all directories
// when Root.PluginPath contains multiple colon-separated directory paths.
TEST(TPluginManager, MultiDirectoryPluginPath)
{
   // Save original Root.PluginPath
   const char *origPluginPath = gEnv->GetValue("Root.PluginPath", (char *)nullptr);
   TString savedPath = origPluginPath ? origPluginPath : "";

   // Construct default plugin path ($ROOTSYS/etc/plugins)
   TString defaultPath = "plugins";
   gSystem->PrependPathName(TROOT::GetEtcDir(), defaultPath);

   // Create a temporary custom plugin directory with a custom macro
   TUUID uuid;
   TString tempDir = gSystem->TempDirectory();
   TString customDir = TString::Format("test_plugins_%s", uuid.AsString());
   gSystem->PrependPathName(tempDir.Data(), customDir);

   TString baseSubdir = "TCustomPluginBase";
   gSystem->PrependPathName(customDir.Data(), baseSubdir);
   ASSERT_EQ(0, gSystem->mkdir(baseSubdir.Data(), /*recursive=*/true));

   TString macroFile = "P010_TCustomPlugin.C";
   gSystem->PrependPathName(baseSubdir.Data(), macroFile);

   {
      std::ofstream f(macroFile.Data());
      ASSERT_TRUE(f.is_open());
      f << "void P010_TCustomPlugin()\n"
        << "{\n"
        << "   gPluginMgr->AddHandler(\"TCustomPluginBase\", \".*\", \"TCustomClass\",\n"
        << "      \"CustomLib\", \"TCustomClass()\");\n"
        << "}\n";
   }

   // Set multi-directory Root.PluginPath: custom directory prepended before default directory
   TString multiPath = customDir + ":" + defaultPath;
   gEnv->SetValue("Root.PluginPath", multiPath.Data());

   // 1. Verify that the custom plugin from the first directory is resolved
   TPluginHandler *hCustom = gPluginMgr->FindHandler("TCustomPluginBase", "custom://test");
   ASSERT_NE(hCustom, nullptr);
   EXPECT_STREQ(hCustom->GetClass(), "TCustomClass");

   // 2. Verify that built-in plugins from subsequent directories are resolved properly (not dropped or corrupted)
   TPluginHandler *hArchive = gPluginMgr->FindHandler("TArchiveFile", "test.zip");
   ASSERT_NE(hArchive, nullptr);
   EXPECT_STREQ(hArchive->GetClass(), "TZIPFile");

   TPluginHandler *hSQL = gPluginMgr->FindHandler("TSQLServer", "sqlite://test.db");
   ASSERT_NE(hSQL, nullptr);
   EXPECT_STREQ(hSQL->GetClass(), "TSQLiteServer");

   // Clean up
   gSystem->Unlink(macroFile.Data());
   gSystem->Unlink(baseSubdir.Data());
   gSystem->Unlink(customDir.Data());

   if (!savedPath.IsNull()) {
      gEnv->SetValue("Root.PluginPath", savedPath.Data());
   }
}
