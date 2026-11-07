#include "gtest/gtest.h"

#include "TEnv.h"
#include "TPluginManager.h"
#include "TROOT.h"
#include "TString.h"
#include "TSystem.h"
#include "TUUID.h"

#include <fstream>
#include <vector>

namespace {

class TPluginManagerTest : public testing::Test {
protected:
   TEnv fEnv;
   TPluginManager fPluginManager;
   TEnv *fSavedEnv = nullptr;
   TPluginManager *fSavedPluginManager = nullptr;
   std::vector<TString> fPaths;
   TString fBase;

   void SetUp() override
   {
      // Initialize ROOT before replacing the globals used by interpreted plugin macros.
      (void)gROOT->GetPluginManager();
      fSavedEnv = gEnv;
      fSavedPluginManager = gPluginMgr;
      gEnv = &fEnv;
      gPluginMgr = &fPluginManager;

      TUUID uuid;
      TString id = uuid.AsString();
      id.ReplaceAll("-", "_");
      fBase = "TestPluginBase_" + id;
      TString rootDir = "test_plugins_" + id;
      gSystem->PrependPathName(gSystem->TempDirectory(), rootDir);
      ASSERT_EQ(0, gSystem->mkdir(rootDir));
      fPaths.push_back(rootDir);

      TString pluginPath;
      for (int i = 0; i < 3; ++i) {
         TString dir = TString::Format("dir%d", i);
         gSystem->PrependPathName(rootDir, dir);
         ASSERT_EQ(0, gSystem->mkdir(dir));
         fPaths.push_back(dir);

         TString baseDir = fBase;
         gSystem->PrependPathName(dir, baseDir);
         ASSERT_EQ(0, gSystem->mkdir(baseDir));
         fPaths.push_back(baseDir);

         TString macroName = TString::Format("P010_TestPlugin%d_", i) + id;
         TString macroFile = macroName + ".C";
         gSystem->PrependPathName(baseDir, macroFile);
         fPaths.push_back(macroFile);
         std::ofstream macro(macroFile.Data());
         ASSERT_TRUE(macro.is_open());
         macro << "#include \"TPluginManager.h\"\n"
               << "void " << macroName << "()\n{\n"
               << "   gPluginMgr->AddHandler(\"" << fBase << "\", \"^test" << i << ":\", \"TestPlugin" << i
               << "\", \"TestPluginLib\", \"TestPlugin" << i << "()\");\n}\n";
         macro.close();
         ASSERT_TRUE(macro.good());

         if (i != 0) {
#ifdef R__WIN32
            pluginPath += ";";
#else
            pluginPath += ":";
#endif
         }
         pluginPath += dir;
      }
      fEnv.SetValue("Root.PluginPath", pluginPath);
   }

   void TearDown() override
   {
      gPluginMgr = fSavedPluginManager;
      gEnv = fSavedEnv;
      for (auto it = fPaths.rbegin(); it != fPaths.rend(); ++it)
         EXPECT_EQ(0, gSystem->Unlink(*it)) << *it;
   }
};

// Regression test for https://github.com/root-project/root/issues/23536.
TEST_F(TPluginManagerTest, MultiDirectoryPluginPath)
{
   for (int i = 0; i < 3; ++i) {
      auto *handler = fPluginManager.FindHandler(fBase, TString::Format("test%d:example", i));
      ASSERT_NE(handler, nullptr) << "Plugin directory " << i;
      EXPECT_STREQ(handler->GetClass(), TString::Format("TestPlugin%d", i).Data());
   }
}

} // namespace
