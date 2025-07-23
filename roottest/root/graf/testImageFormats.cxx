#include <TGClient.h>
#include <TGPicture.h>

#include <gtest/gtest.h>

// Import a few icons from ROOT's icon directory

TEST(ImageFormats, GIF)
{
   if (!gClient)
      GTEST_SKIP() << "Graphics system not enabled";
   auto icon = gClient->GetPicture("GoBack.gif");
   ASSERT_NE(icon, nullptr);
   ASSERT_NE(icon->GetWidth(), 0u);
   delete icon;
}

TEST(ImageFormats, PNG)
{
   if (!gClient)
      GTEST_SKIP() << "Graphics system not enabled";
   auto icon = gClient->GetPicture("pause.png");
   ASSERT_NE(icon, nullptr);
   ASSERT_NE(icon->GetWidth(), 0u);
   delete icon;
}
