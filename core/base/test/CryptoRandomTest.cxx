#include "gtest/gtest.h"

#include <ROOT/RCryptoRandom.hxx>

TEST(TSystem, CryptoRandom)
{
   // test with 512 bits, longer keys may not work

   const int len = 64;
   uint8_t buf[64];

   for (int n = 0; n < len; n++)
      buf[n] = 0;

   EXPECT_TRUE(ROOT::Internal::GetCryptoRandom(buf, len));

   int nmatch = 0;

   for (int n = 0; n < len; n++)
      if (buf[n] == 0)
         nmatch++;

   // check that values in buffer changed
   EXPECT_TRUE(nmatch != len);

   for (int n = 0; n < len; n++)
      buf[n] = n;

   EXPECT_TRUE(ROOT::Internal::GetCryptoRandom(buf, len));

   nmatch = 0;

   for (int n = 0; n < len; n++)
      if (buf[n] == n)
         nmatch++;

   // check that values in buffer changed
   EXPECT_TRUE(nmatch != len);
}
