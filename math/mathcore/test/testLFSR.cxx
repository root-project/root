
#include <Math/LFSR.h>
#include "gtest/gtest.h"

TEST(LFSR, GenerateSequence)
{
   // PRBS3
   std::array<std::uint16_t, 2> taps3 = {3, 2}; // Exponents of the monic polynomial
   auto prbs3 = ROOT::Math::LFSR::GenerateSequence<3, 2, bool>(std::bitset<3>().flip(), taps3); // Start value all high
   EXPECT_EQ(prbs3, std::vector<bool>({false, false, true, false, true, true, true}));

   // PRBS4
   std::array<std::uint16_t, 2> taps4 = {4, 3}; // Exponents of the monic polynomial
   auto prbs4 = ROOT::Math::LFSR::GenerateSequence<4, 2, bool>(std::bitset<4>().flip(), taps4); // Start value all high
   EXPECT_EQ(prbs4, std::vector<bool>({false, false, false, true, false, false, true, true, false, true, false, true,
                                       true, true, true}));

   // PRBS5
   std::array<std::uint16_t, 2> taps5 = {5, 3}; // Exponents of the monic polynomial
   auto prbs5 = ROOT::Math::LFSR::GenerateSequence<5, 2, bool>(std::bitset<5>().flip(), taps5); // Start value all high
   EXPECT_EQ(prbs5, std::vector<bool>({false, false, false, true,  true,  false, true, true,  true,  false, true,
                                       false, true,  false, false, false, false, true, false, false, true,  false,
                                       true,  true,  false, false, true,  true,  true, true,  true}));

   // Exponents of the monic polynomial were extracted from
   // Keysight Trueform Series Operating and Service Guide p. 284-285. More examples in tutorials/math/PRBS.C
}
