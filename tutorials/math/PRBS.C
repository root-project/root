/// \file
/// \ingroup tutorial_math
/// \notebook -nodraw
/// Tutorial illustrating the use of TRandomBinary::GenerateSequence
/// can be run with:
///
/// ~~~{.cpp}
/// root > .x PRBS.C
/// root > .x PRBS.C+ with ACLIC
/// ~~~
///
/// \macro_output
/// \macro_code
///
/// \author Fernando Hueso-González

#include <TRandomBinary.h>
#include <iostream>

void PRBS ()
{
   printf("\nTRandomBinary::GenerateSequence PRBS3, PRBS4 and PRBS5 tests\n");
   printf("==========================\n");

   //PRBS3
   std::array<uint16_t, 2> taps3 = {2, 3}; // Exponents of the monic polynomial
   auto prbs3 = TRandomBinary::GenerateSequence(std::bitset<3>().flip(), taps3);// Start value all high

   //PRBS4
   std::array<uint16_t, 2> taps4 = {3, 4}; // Exponents of the monic polynomial
   auto prbs4 = TRandomBinary::GenerateSequence(std::bitset<4>().flip(), taps4);// Start value all high

   //PRBS7
   std::array<uint16_t, 2> taps5 = {5, 3}; // Exponents of the monic polynomial
   auto prbs5 = TRandomBinary::GenerateSequence(std::bitset<5>().flip(), taps5);// Start value all high

   for(auto prbs : {prbs3, prbs4, prbs5})
   {
      std::cout << "PRBS period " << prbs.size() << ":\t";
      for(auto p : prbs)
      {
         std::cout << p << " ";
      }
      std::cout << std::endl;
   }
}
