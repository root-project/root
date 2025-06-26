// @(#)root/mathcore:$Id$
// Author: Fernando Hueso-González   04/08/2021

#ifndef ROOT_TRandomBinary
#define ROOT_TRandomBinary

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRandomBinary                                                        //
//                                                                      //
// Pseudo Random Binary Sequence generator class (periodicity = 2**n-1) //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <array>
#include <bitset>
#include <vector>
#include <set>
#include <cmath>
#include "Rtypes.h"
#include "TError.h"

class TRandomBinary  {
public:
   /**
    * @brief Generate the next pseudo-random bit using the current state of a linear feedback shift register (LFSR) and update it
    * @tparam k the length of the LFSR, usually also the order of the monic polynomial PRBS-k (last exponent)
    * @tparam nTaps the number of taps
    * @param lfsr the current value of the LFSR. Passed by reference, it will be updated with the next value
    * @param taps the taps that will be XOR-ed to calculate the new bit. They are the exponents of the monic polynomial. Ordering is unimportant. Note that an exponent E in the polynom maps to bit index E-1 in the LFSR.
    * @param left if true, the direction of the register shift is to the left <<, the newBit is set on lfsr at bit position 0 (right). If false, shift is to the right and the newBit is stored at bit position (k-1)
    * @return the new random bit
    * @throw an exception is thrown if taps are out of the range [1, k]
    * @see https://en.wikipedia.org/wiki/Monic_polynomial
    * @see https://en.wikipedia.org/wiki/Linear-feedback_shift_register
    * @see https://en.wikipedia.org/wiki/Pseudorandom_binary_sequence
    */
   template <size_t k, size_t nTaps>
   static bool
   NextLFSR(std::bitset<k>& lfsr, const std::array<uint16_t, nTaps> taps, const bool left = true)
   {
      static_assert(k <= 32, "For the moment, only supported until k == 32.");
      static_assert(k > 0, "Non-zero degree is needed for the LFSR.");
      static_assert(nTaps > 0, "At least one tap is needed for the LFSR.");
      static_assert(nTaps <= k, "Cannot use more taps than polynomial order");

      // First, calculate the XOR (^) of all selected bits (marked by the taps)
      bool newBit = lfsr[taps.at(0) - 1]; // the exponent E of the polynomial correspond to index E - 1 in the bitset
      for(uint16_t j = 1; j < nTaps ; ++j)
      {
         newBit ^= lfsr[taps.at(j) - 1];
      }

      //Apply the shift to the register in the right direction, and overwrite the empty one with newBit
      if(left)
      {
         lfsr <<= 1;
         lfsr[0] = newBit;
      }
      else
      {
         lfsr >>= 1;
         lfsr[k-1] = newBit;
      }

      return newBit;
   }

   /**
    * @brief Generation of a sequence of pseudo-random bits using a linear feedback shift register (LFSR), until a register value is repeated (or maxPeriod is reached)
    * @tparam k the length of the LFSR, usually also the order of the monic polynomial PRBS-k (last exponent)
    * @tparam nTaps the number of taps
    * @param start the start value (seed) of the LFSR
    * @param taps the taps that will be XOR-ed to calculate the new bit. They are the exponents of the monic polynomial. Ordering is unimportant. Note that an exponent E in the polynom maps to bit index E-1 in the LFSR.
    * @param left if true, the direction of the register shift is to the left <<, the newBit is set on lfsr at bit position 0 (right). If false, shift is to the right and the newBit is stored at bit position (k-1)
    * @param wrapping if true, allow repetition of values in the LFSRhistory, until maxPeriod is reached or the repeated value == start. Enabling this option saves memory as no history is kept
    * @param oppositeBit if true, use the high/low bit of the LFSR to store output (for left=true/false, respectively) instead of the newBit returned by ::NextLFSR
    * @return the array of pseudo random bits, or an empty array if input was incorrect
    * @see https://en.wikipedia.org/wiki/Monic_polynomial
    * @see https://en.wikipedia.org/wiki/Linear-feedback_shift_register
    * @see https://en.wikipedia.org/wiki/Pseudorandom_binary_sequence
    */
   template <size_t k, size_t nTaps>
   static std::vector<bool>
   GenerateSequence(const std::bitset<k> start, const std::array<uint16_t, nTaps> taps, const bool left = true, const bool wrapping = false, const bool oppositeBit = false)
   {
      std::vector<bool> result; // Store result here

      //Sanity-checks
      static_assert(k <= 32, "For the moment, only supported until k == 32.");
      static_assert(k > 0, "Non-zero degree is needed for the LFSR.");
      static_assert(nTaps >= 2, "At least two taps are needed for a proper sequence");
      static_assert(nTaps <= k, "Cannot use more taps than polynomial order");
      for(auto tap : taps) {
         if(tap > k || tap == 0) {
            Error("TRandomBinary", "Tap %u is out of range [1,%lu]", tap, k);
            return result;
         }
      }
      if(start.none()) {
         Error("TRandomBinary", "A non-zero start value is needed");
         return result;
      }

      // Calculate maximum period and pre-allocate space in result
      const uint32_t maxPeriod = pow(2,k) - 1;
      result.reserve(maxPeriod);

      std::set<uint32_t> lfsrHistory; // a placeholder to store the history of all different values of the LFSR
      std::bitset<k> lfsr(start); // a variable storing the current value of the LFSR
      uint32_t i = 0; // a loop counter
      if(oppositeBit) // if oppositeBit enabled, first value is already started with the seed
         result.emplace_back(left ? lfsr[k-1] : lfsr[0]);

      //Loop now until maxPeriod or a lfsr value is repeated. If wrapping enabled, allow repeated values if not equal to seed
      do {
         bool newBit = NextLFSR(lfsr, taps, left);

         if(!oppositeBit)
            result.emplace_back(newBit);
         else
            result.emplace_back(left ? lfsr[k-1] : lfsr[0]);

         ++i;

         if(!wrapping) // If wrapping not allowed, break the loop once a repeated value is encountered
         {
            if(lfsrHistory.count(lfsr.to_ulong()))
               break;

            lfsrHistory.insert(lfsr.to_ulong()); // Add to the history
         }
      }
      while(lfsr != start && i < maxPeriod);

      if(oppositeBit)
         result.pop_back();// remove last element, as we already pushed the one from the seed above the while loop

      result.shrink_to_fit();//only some special taps will lead to the maxPeriod, others will stop earlier

      return result;
   }

   TRandomBinary() = default;
   virtual ~TRandomBinary() = default;

   ClassDef(TRandomBinary,0)  //Pseudo Random Binary Sequence (periodicity = 2**n - 1)
};

#endif
