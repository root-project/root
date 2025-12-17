// @(#)root/mathcore:$Id$
// Author: Fernando Hueso-González   04/08/2021

#ifndef ROOT_Math_LFSR
#define ROOT_Math_LFSR

#include <array>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <vector>
#include <set>
#include <cmath>
#include <cstdint> // for std::uint16_t
#include "TError.h"

/// Pseudo Random Binary Sequence (PRBS) generator namespace with functions based
/// on linear feedback shift registers (LFSR) with a periodicity of 2^n-1
///
/// @note It should NOT be used for general-purpose random number generation or any
/// statistical study, for those cases see e.g. std::mt19937 instead.
///
/// The goal is to generate binary bit sequences with the same algorithm as the ones usually implemented
/// in electronic chips, so that the theoretically expected ones can be compared with the acquired sequences.
///
/// The main ingredients of a PRBS generator are a monic polynomial of maximum degree \f$n\f$, with coefficients
/// either 0 or 1, and a <a href="https://www.nayuki.io/page/galois-linear-feedback-shift-register">Galois</a>
/// linear-feedback shift register with a non-zero seed. When the monic polynomial exponents are chosen appropriately,
/// the period of the resulting bit sequence (0s and 1s) yields \f$2^n - 1\f$.
///
/// @sa https://gist.github.com/mattbierner/d6d989bf26a7e54e7135,
/// https://root.cern/doc/master/civetweb_8c_source.html#l06030,
/// https://cryptography.fandom.com/wiki/Linear_feedback_shift_register,
/// https://www3.advantest.com/documents/11348/33b24c8a-c8cb-40b8-a2a7-37515ba4abc8,
/// https://www.reddit.com/r/askscience/comments/63a10q/for_prbs3_with_clock_input_on_each_gate_how_can/,
/// https://es.mathworks.com/help/serdes/ref/prbs.html, https://metacpan.org/pod/Math::PRBS,
/// https://ez.analog.com/data_converters/high-speed_adcs/f/q-a/545335/ad9689-pn9-and-pn23

namespace ROOT::Math::LFSR {

/**
 * @brief Generate the next pseudo-random bit using the current state of a linear feedback shift register (LFSR) and
 * update it
 * @tparam k the length of the LFSR, usually also the order of the monic polynomial PRBS-k (last exponent)
 * @tparam nTaps the number of taps
 * @param lfsr the current value of the LFSR. Passed by reference, it will be updated with the next value
 * @param taps the taps that will be XOR-ed to calculate the new bit. They are the exponents of the monic polynomial.
 * Ordering is unimportant. Note that an exponent E in the polynom maps to bit index E-1 in the LFSR.
 * @param left if true, the direction of the register shift is to the left <<, the newBit is set on lfsr at bit position
 * 0 (right). If false, shift is to the right and the newBit is stored at bit position (k-1)
 * @return the new random bit
 * @throw an exception is thrown if taps are out of the range [1, k]
 * @see https://en.wikipedia.org/wiki/Monic_polynomial, https://en.wikipedia.org/wiki/Linear-feedback_shift_register,
 * https://en.wikipedia.org/wiki/Pseudorandom_binary_sequence
 */
template <size_t k, size_t nTaps>
bool NextLFSR(std::bitset<k> &lfsr, std::array<std::uint16_t, nTaps> taps, bool left = true)
{
   static_assert(k <= 32, "For the moment, only supported until k == 32.");
   static_assert(k > 0, "Non-zero degree is needed for the LFSR.");
   static_assert(nTaps > 0, "At least one tap is needed for the LFSR.");
   static_assert(nTaps <= k, "Cannot use more taps than polynomial order");
   for (std::uint16_t j = 0; j < nTaps; ++j) {
      assert(static_cast<size_t>(taps[j] - 1) <= k && static_cast<size_t>(taps[j] - 1) > 0 &&
             "Tap value is out of range [1,k]");
   }

   // First, calculate the XOR (^) of all selected bits (marked by the taps)
   bool newBit = lfsr[taps[0] - 1]; // the exponent E of the polynomial correspond to index E - 1 in the bitset
   for (std::uint16_t j = 1; j < nTaps; ++j) {
      newBit ^= lfsr[taps[j] - 1];
   }

   // Apply the shift to the register in the right direction, and overwrite the empty one with newBit
   if (left) {
      lfsr <<= 1;
      lfsr[0] = newBit;
   } else {
      lfsr >>= 1;
      lfsr[k - 1] = newBit;
   }

   return newBit;
}

/**
 * @brief Generation of a sequence of pseudo-random bits using a linear feedback shift register (LFSR), until a
 * register value is repeated (or maxPeriod is reached)
 * @tparam k the length of the LFSR, usually also the order of the monic polynomial PRBS-k (last exponent)
 * @tparam nTaps the number of taps
 * @tparam Output the type of the container where the bit result (0 or 1) is stored (e.g. char, bool). It's unsigned
 * char by default, use bool instead if you want to save memory
 * @param start the start value (seed) of the LFSR
 * @param taps the taps that will be XOR-ed to calculate the new bit. They are the exponents of the monic polynomial.
 * Ordering is unimportant. Note that an exponent E in the polynom maps to bit index E-1 in the LFSR.
 * @param left if true, the direction of the register shift is to the left <<, the newBit is set on lfsr at bit
 * position 0 (right). If false, shift is to the right and the newBit is stored at bit position (k-1)
 * @param wrapping if true, allow repetition of values in the LFSRhistory, until maxPeriod is reached or the repeated
 * value == start. Enabling this option saves memory as no history is kept
 * @param oppositeBit if true, use the high/low bit of the LFSR to store output (for left=true/false, respectively)
 * instead of the newBit returned by ::NextLFSR
 * @return the array of pseudo random bits, or an empty array if input was incorrect
 * @see https://en.wikipedia.org/wiki/Monic_polynomial, https://en.wikipedia.org/wiki/Linear-feedback_shift_register,
 * https://en.wikipedia.org/wiki/Pseudorandom_binary_sequence
 */
template <size_t k, size_t nTaps, typename Output = unsigned char>
std::vector<Output> GenerateSequence(std::bitset<k> start, std::array<std::uint16_t, nTaps> taps, bool left = true,
                                     bool wrapping = false, bool oppositeBit = false)
{
   std::vector<Output> result; // Store result here

   // Sanity-checks
   static_assert(k <= 32, "For the moment, only supported until k == 32.");
   static_assert(k > 0, "Non-zero degree is needed for the LFSR.");
   static_assert(nTaps >= 2, "At least two taps are needed for a proper sequence");
   static_assert(nTaps <= k, "Cannot use more taps than polynomial order");
   for (auto tap : taps) {
      if (tap > k || tap == 0) {
         Error("ROOT::Math::LFSR", "Tap %u is out of range [1,%lu]", tap, k);
         return result;
      }
   }
   if (start.none()) {
      Error("ROOT::Math::LFSR", "A non-zero start value is needed");
      return result;
   }

   // Calculate maximum period and pre-allocate space in result
   const std::uint32_t maxPeriod = pow(2, k) - 1;
   result.reserve(maxPeriod);

   std::set<uint32_t> lfsrHistory; // a placeholder to store the history of all different values of the LFSR
   std::bitset<k> lfsr(start);     // a variable storing the current value of the LFSR
   std::uint32_t i = 0;            // a loop counter
   if (oppositeBit)                // if oppositeBit enabled, first value is already started with the seed
      result.emplace_back(left ? lfsr[k - 1] : lfsr[0]);

   // Loop now until maxPeriod or a lfsr value is repeated. If wrapping enabled, allow repeated values if not equal
   // to seed
   do {
      bool newBit = NextLFSR(lfsr, taps, left);

      if (!oppositeBit)
         result.emplace_back(newBit);
      else
         result.emplace_back(left ? lfsr[k - 1] : lfsr[0]);

      ++i;

      if (!wrapping) // If wrapping not allowed, break the loop once a repeated value is encountered
      {
         if (lfsrHistory.count(lfsr.to_ulong()))
            break;

         lfsrHistory.insert(lfsr.to_ulong()); // Add to the history
      }
   } while (lfsr != start && i < maxPeriod);

   if (oppositeBit)
      result.pop_back(); // remove last element, as we already pushed the one from the seed above the while loop

   result.shrink_to_fit(); // only some special taps will lead to the maxPeriod, others will stop earlier

   return result;
}
} // namespace ROOT::Math::LFSR

#endif
