// @(#)root/mathcore:$Id$
// Author: Jonas Hahnfeld 11/2020

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class ROOT::Math::RanluxppEngine
Implementation of the RANLUX++ generator

RANLUX++ is an LCG equivalent of RANLUX using 576 bit numbers.

The idea of the generator (such as the initialization method) and the algorithm
for the modulo operation are described in
A. Sibidanov, *A revision of the subtract-with-borrow random numbergenerators*,
*Computer Physics Communications*, 221(2017), 299-303,
preprint https://arxiv.org/pdf/1705.03123.pdf

The code is loosely based on the Assembly implementation by A. Sibidanov
available at https://github.com/sibidanov/ranluxpp/.

Compared to the original generator, this implementation contains a fix to ensure
that the modulo operation of the LCG always returns the smallest value congruent
to the modulus (based on notes by M. Lüscher). Also, the generator converts the
LCG state back to RANLUX numbers (implementation based on notes by M. Lüscher).
This avoids a bias in the generated numbers because the upper bits of the LCG
state, that is smaller than the modulus \f$ m = 2^{576} - 2^{240} + 1 \f$ (not
a power of 2!), have a higher probability of being 0 than 1. And finally, this
implementation draws 48 random bits for each generated floating point number
(instead of 52 bits as in the original generator) to maintain the theoretical
properties from understanding the original transition function of RANLUX as a
chaotic dynamical system.
*/

#include "Math/RanluxppEngine.h"

#include "ranluxpp/mulmod.h"
#include "ranluxpp/ranlux_lcg.h"

#include <cassert>
#include <cstdint>

namespace {

// Variable templates are a feature of C++14, use the older technique of having
// a static member in a template class.

// The coefficients have been determined using Python, and in parts compared to the values given by Sibidanov.
//
//     >>> def print_hex(a):
//     ...     while a > 0:
//     ...         print('{0:#018x}'.format(a & 0xffffffffffffffff))
//     ...         a >>= 64
//     ...
//     >>> m = 2 ** 576 - 2 ** 240 + 1
//     >>> a = m - (m - 1) // 2 ** 24
//     >>> kA = pow(a, <w>, m)
//     >>> print_hex(kA)

template <int p>
struct RanluxppData;

template <>
struct RanluxppData<24> {
   static const uint64_t kA[9];
};
// Also given by Sibidanov
const uint64_t RanluxppData<24>::kA[] = {
   0x0000000000000000, 0x0000000000000000, 0x0000000000010000, 0xfffe000000000000, 0xffffffffffffffff,
   0xffffffffffffffff, 0xffffffffffffffff, 0xfffffffeffffffff, 0xffffffffffffffff,
};

template <>
struct RanluxppData<218> {
   static const uint64_t kA[9];
};
const uint64_t RanluxppData<218>::kA[] = {
	0xf445fffffffffd94, 0xfffffd74ffffffff, 0x000000000ba5ffff, 0xfc76000000000942, 0xfffffaaaffffffff,
	0x0000000000b0ffff, 0x027b0000000007d1, 0xfffff96000000000, 0xfffffffff8e4ffff,
};

template <>
struct RanluxppData<223> {
   static const uint64_t kA[9];
};
// Also given by Sibidanov
const uint64_t RanluxppData<223>::kA[] = {
   0x0000000ba6000000, 0x0a00000000094200, 0xffeef0fffffffffa, 0xfffffffe25ffffff, 0x7b0000000007d0ff,
   0xfff9600000000002, 0xfffffff8e4ffffff, 0xba00000000026cff, 0x00028b000000000b,
};

template <>
struct RanluxppData<389> {
   static const uint64_t kA[9];
};
// Also given by Sibidanov
const uint64_t RanluxppData<389>::kA[] = {
   0x00002ecac9000000, 0x740000002c389600, 0xb9c8a6ffffffe525, 0xfffff593cfffffff, 0xab0000001e93f2ff,
   0xe4ab160000000d92, 0xffffdf6604ffffff, 0x020000000b9242ff, 0x0df0600000002ee0,
};

template <>
struct RanluxppData<404> {
   static const uint64_t kA[9];
};
const uint64_t RanluxppData<404>::kA[] = {
	0x2eabffffffc9d08b, 0x00012612ffffff99, 0x0000007c3ebe0000, 0x353600000047bba1, 0xffd3c769ffffffd1,
	0x0000001ada8bffff, 0x6c30000000463759, 0xffb2a1440000000a, 0xffffffc634beffff,
};

template <>
struct RanluxppData<778> {
   static const uint64_t kA[9];
};
const uint64_t RanluxppData<778>::kA[] = {
   0x872de42d9dca512b, 0xdbf015ea1662f8a0, 0x01f48f0d28482e96, 0x392fca0b3be2ae04, 0xed00881af896ce54,
   0x14f0a768664013f3, 0x9489f52deb1f7f80, 0x72139804e09c0f37, 0x2146b0bb92a2f9a4,
};

template <>
struct RanluxppData<794> {
   static const uint64_t kA[9];
};
const uint64_t RanluxppData<794>::kA[] = {
	0x428df7227a2ca7c9, 0xde32225faaa74b1a, 0x4b9d965ca1ebd668, 0x78d15f59e58e2aff, 0x240fea15e99d075f,
	0xfe0b70f2d7b7d169, 0x75a535f4c41d51fb, 0x1a5ef0b7233b93e1, 0xbc787ca783d5d5a9,
};

template <>
struct RanluxppData<2048> {
   static const uint64_t kA[9];
};
// Also given by Sibidanov
const uint64_t RanluxppData<2048>::kA[] = {
   0xed7faa90747aaad9, 0x4cec2c78af55c101, 0xe64dcb31c48228ec, 0x6d8a15a13bee7cb0, 0x20b2ca60cb78c509,
   0x256c3d3c662ea36c, 0xff74e54107684ed2, 0x492edfcc0cc8e753, 0xb48c187cf5b22097,
};

} // end anonymous namespace

namespace ROOT {
namespace Math {

template <int w, int p, int u>
class RanluxppEngineImpl {
   // Needs direct access to private members to initialize its four states.
   friend class RanluxppCompatEngineLuescherImpl<w, p>;

private:
   uint64_t fState[9]; ///< RANLUX state of the generator
   unsigned fCarry;    ///< Carry bit of the RANLUX state
   int fPosition = 0;  ///< Current position in bits

   static constexpr const uint64_t *kA = RanluxppData<p>::kA;
   static constexpr int kMaxPos = (u == 0) ? 9 * 64 : u * w;
   static_assert(kMaxPos <= 576, "maximum position larger than 576 bits");

   /// Advance with given multiplier
   void Advance(const uint64_t *a)
   {
      uint64_t lcg[9];
      to_lcg(fState, fCarry, lcg);
      mulmod(a, lcg);
      to_ranlux(lcg, fState, fCarry);
      fPosition = 0;
   }

   /// Produce next block of random bits
   void Advance()
   {
      Advance(kA);
   }

   /// Skip 24 RANLUX numbers
   void Skip24()
   {
      Advance(RanluxppData<24>::kA);
   }

public:
   /// Return the next random bits, generate a new block if necessary
   uint64_t NextRandomBits()
   {
      if (fPosition + w > kMaxPos) {
         Advance();
      }

      int idx = fPosition / 64;
      int offset = fPosition % 64;
      int numBits = 64 - offset;

      uint64_t bits = fState[idx] >> offset;
      if (numBits < w) {
         bits |= fState[idx + 1] << numBits;
      }
      bits &= ((uint64_t(1) << w) - 1);

      fPosition += w;
      assert(fPosition <= kMaxPos && "position out of range!");

      return bits;
   }

   /// Return a floating point number, converted from the next random bits.
   double NextRandomFloat()
   {
      static constexpr double div = 1.0 / (uint64_t(1) << w);
      uint64_t bits = NextRandomBits();
      return bits * div;
   }

   /// Initialize and seed the state of the generator as in James' implementation
   void SetSeedJames(uint64_t s)
   {
      // Multiplicative Congruential generator using formula constants of L'Ecuyer
      // as described in "A review of pseudorandom number generators" (Fred James)
      // published in Computer Physics Communications 60 (1990) pages 329-344.
      int64_t seed = s;
      auto next = [&]() {
         const int a = 0xd1a4, b = 0x9c4e, c = 0x2fb3, d = 0x7fffffab;
         int64_t k = seed / a;
         seed = b * (seed - k * a) - k * c ;
         if (seed < 0) seed += d;
         return seed & 0xffffff;
      };

      // Iteration is reversed because the first number from the MCG goes to the
      // highest position.
      for (int i = 6; i >= 0; i -= 3) {
         uint64_t r[8];
         for (int j = 0; j < 8; j++) {
            r[j] = next();
         }

         fState[i+0] = r[7] + (r[6] << 24) + (r[5] << 48);
         fState[i+1] = (r[5] >> 16) + (r[4] << 8) + (r[3] << 32) + (r[2] << 56);
         fState[i+2] = (r[2] >> 8) + (r[1] << 16) + (r[0] << 40);
      }
      fCarry = !seed;

      Skip24();
   }

   /// Initialize and seed the state of the generator as in gsl_rng_ranlx*
   void SetSeedGsl(uint32_t s, bool ranlxd)
   {
      if (s == 0) {
         // The default seed for gsl_rng_ranlx* is 1.
         s = 1;
      }

      uint32_t bits = s;
      auto next_bit = [&]() {
         int b13 = (bits >> 18) & 0x1;
         int b31 = bits & 0x1;
         uint32_t bn = b13 ^ b31;
         bits = (bn << 30) + (bits >> 1);
         return b31;
      };
      auto next = [&]() {
         uint64_t ix = 0;
         for (int i = 0; i < 48; i++) {
            int iy = next_bit();
            if (ranlxd) {
               iy = (iy + 1) % 2;
            }
            ix = 2 * ix + iy;
         }
         return ix;
      };

      for (int i = 0; i < 9; i += 3) {
         uint64_t r[4];
         for (int j = 0; j < 4; j++) {
            r[j] = next();
         }

         fState[i+0] = r[0] + (r[1] << 48);
         fState[i+1] = (r[1] >> 16) + (r[2] << 32);
         fState[i+2] = (r[2] >> 32) + (r[3] << 16);
      }

      fCarry = 0;
      fPosition = 0;
      Advance();
   }

   /// Initialize and seed the state of the generator as proposed by Sibidanov
   void SetSeedSibidanov(uint64_t s)
   {
      uint64_t lcg[9];
      lcg[0] = 1;
      for (int i = 1; i < 9; i++) {
         lcg[i] = 0;
      }

      uint64_t a_seed[9];
      // Skip 2 ** 96 states.
      powermod(kA, a_seed, uint64_t(1) << 48);
      powermod(a_seed, a_seed, uint64_t(1) << 48);
      // Skip another s states.
      powermod(a_seed, a_seed, s);
      mulmod(a_seed, lcg);

      to_ranlux(lcg, fState, fCarry);
      fPosition = 0;
   }

   /// Initialize and seed the state of the generator as described by the C++ standard
   void SetSeedStd24(uint64_t s)
   {
      // Seed LCG with given parameters.
      uint64_t seed = s;
      const uint64_t a = 40014, m = 2147483563;
      auto next = [&]() {
         seed = (a * seed) % m;
         return seed & 0xffffff;
      };

      for (int i = 0; i < 9; i += 3) {
         uint64_t r[8];
         for (int j = 0; j < 8; j++) {
            r[j] = next();
         }

         fState[i+0] = r[0] + (r[1] << 24) + (r[2] << 48);
         fState[i+1] = (r[2] >> 16) + (r[3] << 8) + (r[4] << 32) + (r[5] << 56);
         fState[i+2] = (r[5] >> 8) + (r[6] << 16) + (r[7] << 40);
      }
      fCarry = !seed;

      Skip24();
   }

   /// Initialize and seed the state of the generator as described by the C++ standard
   void SetSeedStd48(uint64_t s)
   {
      // Seed LCG with given parameters.
      uint64_t seed = s;
      const uint64_t a = 40014, m = 2147483563;
      auto next = [&]() {
         seed = (a * seed) % m;
         uint64_t result = seed;
         seed = (a * seed) % m;
         result += seed << 32;
         return result & 0xffffffffffff;
      };

      for (int i = 0; i < 9; i += 3) {
         uint64_t r[4];
         for (int j = 0; j < 4; j++) {
            r[j] = next();
         }

         fState[i+0] = r[0] + (r[1] << 48);
         fState[i+1] = (r[1] >> 16) + (r[2] << 32);
         fState[i+2] = (r[2] >> 32) + (r[3] << 16);
      }
      fCarry = !seed;

      Skip24();
   }

   /// Skip `n` random numbers without generating them
   void Skip(uint64_t n)
   {
      int left = (kMaxPos - fPosition) / w;
      assert(left >= 0 && "position was out of range!");
      if (n < (uint64_t)left) {
         // Just skip the next few entries in the currently available bits.
         fPosition += n * w;
         assert(fPosition <= kMaxPos && "position out of range!");
         return;
      }

      n -= left;
      // Need to advance and possibly skip over blocks.
      int nPerState = kMaxPos / w;
      int skip = (n / nPerState);

      uint64_t a_skip[9];
      powermod(kA, a_skip, skip + 1);

      uint64_t lcg[9];
      to_lcg(fState, fCarry, lcg);
      mulmod(a_skip, lcg);
      to_ranlux(lcg, fState, fCarry);

      // Potentially skip numbers in the freshly generated block.
      int remaining = n - skip * nPerState;
      assert(remaining >= 0 && "should not end up at a negative position!");
      fPosition = remaining * w;
      assert(fPosition <= kMaxPos && "position out of range!");
   }
};

template <int p>
RanluxppEngine<p>::RanluxppEngine(uint64_t seed) : fImpl(new ImplType)
{
   this->SetSeed(seed);
}

template <int p>
RanluxppEngine<p>::~RanluxppEngine() = default;

template <int p>
double RanluxppEngine<p>::Rndm()
{
   return (*this)();
}

template <int p>
double RanluxppEngine<p>::operator()()
{
   return fImpl->NextRandomFloat();
}

template <int p>
uint64_t RanluxppEngine<p>::IntRndm()
{
   return fImpl->NextRandomBits();
}

template <int p>
void RanluxppEngine<p>::SetSeed(uint64_t seed)
{
   fImpl->SetSeedSibidanov(seed);
}

template <int p>
void RanluxppEngine<p>::Skip(uint64_t n)
{
   fImpl->Skip(n);
}

template class RanluxppEngine<24>;
template class RanluxppEngine<2048>;


template <int p>
RanluxppCompatEngineJames<p>::RanluxppCompatEngineJames(uint64_t seed) : fImpl(new ImplType)
{
   this->SetSeed(seed);
}

template <int p>
RanluxppCompatEngineJames<p>::~RanluxppCompatEngineJames() = default;

template <int p>
double RanluxppCompatEngineJames<p>::Rndm()
{
   return (*this)();
}

template <int p>
double RanluxppCompatEngineJames<p>::operator()()
{
   return fImpl->NextRandomFloat();
}

template <int p>
uint64_t RanluxppCompatEngineJames<p>::IntRndm()
{
   return fImpl->NextRandomBits();
}

template <int p>
void RanluxppCompatEngineJames<p>::SetSeed(uint64_t seed)
{
   fImpl->SetSeedJames(seed);
}

template <int p>
void RanluxppCompatEngineJames<p>::Skip(uint64_t n)
{
   fImpl->Skip(n);
}

template class RanluxppCompatEngineJames<223>;
template class RanluxppCompatEngineJames<389>;


template <int p>
RanluxppCompatEngineGslRanlxs<p>::RanluxppCompatEngineGslRanlxs(uint64_t seed) : fImpl(new ImplType)
{
   this->SetSeed(seed);
}

template <int p>
RanluxppCompatEngineGslRanlxs<p>::~RanluxppCompatEngineGslRanlxs() = default;

template <int p>
double RanluxppCompatEngineGslRanlxs<p>::Rndm()
{
   return (*this)();
}

template <int p>
double RanluxppCompatEngineGslRanlxs<p>::operator()()
{
   return fImpl->NextRandomFloat();
}

template <int p>
uint64_t RanluxppCompatEngineGslRanlxs<p>::IntRndm()
{
   return fImpl->NextRandomBits();
}

template <int p>
void RanluxppCompatEngineGslRanlxs<p>::SetSeed(uint64_t seed)
{
   fImpl->SetSeedGsl(seed, /*ranlxd=*/false);
}

template <int p>
void RanluxppCompatEngineGslRanlxs<p>::Skip(uint64_t n)
{
   fImpl->Skip(n);
}

template class RanluxppCompatEngineGslRanlxs<218>;
template class RanluxppCompatEngineGslRanlxs<404>;
template class RanluxppCompatEngineGslRanlxs<794>;


template <int p>
RanluxppCompatEngineGslRanlxd<p>::RanluxppCompatEngineGslRanlxd(uint64_t seed) : fImpl(new ImplType)
{
   this->SetSeed(seed);
}

template <int p>
RanluxppCompatEngineGslRanlxd<p>::~RanluxppCompatEngineGslRanlxd() = default;

template <int p>
double RanluxppCompatEngineGslRanlxd<p>::Rndm()
{
   return (*this)();
}

template <int p>
double RanluxppCompatEngineGslRanlxd<p>::operator()()
{
   return fImpl->NextRandomFloat();
}

template <int p>
uint64_t RanluxppCompatEngineGslRanlxd<p>::IntRndm()
{
   return fImpl->NextRandomBits();
}

template <int p>
void RanluxppCompatEngineGslRanlxd<p>::SetSeed(uint64_t seed)
{
   fImpl->SetSeedGsl(seed, /*ranlxd=*/true);
}

template <int p>
void RanluxppCompatEngineGslRanlxd<p>::Skip(uint64_t n)
{
   fImpl->Skip(n);
}

template class RanluxppCompatEngineGslRanlxd<404>;
template class RanluxppCompatEngineGslRanlxd<794>;


template <int w, int p>
class RanluxppCompatEngineLuescherImpl {

private:
  RanluxppEngineImpl<w, p> fStates[4]; ///< The states of this generator
  int fNextState = 0;                  ///< The index of the next state

public:
   /// Return the next random bits, generate a new block if necessary
   uint64_t NextRandomBits()
   {
      uint64_t bits = fStates[fNextState].NextRandomBits();
      fNextState = (fNextState + 1) % 4;
      return bits;
   }

   /// Return a floating point number, converted from the next random bits.
   double NextRandomFloat()
   {
      double number = fStates[fNextState].NextRandomFloat();
      fNextState = (fNextState + 1) % 4;
      return number;
   }

   /// Initialize and seed the state of the generator as in Lüscher's ranlxs
   void SetSeed(uint32_t s, bool ranlxd)
   {
      uint32_t bits = s;
      auto next_bit = [&]() {
         int b13 = (bits >> 18) & 0x1;
         int b31 = bits & 0x1;
         uint32_t bn = b13 ^ b31;
         bits = (bn << 30) + (bits >> 1);
         return b31;
      };
      auto next = [&]() {
         uint64_t ix = 0;
         for (int l = 0; l < 24; l++) {
            ix = 2 * ix + next_bit();
         }
         return ix;
      };

      for (int i = 0; i < 4; i++) {
         auto &state = fStates[i];
         for (int j = 0; j < 9; j += 3) {
            uint64_t r[8];
            for (int m = 0; m < 8; m++) {
               uint64_t ix = next();
               // Lüscher's implementation uses k = (j / 3) * 8 + m, so only
               // the value of m is important for (k % 4).
               if ((!ranlxd && (m % 4) == i) || (ranlxd && (m % 4) != i)) {
                  ix = 16777215 - ix;
               }
               r[m] = ix;
            }

            state.fState[j+0] = r[0] + (r[1] << 24) + (r[2] << 48);
            state.fState[j+1] = (r[2] >> 16) + (r[3] << 8) + (r[4] << 32) + (r[5] << 56);
            state.fState[j+2] = (r[5] >> 8) + (r[6] << 16) + (r[7] << 40);
         }

         state.fCarry = 0;
         state.fPosition = 0;
         state.Advance();
      }

      fNextState = 0;
   }

   /// Skip `n` random numbers without generating them
   void Skip(uint64_t n)
   {
      uint64_t nPerState = n / 4;
      int remainder = n % 4;
      for (int i = 0; i < 4; i++) {
         int idx = (fNextState + i) % 4;
         uint64_t nForThisState = nPerState;
         if (i < remainder) {
            nForThisState++;
         }
         fStates[idx].Skip(nForThisState);
      }
      // Switch the next state according to the remainder.
      fNextState = (fNextState + remainder) % 4;
   }
};

template <int p>
RanluxppCompatEngineLuescherRanlxs<p>::RanluxppCompatEngineLuescherRanlxs(uint64_t seed) : fImpl(new ImplType)
{
   this->SetSeed(seed);
}

template <int p>
RanluxppCompatEngineLuescherRanlxs<p>::~RanluxppCompatEngineLuescherRanlxs() = default;

template <int p>
double RanluxppCompatEngineLuescherRanlxs<p>::Rndm()
{
   return (*this)();
}

template <int p>
double RanluxppCompatEngineLuescherRanlxs<p>::operator()()
{
   return fImpl->NextRandomFloat();
}

template <int p>
uint64_t RanluxppCompatEngineLuescherRanlxs<p>::IntRndm()
{
   return fImpl->NextRandomBits();
}

template <int p>
void RanluxppCompatEngineLuescherRanlxs<p>::SetSeed(uint64_t seed)
{
   fImpl->SetSeed(seed, /*ranlxd=*/false);
}

template <int p>
void RanluxppCompatEngineLuescherRanlxs<p>::Skip(uint64_t n)
{
   fImpl->Skip(n);
}

template class RanluxppCompatEngineLuescherRanlxs<218>;
template class RanluxppCompatEngineLuescherRanlxs<404>;
template class RanluxppCompatEngineLuescherRanlxs<794>;


template <int p>
RanluxppCompatEngineLuescherRanlxd<p>::RanluxppCompatEngineLuescherRanlxd(uint64_t seed) : fImpl(new ImplType)
{
   this->SetSeed(seed);
}

template <int p>
RanluxppCompatEngineLuescherRanlxd<p>::~RanluxppCompatEngineLuescherRanlxd() = default;

template <int p>
double RanluxppCompatEngineLuescherRanlxd<p>::Rndm()
{
   return (*this)();
}

template <int p>
double RanluxppCompatEngineLuescherRanlxd<p>::operator()()
{
   return fImpl->NextRandomFloat();
}

template <int p>
uint64_t RanluxppCompatEngineLuescherRanlxd<p>::IntRndm()
{
   return fImpl->NextRandomBits();
}

template <int p>
void RanluxppCompatEngineLuescherRanlxd<p>::SetSeed(uint64_t seed)
{
   fImpl->SetSeed(seed, /*ranlxd=*/true);
}

template <int p>
void RanluxppCompatEngineLuescherRanlxd<p>::Skip(uint64_t n)
{
   fImpl->Skip(n);
}

template class RanluxppCompatEngineLuescherRanlxd<404>;
template class RanluxppCompatEngineLuescherRanlxd<794>;


RanluxppCompatEngineStdRanlux24::RanluxppCompatEngineStdRanlux24(uint64_t seed) : fImpl(new ImplType)
{
   this->SetSeed(seed);
}

RanluxppCompatEngineStdRanlux24::~RanluxppCompatEngineStdRanlux24() = default;

double RanluxppCompatEngineStdRanlux24::Rndm()
{
   return (*this)();
}

double RanluxppCompatEngineStdRanlux24::operator()()
{
   return fImpl->NextRandomFloat();
}

uint64_t RanluxppCompatEngineStdRanlux24::IntRndm()
{
   return fImpl->NextRandomBits();
}

void RanluxppCompatEngineStdRanlux24::SetSeed(uint64_t seed)
{
   fImpl->SetSeedStd24(seed);
}

void RanluxppCompatEngineStdRanlux24::Skip(uint64_t n)
{
   fImpl->Skip(n);
}


RanluxppCompatEngineStdRanlux48::RanluxppCompatEngineStdRanlux48(uint64_t seed) : fImpl(new ImplType)
{
   this->SetSeed(seed);
}

RanluxppCompatEngineStdRanlux48::~RanluxppCompatEngineStdRanlux48() = default;

double RanluxppCompatEngineStdRanlux48::Rndm()
{
   return (*this)();
}

double RanluxppCompatEngineStdRanlux48::operator()()
{
   return fImpl->NextRandomFloat();
}

uint64_t RanluxppCompatEngineStdRanlux48::IntRndm()
{
   return fImpl->NextRandomBits();
}

void RanluxppCompatEngineStdRanlux48::SetSeed(uint64_t seed)
{
   fImpl->SetSeedStd48(seed);
}

void RanluxppCompatEngineStdRanlux48::Skip(uint64_t n)
{
   fImpl->Skip(n);
}

} // end namespace Math
} // end namespace ROOT
