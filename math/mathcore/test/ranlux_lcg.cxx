// @(#)root/mathcore:$Id$
// Author: Jonas Hahnfeld 05/2021

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "../src/ranluxpp/ranlux_lcg.h"

#include "gtest/gtest.h"

#include <cstdint>

TEST(to_lcg, zero)
{
   // RANLUX state with zero in all components.
   uint64_t ranlux[9] = {0};
   unsigned c = 0;
   uint64_t lcg[9];

   to_lcg(ranlux, c, lcg);

   for (int i = 0; i < 9; i++) {
      EXPECT_EQ(lcg[i], 0);
   }
}

TEST(to_lcg, one)
{
   // RANLUX state with a one in the last component.
   uint64_t ranlux[9] = {1, 0, 0, 0, 0, 0, 0, 0, 0};
   unsigned c = 0;
   uint64_t lcg[9];

   to_lcg(ranlux, c, lcg);

   EXPECT_EQ(lcg[0], 1);
   for (int i = 1; i < 9; i++) {
      EXPECT_EQ(lcg[i], 0);
   }
}

TEST(to_lcg, carry)
{
   // RANLUX state with zero in all components and the carry bit set.
   uint64_t ranlux[9] = {0};
   unsigned c = 1;
   uint64_t lcg[9];

   to_lcg(ranlux, c, lcg);

   // Note that this state is the same as to_lcg::one above. When passed into
   // the function to_ranlux, it will return the input state of to_lcg::one and
   // not this one with the carry bit set.
   EXPECT_EQ(lcg[0], 1);
   for (int i = 1; i < 9; i++) {
      EXPECT_EQ(lcg[i], 0);
   }
}

TEST(to_lcg, all_ones)
{
   // RANLUX state with a one in every component.
   uint64_t ranlux[9];
   for (int i = 0; i < 9; i += 3) {
      ranlux[i+0] = 1 + (1 << 24) + (uint64_t(1) << 48);
      ranlux[i+1] = (1 << 8) + (uint64_t(1) << 32) + (uint64_t(1) << 56);
      ranlux[i+2] = (1 << 16) + (uint64_t(1) << 40);
   }
   unsigned c = 1;
   uint64_t lcg[9];

   to_lcg(ranlux, c, lcg);

   EXPECT_EQ(lcg[0], 1);
   EXPECT_EQ(lcg[1], 0);
   EXPECT_EQ(lcg[2], 0);
   EXPECT_EQ(lcg[3], 0x0001000000000000);
   EXPECT_EQ(lcg[4], 0x0100000100000100);
   EXPECT_EQ(lcg[5], 0x0000010000010000);
   EXPECT_EQ(lcg[6], 0x0001000001000001);
   EXPECT_EQ(lcg[7], 0x0100000100000100);
   EXPECT_EQ(lcg[8], 0x0000010000010000);
}

TEST(to_lcg, all_ones_alt)
{
   // The RANLUX state that the assembly version returns for the output of
   // to_lcg::all_ones.
   uint64_t ranlux[9];
   for (int i = 0; i < 9; i += 3) {
      ranlux[i+0] = 1 + (1 << 24) + (uint64_t(1) << 48);
      ranlux[i+1] = (1 << 8) + (uint64_t(1) << 32) + (uint64_t(1) << 56);
      ranlux[i+2] = (1 << 16) + (uint64_t(1) << 40);
   }
   ranlux[0]++;
   unsigned c = 0;
   uint64_t lcg[9];

   to_lcg(ranlux, c, lcg);

   EXPECT_EQ(lcg[0], 1);
   EXPECT_EQ(lcg[1], 0);
   EXPECT_EQ(lcg[2], 0);
   EXPECT_EQ(lcg[3], 0x0001000000000000);
   EXPECT_EQ(lcg[4], 0x0100000100000100);
   EXPECT_EQ(lcg[5], 0x0000010000010000);
   EXPECT_EQ(lcg[6], 0x0001000001000001);
   EXPECT_EQ(lcg[7], 0x0100000100000100);
   EXPECT_EQ(lcg[8], 0x0000010000010000);
}

TEST(to_lcg, all_ones_no_carry)
{
   // RANLUX state with a one in every component except the carry.
   uint64_t ranlux[9];
   for (int i = 0; i < 9; i += 3) {
      ranlux[i+0] = 1 + (1 << 24) + (uint64_t(1) << 48);
      ranlux[i+1] = (1 << 8) + (uint64_t(1) << 32) + (uint64_t(1) << 56);
      ranlux[i+2] = (1 << 16) + (uint64_t(1) << 40);
   }
   unsigned c = 0;
   uint64_t lcg[9];

   to_lcg(ranlux, c, lcg);

   EXPECT_EQ(lcg[0], 0);
   EXPECT_EQ(lcg[1], 0);
   EXPECT_EQ(lcg[2], 0);
   EXPECT_EQ(lcg[3], 0x0001000000000000);
   EXPECT_EQ(lcg[4], 0x0100000100000100);
   EXPECT_EQ(lcg[5], 0x0000010000010000);
   EXPECT_EQ(lcg[6], 0x0001000001000001);
   EXPECT_EQ(lcg[7], 0x0100000100000100);
   EXPECT_EQ(lcg[8], 0x0000010000010000);
}

TEST(to_lcg, all_ones_no_carry_alt)
{
   // The RANLUX state that to_ranlux and the assembly version returns for the
   // output of to_lcg::all_ones_no_carry.
   uint64_t ranlux[9];
   for (int i = 0; i < 9; i += 3) {
      ranlux[i+0] = 1 + (1 << 24) + (uint64_t(1) << 48);
      ranlux[i+1] = (1 << 8) + (uint64_t(1) << 32) + (uint64_t(1) << 56);
      ranlux[i+2] = (1 << 16) + (uint64_t(1) << 40);
   }
   ranlux[0]--;
   unsigned c = 1;
   uint64_t lcg[9];

   to_lcg(ranlux, c, lcg);

   EXPECT_EQ(lcg[0], 0);
   EXPECT_EQ(lcg[1], 0);
   EXPECT_EQ(lcg[2], 0);
   EXPECT_EQ(lcg[3], 0x0001000000000000);
   EXPECT_EQ(lcg[4], 0x0100000100000100);
   EXPECT_EQ(lcg[5], 0x0000010000010000);
   EXPECT_EQ(lcg[6], 0x0001000001000001);
   EXPECT_EQ(lcg[7], 0x0100000100000100);
   EXPECT_EQ(lcg[8], 0x0000010000010000);
}

TEST(to_lcg, ascending)
{
   uint64_t ranlux[9];
   for (int i = 0; i < 9; i += 3) {
      uint64_t ii = 8 * (i / 3);
      ranlux[i+0] = (ii + 1) + ((ii + 2) << 24) + ((ii + 3) << 48);
      ranlux[i+1] = ((ii + 4) << 8) + ((ii + 5) << 32) + ((ii + 6) << 56);
      ranlux[i+2] = ((ii + 7) << 16) + ((ii + 8) << 40);
   }
   unsigned c = 0;
   uint64_t lcg[9];

   to_lcg(ranlux, c, lcg);

   EXPECT_EQ(lcg[0], 0xfff1fffff1fffff2);
   EXPECT_EQ(lcg[1], 0xf1fffff1fffff1ff);
   EXPECT_EQ(lcg[2], 0xfffff1fffff1ffff);
   EXPECT_EQ(lcg[3], 0x000afffff1fffff1);
   EXPECT_EQ(lcg[4], 0x0e00000d00000c00);
   EXPECT_EQ(lcg[5], 0x00001000000f0000);
   EXPECT_EQ(lcg[6], 0x0013000012000011);
   EXPECT_EQ(lcg[7], 0x1600001500001400);
   EXPECT_EQ(lcg[8], 0x0000180000170000);
}

TEST(to_lcg, ascending_alt)
{
   uint64_t ranlux[9];
   for (int i = 0; i < 9; i += 3) {
      uint64_t ii = 8 * (i / 3);
      ranlux[i+0] = (ii + 1) + ((ii + 2) << 24) + ((ii + 3) << 48);
      ranlux[i+1] = ((ii + 4) << 8) + ((ii + 5) << 32) + ((ii + 6) << 56);
      ranlux[i+2] = ((ii + 7) << 16) + ((ii + 8) << 40);
   }
   ranlux[0]--;
   unsigned c = 1;
   uint64_t lcg[9];

   to_lcg(ranlux, c, lcg);

   EXPECT_EQ(lcg[0], 0xfff1fffff1fffff2);
   EXPECT_EQ(lcg[1], 0xf1fffff1fffff1ff);
   EXPECT_EQ(lcg[2], 0xfffff1fffff1ffff);
   EXPECT_EQ(lcg[3], 0x000afffff1fffff1);
   EXPECT_EQ(lcg[4], 0x0e00000d00000c00);
   EXPECT_EQ(lcg[5], 0x00001000000f0000);
   EXPECT_EQ(lcg[6], 0x0013000012000011);
   EXPECT_EQ(lcg[7], 0x1600001500001400);
   EXPECT_EQ(lcg[8], 0x0000180000170000);
}

TEST(to_lcg, descending)
{
   uint64_t ranlux[9];
   for (int i = 0; i < 9; i += 3) {
      uint64_t ii = 8 * (2 - i / 3);
      ranlux[i+0] = (ii + 8) + ((ii + 7) << 24) + ((ii + 6) << 48);
      ranlux[i+1] = ((ii + 5) << 8) + ((ii + 4) << 32) + ((ii + 3) << 56);
      ranlux[i+2] = ((ii + 2) << 16) + ((ii + 1) << 40);
   }
   unsigned c = 0;
   uint64_t lcg[9];

   to_lcg(ranlux, c, lcg);

   EXPECT_EQ(lcg[0], 0x000e00000e00000e);
   EXPECT_EQ(lcg[1], 0x0e00000e00000e00);
   EXPECT_EQ(lcg[2], 0x00000e00000e0000);
   EXPECT_EQ(lcg[3], 0x000e00000e00000e);
   EXPECT_EQ(lcg[4], 0x0b00000c00000d00);
   EXPECT_EQ(lcg[5], 0x00000900000a0000);
   EXPECT_EQ(lcg[6], 0x0006000007000008);
   EXPECT_EQ(lcg[7], 0x0300000400000500);
   EXPECT_EQ(lcg[8], 0x0000010000020000);
}

TEST(to_lcg, max)
{
   uint64_t max = UINT64_MAX;
   uint64_t ranlux[9] = {max, max, max, max, max, max, max, max, max};
   unsigned c = 0;
   uint64_t lcg[9];

   to_lcg(ranlux, c, lcg);

   EXPECT_EQ(lcg[0], 0);
   EXPECT_EQ(lcg[1], 0);
   EXPECT_EQ(lcg[2], 0);
   EXPECT_EQ(lcg[3], 0xffff000000000000);
   for (int i = 4; i < 9; i++) {
      EXPECT_EQ(lcg[i], 0xffffffffffffffff);
   }
}

TEST(to_ranlux, zero)
{
   uint64_t lcg[9] = {0};
   uint64_t ranlux[9];
   unsigned c = 0;

   to_ranlux(lcg, ranlux, c);

   for (int i = 0; i < 9; i++) {
      EXPECT_EQ(ranlux[i], 0);
   }
   EXPECT_EQ(c, 0);
}

TEST(to_ranlux, one)
{
   uint64_t lcg[9] = {1, 0, 0, 0, 0, 0, 0, 0, 0};
   uint64_t ranlux[9];
   unsigned c = 0;

   to_ranlux(lcg, ranlux, c);

   EXPECT_EQ(ranlux[0], 1);
   for (int i = 1; i < 9; i++) {
      EXPECT_EQ(ranlux[i], 0);
   }
   EXPECT_EQ(c, 0);
}

TEST(to_ranlux, all_ones)
{
   // See to_lcg::all_ones
   uint64_t lcg[9] = {
      1, 0, 0, 0x0001000000000000, 0x0100000100000100, 0x0000010000010000,
      0x0001000001000001, 0x0100000100000100, 0x0000010000010000,
   };
   uint64_t ranlux[9];
   unsigned c = 0;

   to_ranlux(lcg, ranlux, c);

   EXPECT_EQ(ranlux[0], 2 + (1 << 24) + (uint64_t(1) << 48));
   for (int i = 0; i < 9; i += 3) {
      if (i != 0) {
         EXPECT_EQ(ranlux[i+0], 1 + (1 << 24) + (uint64_t(1) << 48));
      }
      EXPECT_EQ(ranlux[i+1], (1 << 8) + (uint64_t(1) << 32) + (uint64_t(1) << 56));
      EXPECT_EQ(ranlux[i+2], (1 << 16) + (uint64_t(1) << 40));
   }
   EXPECT_EQ(c, 0);
}

TEST(to_ranlux, all_ones_no_carry)
{
   // See to_lcg::all_ones_no_carry and to_lcg::all_ones_no_carry_alt
   uint64_t lcg[9] = {
      0, 0, 0, 0x0001000000000000, 0x0100000100000100, 0x0000010000010000,
      0x0001000001000001, 0x0100000100000100, 0x0000010000010000,
   };
   uint64_t ranlux[9];
   unsigned c = 0;

   to_ranlux(lcg, ranlux, c);

   EXPECT_EQ(ranlux[0], (1 << 24) + (uint64_t(1) << 48));
   for (int i = 0; i < 9; i += 3) {
      if (i != 0) {
         EXPECT_EQ(ranlux[i+0], 1 + (1 << 24) + (uint64_t(1) << 48));
      }
      EXPECT_EQ(ranlux[i+1], (1 << 8) + (uint64_t(1) << 32) + (uint64_t(1) << 56));
      EXPECT_EQ(ranlux[i+2], (1 << 16) + (uint64_t(1) << 40));
   }
   EXPECT_EQ(c, 1);
}

TEST(to_ranlux, ascending)
{
   // See to_lcg::ascending and to_lcg::ascending_alt
   uint64_t lcg[9] = {
      0xfff1fffff1fffff2, 0xf1fffff1fffff1ff, 0xfffff1fffff1ffff,
      0x000afffff1fffff1, 0x0e00000d00000c00, 0x00001000000f0000,
      0x0013000012000011, 0x1600001500001400, 0x0000180000170000,
   };
   uint64_t ranlux[9];
   unsigned c = 0;

   to_ranlux(lcg, ranlux, c);

   EXPECT_EQ(ranlux[0], (2 << 24) + (uint64_t(3) << 48));
   for (int i = 0; i < 9; i += 3) {
      int ii = 8 * (i / 3);
      if (i != 0) {
         EXPECT_EQ(ranlux[i+0], (ii + 1) + ((ii + 2) << 24) + (uint64_t(ii + 3) << 48));
      }
      EXPECT_EQ(ranlux[i+1], ((ii + 4) << 8) + (uint64_t(ii + 5) << 32) + (uint64_t(ii + 6) << 56));
      EXPECT_EQ(ranlux[i+2], ((ii + 7) << 16) + (uint64_t(ii + 8) << 40));
   }
   EXPECT_EQ(c, 1);
}

TEST(to_ranlux, descending)
{
   // See to_lcg::decending
   uint64_t lcg[9] = {
      0x000e00000e00000e, 0x0e00000e00000e00, 0x00000e00000e0000,
      0x000e00000e00000e, 0x0b00000c00000d00, 0x00000900000a0000,
      0x0006000007000008, 0x0300000400000500, 0x0000010000020000,
   };
   uint64_t ranlux[9];
   unsigned c = 0;

   to_ranlux(lcg, ranlux, c);

   for (int i = 0; i < 9; i += 3) {
      uint64_t ii = 8 * (2 - i / 3);
      EXPECT_EQ(ranlux[i+0], (ii + 8) + ((ii + 7) << 24) + ((ii + 6) << 48));
      EXPECT_EQ(ranlux[i+1], ((ii + 5) << 8) + ((ii + 4) << 32) + ((ii + 3) << 56));
      EXPECT_EQ(ranlux[i+2], ((ii + 2) << 16) + ((ii + 1) << 40));
   }
   EXPECT_EQ(c, 0);
}

TEST(to_ranlux, max)
{
   // See to_lcg::max
   uint64_t max = UINT64_MAX;
   uint64_t lcg[9] = {0, 0, 0, 0xffff000000000000, max, max, max, max, max};
   uint64_t ranlux[9];
   unsigned c = 0;

   to_ranlux(lcg, ranlux, c);

   EXPECT_EQ(ranlux[0], max - 1);
   for (int i = 1; i < 9; i++) {
      EXPECT_EQ(ranlux[i], max);
   }
   EXPECT_EQ(c, 1);
}
