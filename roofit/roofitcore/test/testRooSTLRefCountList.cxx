// Tests for the RooSTLRefCountList
// Author: Jonas Rembser, CERN  2021

#include "RooRealVar.h"
#include "RooSTLRefCountList.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <random>
#include <string>
#include <vector>

// Test whether the RooSTLRefCountList by RooAbsArg::namePtr() still works
// after renaming the arguments.
TEST(RooSTLRefCountList, TestRenaming)
{

   const std::size_t nElements = 2 * RooSTLRefCountList<RooAbsArg>::minSizeForNamePointerOrdering;

   std::vector<RooRealVar> vars;
   std::vector<std::size_t> newNameIndices;
   RooSTLRefCountList<RooAbsArg> list;

   vars.reserve(nElements);
   newNameIndices.reserve(nElements);
   list.reserve(nElements);

   for (std::size_t i = 0; i < nElements; ++i) {
      auto name = std::string("v") + std::to_string(i);
      vars.emplace_back(name.c_str(), name.c_str(), i);
      newNameIndices.push_back(i);
      list.Add(&vars.back());
   }

   auto rng = std::default_random_engine{};
   std::shuffle(std::begin(newNameIndices), std::end(newNameIndices), rng);

   // To trigger the creation of the internal ordered storage
   list.findByNamePointer(&vars[0]);

   // Now do the renaming
   for (std::size_t i = 0; i < nElements; ++i) {
      std::size_t iNew = newNameIndices[i];
      auto name = std::string("v") + std::to_string(iNew);
      vars[i].SetName(name.c_str());
   }

   // And now see if the lookup works
   std::size_t nMatches = 0;
   for (std::size_t i = 0; i < nElements; ++i) {
      std::size_t iNew = newNameIndices[i];
      auto name = std::string("v") + std::to_string(iNew);
      RooRealVar varWithSameName(name.c_str(), name.c_str(), i);
      nMatches += list.findByNamePointer(&varWithSameName) == &vars[i];
   }
   EXPECT_EQ(nMatches, nElements);
}
