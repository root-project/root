/*
 * Project: RooFit
 * Authors:
 *   Stephan Hageboeck, CERN 2020
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_TreeReadBuffer_h
#define RooFit_TreeReadBuffer_h

#include <TTree.h>

struct TreeReadBuffer {
   virtual ~TreeReadBuffer() = default;
   virtual operator double() = 0;
   virtual operator int() = 0;
};

/// Helper for reading branches with various types from a TTree, and convert all to double.
template <typename T>
struct TypedTreeReadBuffer final : public TreeReadBuffer {
   operator double() override { return _value; }
   operator int() override { return _value; }
   T _value;
};

/// Create a TreeReadBuffer to hold the specified type, and attach to the branch passed as argument.
/// \tparam T Type of branch to be read.
/// \param[in] branchName Attach to this branch.
/// \param[in] tree Tree to attach to.
template <typename T>
std::unique_ptr<TreeReadBuffer> createTreeReadBuffer(const TString &branchName, TTree &tree)
{
   auto buf = new TypedTreeReadBuffer<T>();
   tree.SetBranchAddress(branchName.Data(), &buf->_value);
   return std::unique_ptr<TreeReadBuffer>(buf);
}

#endif
