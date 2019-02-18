// Author: Giulio Eulisse CERN  2/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// clang-format off
/** \class ROOT::RDF::RCombinedDS
    \ingroup dataframe
    \brief RDataSource which does the cartesian product of entries in two other datasources

This RDataSource takes two input datasources, and iterates on all the couples of the 
cartesian product between the set of entries of the two. This is effectively mimicking a
double loop on the entries of the two RDataSources.

*/
// clang-format on

#define protected public
#include <ROOT/RCombinedDS.hxx>

#include <ROOT/RDFUtils.hxx>
#include <ROOT/TSeq.hxx>
#include <ROOT/RMakeUnique.hxx>
#include <ROOT/RDataFrame.hxx>

#include <algorithm>
#include <sstream>
#include <string>

using namespace ROOT::RDF;

namespace ROOT
{
namespace RDF
{

std::vector<std::pair<ULong64_t, ULong64_t>>
  RCombindedDSCrossJoinIndex::BuildIndex(std::unique_ptr<RDataFrame>& left,
                                         std::unique_ptr<RDataFrame>& right)
{
  std::vector<std::pair<ULong64_t, ULong64_t>> ranges;
  fLeftCount = *left->Count();
  fRightCount = *right->Count();
  ranges.reserve(fLeftCount);
  for (ULong64_t i = 0; i < fLeftCount; ++i) {
    ranges.emplace_back(std::make_pair<ULong64_t, ULong64_t>(fRightCount * i, fRightCount * (i + 1)));
  }
  return ranges;
}

std::vector<std::pair<ULong64_t, ULong64_t>>
  RCombindedDSFriendIndex::BuildIndex(std::unique_ptr<RDataFrame>& left,
                                      std::unique_ptr<RDataFrame>& right)
{
  auto leftCount = *left->Count();
  auto rightCount = *right->Count();
  if (leftCount != rightCount) {
    throw std::runtime_error("Union can be performed only with two datasources which have the same amount of entries");
  }
  std::vector<std::pair<ULong64_t, ULong64_t>> ranges;
  // FIXME: should we really use the min between two number of slots?
  auto nSlots = std::min(left->GetLoopManager()->GetNSlots(), right->GetLoopManager()->GetNSlots());
  assert(nSlots > 0);
  auto deltaSize = rightCount / nSlots;
  ULong64_t i = 0;
  for (; i < (nSlots - 1); ++i) {
    ranges.emplace_back(std::pair<ULong64_t, ULong64_t>(deltaSize * i, deltaSize * (i + 1)));
  }
  ranges.emplace_back(std::pair<ULong64_t, ULong64_t>(i*deltaSize, rightCount));
  return ranges;
}

////////////////////////////////////////////////////////////////////////
/// Constructor to create an Arrow RDataSource for RDataFrame.
/// \param[in] left the first table we iterate on, i.e. the outer loop
/// \param[in] right the second table we iterate on, i.e. the inner loop
/// \param[in] leftPrefix the prefix to prepend to the element of the first table
/// \param[in] right the second table we iterate on, i.e. the inner loop
/// \param[in] rightPrefix the prefix to prepend to the element of the second table
RCombinedDS::RCombinedDS(std::unique_ptr<RDataSource> inLeft, std::unique_ptr<RDataSource> inRight,
                         std::unique_ptr<RCombinedDSIndex> inIndex,
                         std::string inLeftPrefix, std::string inRightPrefix)
  : // FIXME: we cache the bare pointers, under the assumption that
    // the dataframes fLeftDF, fRightDF have longer lifetime as
    // they actually own them.
    fLeft{ inLeft.get() },
    fRight{ inRight.get() },
    fLeftDF{ std::make_unique<RDataFrame>(std::move(inLeft)) },
    fRightDF{ std::make_unique<RDataFrame>(std::move(inRight)) },
    fLeftPrefix{ inLeftPrefix },
    fRightPrefix{ inRightPrefix },
    fIndex{ std::move(inIndex) }
{
  fColumnNames.reserve(fLeft->GetColumnNames().size() + fRight->GetColumnNames().size());
  for (auto& c : fLeft->GetColumnNames()) {
    fColumnNames.push_back(fLeftPrefix + c);
  }
  for (auto& c : fRight->GetColumnNames()) {
    fColumnNames.push_back(fRightPrefix + c);
  }
}

////////////////////////////////////////////////////////////////////////
/// Destructor.
RCombinedDS::~RCombinedDS() = default;

const std::vector<std::string>& RCombinedDS::GetColumnNames() const
{
  return fColumnNames;
}

std::vector<std::pair<ULong64_t, ULong64_t>> RCombinedDS::GetEntryRanges()
{
  auto entryRanges(std::move(fEntryRanges)); // empty fEntryRanges
  return entryRanges;
}

std::string RCombinedDS::GetTypeName(std::string_view colName) const
{
  if (colName.compare(0, fLeftPrefix.size(), fLeftPrefix) == 0) {
    colName.remove_prefix(fLeftPrefix.size());
    return fLeft->GetTypeName(colName);
  }
  if (colName.compare(0, fRightPrefix.size(), fRightPrefix) == 0) {
    colName.remove_prefix(fRightPrefix.size());
    return fRight->GetTypeName(colName);
  }
  throw std::runtime_error("Column not found: " + colName);
}

bool RCombinedDS::HasColumn(std::string_view colName) const
{
  if (colName.compare(0, fLeftPrefix.size(), fLeftPrefix) == 0) {
    colName.remove_prefix(fLeftPrefix.size());
    return fLeft->HasColumn(colName);
  }
  if (colName.compare(0, fRightPrefix.size(), fRightPrefix) == 0) {
    colName.remove_prefix(fRightPrefix.size());
    return fRight->HasColumn(colName);
  }
  return false;
}

bool RCombinedDS::SetEntry(unsigned int slot, ULong64_t entry)
{
  std::pair<ULong64_t, ULong64_t> association = fIndex->GetAssociatedEntries(entry);
  fLeft->SetEntry(slot, association.first);
  fRight->SetEntry(slot, association.second);
  return true;
}

void RCombinedDS::InitSlot(unsigned int slot, ULong64_t entry)
{
  std::pair<ULong64_t, ULong64_t> association = fIndex->GetAssociatedEntries(entry);
  fLeft->InitSlot(slot, association.first);
  fRight->InitSlot(slot, association.second);
}

void RCombinedDS::SetNSlots(unsigned int nSlots)
{
  assert(0U == fNSlots && "Setting the number of slots even if the number of slots is different from zero.");
  /// FIXME: For the moment we simply forward the nSlots, not sure this is the
  ///        right thing to do.
  fLeft->SetNSlots(nSlots);
  fRight->SetNSlots(nSlots);
}

/// This should never be called, since we did a template overload for GetColumnReaders()
std::vector<void*> RCombinedDS::GetColumnReadersImpl(std::string_view colName, const std::type_info& info)
{
  if (colName.compare(0, fLeftPrefix.size(), fLeftPrefix) == 0) {
    colName.remove_prefix(fLeftPrefix.size());
    return fLeft->GetColumnReadersImpl(colName, info);
  }
  if (colName.compare(0, fRightPrefix.size(), fRightPrefix) == 0) {
    colName.remove_prefix(fRightPrefix.size());
    return fRight->GetColumnReadersImpl(colName, info);
  }
  assert(false);
}

void RCombinedDS::Initialise()
{
  fEntryRanges = fIndex->BuildIndex(fLeftDF, fRightDF);

  fLeft->Initialise();
  fRight->Initialise();
}

/// Creates a RDataFrame using an arrow::Table as input.
/// \param[in] table the arrow Table to observe.
/// \param[in] columnNames the name of the columns to use
/// In case columnNames is empty, we use all the columns found in the table
RDataFrame MakeCombinedDataFrame(std::unique_ptr<RDataSource> left, std::unique_ptr<RDataSource> right,
                                 std::unique_ptr<RCombinedDSIndex> index,
                                 std::string leftPrefix, std::string rightPrefix)
{
  ROOT::RDataFrame tdf(std::make_unique<RCombinedDS>(std::move(left), std::move(right), std::move(index), leftPrefix, rightPrefix));
  return tdf;
}

RDataFrame MakeCrossProductDataFrame(std::unique_ptr<RDataSource> left, std::unique_ptr<RDataSource> right,
                                     std::string leftPrefix, std::string rightPrefix)
{
  ROOT::RDataFrame tdf(std::make_unique<RCombinedDS>(std::move(left), std::move(right), std::move(std::make_unique<RCombindedDSCrossJoinIndex>()), leftPrefix, rightPrefix));
  return tdf;
}

RDataFrame MakeColumnIndexedDataFrame(std::unique_ptr<RDataSource> left, std::unique_ptr<RDataSource> right,
                                      std::string indexColumnName,
                                      std::string leftPrefix, std::string rightPrefix)
{
  ROOT::RDataFrame tdf(std::make_unique<RCombinedDS>(std::move(left), std::move(right), std::move(std::make_unique<RCombindedDSColumnJoinIndex<int>>(indexColumnName)), leftPrefix, rightPrefix));
  return tdf;
}

RDataFrame MakeBlockAntiDataFrame(std::unique_ptr<RDataSource> left, std::unique_ptr<RDataSource> right,
                                  std::string indexColumnName,
                                  std::string leftPrefix, std::string rightPrefix)
{
  ROOT::RDataFrame tdf(std::make_unique<RCombinedDS>(std::move(left), std::move(right), std::move(std::make_unique<RCombindedDSBlockCrossJoinIndex<int>>(indexColumnName)), leftPrefix, rightPrefix));
  return tdf;
}

RDataFrame MakeFriendDataFrame(std::unique_ptr<RDataSource> left, std::unique_ptr<RDataSource> right,
                               std::string leftPrefix, std::string rightPrefix)
{
  ROOT::RDataFrame tdf(std::make_unique<RCombinedDS>(std::move(left), std::move(right), std::move(std::make_unique<RCombindedDSFriendIndex>()), leftPrefix, rightPrefix));
  return tdf;
}


} // namespace RDF
} // namespace ROOT
