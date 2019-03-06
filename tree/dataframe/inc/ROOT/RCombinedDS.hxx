#ifndef ROOT_RCOMBINEDDS
#define ROOT_RCOMBINEDDS

#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDataSource.hxx"

#include <memory>
#include <string>
#include <vector>

namespace ROOT
{

namespace RDF
{

/// This is the baseclass which actually implements how two tables should be
/// combined together, via the GetAssociatedEntries() method. The BuildIndex()
/// method is a convenience method which gets invoked only once per
/// Initialise() and that can be used to precompute the index itself, if the
/// mapping combinedEntry -> (leftEntry, rightEntry) cannot be computed on
/// the fly quickly.
class RCombinedDSIndex
{
 public:
  virtual ~RCombinedDSIndex() = default;
  /// This is invoked on every Inititialise of the RCombinedDS to
  /// allow constructing the index associated to it.
  /// \param[in]left is the dataframe constructed on top of the left input.
  /// \param[in]right is the dataframe constructed on top of the right input.
  /// \result the vector with the ranges of the combined dataset.
  virtual std::vector<std::pair<ULong64_t, ULong64_t>> BuildIndex(std::unique_ptr<RDataFrame>& left,
                                                                  std::unique_ptr<RDataFrame>& right) = 0;
  /// This is invoked on every GetEntry() of the RCombinedDS and
  /// it's used to effectively enumerate all the pairs of the combination.
  /// \param[in]entry is the entry in the combined table
  /// \result a pair where first is the entry in the associated left table, while
  ///         right is an entry in the associated right table.
  virtual std::pair<ULong64_t, ULong64_t> GetAssociatedEntries(ULong64_t entry) = 0;
};

/// An index which allows doing a inner join on the row number for two tables,
/// i.e.  putting the rows of one next to the rows of other.
class RCombindedDSFriendIndex : public RCombinedDSIndex
{
 public:
  std::pair<ULong64_t, ULong64_t> GetAssociatedEntries(ULong64_t entry) final
  {
    return std::pair<ULong64_t, ULong64_t>(entry, entry);
  }
  std::vector<std::pair<ULong64_t, ULong64_t>> BuildIndex(std::unique_ptr<RDataFrame>& left,
                                                          std::unique_ptr<RDataFrame>& right) final;
};

/// An index which allows doing a cross join between two tables. I.e. all the
/// entries of one coupled with all the entries of the other
class RCombindedDSCrossJoinIndex : public RCombinedDSIndex
{
 public:
  std::pair<ULong64_t, ULong64_t> GetAssociatedEntries(ULong64_t entry) final
  {
    return std::make_pair<ULong64_t, ULong64_t>(entry / fRightCount, entry % fRightCount);
  }
  std::vector<std::pair<ULong64_t, ULong64_t>> BuildIndex(std::unique_ptr<RDataFrame>& left,
                                                          std::unique_ptr<RDataFrame>& right) final;

 private:
  ULong64_t fLeftCount;
  ULong64_t fRightCount;
};

/// An index which allows doing a join using a column on the right.
/// FIXME: the need for templation is due to the fact RDataFrame::GetColumnType
///        was introduced only in ROOT 6.16.x. We can remove it once
///        we have a proper build of ROOT.
template <typename INDEX_TYPE = int>
class RCombindedDSColumnJoinIndex : public RCombinedDSIndex
{
 public:
  RCombindedDSColumnJoinIndex(std::string const& indexColumnName)
    : fIndexColumnName{ indexColumnName }
  {
  }

  std::pair<ULong64_t, ULong64_t> GetAssociatedEntries(ULong64_t entry) final
  {
    auto left = fAssociations[entry];
    return std::pair<ULong64_t, ULong64_t>(left, entry);
  }

  std::vector<std::pair<ULong64_t, ULong64_t>>
    BuildIndex(std::unique_ptr<RDataFrame>& left,
               std::unique_ptr<RDataFrame>& right) final
  {
    std::vector<std::pair<ULong64_t, ULong64_t>> ranges;
    auto nEntries = *right->Count();
    fAssociations.reserve(nEntries);
    // Fill the index with the associations
    auto filler = [& assoc = fAssociations](INDEX_TYPE ri) { assoc.push_back(ri); };
    right->Foreach(filler, std::vector<std::string>{ fIndexColumnName });

    // Create the ranges by processing 64 entries per range
    auto deltaRange = 64;
    ranges.reserve(nEntries / deltaRange + 1);
    ULong64_t i = 0;
    while (deltaRange * (i + 1) < nEntries) {
      ranges.emplace_back(std::pair<ULong64_t, ULong64_t>(deltaRange * i, deltaRange * (i + 1)));
    }
    ranges.emplace_back(std::pair<ULong64_t, ULong64_t>(deltaRange * i, nEntries)); // Last entry
    return ranges;
  }

 private:
  std::string fIndexColumnName;
  std::vector<INDEX_TYPE> fAssociations;
};

/// An index which allows doing a cross join of all entries belonging to the
/// same category, where the category is defined by a two given columns.
///
/// This can be used to do double loops on events when using the event id column.
/// FIXME: the need for templation is due to the fact RDataFrame::GetColumnType
///        was introduced only in ROOT 6.16.x. We can remove it once
///        we have a proper build of ROOT.
/// FIXME: for the moment this only works when the inputs are actually from the same
///        table.
template <typename INDEX_TYPE = int>
class RCombindedDSBlockCrossJoinIndex : public RCombinedDSIndex
{

  using Association = std::pair<ULong64_t, ULong64_t>;
 public:
  enum struct BlockCombinationRule {
    Full,
    Upper,
    StrictlyUpper,
    Diagonal,
    Anti
  };

  RCombindedDSBlockCrossJoinIndex(std::string const& leftCategoryColumn,
                                  bool self = true,
                                  BlockCombinationRule combinationType = BlockCombinationRule::Anti,
                                  std::string const& rightCategoryColumn = "")
    : fLeftCategoryColumn{ leftCategoryColumn },
      fRightCategoryColumn{ rightCategoryColumn.empty() ? leftCategoryColumn : rightCategoryColumn },
      fSelf{self},
      fCombinationType{combinationType}
  {
  }

  std::pair<ULong64_t, ULong64_t> GetAssociatedEntries(ULong64_t entry) final
  {
    return fAssociations[entry];
  }

  std::vector<std::pair<ULong64_t, ULong64_t>>
    BuildIndex(std::unique_ptr<RDataFrame>& left,
               std::unique_ptr<RDataFrame>& right) final
  {
    std::vector<std::pair<ULong64_t, ULong64_t>> ranges;
    std::vector<INDEX_TYPE> leftCategories;
    std::vector<INDEX_TYPE> rightCategories;
    std::vector<Association> leftPairs;
    std::vector<Association> rightPairs;

    computePairsAndCategories(left, leftCategories, leftPairs, fLeftCategoryColumn);
    /// FIXME: we should avoid the memory copy here. OK for now.
    if (fSelf) {
      rightCategories = leftCategories;
      rightPairs = leftPairs;
    } else {
      computePairsAndCategories(right, rightCategories, rightPairs, fRightCategoryColumn);
    }

    auto same = [](std::pair<ULong64_t, ULong64_t> const& a, std::pair<ULong64_t, ULong64_t> const& b) {
      return a.first < b.first;
    };

    /// For all categories, do the full set of permutations.
    /// In case the two inputs are the same, we can simply reuse the same range
    /// of entries.
    int startSize = fAssociations.size();
    for (auto categoryValue : leftCategories) {
      std::pair<ULong64_t, ULong64_t> p{categoryValue, 0};
      auto outerRange = std::equal_range(leftPairs.begin(), leftPairs.end(), p, same);
      decltype(outerRange) innerRange;
      if (fSelf) {
        innerRange = outerRange;
      } else {
        innerRange = std::equal_range(rightPairs.begin(), rightPairs.end(), p, same);
      }
      int offset = 0;
      switch (fCombinationType) {
        case BlockCombinationRule::Full:
          for (auto out = outerRange.first; out != outerRange.second; ++out) {
            for (auto in = innerRange.first; in != innerRange.second; ++in) {
              fAssociations.emplace_back(Association{ out->second, in->second });
            }
          }
          break;
        case BlockCombinationRule::Upper:
          offset = 0;
          for (auto out = outerRange.first; out != outerRange.second; ++out) {
            if (innerRange.first == innerRange.second) {
              break;
            }
            for (auto in = innerRange.first + offset; in != innerRange.second; ++in) {
              fAssociations.emplace_back(Association{ out->second, in->second });
            }
            offset++;
          }
          break;
        case BlockCombinationRule::StrictlyUpper:
          offset = 1;
          for (auto out = outerRange.first; out != outerRange.second; ++out) {
            if (innerRange.first == innerRange.second || innerRange.first + 1 == innerRange.second) {
              break;
            }
            for (auto in = innerRange.first + offset; in != innerRange.second; ++in) {
              fAssociations.emplace_back(Association{ out->second, in->second });
            }
            offset++;
          }
          break;
        case BlockCombinationRule::Anti:
          for (auto out = outerRange.first; out != outerRange.second; ++out) {
            for (auto in = innerRange.first; in != innerRange.second; ++in) {
              if (std::distance(innerRange.first, in) == std::distance(outerRange.first, out)) {
                continue;
              }
              fAssociations.emplace_back(Association{ out->second, in->second });
            }
            offset++;
          }
          break;
        case BlockCombinationRule::Diagonal:
          auto sizeRow = std::distance(outerRange.first, outerRange.second);
          auto sizeCol = std::distance(innerRange.first, innerRange.second);
          for (size_t i = 0, e = std::min(sizeRow, sizeCol); i < e; ++i) {
            fAssociations.emplace_back(Association{ (outerRange.first + i)->second, (innerRange.first + i)->second });
          }
          break;
      }
      auto rangeFirst = startSize;
      auto rangeSecond = fAssociations.size();
      startSize = fAssociations.size();
      ranges.emplace_back(std::make_pair<ULong64_t, ULong64_t>(rangeFirst, rangeSecond));
    }
    return ranges;
  }
 private:
  std::string fLeftCategoryColumn;
  std::string fRightCategoryColumn;
  bool fSelf;
  BlockCombinationRule fCombinationType;
  std::vector<Association> fAssociations;
  void computePairsAndCategories(std::unique_ptr<RDataFrame>& df,
                                 std::vector<INDEX_TYPE>& categories,
                                 std::vector<Association>& pairs,
                                 std::string const& column) {
    categories = *df->template Take<INDEX_TYPE>(column);
    // Fill the pairs according tho the actual category
    for (size_t i = 0; i < categories.size(); ++i) {
      pairs.emplace_back(categories[i], i);
    }
    // Do a stable sort so that same categories entries are
    // grouped together.
    std::stable_sort(pairs.begin(), pairs.end());
    // Keep only the categories.
    std::stable_sort(categories.begin(), categories.end());
    auto last = std::unique(categories.begin(), categories.end());
    categories.erase(last, categories.end());
  }
};

/// An RDataSource which combines the rows of two other RDataSources
/// between them. The actual logic to do the combination is specified by
/// the provided RCombinedDSIndex implementation.
/// By default it simply pairs same position rows of two tables with the same
/// length.
class RCombinedDS final : public ROOT::RDF::RDataSource
{
 private:
  /// We need bare pointers because we need to move the ownership of
  /// the datasource to the dataframe
  RDataSource* fLeft;
  RDataSource* fRight;
  std::string fLeftPrefix;
  std::string fRightPrefix;
  std::unique_ptr<RDataFrame> fLeftDF;
  std::unique_ptr<RDataFrame> fRightDF;
  ULong64_t fLeftCount;
  ULong64_t fRightCount;
  size_t fNSlots = 0U;
  std::vector<std::string> fColumnNames;
  std::vector<std::pair<ULong64_t, ULong64_t>> fEntryRanges;
  std::unique_ptr<RCombinedDSIndex> fIndex;

 protected:
  std::vector<void*> GetColumnReadersImpl(std::string_view colName, const std::type_info& info) override;

 public:
  RCombinedDS(std::unique_ptr<RDataSource> left,
              std::unique_ptr<RDataSource> right,
              std::unique_ptr<RCombinedDSIndex> index = std::make_unique<RCombindedDSFriendIndex>(),
              std::string leftPrefix = std::string{ "left_" },
              std::string rightPrefix = std::string{ "right_" });
  ~RCombinedDS() override;

  template <typename T>
  std::vector<T**> GetColumnReaders(std::string_view colName)
  {
    if (colName.compare(0, fLeftPrefix.size(), fLeftPrefix)) {
      colName.remove_prefix(fLeftPrefix.size());
      return fLeft->GetColumnReaders<T>(colName);
    }
    if (colName.compare(0, fRightPrefix.size(), fRightPrefix)) {
      colName.remove_prefix(fRightPrefix.size());
      return fRight->GetColumnReaders<T>(colName);
    }
    throw std::runtime_error("Column not found: " + colName);
  }
  const std::vector<std::string>& GetColumnNames() const override;
  std::vector<std::pair<ULong64_t, ULong64_t>> GetEntryRanges() override;
  std::string GetTypeName(std::string_view colName) const override;
  bool HasColumn(std::string_view colName) const override;
  bool SetEntry(unsigned int slot, ULong64_t entry) override;
  void InitSlot(unsigned int slot, ULong64_t firstEntry) override;
  void SetNSlots(unsigned int nSlots) override;
  void Initialise() override;
};

////////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Factory method to create a Apache Arrow RDataFrame.
/// \param[in] table an apache::arrow table to use as a source.
RDataFrame MakeCombinedDataFrame(std::unique_ptr<RDataSource> left, std::unique_ptr<RDataSource>, std::unique_ptr<RCombinedDSIndex> index, std::string leftPrefix = "left_", std::string rightPrefix = "right_");
RDataFrame MakeCrossProductDataFrame(std::unique_ptr<RDataSource> left, std::unique_ptr<RDataSource>, std::string leftPrefix = "left_", std::string rightPrefix = "right_");
RDataFrame MakeColumnIndexedDataFrame(std::unique_ptr<RDataSource> left, std::unique_ptr<RDataSource>, std::string indexColName, std::string leftPrefix = "left_", std::string rightPrefix = "right_");
RDataFrame MakeFriendDataFrame(std::unique_ptr<RDataSource> left, std::unique_ptr<RDataSource> right, std::string leftPrefix = "left_", std::string rightPrefix = "right_");
RDataFrame MakeBlockAntiDataFrame(std::unique_ptr<RDataSource> left, std::unique_ptr<RDataSource> right, std::string indexColumnName, std::string leftPrefix = "left_", std::string rightPrefix = "right_");

} // namespace RDF

} // namespace ROOT

#endif
