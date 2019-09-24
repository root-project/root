/// \file ROOT/RPageStorageChain.hxx
/// \ingroup NTuple ROOT7
/// \author Simon Leisibach <simon.satoshi.rene.leisibach@cern.ch>
/// \date 2019-09-09
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RPageStorageChain
#define ROOT7_RPageStorageChain

#include <ROOT/RNTupleOptions.hxx>
#include <ROOT/RPageStorage.hxx>

#include <memory>

namespace ROOT {
namespace Experimental {
namespace Detail {

// clang-format off
/**
\class ROOT::Experimental::Detail::RPageSourceChain
\ingroup NTuple
\brief A pagesource generated from multiple files with the same fields and columns. It acts like a PageSource for a file where multiple files were merged into one.
 
 An instance of RPageSourceChain is created in RPageStorage::Create() when a std::vector of filenames is passed as an argument instead of a filename-string. It first creates a RPageSource (including its descriptor) for each filename passed and initalizes its members. Later it merges the information from all the descriptors to create a single descriptor containing all information. It's main job is to assign PopulatePage() and ReleasePage() to the correct RPageSource.
*/
// clang-format on
class RPageSourceChain : public RPageSource {
private:
   /// Holds a RPageSource pointer for each file.
   std::vector<std::unique_ptr<RPageSource>> fSources;
   /// Holds the cumulative number of entries per file. It's size is number of files + 1.
   /// fNEntryPerSource[i] holds the number of entries of the i-th file, fNEntryPerSource[0] = 0
   std::vector<std::size_t> fNEntryPerSource;
   /// Holds the cumulative number of clusters per file. It's size is number of files + 1.
   /// fNClusterPerSource[i] holds the number of clusters of the i-th file, fNClusterPerSource[0] = 0
   std::vector<std::size_t> fNClusterPerSource;
   /// The number of elements can vary between different columns of the same RPageSource, e.g. for a int and std::string
   /// column. fNElementsPerColumnPerSource.at(i).at(j) holds the entry of the i-th source and (j-1)-th column (start
   /// counting i and j from 1 instead of 0) fNElementsPerColumnPerSource.at(0).at(j) = 0.
   std::vector<std::vector<std::size_t>> fNElementsPerColumnPerSource;
   /// Maps the buffer of a RPage (void*) to its RPageSource (std::size_t = index of fSources)
   std::unordered_map<void *, std::size_t> fPageMapper;
   /// Is set to true when the meta-data of the fields and columns don't match.
   /// Getting pages from a unsafe RPageStorageChain can lead to undefined behaviour.
   bool fUnsafe = false;

protected:
   RNTupleDescriptor DoAttach();
   /// Checks if the field and column meta-data of all files are the same. If the meta-data doesn't match it allows
   /// further work but warns the user of undefined behaviour.
   void CompareFileMetaData();
   void InitializeVariables();

public:
   RPageSourceChain(std::string_view ntupleName, std::vector<std::string> locationVec,
                    const RNTupleReadOptions &options);
   RPageSourceChain(std::string_view ntupleName, std::vector<RPageSource *> sources, const RNTupleReadOptions &options);
   RPageSourceChain(std::string_view ntupleName, std::vector<std::unique_ptr<RPageSource>> &&sources,
                    const RNTupleReadOptions &options);
   RPageSourceChain(const RPageSourceChain &other) = delete;
   RPageSourceChain &operator=(const RPageSourceChain &other) = delete;
   RPageSourceChain(RPageSourceChain &&other) = default;
   RPageSourceChain &operator=(RPageSourceChain &&other) = default;
   ~RPageSourceChain() = default;

   std::unique_ptr<RPageSource> Clone() const;

   RPage PopulatePage(ColumnHandle_t columnHandle, NTupleSize_t globalIndex);
   RPage PopulatePage(ColumnHandle_t columnHandle, const RClusterIndex &clusterIndex);
   void ReleasePage(RPage &page);

   void GetHeaderAndFooter(RNTupleDescriptorBuilder &descBuilder);
   bool IsSafe() const { return fUnsafe; }
};

} // namespace Detail
} // namespace Experimental
} // namespace ROOT

#endif
