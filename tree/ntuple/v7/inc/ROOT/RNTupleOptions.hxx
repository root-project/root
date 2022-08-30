/// \file ROOT/RNTupleOptions.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2019-08-25
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RNTupleOptions
#define ROOT7_RNTupleOptions

#include <Compression.h>
#include <ROOT/RNTupleUtil.hxx>

#include <memory>

namespace ROOT {
namespace Experimental {

// clang-format off
/**
\class ROOT::Experimental::ENTupleContainerFormat
\ingroup NTuple
\brief Describes the options for wrapping RNTuple data in files
*/
// clang-format on
enum class ENTupleContainerFormat {
   kTFile, // ROOT TFile
   kBare, // A thin envelope supporting a single RNTuple only
};


// clang-format off
/**
\class ROOT::Experimental::RNTupleWriteOptions
\ingroup NTuple
\brief Common user-tunable settings for storing ntuples

All page sink classes need to support the common options.
*/
// clang-format on
class RNTupleWriteOptions {
   int fCompression{RCompressionSetting::EDefaults::kUseAnalysis};
   ENTupleContainerFormat fContainerFormat{ENTupleContainerFormat::kTFile};
   /// Approximation of the target compressed cluster size
   std::size_t fApproxZippedClusterSize = 50 * 1000 * 1000;
   /// Memory limit for committing a cluster: with very high compression ratio, we need a limit
   /// on how large the I/O buffer can grow during writing.
   std::size_t fMaxUnzippedClusterSize = 512 * 1024 * 1024;
   /// Should be just large enough so that the compression ratio does not benefit much more from larger pages.
   /// Unless the cluster is too small to contain a sufficiently large page, pages are
   /// fApproxUnzippedPageSize in size and tail pages (the last page in a cluster) is between
   /// fApproxUnzippedPageSize/2 and fApproxUnzippedPageSize * 1.5 in size.
   std::size_t fApproxUnzippedPageSize = 64 * 1024;
   bool fUseBufferedWrite = true;

public:
   virtual ~RNTupleWriteOptions() = default;
   virtual std::unique_ptr<RNTupleWriteOptions> Clone() const;

   int GetCompression() const { return fCompression; }
   void SetCompression(int val) { fCompression = val; }
   void SetCompression(RCompressionSetting::EAlgorithm::EValues algorithm, int compressionLevel) {
      fCompression = CompressionSettings(algorithm, compressionLevel);
   }

   ENTupleContainerFormat GetContainerFormat() const { return fContainerFormat; }
   void SetContainerFormat(ENTupleContainerFormat val) { fContainerFormat = val; }

   std::size_t GetApproxZippedClusterSize() const { return fApproxZippedClusterSize; }
   void SetApproxZippedClusterSize(std::size_t val);

   std::size_t GetMaxUnzippedClusterSize() const { return fMaxUnzippedClusterSize; }
   void SetMaxUnzippedClusterSize(std::size_t val);

   std::size_t GetApproxUnzippedPageSize() const { return fApproxUnzippedPageSize; }
   void SetApproxUnzippedPageSize(std::size_t val);

   bool GetUseBufferedWrite() const { return fUseBufferedWrite; }
   void SetUseBufferedWrite(bool val) { fUseBufferedWrite = val; }
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleWriteOptionsDaos
\ingroup NTuple
\brief DAOS-specific user-tunable settings for storing ntuples
*/
// clang-format on
class RNTupleWriteOptionsDaos : public RNTupleWriteOptions {
  std::string fObjectClass{"SX"};

public:
   ~RNTupleWriteOptionsDaos() override = default;
   std::unique_ptr<RNTupleWriteOptions> Clone() const override
   { return std::make_unique<RNTupleWriteOptionsDaos>(*this); }

   const std::string &GetObjectClass() const { return fObjectClass; }
   /// Set the object class used to generate OIDs that relate to user data. Any
   /// `OC_xxx` constant defined in `daos_obj_class.h` may be used here without
   /// the OC_ prefix.
   void SetObjectClass(const std::string &val) { fObjectClass = val; }
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleReadOptions
\ingroup NTuple
\brief Common user-tunable settings for reading ntuples

All page source classes need to support the common options.
*/
// clang-format on
class RNTupleReadOptions {
public:
   enum EClusterCache {
      kOff,
      kOn,
      kDefault = kOn,
   };

private:
   EClusterCache fClusterCache = EClusterCache::kDefault;
   unsigned int fClusterBunchSize = 1;

public:
   EClusterCache GetClusterCache() const { return fClusterCache; }
   void SetClusterCache(EClusterCache val) { fClusterCache = val; }
   unsigned int GetClusterBunchSize() const  { return fClusterBunchSize; }
   void SetClusterBunchSize(unsigned int val) { fClusterBunchSize = val; }
};

} // namespace Experimental
} // namespace ROOT

#endif
