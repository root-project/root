/// \file ROOT/RNTuplerImporter.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2022-11-22
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RNTuplerImporter
#define ROOT7_RNTuplerImporter

#include <ROOT/REntry.hxx>
#include <ROOT/RError.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleOptions.hxx>
#include <ROOT/RStringView.hxx>

#include <TFile.h>
#include <TTree.h>

#include <cstdlib>
#include <map>
#include <memory>
#include <vector>

class TLeaf;

namespace ROOT {
namespace Experimental {

// clang-format off
/**
\class ROOT::Experimental::Detail::RNTupleImporter
\ingroup NTuple
\brief Converts a TTree into an RNTuple

Example usage:

~~~ {.cpp}
#include <ROOT/RNTupleImporter.hxx>
using ROOT::Experimental::RNTupleImporter;

auto importer = RNTupleImporter::Create("data.root", "TreeName", "output.root").Unwrap();
// As required: importer->SetNTupleName(), importer->SetWriteOptions(), ...
importer->Import().ThrowOnError();
~~~

The output file is created if it does not exist, otherwise the ntuple is added to the existing file.
Note that input file and output file can be identical if the ntuple is stored under a different name than the tree
(use `SetNTupleName()`).

By default, the RNTuple is compressed with zstd, independent of the input compression. The compression settings
(and other output parameters) can be changed by `SetWriteOptions()`.

Most RNTuple fields have a type identical to the corresponding TTree input branch. Exceptions are
  - C string branches are translated to std::string fields
  - C style arrays are translated to std::array<...> fields
  - Leaf lists are translated to untyped records
  - Leaf count arrays are translated to anonymous collections with generic names (_collection0, collection1, etc.).
    In order to keep field names and branch names aligned, RNTuple projects the members of these collections and
    its collection counter to the input branch names. For instance, the input leafs
      Int_t njets;
      float jet_pt[njets]
      float jet_eta[njets]
    will be converted to the following RNTuple schema:
      _collection0 (untyped collection)
      |- float jet_pt
      |- float jet_eta
      std::size_t (RNTupleCardinality) njets (projected from _collection0 without subfields)
      ROOT::RVec<float> jet_pt (projected from _collection0.jet_pt)
      ROOT::RVec<float> jet_eta (projected from _collection0.jet_eta)
    These projections are meta-data only operations and don't involve duplicating the data.

Current limitations of the importer:
  - Importing collection proxies is untested
  - No support for trees containing TObject (or derived classes) or TClonesArray collections
  - Due to RNTuple currently storing data fully split, "don't split" markers are ignored
  - Some types are not (yet) available in RNTuple, such as pointers, Double32_t or std::map
*/
// clang-format on
class RNTupleImporter {
public:
   /// Used to report every ~50MB (compressed), and at the end about the status of the import.
   class RProgressCallback {
   public:
      virtual ~RProgressCallback() = default;
      void operator()(std::uint64_t nbytesWritten, std::uint64_t neventsWritten)
      {
         Call(nbytesWritten, neventsWritten);
      }
      virtual void Call(std::uint64_t nbytesWritten, std::uint64_t neventsWritten) = 0;
      virtual void Finish(std::uint64_t nbytesWritten, std::uint64_t neventsWritten) = 0;
   };

private:
   /// By default, compress RNTuple with zstd, level 5
   static constexpr int kDefaultCompressionSettings = 505;

   struct RImportBranch {
      RImportBranch() = default;
      RImportBranch(const RImportBranch &other) = delete;
      RImportBranch(RImportBranch &&other) = default;
      RImportBranch &operator=(const RImportBranch &other) = delete;
      RImportBranch &operator=(RImportBranch &&other) = default;
      std::string fBranchName;                        ///< Top-level branch name from the input TTree
      std::unique_ptr<unsigned char[]> fBranchBuffer; ///< The destination of SetBranchAddress() for `fBranchName`
   };

   struct RImportField {
      RImportField() = default;
      ~RImportField()
      {
         if (fOwnsFieldBuffer)
            free(fFieldBuffer);
      }
      RImportField(const RImportField &other) = delete;
      RImportField(RImportField &&other)
         : fField(other.fField),
           fFieldBuffer(other.fFieldBuffer),
           fOwnsFieldBuffer(other.fOwnsFieldBuffer),
           fIsInUntypedCollection(other.fIsInUntypedCollection),
           fIsClass(other.fIsClass)
      {
         other.fOwnsFieldBuffer = false;
      }
      RImportField &operator=(const RImportField &other) = delete;
      RImportField &operator=(RImportField &&other)
      {
         fField = other.fField;
         fFieldBuffer = other.fFieldBuffer;
         fOwnsFieldBuffer = other.fOwnsFieldBuffer;
         fIsInUntypedCollection = other.fIsInUntypedCollection;
         fIsClass = other.fIsClass;
         other.fOwnsFieldBuffer = false;
         return *this;
      }

      /// The field is kept during schema preparation and transferred to the fModel before the writing starts
      Detail::RFieldBase *fField = nullptr;
      void *fFieldBuffer = nullptr; ///< Usually points to the corresponding RImportBranch::fBranchBuffer but not always
      bool fOwnsFieldBuffer = false;       ///< Whether or not `fFieldBuffer` needs to be freed on destruction
      bool fIsInUntypedCollection = false; ///< Sub-fields of untyped collections (leaf count arrays in the input)
      bool fIsClass = false; ///< Field imported from a branch with stramer info (e.g., STL, user-defined class)
   };

   /// Base class to perform data transformations from TTree branches to RNTuple fields if necessary
   struct RImportTransformation {
      std::size_t fImportBranchIdx = 0;
      std::size_t fImportFieldIdx = 0;

      RImportTransformation(std::size_t branchIdx, std::size_t fieldIdx)
         : fImportBranchIdx(branchIdx), fImportFieldIdx(fieldIdx)
      {
      }
      virtual ~RImportTransformation() = default;
      virtual RResult<void> Transform(const RImportBranch &branch, RImportField &field) = 0;
      virtual void ResetEntry() = 0; // called at the end of an entry
   };

   /// Leaf count arrays require special treatment. They are translated into RNTuple untyped collections.
   /// This class does the bookkeeping of the sub-schema for these collections.
   struct RImportLeafCountCollection {
      RImportLeafCountCollection() = default;
      RImportLeafCountCollection(const RImportLeafCountCollection &other) = delete;
      RImportLeafCountCollection(RImportLeafCountCollection &&other) = default;
      RImportLeafCountCollection &operator=(const RImportLeafCountCollection &other) = delete;
      RImportLeafCountCollection &operator=(RImportLeafCountCollection &&other) = default;
      std::unique_ptr<RNTupleModel> fCollectionModel;             ///< The model for the collection itself
      std::shared_ptr<RCollectionNTupleWriter> fCollectionWriter; ///< Used to fill the collection elements per event
      std::unique_ptr<REntry> fCollectionEntry; ///< Keeps the memory location of the collection members
      /// The number of elements for the collection for a particular event. Used as a destination for SetBranchAddress()
      /// of the count leaf
      std::unique_ptr<Int_t> fCountVal;
      std::vector<size_t> fImportFieldIndexes; ///< Points to the correspondings fields in fImportFields
      /// One transformation for every field, to copy the content of the array one by one
      std::vector<std::unique_ptr<RImportTransformation>> fTransformations;
      Int_t fMaxLength = 0;   ///< Stores count leaf GetMaximum() to create large enough buffers for the array leafs
      std::string fFieldName; ///< name of the untyped collection, e.g. _collection0, _collection1, etc.
   };

   /// Transform a NULL terminated C string branch into an std::string field
   struct RCStringTransformation : public RImportTransformation {
      RCStringTransformation(std::size_t b, std::size_t f) : RImportTransformation(b, f) {}
      virtual ~RCStringTransformation() = default;
      RResult<void> Transform(const RImportBranch &branch, RImportField &field) final;
      void ResetEntry() final {}
   };

   /// When writing the elements of a leaf count array, moves the data from the input array one-by-one
   /// to the memory locations of the fields of the corresponding untyped collection.
   /// TODO(jblomer): write arrays as a whole to RNTuple
   struct RLeafArrayTransformation : public RImportTransformation {
      std::int64_t fNum = 0;
      RLeafArrayTransformation(std::size_t b, std::size_t f) : RImportTransformation(b, f) {}
      virtual ~RLeafArrayTransformation() = default;
      RResult<void> Transform(const RImportBranch &branch, RImportField &field) final;
      void ResetEntry() final { fNum = 0; }
   };

   RNTupleImporter() = default;

   std::unique_ptr<TFile> fSourceFile;
   std::vector<TTree *> fSourceTrees;

   /// Maps the name of the original TTree to the desired RNTuple name.
   std::map<std::string_view, std::string> fNTupleNames;
   std::string fDestFileName;
   std::unique_ptr<TFile> fDestFile;
   RNTupleWriteOptions fWriteOptions;

   /// The maximum number of entries to import. When this value is -1 (default), import all entries.
   std::int64_t fMaxEntries = -1;

   /// No standard output, conversely if set to false, schema information and progress is printed.
   bool fIsQuiet = false;
   std::unique_ptr<RProgressCallback> fProgressCallback;

   std::vector<RImportBranch> fImportBranches;
   std::vector<RImportField> fImportFields;
   /// Maps the count leaf to the information about the corresponding untyped collection
   std::map<std::string, RImportLeafCountCollection> fLeafCountCollections;
   /// The list of transformations to be performed for every entry
   std::vector<std::unique_ptr<RImportTransformation>> fImportTransformations;
   std::unique_ptr<RNTupleModel> fModel;
   std::unique_ptr<REntry> fEntry;

   ROOT::Experimental::RResult<void> InitDestination(std::string_view destFileName);

   void ResetSchema();
   /// Sets up the connection from TTree branches to RNTuple fields, including initialization of the memory
   /// buffers used for reading and writing.
   RResult<void> PrepareSchema(TTree *sourceTree);
   void ReportSchema();

   RResult<void> Import(TTree *sourceTree);

public:
   RNTupleImporter(const RNTupleImporter &other) = delete;
   RNTupleImporter &operator=(const RNTupleImporter &other) = delete;
   RNTupleImporter(RNTupleImporter &&other) = delete;
   RNTupleImporter &operator=(RNTupleImporter &&other) = delete;
   ~RNTupleImporter() = default;

   /// Opens the input file for reading and the output file for writing (update).
   static RResult<std::unique_ptr<RNTupleImporter>>
   Create(std::string_view sourceFileName, std::string_view treeName, std::string_view destFileName);

   /// Directly uses the provided tree and opens the output file for writing (update).
   static RResult<std::unique_ptr<RNTupleImporter>> Create(TTree *sourceTree, std::string_view destFileName);

   /// Imports all TTree objects present in `sourceFile` as RNTuple objects to `destFile`.
   static RResult<std::unique_ptr<RNTupleImporter>> Create(std::string_view sourceFile, std::string_view destFile);

   RNTupleWriteOptions GetWriteOptions() const { return fWriteOptions; }
   void SetWriteOptions(RNTupleWriteOptions options) { fWriteOptions = options; }
   void SetMaxEntries(std::uint64_t maxEntries) { fMaxEntries = maxEntries; }

   ROOT::Experimental::RResult<void> SetNTupleName(std::string_view treeName, const std::string &ntupleName);

   /// Whether or not information and progress is printed to stdout.
   void SetIsQuiet(bool value) { fIsQuiet = value; }

   /// Import works in two steps:
   /// 1. PrepareSchema() calls SetBranchAddress() on all the TTree branches and creates the corresponding RNTuple
   ///    fields and the model
   /// 2. An event loop reads every entry from the TTree, applies transformations where necessary, and writes the
   ///    output entry to the RNTuple.
   RResult<void> Import();
}; // class RNTupleImporter

} // namespace Experimental
} // namespace ROOT

#endif
