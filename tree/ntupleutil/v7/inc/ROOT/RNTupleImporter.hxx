/// \file ROOT/RNTupleImporter.hxx
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
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriteOptions.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <string_view>

#include <TFile.h>
#include <TTree.h>

#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <vector>

class TLeaf;

namespace ROOT {
namespace Experimental {

// clang-format off
/**
\class ROOT::Experimental::RNTupleImporter
\ingroup NTuple
\brief Converts a TTree into an RNTuple

Example usage (see the ntpl008_import.C tutorial for a full example):

~~~ {.cpp}
#include <ROOT/RNTupleImporter.hxx>
using ROOT::Experimental::RNTupleImporter;

auto importer = RNTupleImporter::Create("data.root", "TreeName", "output.root");
// As required: importer->SetNTupleName(), importer->SetWriteOptions(), ...
importer->Import();
~~~

The output file is created if it does not exist, otherwise the ntuple is added to the existing file.
Note that input file and output file can be identical if the ntuple is stored under a different name than the tree
(use `SetNTupleName()`).

By default, the RNTuple is compressed with zstd, independent of the input compression. The compression settings
(and other output parameters) can be changed by `SetWriteOptions()`. For example, to compress the imported RNTuple
using lz4 (with compression level 4) instead:

~~~ {.cpp}
auto writeOptions = importer->GetWriteOptions();
writeOptions.SetCompression(404);
importer->SetWriteOptions(writeOptions);
~~~

Most RNTuple fields have a type identical to the corresponding TTree input branch. Exceptions are
  - C string branches are translated to `std::string` fields
  - C style arrays are translated to `std::array<...>` fields
  - Leaf lists are translated to untyped records
  - Leaf count arrays are translated to anonymous collections with generic names (`_collection0`, `_collection1`, etc.).
    In order to keep field names and branch names aligned, RNTuple projects the members of these collections and
    its collection counter to the input branch names. For instance, the following input leafs:
~~~
Int_t njets
float jet_pt[njets]
float jet_eta[njets]
~~~
    will be converted to the following RNTuple schema:
~~~
      _collection0 (untyped collection)
      |- float jet_pt
      |- float jet_eta
      std::size_t (RNTupleCardinality) njets   (projected from _collection0 without subfields)
      ROOT::RVec<float>                jet_pt  (projected from _collection0.jet_pt)
      ROOT::RVec<float>                jet_eta (projected from _collection0.jet_eta)
~~~
    These projections are meta-data only operations and don't involve duplicating the data.

Current limitations of the importer:
  - No support for trees containing TClonesArray collections
  - Due to RNTuple currently storing data fully split, "don't split" markers are ignored
  - Some types are not available in RNTuple. Please refer to the
    [RNTuple specification](https://github.com/root-project/root/blob/master/tree/ntuple/v7/doc/specifications.md) for
    an overview of all types currently supported.
*/
// clang-format on
class RNTupleImporter {
public:
   /// Used to make adjustments to the fields of the output model.
   using FieldModifier_t = std::function<void(RFieldBase &)>;

   /// Used to report every ~100 MB (compressed), and at the end about the status of the import.
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
      ~RImportField() = default;
      RImportField(const RImportField &other) = delete;
      RImportField(RImportField &&other) = default;
      RImportField &operator=(const RImportField &other) = delete;
      RImportField &operator=(RImportField &&other) = default;

      /// The field is kept during schema preparation and transferred to the fModel before the writing starts
      RFieldBase *fField = nullptr;
      std::unique_ptr<RFieldBase::RValue> fValue; ///< Set if a value is generated, only for transformed fields
      void *fFieldBuffer = nullptr; ///< Usually points to the corresponding RImportBranch::fBranchBuffer but not always
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
   };

   /// When the schema is set up and the import started, it needs to be reset before the next Import() call
   /// can start. This RAII guard ensures that ResetSchema is called.
   struct RImportGuard {
      RNTupleImporter &fImporter;

      explicit RImportGuard(RNTupleImporter &importer) : fImporter(importer) {}
      RImportGuard(const RImportGuard &) = delete;
      RImportGuard &operator=(const RImportGuard &) = delete;
      RImportGuard(RImportGuard &&) = delete;
      RImportGuard &operator=(RImportGuard &&) = delete;
      ~RImportGuard() { fImporter.ResetSchema(); }
   };

   /// Leaf count arrays require special treatment. They are translated into untyped collections of untyped records.
   /// This class does the bookkeeping of the sub-schema for these collections.
   struct RImportLeafCountCollection {
      RImportLeafCountCollection() = default;
      RImportLeafCountCollection(const RImportLeafCountCollection &other) = delete;
      RImportLeafCountCollection(RImportLeafCountCollection &&other) = default;
      RImportLeafCountCollection &operator=(const RImportLeafCountCollection &other) = delete;
      RImportLeafCountCollection &operator=(RImportLeafCountCollection &&other) = default;
      std::string fFieldName; ///< name of the untyped collection, e.g. `_collection0`, `_collection1`, etc.
      /// Stores count leaf GetMaximum() to create large enough buffers for the array leafs.
      /// Uses Int_t because that is the return type if TLeaf::GetMaximum().
      Int_t fMaxLength = 0;
      /// The number of elements for the collection for a particular event. Used as a destination for SetBranchAddress()
      /// of the count leaf
      std::unique_ptr<Int_t> fCountVal;
      /// The leafs of the array as we encounter them traversing the TTree schema.
      /// Eventually, the fields are moved as leaves to an untyped collection of untyped records that in turn
      /// is attached to the RNTuple model.
      std::vector<std::unique_ptr<RFieldBase>> fLeafFields;
      std::vector<size_t> fLeafBranchIndexes; ///< Points to the correspondings leaf branches in fImportBranches
      RRecordField *fRecordField = nullptr; ///< Points to the item field of the untyped collection field in the model.
      std::vector<unsigned char> fFieldBuffer; ///< The collection field memory representation. Bound to the entry.
   };

   /// Transform a NULL terminated C string branch into an `std::string` field
   struct RCStringTransformation : public RImportTransformation {
      RCStringTransformation(std::size_t b, std::size_t f) : RImportTransformation(b, f) {}
      ~RCStringTransformation() override = default;
      RResult<void> Transform(const RImportBranch &branch, RImportField &field) final;
   };

   RNTupleImporter() = default;

   std::unique_ptr<TFile> fSourceFile;
   TTree *fSourceTree;

   std::string fDestFileName;
   std::string fNTupleName;
   std::unique_ptr<TFile> fDestFile;
   ROOT::RNTupleWriteOptions fWriteOptions;

   /// Whether or not dot characters in branch names should be converted to underscores. If this option is not set and a
   /// branch with a '.' is encountered, the importer will throw an exception.
   bool fConvertDotsInBranchNames = false;

   /// The maximum number of entries to import. When this value is -1 (default), import all entries.
   std::int64_t fMaxEntries = -1;

   /// No standard output, conversely if set to false, schema information and progress is printed.
   bool fIsQuiet = false;
   std::unique_ptr<RProgressCallback> fProgressCallback;
   FieldModifier_t fFieldModifier;

   std::unique_ptr<RNTupleModel> fModel;
   std::unique_ptr<REntry> fEntry;
   std::vector<RImportBranch> fImportBranches;
   std::vector<RImportField> fImportFields;
   /// Maps the count leaf to the information about the corresponding untyped collection
   std::map<std::string, RImportLeafCountCollection> fLeafCountCollections;
   /// The list of transformations to be performed for every entry
   std::vector<std::unique_ptr<RImportTransformation>> fImportTransformations;

   ROOT::RResult<void> InitDestination(std::string_view destFileName);

   void ResetSchema();
   /// Sets up the connection from TTree branches to RNTuple fields, including initialization of the memory
   /// buffers used for reading and writing.
   ROOT::RResult<void> PrepareSchema();
   void ReportSchema();

public:
   RNTupleImporter(const RNTupleImporter &other) = delete;
   RNTupleImporter &operator=(const RNTupleImporter &other) = delete;
   RNTupleImporter(RNTupleImporter &&other) = delete;
   RNTupleImporter &operator=(RNTupleImporter &&other) = delete;
   ~RNTupleImporter() = default;

   /// Opens the input file for reading and the output file for writing (update).
   static std::unique_ptr<RNTupleImporter>
   Create(std::string_view sourceFileName, std::string_view treeName, std::string_view destFileName);

   /// Directly uses the provided tree and opens the output file for writing (update).
   static std::unique_ptr<RNTupleImporter> Create(TTree *sourceTree, std::string_view destFileName);

   ROOT::RNTupleWriteOptions GetWriteOptions() const { return fWriteOptions; }
   void SetWriteOptions(ROOT::RNTupleWriteOptions options) { fWriteOptions = options; }
   void SetNTupleName(const std::string &name) { fNTupleName = name; }
   void SetMaxEntries(std::uint64_t maxEntries) { fMaxEntries = maxEntries; };

   /// Whereas branch names may contain dots, RNTuple field names may not. By setting this option, dot characters are
   /// automatically converted into underscores to prevent the importer from throwing an exception.
   void SetConvertDotsInBranchNames(bool value) { fConvertDotsInBranchNames = value; }

   /// Whether or not information and progress is printed to stdout.
   void SetIsQuiet(bool value) { fIsQuiet = value; }

   /// Add custom method to adjust column representations.  Will be called for every field of the frozen model
   /// before it is attached to the page sink
   void SetFieldModifier(FieldModifier_t modifier) { fFieldModifier = modifier; }

   /// Import works in two steps:
   /// 1. PrepareSchema() calls SetBranchAddress() on all the TTree branches and creates the corresponding RNTuple
   ///    fields and the model
   /// 2. An event loop reads every entry from the TTree, applies transformations where necessary, and writes the
   ///    output entry to the RNTuple.
   void Import();
}; // class RNTupleImporter

} // namespace Experimental
} // namespace ROOT

#endif
