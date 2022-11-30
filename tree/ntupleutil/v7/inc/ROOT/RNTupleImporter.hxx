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
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleOptions.hxx>
#include <ROOT/RStringView.hxx>

#include <TFile.h>
#include <TTree.h>

#include <cstdlib>
#include <memory>
#include <vector>

namespace ROOT {
namespace Experimental {

// clang-format off
/**
\class ROOT::Experimental::Detail::RNTupleImporter
\ingroup NTuple
\brief Converts a TTree into an RNTuple

The class steers the conversion of a TTree into an RNTuple.
*/
// clang-format on
class RNTupleImporter {
public:
   class RProgressCallback {
   public:
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
      std::string fBranchName;
      std::unique_ptr<unsigned char[]> fBranchBuffer;
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
         : fField(other.fField), fFieldBuffer(other.fFieldBuffer), fOwnsFieldBuffer(other.fOwnsFieldBuffer)
      {
         other.fOwnsFieldBuffer = false;
      }
      RImportField &operator=(const RImportField &other) = delete;
      RImportField &operator=(RImportField &&other)
      {
         fField = other.fField;
         fFieldBuffer = other.fFieldBuffer;
         fOwnsFieldBuffer = other.fOwnsFieldBuffer;
         other.fOwnsFieldBuffer = false;
         return *this;
      }

      Detail::RFieldBase *fField = nullptr;
      void *fFieldBuffer = nullptr;
      bool fOwnsFieldBuffer = false;
   };

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

   struct RCStringTransformation : public RImportTransformation {
      RCStringTransformation(std::size_t b, std::size_t f) : RImportTransformation(b, f) {}
      virtual ~RCStringTransformation() = default;
      RResult<void> Transform(const RImportBranch &branch, RImportField &field) final;
   };

   RNTupleImporter() = default;

   std::string fNTupleName;
   std::unique_ptr<TFile> fSourceFile;
   std::unique_ptr<TTree> fSourceTree;

   std::string fDestFileName;
   std::unique_ptr<TFile> fDestFile;
   RNTupleWriteOptions fWriteOptions;

   /// No standard output, conversly if set to false, schema information and progress is printed
   bool fIsQuiet = false;
   std::unique_ptr<RProgressCallback> fProgressCallback;

   std::vector<RImportBranch> fImportBranches;
   std::vector<RImportField> fImportFields;
   std::vector<std::unique_ptr<RImportTransformation>> fImportTransformations;
   std::unique_ptr<RNTupleModel> fModel;
   std::unique_ptr<REntry> fEntry;

   void ResetSchema();
   RResult<void> PrepareSchema();
   void ReportSchema();

public:
   RNTupleImporter(const RNTupleImporter &other) = delete;
   RNTupleImporter &operator=(const RNTupleImporter &other) = delete;
   RNTupleImporter(RNTupleImporter &&other) = delete;
   RNTupleImporter &operator=(RNTupleImporter &&other) = delete;
   ~RNTupleImporter() = default;

   static RResult<std::unique_ptr<RNTupleImporter>>
   Create(std::string_view sourceFile, std::string_view treeName, std::string_view destFile);

   RNTupleWriteOptions GetWriteOptions() const { return fWriteOptions; }
   void SetWriteOptions(RNTupleWriteOptions options) { fWriteOptions = options; }
   void SetNTupleName(const std::string &name) { fNTupleName = name; }

   void SetIsQuiet(bool value) { fIsQuiet = value; }

   RResult<void> Import();
}; // class RNTupleImporter

} // namespace Experimental
} // namespace ROOT

#endif
