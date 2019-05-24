/// \file TClingRdictModuleFileExtension.h
///
/// \brief The file contains facilities to work with C++ module files extensions
///        used to store rdict files.
///
/// \author Vassil Vassilev <vvasilev@cern.ch>
///
/// \date May, 2019
///
/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_CLING_RDICT_MODULE_FILE_EXTENSION_H
#define ROOT_CLING_RDICT_MODULE_FILE_EXTENSION_H

#include "clang/Serialization/ModuleFileExtension.h"

#include "llvm/Bitcode/BitstreamReader.h"

/// A module file extension used for testing purposes.
class TClingRdictModuleFileExtension : public clang::ModuleFileExtension {
   std::string UserInfo;
   std::string RdictFileName;

   class Writer : public clang::ModuleFileExtensionWriter {
   public:
      Writer(ModuleFileExtension *Ext) : ModuleFileExtensionWriter(Ext) {}
      ~Writer() override;

      void writeExtensionContents(clang::Sema &SemaRef, llvm::BitstreamWriter &Stream) override;
   };

   class Reader : public clang::ModuleFileExtensionReader {
      llvm::BitstreamCursor Stream;

   public:
      ~Reader() override;

      Reader(clang::ModuleFileExtension *Ext, clang::ASTReader &Reader, clang::serialization::ModuleFile &Mod,
             const llvm::BitstreamCursor &InStream);
   };

public:
   TClingRdictModuleFileExtension(llvm::StringRef UserInfo) : UserInfo(UserInfo) {}

   ~TClingRdictModuleFileExtension() override;

   clang::ModuleFileExtensionMetadata getExtensionMetadata() const override;

   llvm::hash_code hashExtension(llvm::hash_code Code) const override;

   std::unique_ptr<clang::ModuleFileExtensionWriter> createExtensionWriter(clang::ASTWriter &Writer) override;

   std::unique_ptr<clang::ModuleFileExtensionReader>
   createExtensionReader(const clang::ModuleFileExtensionMetadata &Metadata, clang::ASTReader &Reader,
                         clang::serialization::ModuleFile &Mod, const llvm::BitstreamCursor &Stream) override;
};

#endif
