// @(#)root/utils:$Id$
// Author: Axel Naumann, 2-13-07-02

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////
//
// PCM writer.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TModuleGenerator
#define ROOT_TModuleGenerator

#include <ostream>
#include <string>
#include <vector>

namespace clang {
   class CompilerInstance;
}

namespace ROOT {

//______________________________________________________________________________
class TModuleGenerator {
public:
   enum ESourceFileKind {
      kSFKNotC,
      kSFKHeader,
      kSFKSource,
      kSFKLinkdef
   };

   TModuleGenerator(clang::CompilerInstance* CI,
                    const char* shLibFileName,
                    const std::string &currentDirectory);

   // FIXME: remove once PCH is gone.
   bool IsPCH() const { return fIsPCH; }
   void ParseArgs(const std::vector<std::string>& args);

   void WritePPCode(std::ostream& out) const;
   void WriteHeaderArray(std::ostream& out) const {
      // Write "header1.h",\n"header2.h",\n0\n
      WriteStringVec(fHeaders);
   }
   void WriteIncludeArray(std::ostream& out) const {
      // Write "./include",\n"/usr/include",\n0\n
      WriteStringVec(fCompI);
   }
   void WriteDefinesArray(std::ostream& out) const {
      // Write "DEFINED",\n"NAME=\"VALUE\"",\n0\n
      WriteStringPairVec(fCompD);
   }
   void WriteUndefinesArray(std::ostream& out) const {
      // Write "UNDEFINED",\n"NAME",\n0\n
      WriteStringVec(fCompD);
   }
   void WriteUndefinesArray(std::ostream& out);
   const std::string& GetDictionaryName() const { return fDictionaryName; }
   const std::string& GetModuleFileName() const { return fModuleFileName; }
   const std::string& GetModuleDirName() const { return fModuleDirName; }
   const std::string& GetUmbrellaName() const { return fUmbrellaName; }
   const std::string& GetContentName() const { return fContentName; }

private:
   typedef std::vector<std::pair<std::string, std::string> > StringPairVec_t;

   ESourceFileKind GetSourceFileKind(const char* filename) const;
   void WriteStringVec(const std::vector<std::string>& vec,
                       std::ostream& out) const;
   void WriteStringPairVec(const StringPairVec_t& vecP,
                           std::ostream& out) const;

   clang::CompilerInstance* fCI;
   const char* fShLibFileName;
   const std::string &fCurrentDirectory;
   bool fIsPCH;

   std::string fDictionaryName; // Name of the dictionary, e.g. "Base"
   std::string fModuleFileName; // PCM file name
   std::string fModuleDirName; // PCM output directory
   std::string fUmbrellaName; // name of umbrella header in PCM
   std::string fContentName; // name of content description header in PCM

   std::vector<std::string> fHeaders; // exported headers in PCM
   std::vector<std::string> fCompI; // -I; needed only for ACLiC without PCMs

   StringPairVec_t fCompD; // -Dfirst=second
   std::vector<std::string> fCompU; // -Ufirst
};

} // namespace ROOT

#endif // ROOT_TModuleGenerator
