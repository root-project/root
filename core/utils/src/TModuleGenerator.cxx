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

#include "TModuleGenerator.h"

#include "TMetaUtils.h"
#include "RConfigure.h"
#include "RConfig.h"

#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Support/Path.h"

#ifndef R__WIN32
#include <unistd.h>
#endif

using namespace ROOT;
using namespace clang;

TModuleGenerator::TModuleGenerator(CompilerInstance* CI,
                                   const char* shLibFileName):
   fCI(CI),
   fShLibFileName(shLibFileName),
   fIsPCH(!strcmp(shLibFileName, "etc/allDict.cxx")),
   fDictionaryName(llvm::sys::path::stem(shLibFileName)),
   fModuleDirName(llvm::sys::path::parent_path(shLibFileName))
{
      
   // Need to resolve _where_ to create the pcm
   // We default in the lib subdirectory
   // otherwise we put it in the same directory as the dictionary file (for ACLiC)
   if (fModuleDirName.empty()) {
      fModuleDirName = "./";
   } else {
      fModuleDirName += "/";
   }

   fModuleFileName = fModuleDirName
      + ROOT::TMetaUtils::GetModuleFileName(fDictionaryName.c_str());
   // .pcm -> .pch
   if (IsPCH()) fModuleFileName[fModuleFileName.length() - 1] = 'h';

   fUmbrellaName = fModuleDirName + fDictionaryName + "_dictUmbrella.h";
   fContentName = fModuleDirName + fDictionaryName + "_dictContent.h";
}

TModuleGenerator::~TModuleGenerator() {
   unlink(fUmbrellaName.c_str());
   unlink(fContentName.c_str());
}

//______________________________________________________________________________
TModuleGenerator::ESourceFileKind
TModuleGenerator::GetSourceFileKind(const char* filename) const
{
   // Check whether the file's extension is compatible with C or C++.
   // Return whether source, header, Linkdef or nothing.
   if (filename[0] == '-') return kSFKNotC;

   const size_t len = strlen(filename);
   const char* ext = filename + len - 1;
   while (ext >= filename && *ext != '.') --ext;
   if (ext < filename || *ext != '.') {
      // This might still be a system header, let's double check
      // via the FileManager.
      clang::Preprocessor& PP = fCI->getPreprocessor();
      clang::HeaderSearch& HdrSearch = PP.getHeaderSearchInfo();
      const clang::DirectoryLookup* CurDir = 0;
      const clang::FileEntry* hdrFileEntry
         =  HdrSearch.LookupFile(filename, true /*isAngled*/, 0 /*FromDir*/,
                                 CurDir, 0 /*CurFileEnt*/, 0 /*SearchPath*/,
                                 0 /*RelativePath*/, 0 /*SuggestedModule*/);
      if (hdrFileEntry) {
         return kSFKHeader;
      }
      return kSFKNotC;
   }
   ++ext;
   const size_t lenExt = filename + len - ext;

   ESourceFileKind ret = kSFKNotC;
   switch (lenExt) {
   case 1: {
      const char last = toupper(filename[len - 1]);
      if (last == 'H') ret = kSFKHeader;
      else if (last == 'C') ret = kSFKSource;
      break;
   }
   case 2: {
      if (filename[len - 2] == 'h' && filename[len - 1] == 'h')
         ret = kSFKHeader;
      else if (filename[len - 2] == 'c' && filename[len - 1] == 'c')
         ret = kSFKSource;
      break;
   }
   case 3: {
      const char last = filename[len - 1];
      if ((last == 'x' || last == 'p')
          && filename[len - 2] == last) {
         if (filename[len - 3] == 'h') ret = kSFKHeader;
         else if (filename[len - 3] == 'c') ret = kSFKSource;
      }
   }
   } // switch extension length

   /*
   static const size_t lenLinkdefdot = 8;
   if (ret == kSFKHeader && len - lenExt >= lenLinkdefdot) {
      if ((strstr(filename,"LinkDef") || strstr(filename,"Linkdef") ||
           strstr(filename,"linkdef")) && strstr(filename,".h")) {
         ret = kSFKLinkdef;
      }
   }*/
   return ret;
}


static
std::pair<std::string, std::string> SplitPPDefine(const std::string& in) {
   std::string::size_type posEq = in.find('=');
   if (posEq == std::string::npos)
      return std::make_pair(in, "");
   return std::pair<std::string, std::string>
      (in.substr(0, posEq), in.substr(posEq + 1, std::string::npos));
}


void TModuleGenerator::ParseArgs(const std::vector<std::string>& args)
{
   // Parse -I -D -U headers.h SomethingLinkdef.h.
   for (size_t iPcmArg = 1 /*skip argv0*/, nPcmArg = args.size();
        iPcmArg < nPcmArg; ++iPcmArg) {
      ESourceFileKind sfk = GetSourceFileKind(args[iPcmArg].c_str());
      if (sfk == kSFKHeader || sfk == kSFKSource) {
         fHeaders.push_back(args[iPcmArg]);
      } else if (sfk == kSFKNotC && args[iPcmArg][0] == '-') {
         switch (args[iPcmArg][1]) {
         case 'I':
            if (args[iPcmArg] != "-I." &&  args[iPcmArg] != "-Iinclude") {
               fCompI.push_back(args[iPcmArg].c_str() + 2);
            }
            break;
         case 'D':
            if (args[iPcmArg] != "-DTRUE=1" && args[iPcmArg] != "-DFALSE=0"
                && args[iPcmArg] != "-DG__NOCINTDLL") {
               // keep -DROOT_Math_VectorUtil_Cint -DG__VECTOR_HAS_CLASS_ITERATOR?
               fCompD.push_back(SplitPPDefine(args[iPcmArg].c_str() + 2));
            }
            break;
         case 'U':
            fCompU.push_back(args[iPcmArg].c_str() + 2);
            break;
         }
      }
   }
}

std::ostream& TModuleGenerator::WritePPDefines(std::ostream& out) const
{
   // Write
   // #ifndef FOO
   // # define FOO=bar
   // #endif
   for (StringPairVec_t::const_iterator i = fCompD.begin(),
           e = fCompD.end(); i != e; ++i) {
      out << "#ifndef " << i->first << "\n"
         "  #define " << i->first;
      if (!i->second.empty()) {
         out << " " << i->second;
      }
      out << "\n"
         "#endif\n";
   }
   out << std::endl;
   return out;
}

std::ostream& TModuleGenerator::WritePPUndefines(std::ostream& out) const
{
   // Write
   // #ifdef FOO
   // # undef FOO
   // #endif
   for (std::vector<std::string>::const_iterator i = fCompU.begin(),
           e = fCompU.end(); i != e; ++i) {
      out << "#ifdef " << *i << "\n"
         "  #undef " << *i << "\n"
         "#endif\n";
   }
   out << std::endl;
   return out;
}

std::ostream& TModuleGenerator::WritePPIncludes(std::ostream& out) const
{
   // Write
   // #include "header1.h"
   // #include "header2.h"
   for (std::vector<std::string>::const_iterator i = fHeaders.begin(),
           e = fHeaders.end(); i != e; ++i) {
      out << "#include \"" << *i << "\"\n";
   }
   out << std::endl;
   return out;
}

std::ostream& TModuleGenerator::WriteAllSeenHeadersArray(std::ostream& out) const
{
   SourceManager& srcMgr = fCI->getSourceManager();
   for (SourceManager::fileinfo_iterator i = srcMgr.fileinfo_begin(),
           e = srcMgr.fileinfo_end(); i != e; ++i) {
      const FileEntry* fileEntry = i->first;
      out << "\"" << fileEntry->getName() << "\",\n";
   }
   out << "0" << std::endl;
   return out;
}

std::ostream& TModuleGenerator::WriteStringVec(const std::vector<std::string>& vec,
                                      std::ostream& out) const
{
   for (std::vector<std::string>::const_iterator i = vec.begin(),
           e = vec.end(); i != e; ++i) {
      out << "\"" << *i << "\",\n";
   }
   out << "0" << std::endl;
   return out;
}

std::ostream& TModuleGenerator::WriteStringPairVec(const StringPairVec_t& vec,
                                          std::ostream& out) const
{
   for (StringPairVec_t::const_iterator i = vec.begin(),
           e = vec.end(); i != e; ++i) {
      out << "\"" << i->first;
      if (!i->second.empty()) {
         out << "=";
         // Need to escape the embedded quotes.
         for (const char *c = i->second.c_str(); *c != '\0'; ++c) {
            if ( *c == '"' ) {
               out << "\\\"";
            } else {
               out << *c;
            }
         }
      }
      out << "\",\n";
   }
   out << "0" << std::endl;
   return out;
}

//______________________________________________________________________________
void TModuleGenerator::WriteRegistrationSource(std::ostream& out, bool inlineHeader) const
{  
   
   // Dictionary initialization code for loading the module
   out << "void TriggerDictionaryInitalization_"
       << GetDictionaryName() << "() {\n"
      "      static const char* headers[] = {\n";
   WriteHeaderArray(out) <<
      "      };\n"
      "      static const char* allHeaders[] = {\n";
   WriteAllSeenHeadersArray(out) << 
      "      };\n"
      "      static const char* includePaths[] = {\n";
   WriteIncludePathArray(out) << 
      "      };\n"
      "      static const char* macroDefines[] = {\n";
   WriteDefinesArray(out) << 
      "      };\n"
      "      static const char* macroUndefines[] = {\n";
   WriteUndefinesArray(out) << 
      "      };\n"
      "      static bool sInitialized = false;\n"
      "      if (!sInitialized) {\n"
      "        TROOT::RegisterModule(\"" << GetDictionaryName() << "\",\n"
      "          headers, allHeaders, includePaths, macroDefines, macroUndefines,\n"
      "          TriggerDictionaryInitalization_" << GetDictionaryName() << ");\n"
      "        sInitialized = true;\n"
      "      }\n"
      "    }\n"
      "namespace {\n"
      "  static struct DictInit {\n"
      "    DictInit() {\n"
      "      TriggerDictionaryInitalization_" << GetDictionaryName() << "();\n"
      "    }\n"
      "  } __TheDictionaryInitializer;\n"
      "}" << std::endl;
}
#include <iostream>
//______________________________________________________________________________
void TModuleGenerator::WriteSingleHeaderRegistrationSource(std::ostream& out) const
{

   // Produce the dictionary code and inline the header
   // Fixme: extract the manipulation of text.
   
   const std::string& headerName = fHeaders[0];
   
   // Build the defines and undefines
   std::ostringstream definesAndUndefines;
   definesAndUndefines << "// Defines and undefines\n";
   WritePPDefines(definesAndUndefines);
   WritePPUndefines(definesAndUndefines);
   definesAndUndefines << "// Input Header Content\n";
   
   // Read in the header, replace the " with \" and inline
   std::ifstream headerFile(headerName.c_str());  
   std::string headerFileContent;
   headerFileContent.reserve(3000);
   headerFileContent+="\"";
   char c;
   char cminus1 = '@';
   bool addChar=true;
   while (EOF != (c = headerFile.get())){

      if (cminus1 == '\n' && c == '\n') {
         addChar=false;
      }

      // If char is a ", just put an escaped "
      if (c == '"'){
         headerFileContent+="\\\"";
         addChar=false;
      }
      // If a char is a \, just put an escaped
      if (c == '\\'){
         headerFileContent+="\\\\";
         addChar=false;
      }

      // if char is a \n just add a " and goto next line and add a "
      if (c=='\n' && cminus1 != '\n'){
         headerFileContent+="\\n\"\n\"";
         addChar=false;
      }

      cminus1=c;
      if (addChar){
         headerFileContent+=c;
      }

      addChar=true;
   }
   // To fix headers not having a trailing \n
   if (cminus1!='\n') {
      headerFileContent+="\"";
   }
   else if (!headerFileContent.empty()){
      headerFileContent.erase(headerFileContent.size()-1);
   }

   // Dictionary initialization code for loading the module
   out << "void TriggerDictionaryInitalization_"
   << GetDictionaryName() << "() {\n"
   "      static const char* header = "<< headerFileContent << ";\n"
   "      static const char* includePaths[] = {\n";
   WriteIncludePathArray(out) <<
   "      };\n"
   "      static bool sInitialized = false;\n"
   "      if (!sInitialized) {\n"
   "        TROOT::RegisterModule(\"" << GetDictionaryName() << "\",\n"
   "          header, includePaths);\n"
   "        sInitialized = true;\n"
   "      }\n"
   "    }\n"
   "namespace {\n"
   "  static struct DictInit {\n"
   "    DictInit() {\n"
   "      TriggerDictionaryInitalization_" << GetDictionaryName() << "();\n"
   "    }\n"
   "  } __TheDictionaryInitializer;\n"
   "}" << std::endl;
   }
   
//______________________________________________________________________________
void TModuleGenerator::WriteContentHeader(std::ostream& out) const
{
   // Write a header file describing the content of this module
   // through a series of variables inside the namespace
   // ROOT::Dict::[DictionaryName]. Each variable is an array of string
   // literals, with a const char* of 0 being the last element, e.g.
   // ROOT::Dict::_DictName::arrIncludes[] = { "A.h", "B.h", 0 };

   out << "namespace ROOT { namespace Dict { namespace _"
       << GetDictionaryName() << "{\n";

   out << "const char* arrIncludes[] = {\n";
   WriteHeaderArray(out) << "};\n";

   out << "const char* arrAllHeaders[] = {\n";
   WriteAllSeenHeadersArray(out) << "};\n";

   out << "const char* arrIncludePaths[] = {\n";
   WriteIncludePathArray(out) << "};\n";

   out << "const char* arrDefines[] = {\n";
   WriteDefinesArray(out) << "};\n";

   out << "const char* arrUndefines[] = {\n";
   WriteUndefinesArray(out) << "};\n";

   out << "} } }" << std::endl;
}

//______________________________________________________________________________
void TModuleGenerator::WriteUmbrellaHeader(std::ostream& out) const
{
   // Write a header file pulling in the content of this module
   // through a series of #defined, #undefs and #includes.
   // The sequence corrsponds to a rootcling invocation with
   //   -c -DFOO -UBAR header.h
   // I.e. defines, undefines and finally includes.

   WritePPDefines(out);
   WritePPUndefines(out);
   WritePPIncludes(out);
}
