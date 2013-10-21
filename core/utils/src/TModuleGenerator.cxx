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

#include <map>

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
void TModuleGenerator::ConvertToCppString(std::string& text) const{
   // Convert input string to cpp code
   // FIXME: Optimisations for size could be put in
   // * Remove empty lines
   // * Remove comments
   
   typedef std::vector<std::pair<std::string,std::string> > strPairs;
   
   // Will be replaced by an initialiser list
   strPairs fromToPatterns;
   fromToPatterns.reserve(4);
   // \r -> "" (carriage return, empty char )
   fromToPatterns.push_back(std::make_pair("\r",""));   
   // \ -> \\'
   fromToPatterns.push_back(std::make_pair("\\","\\\\"));
   // " -> \"
   fromToPatterns.push_back(std::make_pair("\"","\\\""));   
   // \n -> \\n"\n" (new line, ",new line, ")
   fromToPatterns.push_back(std::make_pair("\n","\\n\"\n\""));
   
   for (strPairs::iterator fromToIter=fromToPatterns.begin();
        fromToIter!=fromToPatterns.end();fromToIter++){
      size_t start_pos = 0;
      const std::string& from = fromToIter->first;
      const std::string& to = fromToIter->second;
      while((start_pos = text.find(from, start_pos)) != std::string::npos) {
         text.replace(start_pos, from.length(), to);
         start_pos += to.length();
      }
   }
   size_t textSize (text.length());
   if (textSize>=2 && text[textSize-1] == '"'&& text[textSize-2] == '\n'){
      text.erase(textSize-1);
      textSize-=1;
   }
   if (textSize>=1 && text[textSize-1] != '\"' && text[textSize-1] != '\n'){
      text+='\"';
   }
      

   text="\""+text;
   
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
void TModuleGenerator::WriteRegistrationSource(std::ostream& out, bool inlineHeaders) const
{  

   std::string payloadCode;
   
   // Add defines and undefines to the payloadCode
   std::ostringstream definesAndUndefines;
   WritePPDefines(definesAndUndefines);
   WritePPUndefines(definesAndUndefines);
   payloadCode += definesAndUndefines.str();
   
   // If necessary, inline the first header
   if (inlineHeaders){
      for (std::vector<std::string>::const_iterator hdrNameIt=fHeaders.begin();
           hdrNameIt!=fHeaders.end();hdrNameIt++){
         std::ifstream headerFile(hdrNameIt->c_str());
         const std::string headerFileAsStr((std::istreambuf_iterator<char>(headerFile)),
                                            std::istreambuf_iterator<char>());
         payloadCode += headerFileAsStr;
      }
   }

   // Make it usable as string
   ConvertToCppString(payloadCode);
   
   // Dictionary initialization code for loading the module
   out << "void TriggerDictionaryInitalization_"
       << GetDictionaryName() << "() {\n"
      "      static const char* headers[] = {\n";
   if ( inlineHeaders ){
      out << 0 ;
   } else {
      WriteHeaderArray(out);
   };
   out << "      };\n"
      "      static const char* allHeaders[] = {\n";
   WriteAllSeenHeadersArray(out) << 
      "      };\n"
      "      static const char* includePaths[] = {\n";
   WriteIncludePathArray(out) << 
      "      };\n"
      "      static const char* payloadCode = "<< payloadCode << ";\n"
      "      static bool sInitialized = false;\n"
      "      if (!sInitialized) {\n"
      "        TROOT::RegisterModule(\"" << GetDictionaryName() << "\",\n"
      "          headers, allHeaders, includePaths, payloadCode,\n"
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
/*
   out << "const char* arrDefines[] = {\n";
   WriteDefinesArray(out) << "};\n";

   out << "const char* arrUndefines[] = {\n";
   WriteUndefinesArray(out) << "};\n";*/

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
