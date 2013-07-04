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

#include "clang/Lex/HeaderSearch.h"
#include "clang/Support/SourceManager.h"
#include "llvm/Support/PathV2.h"

using namespace ROOT;
using namespace clang;

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
      clang::Preprocessor& PP = CI->getPreprocessor();
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

   static const size_t lenLinkdefdot = 8;
   if (ret == kSFKHeader && len - lenExt >= lenLinkdefdot) {
      if ((strstr(filename,"LinkDef") || strstr(filename,"Linkdef") ||
           strstr(filename,"linkdef")) && strstr(filename,".h")) {
         ret = kSFKLinkdef;
      }
   }
   return ret;
}


TModuleGenerator::TModuleGenerator(CompilerInstance* CI,
                                   const char* shLibFileName,
                                   const std::string &currentDirectory):
   fCI(CI),
   fShLibFileName(shLibFileName),
   fCurrentDirectory(currentDirectory),
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

   fModuleFileName = dictDir + ROOT::TMetaUtils::GetModuleFileName(dictname.c_str());
   // .pcm -> .pch
   if (isPCH) moduleFile[moduleFile.length() - 1] = 'h';

   fUmbrellaName = fDictionaryName + "_dictUmbrella.h";
   fContentName = fDictionaryName + "_dictContent.h";
}

static
std::pair<std::string, std::string> SplitPPDefine(const std::string& in) {
   std::string::size_type posEq = in.find('=');
   if (posEq == std::string::npos)
      return std::make_pair(in, "");
   return std::pair<std::string, std::string>
      (in.substr(0, posEq - 1), in.substr(posEq + 1, std::string::npos));
}


void TModuleGenerator::ParseArgs(const std::vector<std::string>& args)
{
   // Parse -I -D -U headers.h SomethingLinkdef.h.
   for (size_t iPcmArg = 1 /*skip argv0*/, nPcmArg = args.size();
        iPcmArg < nPcmArg; ++iPcmArg) {
      ESourceFileKind sfk = GetSourceFileKind(CI, args[iPcmArg].c_str());
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
            fCompU.push_back(SplitPPDefine(args[iPcmArg].c_str() + 2));
            break;
         }
      }
   }
}

void TModuleGenerator::WritePPDefines(std::ostream& out)
{
   // Write
   // #ifndef FOO
   // # define FOO=bar
   // #endif
   for (StringPairVec_t::const_iterator i = fCompD.begin(),
           e = fCompD.end(); i != e; ++i) {
      out << "#ifndef " << i->first << "\n"
         "  #define " << i->first;
      if (i->second.empty()) {
         out << i->second;
      }
      out << "\n"
         "#endif\n";
   }
   out << std::endl()
}

void TModuleGenerator::WritePPUndefines(std::ostream& out)
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
   out << std::endl()
}

void TModuleGenerator::WritePPIncludes(std::ostream& out)
{
   // Write
   // #include "header1.h"
   // #include "header2.h"
   for (std::vector<std::string>::const_iterator i = fHeaders.begin(),
           e = fHeaders.end(); i != e; ++i) {
      out << "#include " << *i << "\n";
   }
   out << std::endl()
}

void TModuleGenerator::WriteAllSeenHeadersArray(std::ostream& out) const
{
   for (clang::fileinfo_iterator i = fSrcMgr->fileinfo_begin(),
           e = fSrcMgr->fileinfo_end(); i != e; ++i) {
   }
}

void TModuleGenerator::WriteStringVec(const std::vector<std::string>& vec,
                                      std::ostream& out) const
{
   for (std::vector<std::string>::const_iterator i = vec.begin(),
           e = vec.end(); i != e; ++i) {
      out << "\"" << *i << "\",\n";
   }
   out << "0" << std::endl();

}

void TModuleGenerator::WriteStringPairVec(const StringPairVec_t& vec,
                                          std::ostream& out) const
{
   for (StringPairVec_t::const_iterator i = vec.begin(),
           e = vec.end(); i != e; ++i) {
      out << "\"" << i->first;
      if (!i->second.empty()) {
         // Need to escape the embedded quotes.
         for (const char *c = i->second.c_str(); *c != '\0'; ++c) {
            if ( *c == '"' ) {
               out << "\\\"";            
            } else {
               out << *c;
            }
         }         
         out << "=\\\"" << value << "\\\"";
      }
      out << "\",\n";
   }
   out << "0" << std::endl();
}

//______________________________________________________________________________
void TModuleGenerator::WriteRegistrationSource(std::ostream& out) const
{
   // Dictionary initialization code for loading the module
   out << "void TriggerDictionaryInitalization_"
       << modGen.GetDictionaryName() << "() {\n"
      "      static const char* headers[] = {\n";
   WriteHeaders(*dictSrcOut);
   out << 
      "      };\n"
      "      static const char* includePaths[] = {\n";
   WriteIncludePathArray(*dictSrcOut);

   out << 
      "      };\n"
      "      static const char* macroDefines[] = {\n";
   WriteDefinesArray(*dictSrcOut);
   out << 
      "      };\n"
      "      static const char* macroUndefines[] = {\n";
   WriteUndefinesArray(*dictSrcOut);

   out << 
      "      0 };\n"
      "      static bool sInitialized = false;\n"
      "      if (!sInitialized) {\n"
      "        TROOT::RegisterModule(\"" << dictname << "\",\n"
      "          headers, includePaths, macroDefines, macroUndefines,\n"
      "          TriggerDictionaryInitalization_" << dictname << ");\n"
      "        sInitialized = true;\n"
      "      }\n"
      "    }\n"
      "namespace {\n"
      "  static struct DictInit {\n"
      "    DictInit() {\n"
      "      TriggerDictionaryInitalization_" << dictname << "();\n"
      "    }\n"
      "  } __TheDictionaryInitializer;\n"
      "}" << std::endl;
}
