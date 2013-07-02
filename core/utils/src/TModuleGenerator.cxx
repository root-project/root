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

void TModuleGenerator::writePPCode(std::ostream& out)
{
   // Write
   // #include "header1.h"
   // #include "header2.h"
   // #ifndef FOO
   // # define FOO=bar
   // #endif
      
}

void TModuleGenerator::WriteStringVec(const std::vector<std::string>& vec,
                                      std::ostream& out)
{
   for (std::vector<std::string>::const_iterator i = vec.begin(),
           e = vec.end(); i != e; ++i) {
      out << "\"" << *i << "\",\n";
   }
   out << "0\n";

}

void TModuleGenerator::WriteStringPairVec(const StringPairVec_t& vec,
                                          std::ostream& out)
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
   out << "0\n";
}

//______________________________________________________________________________
static int GenerateModule(clang::CompilerInstance* CI,
                          const char* libName, const std::vector<std::string>& args,
                          const std::string &currentDirectory)
{
   // Generate the clang module given the arguments.
   // Returns != 0 on error.

   ModuleGenerator modGen(CI, dictSrcFile, currentDirectory, *dictSrcOut);
   modGen.ParseArgs(args);

   std::string dictname = llvm::sys::path::stem(dictSrcFile);

   // Dictionary initialization code for loading the module
   (*dictSrcOut) << "void TriggerDictionaryInitalization_"
                 << modGen.fDictionaryName << "() {\n"
      "      static const char* headers[] = {\n";
   {
      for (size_t iH = 0, eH = headers.size(); iH < eH; ++iH) {
         (*dictSrcOut) << "             \"" << headers[iH] << "\"," << std::endl;
      }
   }

   (*dictSrcOut) << 
      "      0 };\n"
      "      static const char* includePaths[] = {\n";
   for (std::vector<const char*>::const_iterator
           iI = compI.begin(), iE = compI.end(); iI != iE; ++iI) {
      (*dictSrcOut) << "             \"" << *iI << "\"," << std::endl;
   }

   (*dictSrcOut) << 
      "      0 };\n"
      "      static const char* macroDefines[] = {\n";
   for (std::vector<const char*>::const_iterator
           iD = compD.begin(), iDE = compD.end(); iD != iDE; ++iD) {
      (*dictSrcOut) << "             \"";
      // Need to escape the embedded quotes.
      for(const char *c = *iD; *c != '\0'; ++c) {
         if ( *c == '"' ) {
            (*dictSrcOut) << "\\\"";            
         } else {
            (*dictSrcOut) << *c;
         }
      }
      (*dictSrcOut) << "\"," << std::endl;
   }

   (*dictSrcOut) << 
      "      0 };\n"
      "      static const char* macroUndefines[] = {\n";
   for (std::vector<const char*>::const_iterator
           iU = compU.begin(), iUE = compU.end(); iU != iUE; ++iU) {
      (*dictSrcOut) << "             \"" << *iU << "\"," << std::endl;
   }

   (*dictSrcOut) << 
      "      0 };\n"
      "      static bool sInitialized = false;\n"
      "      if (!sInitialized) {\n"
      "        TCling__RegisterModule(\"" << dictname << "\",\n"
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

   if (!isPCH) {
      CI->getPreprocessor().getHeaderSearchInfo().
         setModuleCachePath(dictDir.c_str());
   }
   std::string moduleFile = dictDir + ROOT::TMetaUtils::GetModuleFileName(dictname.c_str());
   // .pcm -> .pch
   if (isPCH) moduleFile[moduleFile.length() - 1] = 'h';

   clang::Module* module = 0;
   if (!isPCH) {
      std::vector<const char*> headersCStr;
      for (std::vector<std::string>::const_iterator
              iH = headers.begin(), eH = headers.end();
           iH != eH; ++iH) {
         headersCStr.push_back(iH->c_str());
      }
      headersCStr.push_back(0);
      module = ROOT::TMetaUtils::declareModuleMap(CI, moduleFile.c_str(), &headersCStr[0]);
   }

   // From PCHGenerator and friends:
   llvm::SmallVector<char, 128> Buffer;
   llvm::BitstreamWriter Stream(Buffer);
   clang::ASTWriter Writer(Stream);
   llvm::raw_ostream *OS
      = CI->createOutputFile(moduleFile, /*Binary=*/true,
                             /*RemoveFileOnSignal=*/false, /*InFile*/"",
                             /*Extension=*/"", /*useTemporary=*/false,
                             /*CreateMissingDirectories*/false);
   if (OS) {
      // Emit the PCH file
      CI->getFrontendOpts().RelocatablePCH = true;
      std::string ISysRoot("/DUMMY_SYSROOT/include/");
#ifdef ROOTBUILD
      ISysRoot = (currentDirectory + "/").c_str();
#endif
      Writer.WriteAST(CI->getSema(), moduleFile, module, ISysRoot.c_str());

      // Write the generated bitstream to "Out".
      OS->write((char *)&Buffer.front(), Buffer.size());

      // Make sure it hits disk now.
      OS->flush();
      bool deleteOutputFile =  CI->getDiagnostics().hasErrorOccurred();
      CI->clearOutputFiles(deleteOutputFile);
    
   }

   // Free up some memory, in case the process is kept alive.
   Buffer.clear();

   return 0;
}

