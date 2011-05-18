// @(#)root/html:$Id$
// Author: Axel Naumann 2007-01-09

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDocOutput
#define ROOT_TDocOutput


////////////////////////////////////////////////////////////////////////////
//                                                                        //
// ROOT_TDocOutput                                                        //
//                                                                        //
// Generates documentation output using XHTML 1.0 transitional            //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_Riosfwd
#include "Riosfwd.h"
#endif
#ifndef ROOT_TDocParser
#include "TDocParser.h"
#endif

class TClass;
class TDataMember;
class TDataType;
class TGClient;
class THtml;
class TModuleDocInfo;
class TString;
class TSubString;
class TVirtualPad;

class TDocOutput: public TObject {
protected:
   enum EGraphvizTool {
      kDot,
      kNeato,
      kFdp,
      kCirco
   };

   THtml*         fHtml; // THtml object we belong to

   int            CaseInsensitiveSort(const void *name1, const void *name2);
   void           AddLink(TSubString& str, TString& link, const char* comment);
   void           ProcessDocInDir(std::ostream& out, const char* indir, const char* outdir, const char* linkdir);
   Bool_t         RunDot(const char* filename, std::ostream* outMap = 0, EGraphvizTool gvwhat = kDot);
   void           WriteHtmlHeader(std::ostream& out, const char *titleNoSpecial,
                                  const char* dir /*=""*/, TClass *cls /*=0*/,
                                  const char* header);
   void           WriteHtmlFooter(std::ostream& out, const char *dir,
                                  const char *lastUpdate, const char *author,
                                  const char *copyright, const char* footer);
   virtual void   WriteSearch(std::ostream& out);
   void           WriteLocation(std::ostream& out, TModuleDocInfo* module, const char* classname = 0);
   void           WriteModuleLinks(std::ostream& out);
   void           WriteModuleLinks(std::ostream& out, TModuleDocInfo* super);
   void           WriteTopLinks(std::ostream& out, TModuleDocInfo* module, const char* classname = 0, Bool_t withLocation = kTRUE);

public:
   enum EFileType { kSource, kInclude, kTree, kDoc };

   TDocOutput(THtml& html);
   virtual ~TDocOutput();

   virtual void   AdjustSourcePath(TString& line, const char* relpath = "../");
   void           Convert(std::istream& in, const char* infilename,
                          const char* outfilename, const char *title,
                          const char *relpath = "../",
                          Int_t includeOutput = 0,
                          const char* context = "",
                          TGClient* gclient = 0);
   Bool_t         CopyHtmlFile(const char *sourceName, const char *destName="");

   virtual void   CreateClassIndex();
   virtual void   CreateModuleIndex();
   virtual void   CreateProductIndex();
   virtual void   CreateTypeIndex();
   virtual void   CreateClassTypeDefs();
   virtual void   CreateHierarchy();

   virtual void   DecorateEntityBegin(TString& str, Ssiz_t& pos, TDocParser::EParseContext type);
   virtual void   DecorateEntityEnd(TString& str, Ssiz_t& pos, TDocParser::EParseContext type);
   virtual void   FixupAuthorSourceInfo(TString& authors);
   const char*    GetExtension() const { return ".html"; }
   THtml*         GetHtml() { return fHtml; }
   virtual Bool_t IsModified(TClass *classPtr, EFileType type);
   virtual void   NameSpace2FileName(TString &name);

   virtual void   ReferenceEntity(TSubString& str, TClass* entity, const char* comment = 0);
   virtual void   ReferenceEntity(TSubString& str, TDataMember* entity, const char* comment = 0);
   virtual void   ReferenceEntity(TSubString& str, TDataType* entity, const char* comment = 0);
   virtual void   ReferenceEntity(TSubString& str, TMethod* entity, const char* comment = 0);
   virtual Bool_t ReferenceIsRelative(const char* reference) const;

   virtual const char* ReplaceSpecialChars(char c);
   void           ReplaceSpecialChars(std::ostream &out, const char *string);
   void           ReplaceSpecialChars(TString& text);
   void           ReplaceSpecialChars(TString& text, Ssiz_t &pos);

   virtual void   WriteHtmlHeader(std::ostream &out, const char *title, const char* dir="", TClass *cls=0);
   virtual void   WriteHtmlFooter(std::ostream &out, const char *dir="", const char *lastUpdate="",
                                  const char *author="", const char *copyright="");
   void           WriteLineNumbers(std::ostream& out, Long_t nLines, const TString& infileBase) const;

   ClassDef(TDocOutput, 0); // generates documentation web pages
};

#endif // ROOT_TDocOutput
