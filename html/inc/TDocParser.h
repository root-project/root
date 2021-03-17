// @(#)root/html:$Id$
// Author: Axel Naumann 2007-01-09

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDocParser
#define ROOT_TDocParser

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TDocParser                                                             //
//                                                                        //
// Parses documentation in source files                                   //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#include <list>
#include <set>
#include <map>

#include "TObject.h"
#include "THashList.h"
#include "TString.h"

class TBaseClass;
class TClass;
class TClassDocOutput;
class TDocOutput;
class THtml;

class TDocMethodWrapper: public TObject {
public:
   virtual TMethod* GetMethod() const = 0;
   virtual Int_t GetOverloadIdx() const = 0;
   enum { kDocumented = 14 };
};

class TDocParser: public TObject {
protected:
   enum EDocContext {
      kIgnore,
      kDocFunc,
      kDocClass,
      kNumDocContexts
   };

public:
   enum ESourceInfo {
      kInfoLastUpdate,
      kInfoAuthor,
      kInfoCopyright,
      kInfoLastChanged,
      kInfoLastGenerated,
      kNumSourceInfos
   };
   enum EAccess {
      kPrivate,
      kProtected,
      kPublic
   };
   enum EParseContext {
      kNoContext,
      kCode,
      kComment,
      kDirective,
      kString,
      kKeyword,
      kCPP,
      kVerbatim,
      kNumParseContexts,
      kParseContextMask = BIT(4) - 1
   };
   enum EParseContextFlag {
      kCXXComment = BIT(4), // kComment is a C++ comment, or macro/html/latex content is surrounded by /* */
      kParseContextFlagMask = (UInt_t)(~(BIT(4) - 1))

   };

protected:
   THtml*         fHtml;            // THtml object using us
   TDocOutput*    fDocOutput;       // TDocOutput invoking us
   UInt_t         fLineNo;          // current line number
   TString        fLineRaw;         // current line
   TString        fLineStripped;    // current line without surrounding spaces
   TString        fLineComment;         // current line with links and directives for doc
   TString        fLineSource;      // current line with links
   TString        fComment;         // current comment
   TString        fFirstClassDoc;   // first class-doc found - per file, taken if fLastClassDoc is empty
   TString        fLastClassDoc;    // last class-doc found - becomes class doc at ClassImp or first method
   TClass*        fCurrentClass;    // current class context of sources being parsed
   TClass*        fRecentClass;     // recently seen class context of sources being parsed, e.g. for Convert()
   TString        fCurrentModule;   // current module context of sources being parsed
   TString        fCurrentMethodTag;// name_idx of the currently parsed method
   Int_t          fDirectiveCount;  // index of directive for current method
   Long_t         fLineNumber;      // source line number
   TString        fCurrentFile;     // current source / header file name
   std::map<std::string /*name*/, Int_t > fMethodCounts;     // number of undocumented overloads
   EDocContext    fDocContext;      // current context of parsed sources for documenting
   std::list<UInt_t> fParseContext; // current context of parsed sources
   Bool_t         fCheckForMethod;  // whether to check the current line for a method
   enum {
      kClassDoc_Uninitialized,
      kClassDoc_LookingNothingFound,
      kClassDoc_LookingHaveSomething,
      kClassDoc_Written,
      kClassDoc_Ignore,
      kClassDoc_NumStates
   }              fClassDocState; // whether we found the class description
   Bool_t         fCommentAtBOL;    // at the beginning of the current line, fParseContext contained kComment
   TString        fClassDescrTag;   // tag for finding the class description
   TString        fSourceInfoTags[kNumSourceInfos]; // tags for source info elements (copyright, last changed, author)
   TList          fDirectiveHandlers;// handler for doc directives (TDocDirective objects)
   Bool_t         fAllowDirectives;  // whether directives are to be interpreted
   std::set<UInt_t> fExtraLinesWithAnchor; // lines that need an additional anchor
   TString        fSourceInfo[kNumSourceInfos];// author, last changed, ...
   THashList      fMethods[3];      // methods as TMethodWrapper objects (by access)
   TList          fDataMembers[6];  // data members (by access, plus enums)

   static std::set<std::string>  fgKeywords; // C++ keywords

   void           AddClassMethodsRecursively(TBaseClass* bc);
   void           AddClassDataMembersRecursively(TBaseClass* bc);
   EParseContext  Context() const { return fParseContext.empty() ? kComment : (EParseContext)(fParseContext.back() & kParseContextMask); }
   virtual void   ExpandCPPLine(TString& line, Ssiz_t& pos);
   virtual Bool_t HandleDirective(TString& keyword, Ssiz_t& pos,
      TString& word, Ssiz_t& copiedToCommentUpTo);
   void           InitKeywords() const;
   virtual TClass* IsDirective(const TString& line, Ssiz_t pos, const TString& word, Bool_t& begin) const;
   TMethod*       LocateMethodInCurrentLine(Ssiz_t& posMethodName, TString& ret,
      TString& name, TString& params, Bool_t& isconst,
      std::ostream &srcOut, TString &anchor,
      std::ifstream& sourcefile, Bool_t allowPureVirtual);
   void           LocateMethodsInSource(std::ostream& out);
   void           LocateMethodsInHeaderInline(std::ostream& out);
   void           LocateMethodsInHeaderClassDecl(std::ostream& out);
   void           LocateMethods(std::ostream& out, const char* filename,
                                Bool_t lookForSourceInfo = kTRUE,
                                Bool_t useDocxxStyle = kFALSE,
                                Bool_t allowPureVirtual = kFALSE,
                                const char* methodPattern = 0,
                                const char* sourceExt = 0);
   virtual Bool_t ProcessComment();
   void           RemoveCommentContext(Bool_t cxxcomment);
   void           WriteClassDoc(std::ostream& out, Bool_t first = kTRUE);
   void           WriteMethod(std::ostream& out, TString& ret,
                              TString& name, TString& params,
                              Bool_t isconst,
                              const char* file, TString& anchor,
                              TString& codeOneLiner);
   void           WriteSourceLine(std::ostream& out);

public:
   TDocParser(TClassDocOutput& docOutput, TClass* cl);
   TDocParser(TDocOutput& docOutput);
   virtual       ~TDocParser();

   static void   AnchorFromLine(const TString& line, TString& anchor);
   void          Convert(std::ostream& out, std::istream& in, const char* relpath,
                         Bool_t isCode, Bool_t interpretDirectives);
   void          DecrementMethodCount(const char* name);
   virtual void  DecorateKeywords(std::ostream& out, const char* text);
   virtual void  DecorateKeywords(TString& text);
   virtual void  DeleteDirectiveOutput() const;
   const TList*  GetMethods(EAccess access) const { return &fMethods[access]; }
   TClass*       GetCurrentClass() const { return fCurrentClass; }
   void          GetCurrentModule(TString& out_module) const;
   TDocOutput*   GetDocOutput() const { return fDocOutput; }
   Long_t        GetLineNumber() const { return fLineNumber; }
   const TList*  GetDataMembers(EAccess access) const { return &fDataMembers[access]; }
   const TList*  GetEnums(EAccess access) const { return &fDataMembers[access+3]; }
   const char*   GetSourceInfo(ESourceInfo type) const { return fSourceInfo[type]; }
   void          SetCurrentModule(const char* module) { fCurrentModule = module; }

   UInt_t        InContext(Int_t context) const;
   static Bool_t IsName(UChar_t c);
   static Bool_t IsWord(UChar_t c);

   virtual void  Parse(std::ostream& out);
   static Bool_t Strip(TString& s);

   ClassDef(TDocParser,0); // parser for reference documentation
};

#endif // ROOT_TDocParser
