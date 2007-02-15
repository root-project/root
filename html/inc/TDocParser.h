// @(#)root/html:$Name:  $:$Id: TDocParser.h,v 1.3 2007/02/13 20:22:06 axel Exp $
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

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TList
#include "TList.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_Riosfwd
#include "Riosfwd.h"
#endif

class TBaseClass;
class TClass;
class TClassDocOutput;
class TDocOutput;
class THtml;

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
      kParseContextFlagMask = ~(BIT(4) - 1)

   };

   class TMethodWrapper: public TObject {
   public:
      virtual const TMethod* GetMethod() const = 0;
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
   TClass*        fCurrentClass;    // current class context of sources being parsed
   TString        fCurrentMethodTag;// name_idx of the currently parsed method
   Int_t          fDirectiveCount;  // index of directive for current method
   TString        fCurrentFile;     // current source / header file name
   std::map<std::string /*name*/, Int_t > fMethodCounts;     // current class's method names
   EDocContext    fDocContext;      // current context of parsed sources for documenting
   std::list<UInt_t> fParseContext; // current context of parsed sources
   Bool_t         fCheckForMethod;  // whether to check the current line for a method
   Bool_t         fFoundClassDescription; // whether we found the class description
   Bool_t         fLookForClassDescription; // whether we are looking for the class description
   Bool_t         fCommentAtBOL;    // at the beginning of the current line, fParseContext contained kComment
   TString        fClassDescrTag;   // tag for finding the class description
   TString        fSourceInfoTags[kNumSourceInfos]; // tags for source info elements (copyright, last changed, author)
   TList          fDirectiveHandlers;// handler for doc directives (TDocDirective objects)
   std::set<UInt_t> fExtraLinesWithAnchor; // lines that need an additional anchor
   TString        fSourceInfo[kNumSourceInfos];// author, last changed, ...
   TList          fMethods[3];      // methods as TMethodWrapper objects (by access)
   TList          fDataMembers[6];  // data members (by access, plus enums)

   static std::set<std::string>  fgKeywords; // C++ keywords

   void           AddClassMethodsRecursively(TBaseClass* bc);
   void           AddClassDataMembersRecursively(TBaseClass* bc);
   void           AnchorFromLine(TString& anchor);
   EParseContext  Context() const { return (EParseContext)(fParseContext.back() & kParseContextMask); }
   virtual void   ExpandCPPLine(TString& line, Ssiz_t& pos);
   virtual Bool_t HandleDirective(TString& keyword, Ssiz_t& pos, 
      TString& word, Ssiz_t& copiedToCommentUpTo);
   virtual void   InitKeywords() const;
   virtual TClass* IsDirective(const TString& line, Ssiz_t pos, const TString& word, Bool_t& begin) const;
   TMethod*       LocateMethodInCurrentLine(Ssiz_t& posMethodName, TString& ret, 
      TString& name, TString& params, std::ostream &srcOut, TString &anchor, 
      std::ifstream& sourcefile, Bool_t allowPureVirtual);
   void           LocateMethodsInSource(std::ostream& out);
   void           LocateMethodsInHeaderInline(std::ostream& out);
   void           LocateMethodsInHeaderClassDecl(std::ostream& out);
   void           LocateMethods(std::ostream& out, const char* filename,
                                Bool_t lookForSourceInfo = kTRUE, 
                                Bool_t useDocxxStyle = kFALSE, 
                                Bool_t lookForClassDescr = kTRUE,
                                Bool_t allowPureVirtual = kFALSE,
                                const char* methodPattern = 0, 
                                const char* sourceExt = 0);
   virtual Bool_t ProcessComment();
   void           RemoveCommentContext(Bool_t cxxcomment);
   void   WriteMethod(std::ostream& out, TString& ret, 
                      TString& name, TString& params,
                      const char* file, TString& anchor,
                      TString& codeOneLiner);
   void  WriteSourceLine(std::ostream& out);

public:
   TDocParser(TClassDocOutput& docOutput, TClass* cl);
   TDocParser(TDocOutput& docOutput);
   virtual       ~TDocParser();

   void          Convert(std::ostream& out, std::istream& in, const char* relpath);
   void          DecrementMethodCount(const char* name);
   virtual void  DecorateKeywords(std::ostream& out, const char* text);
   virtual void  DecorateKeywords(TString& text);
   const TList*  GetMethods(EAccess access) const { return &fMethods[access]; }
   TClass*       GetCurrentClass() const { return fCurrentClass; }
   TDocOutput*   GetDocOutput() const { return fDocOutput; }
   const TList*  GetDataMembers(EAccess access) const { return &fDataMembers[access]; }
   const TList*  GetEnums(EAccess access) const { return &fDataMembers[access+3]; }
   const char*   GetSourceInfo(ESourceInfo type) const { return fSourceInfo[type]; }

   UInt_t        InContext(Int_t context) const;
   static Bool_t IsName(UChar_t c);
   static Bool_t IsWord(UChar_t c);

   virtual void  Parse(std::ostream& out);
   static Bool_t Strip(TString& s);

   ClassDef(TDocParser,0); // parser for reference documentation
};

#endif // ROOT_TDocParser
