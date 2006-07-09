// @(#)root/html:$Name:  $:$Id: THtml.h,v 1.18 2006/07/08 19:47:50 brun Exp $
// Author: Nenad Buncic   18/10/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_THtml
#define ROOT_THtml


////////////////////////////////////////////////////////////////////////////
//                                                                        //
// THtml                                                                  //
//                                                                        //
// Html makes a documentation for all ROOT classes                        //
// using Hypertext Markup Language 2.0                                    //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#ifndef __CINT__
#include "TCint.h"
#include "Api.h"
#undef G__FPROTO_H
#include "fproto.h"
#endif
//#include "Type.h"
//#include "G__ci.h"
//#include "Typedf.h"
//#include "Class.h"

#ifndef ROOT_TROOT
#include "TROOT.h"
#endif
#ifndef ROOT_TDictionary
#include "TDictionary.h"
#endif
#ifndef ROOT_TMap
#include "TMap.h"
#endif
#ifndef ROOT_THashList
#include "THashList.h"
#endif
#include <list>
#include <map>
#include <set>
#include <vector>

class TClass;
class TVirtualPad;
class TPaveText;

class THtml : public TObject {
protected:
   enum ETraverse {
      kUp, kDown, kBoth        // direction to traverse class tree in ClassHtmlTree()
   };
   enum EParseContext {
      kCode,
      kCComment,
      kBeginEndHtml,
      kBeginEndHtmlInCComment,
      kString,
      kNumParseContexts
   };
   enum EDocContext {
      kIgnore,
      kDocFunc,
      kDocClass,
      kNumDocContexts
   };
   enum ESourceInfo {
      kInfoLastUpdate,
      kInfoAuthor,
      kInfoCopyright,
      kNumSourceInfos
   };

   typedef std::map<std::string /*method name*/, Int_t > MethodNames_t;

   TString        fXwho;            // by default http://xwho.cern.ch/WHO/people?
   TString        fSourcePrefix;    // prefix to relative source path
   TString        fSourceDir;       // source path
   TString        fOutputDir;       // output directory
   TString        fLine;            // current line
   TString        fLineExpanded;    // current line with links
   TClass        *fCurrentClass;    // current class context of sources being parsed
   MethodNames_t  fMethodNames;     // current class's method names
   EDocContext    fDocContext;      // current context of parsed sources for documenting
   EParseContext  fParseContext;    // current context of parsed sources
   TString        fSourceInfo[kNumSourceInfos];// author, last changed, ...
   TString        fCounter;         // counter string
   Bool_t         fEscFlag;         // Flag to mark the symbol must be written "as is"
   char           fEsc;             // The special symbol ("backslash" by default) to mark "the next symbol should not be converted"
   Int_t          fHierarchyLines;  // counter for no. lines in hierarchy
   Int_t          fNumberOfClasses; // Number of known classes
   const char   **fClassNames;      // Names of known classes
   Int_t          fNumberOfFileNames;// Number of names of files for known classes
   char         **fFileNames;       // Names of files for known classes
   std::list<std::string> fModules; // Names of modules
   std::map<TClass*,std::string> fGuessedDeclFileNames; // names of additional decl file names
   std::map<TClass*,std::string> fGuessedImplFileNames; // names of additional impl file names
   static std::set<std::string>  fKeywords; // C++ keywords

   virtual void BeautifyLine(std::ostream &srcOut, TString* anchor = 0);
   void    Class2Html(Bool_t force=kFALSE);
   void    ClassDescription(ofstream &out);
   void    ClassHtmlTree(ofstream &out, TClass *classPtr, ETraverse dir=kBoth, int depth=1);
   void    ClassTree(TVirtualPad *canvas, TClass *classPtr, Bool_t force=kFALSE);
   Bool_t  CopyHtmlFile(const char *sourceName, const char *destName="");
   void    CreateIndex(const char **classNames, Int_t numberOfClasses);
   void    CreateIndexByTopic(char **filenames, Int_t numberOfNames);
   void    CreateHierarchy(const char **classNames, Int_t numberOfClasses);
   void    CreateListOfTypes();
   void    CreateListOfClasses(const char* filter);
   void    CreateSourceOutputStream(std::ofstream& out, const char* extension, TString& filename);
   void    CreateStyleSheet();
   void    DescendHierarchy(ofstream &out, TClass* basePtr, 
                  const char **classNames, Int_t numberOfClasses, 
                  Int_t maxLines=0, Int_t depth=1);
   void    ExpandKeywords(ostream& out, const char* line);
   void    ExpandKeywords(TString& text);
   void    ExpandPpLine(ostream &out);
   TClass *GetClass(const char *name, Bool_t load=kFALSE);
   const char   *GetFileName(const char *filename);
   void    GetSourceFileName(TString& filename);
   void    GetHtmlFileName(TClass *classPtr, TString& filename);
   Bool_t  IsModified(TClass *classPtr, const Int_t type);
   static Bool_t  IsName(UChar_t c);
   static Bool_t  IsWord(UChar_t c);
   TMethod*LocateMethodInCurrentLine(Ssiz_t& posMethodName, TString& ret, 
      TString& name, TString& params, std::ostream &srcOut, TString &anchor, 
      std::ifstream& sourcefile, Bool_t allowPureVirtual);
   void    LocateMethods(std::ofstream & out, const char* filename,
                          Bool_t lookForSourceInfo = kTRUE, 
                          Bool_t useDocxxStyle = kFALSE, 
                          Bool_t lookForClassDescr = kTRUE,
                          Bool_t allowPureVirtual = kFALSE,
                          const char* methodPattern = 0, 
                          const char* sourceExt = 0);
   void    LocateMethodsInSource(ofstream & out);
   void    LocateMethodsInHeaderInline(ofstream & out);
   void    LocateMethodsInHeaderClassDecl(ofstream & out);

   void    NameSpace2FileName(TString &name);
   void    ReplaceSpecialChars(ostream &out, const char c);
   void    ReplaceSpecialChars(ostream &out, const char *string);
   void    ReplaceSpecialChars(TString& text, Ssiz_t &pos);
   void    SortNames(const char **strings, Int_t num, Bool_t type=0);
   char   *StrDup(const char *s1, Int_t n = 1);

   friend Int_t CaseSensitiveSort(const void *name1, const void *name2);
   friend Int_t CaseInsensitiveSort(const void *name1, const void *name2);

public:
   THtml();
   virtual      ~THtml();
   void          Convert(const char *filename, const char *title, const char *dirname = "");
   const char   *GetDeclFileName(TClass* cl) const;
   const char   *GetImplFileName(TClass* cl) const;
   const char   *GetSourceDir()  { return fSourceDir; }
   const char   *GetOutputDir()  { return fOutputDir; }
   const char   *GetXwho() const { return fXwho.Data(); }
   void          MakeAll(Bool_t force=kFALSE, const char *filter="*");
   void          MakeClass(const char *className, Bool_t force=kFALSE);
   void          MakeIndex(const char *filter="*");
   void          MakeTree(const char *className, Bool_t force=kFALSE);
   void          SetDeclFileName(TClass* cl, const char* filename);
   void          SetEscape(char esc='\\') { fEsc = esc; }
   void          SetImplFileName(TClass* cl, const char* filename);
   void          SetSourcePrefix(const char *prefix) { fSourcePrefix = prefix; }
   void          SetSourceDir(const char *dir) { fSourceDir = dir; }
   void          SetOutputDir(const char *dir) { fOutputDir = dir; }
   void          SetXwho(const char *xwho) { fXwho = xwho; }
   virtual void  WriteHtmlHeader(ofstream & out, const char *title, const char* dir="", TClass *cls=0);
   virtual void  WriteHtmlFooter(ofstream &out, const char *dir="", const char *lastUpdate="",
                                 const char *author="", const char *copyright="");

   ClassDef(THtml,0)  //Convert class(es) into HTML file(s)
};

R__EXTERN THtml *gHtml;

#endif
