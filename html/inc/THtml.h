// @(#)root/html:$Name:  $:$Id: THtml.h,v 1.29 2006/09/25 08:58:56 brun Exp $
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
#endif

#ifndef ROOT_TClass
#include "TClass.h"
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

   TString        fXwho;            // URL for name lookup
   TString        fSourcePrefix;    // prefix to relative source path
   TString        fSourceDir;       // source path
   TString        fOutputDir;       // output directory
   TString        fDotDir;          // directory of GraphViz's dot binary
   Int_t          fFoundDot;        // whether dot is accessible (-1 dunno, 1 yes, 0 no)
   TString        fLine;            // current line
   UInt_t         fLineNo;          // current line number
   TString        fLineExpanded;    // current line with links
   TString        fLineStripped;    // current line without surrounding spaces
   TClass        *fCurrentClass;    // current class context of sources being parsed
   TString        fCurrentFile;     // current source / header file name
   std::map<std::string /*method name*/, Int_t > fMethodNames;     // current class's method names
   EDocContext    fDocContext;      // current context of parsed sources for documenting
   std::list<EParseContext> fParseContext; // current context of parsed sources
   std::set<UInt_t> fExtraLinesWithAnchor; // lines that need an additional anchor
   TString        fSourceInfo[kNumSourceInfos];// author, last changed, ...
   TString        fCounter;         // counter string
   Bool_t         fEscFlag;         // state flag to mark the next character must be written "as is"
   char           fEsc;             // char to mark the next character must be written "as is"
   Int_t          fHierarchyLines;  // counter for no. lines in hierarchy
   TString        fClassFilter;     // filter used for buidling known classes
   THashList      fClasses;         // known classes
   THashList      fModules;         // known modules
   std::map<TClass*,std::string> fGuessedDeclFileNames; // names of additional decl file names
   std::map<TClass*,std::string> fGuessedImplFileNames; // names of additional impl file names
   static std::set<std::string>  fgKeywords; // C++ keywords

   void    AddClassMethodsRecursive(TBaseClass* bc, TList methodNames[3]);
   void    AddClassDataMembersRecursive(TBaseClass* bc, TList datamembers[6]);
   void    AnchorFromLine(TString& anchor);
   virtual void BeautifyLine(std::ostream &srcOut, const char* relpath = "../");
   void    Class2Html(Bool_t force=kFALSE);
   void    ClassDescription(ofstream &out);
   Bool_t  ClassDotCharts(ofstream & out);
   void    ClassHtmlTree(ofstream &out, TClass *classPtr, ETraverse dir=kBoth, int depth=1);
   void    ClassTree(TVirtualPad *canvas, TClass *classPtr, Bool_t force=kFALSE);
   Bool_t  CopyHtmlFile(const char *sourceName, const char *destName="");
   Bool_t  CreateDotClassChartInh(const char* filename);
   Bool_t  CreateDotClassChartInhMem(const char* filename);
   Bool_t  CreateDotClassChartIncl(const char* filename);
   Bool_t  CreateDotClassChartLib(const char* filename);
   void    CreateIndex();
   void    CreateIndexByTopic();
   void    CreateHierarchy();
   Bool_t  CreateHierarchyDot();
   void    CreateListOfTypes();
   void    CreateListOfClasses(const char* filter);
   void    CreateSourceOutputStream(std::ofstream& out, const char* extension, TString& filename);
   void    DescendHierarchy(ofstream &out, TClass* basePtr, Int_t maxLines=0, Int_t depth=1);
   void    ExpandKeywords(ostream& out, const char* line);
   void    ExpandKeywords(TString& text);
   void    ExpandPpLine(ostream &out);
   Bool_t  ExtractComments(const TString &lineExpandedStripped, 
                           Bool_t &foundClassDescription,
                           const char* classDescrTag, TString& comment);
   TClass *GetClass(const char *name);
   const char   *GetFileName(const char *filename);
   void    GetSourceFileName(TString& filename);
   void    GetHtmlFileName(TClass *classPtr, TString& filename);
   virtual void GetModuleName(TString& module, const char* filename) const;
   Bool_t  HaveDot();
   Bool_t  IsModified(TClass *classPtr, const Int_t type);
   static Bool_t  IsName(UChar_t c);
   static Bool_t  IsWord(UChar_t c);
   TMethod* LocateMethodInCurrentLine(Ssiz_t& posMethodName, TString& ret, 
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

   void    MakeClass(void* cdi, Bool_t force=kFALSE);
   void    NameSpace2FileName(TString &name);
   void    ReplaceSpecialChars(ostream &out, const char c);
   void    ReplaceSpecialChars(ostream &out, const char *string);
   void    ReplaceSpecialChars(TString& text, Ssiz_t &pos);
   Bool_t  RunDot(const char* filename, std::ostream* outMap = 0);
   void    SortNames(const char **strings, Int_t num, Bool_t type=0);
   char   *StrDup(const char *s1, Int_t n = 1);
   static Bool_t Strip(TString& s);
   virtual void WriteMethod(std::ostream & out, TString& ret, 
                            TString& name, TString& params,
                            const char* file, TString& anchor,
                            TString& comment, TString& codeOneLiner);

   friend Int_t CaseSensitiveSort(const void *name1, const void *name2);
   friend Int_t CaseInsensitiveSort(const void *name1, const void *name2);

public:
   THtml();
   virtual      ~THtml();
   void          Convert(const char *filename, const char *title, 
                         const char *dirname = "", const char *relpath="../");
   void          CreateJavascript();
   void          CreateStyleSheet();
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
   void          SetDotDir(const char* dir) { fDotDir = dir; fFoundDot = -1; }
   void          SetXwho(const char *xwho) { fXwho = xwho; }
   virtual void  WriteHtmlHeader(ofstream &out, const char *title, const char* dir="", TClass *cls=0);
   virtual void  WriteHtmlFooter(ofstream &out, const char *dir="", const char *lastUpdate="",
                                 const char *author="", const char *copyright="");

   ClassDef(THtml,0)  //Convert class(es) into HTML file(s)
};

R__EXTERN THtml *gHtml;

#endif
