// @(#)root/html:$Name:  $:$Id: THtml.h,v 1.34 2007/03/19 10:59:41 axel Exp $
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
// Html generates documentation for all ROOT classes                      //
// using XHTML 1.0 transitional                                           //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#ifndef ROOT_THashList
#include "THashList.h"
#endif

#include <map>
#include <set>

class TClass;

class THtml : public TObject {
protected:
   enum ETraverse {
      kUp, kDown, kBoth        // direction to traverse class tree in ClassHtmlTree()
   };

public:

   //__________________________________________________________________________
   // map of lib name to map of module names contained in lib,
   // and their library dependencies. Wraps a long STL name.
   class TMapModuleDepMap: public std::map<std::string, std::set<std::string> > {
   public:
      TMapModuleDepMap() {}
   };
   typedef std::map<std::string, TMapModuleDepMap > LibDep_t;

protected:
   TString        fXwho;            // URL for name lookup
   TString        fROOTURL;         // Root URL for ROOT's reference guide for libs that are not in fLibURLs
   std::map<std::string, TString> fLibURLs; // URL for documentation of external libraries
   TString        fClassDocTag;     // tag for class documentation
   TString        fAuthorTag;       // tag for author
   TString        fLastUpdateTag;   // tag for last update
   TString        fCopyrightTag;    // tag for copyright
   TString        fHeader;          // header file name
   TString        fFooter;          // footerer file name
   TString        fHomepage;        // URL of homepage
   TString        fSearchStemURL;   // URL stem used to build search URL
   TString        fSearchEngine;    // link to search engine
   TString        fViewCVS;         // link to ViewCVS; %f is replaced by the filename (no %f: it's appended)
   TString        fWikiURL;         // URL stem of class's wiki page, %c replaced by mangled class name (no %c: appended)
   TString        fCharset;         // Charset for doc pages
   TString        fDocStyle;        // doc style (only "Doc++" has special treatment)
   
   TString        fSourcePrefix;    // prefix to relative source path
   TString        fSourceDir;       // source path
   TString        fOutputDir;       // output directory
   TString        fDotDir;          // directory of GraphViz's dot binary
   TString        fEtcDir;          // directory containing auxiliary files
   Int_t          fFoundDot;        // whether dot is accessible (-1 dunno, 1 yes, 0 no)
   TString        fCounter;         // counter string
   TString        fCounterFormat;   // counter printf-like format
   TString        fClassFilter;     // filter used for buidling known classes
   TString        fProductName;     // name of the product to document
   TString        fProductDocDir;   // directory containing documentation for the product
   TString        fMacroPath;       // path for macros run via the Begin/End Macro directive
   TString        fModuleDocPath;   // path to check for module documentation
   THashList      fClasses;         // known classes
   THashList      fModules;         // known modules
   std::map<TClass*,std::string> fGuessedDeclFileNames; // names of additional decl file names
   std::map<TClass*,std::string> fGuessedImplFileNames; // names of additional impl file names
   LibDep_t       fSetLibDeps;      // Library dependencies

   virtual void    CreateJavascript() const;
   virtual void    CreateStyleSheet() const;
   void            CreateListOfTypes();
   void            CreateListOfClasses(const char* filter);
   void            MakeClass(void* cdi, Bool_t force=kFALSE);

public:
   THtml();
   virtual      ~THtml();

   // Functions to generate documentation
   void          Convert(const char *filename, const char *title, 
                         const char *dirname = "", const char *relpath="../");
   void          CreateHierarchy();
   void          MakeAll(Bool_t force=kFALSE, const char *filter="*");
   void          MakeClass(const char *className, Bool_t force=kFALSE);
   void          MakeIndex(const char *filter="*");
   void          MakeTree(const char *className, Bool_t force=kFALSE);

   // Configuration setters
   void          SetProductName(const char* product) { fProductName = product; }
   void          SetOutputDir(const char *dir) { fOutputDir = dir; }
   void          SetSourceDir(const char *dir);
   void          SetSourcePrefix(const char *prefix);
   void          SetEtcDir(const char* dir) { fEtcDir = dir; }
   void          SetModuleDocPath(const char* path) { fModuleDocPath = path; }
   void          SetProductDocDir(const char* dir) { fProductDocDir = dir; }
   void          SetDotDir(const char* dir) { fDotDir = dir; fFoundDot = -1; }
   void          SetRootURL(const char* url) { fROOTURL = url; }
   void          SetLibURL(const char* lib, const char* url) { fLibURLs[lib] = url; }
   void          SetXwho(const char *xwho) { fXwho = xwho; }
   void          SetMacroPath(const char* path) {fMacroPath = path;}
   void          AddMacroPath(const char* path);
   void          SetCounterFormat(const char* format) { fCounterFormat = format; }
   void          SetClassDocTag(const char* tag) { fClassDocTag = tag; }
   void          SetAuthorTag(const char* tag) { fAuthorTag = tag; }
   void          SetLastUpdateTag(const char* tag) { fLastUpdateTag = tag; }
   void          SetCopyrightTag(const char* tag) { fCopyrightTag = tag; }
   void          SetHeader(const char* file) { fHeader = file; }
   void          SetFooter(const char* file) { fFooter = file; }
   void          SetHomepage(const char* url) { fHomepage = url; }
   void          SetSearchStemURL(const char* url) { fSearchStemURL = url; }
   void          SetSearchEngine(const char* url) { fSearchEngine = url; }
   void          SetViewCVS(const char* url) { fViewCVS = url; }
   void          SetWikiURL(const char* url) { fWikiURL = url; }
   void          SetCharset(const char* charset) { fCharset = charset; }
   void          SetDocStyle(const char* style) { fDocStyle = style; }

   // Configuration getters
   const TString&      GetProductName() const { return fProductName; }
   const TString&      GetOutputDir() const { return fOutputDir; }
   const TString&      GetSourceDir() const { return fSourceDir; }
   const TString&      GetSourcePrefix() const { return fSourcePrefix; }
   virtual const char* GetEtcDir();
   const TString&      GetModuleDocPath() const { return fModuleDocPath; }
   const TString&      GetProductDocDir() const { return fProductDocDir; }
   const TString&      GetDotDir() const { return fDotDir; }
   const char*         GetURL(const char* lib = 0) const;
   const TString&      GetXwho() const { return fXwho; }
   const TString&      GetMacroPath() const { return fMacroPath; }
   const char*         GetCounterFormat() const { return fCounterFormat; }
   const TString&      GetClassDocTag() const { return fClassDocTag; }
   const TString&      GetAuthorTag() const { return fAuthorTag; }
   const TString&      GetLastUpdateTag() const { return fLastUpdateTag; }
   const TString&      GetCopyrightTag() const { return fCopyrightTag; }
   const TString&      GetHeader() const { return fHeader; }
   const TString&      GetFooter() const { return fFooter; }
   const TString&      GetHomepage() const { return fHomepage; }
   const TString&      GetSearchStemURL() const { return fSearchStemURL; }
   const TString&      GetSearchEngine() const { return fSearchEngine; }
   const TString&      GetViewCVS() const { return fViewCVS; }
   const TString&      GetWikiURL() const { return fWikiURL; }
   const TString&      GetCharset() const { return fCharset; }
   const TString&      GetDocStyle() const { return fDocStyle; }

   // Functions that should only be used by TDocOutput etc.
   Bool_t              CopyFileFromEtcDir(const char* filename) const;
   virtual void        CreateAuxiliaryFiles() const;
   virtual TClass*     GetClass(const char *name);
   const char*         GetCounter() const { return fCounter; }
   virtual const char* GetDeclFileName(TClass* cl) const;
   void                GetDerivedClasses(TClass* cl, std::map<TClass*, Int_t>& derived) const;
   virtual const char* GetImplFileName(TClass* cl) const;
   virtual const char* GetFileName(const char *filename);
   virtual void        GetSourceFileName(TString& filename);
   virtual void        GetHtmlFileName(TClass *classPtr, TString& filename);
   virtual const char* GetHtmlFileName(const char* classname);
   LibDep_t&           GetLibraryDependencies() { return fSetLibDeps; }
   const TList*        GetListOfModules() const { return &fModules; }
   const TList*        GetListOfClasses() const { return &fClasses; }
   virtual void        GetModuleName(TString& module, const char* filename) const;
   virtual void        GetModuleNameForClass(TString& module, TClass* cl) const;
   Bool_t              HaveDot();
   static Bool_t       IsNamespace(const TClass*cl);
   void                SetDeclFileName(TClass* cl, const char* filename);
   void                SetFoundDot(Bool_t found = kTRUE) { fFoundDot = found; }
   void                SetImplFileName(TClass* cl, const char* filename);

   // unused
   void                ReplaceSpecialChars(std::ostream&, const char*) {
      Error("ReplaceSpecialChars",
            "Removed, call TDocOutput::ReplaceSpecialChars() instead!"); }
   void                SetEscape(char /*esc*/ ='\\') {} // for backward comp

   ClassDef(THtml,0)  //Convert class(es) into HTML file(s)
};

R__EXTERN THtml *gHtml;

#endif
