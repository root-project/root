// @(#)root/html:$Name:  $:$Id: THtml.h,v 1.30 2007/01/15 16:57:37 brun Exp $
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
   class MapModuleDepMap: public std::map<std::string, std::set<std::string> > {
   public:
      MapModuleDepMap() {}
   };
   typedef std::map<std::string, MapModuleDepMap > LibDep_t;

protected:
   TString        fXwho;            // URL for name lookup
   TString        fSourcePrefix;    // prefix to relative source path
   TString        fSourceDir;       // source path
   TString        fOutputDir;       // output directory
   TString        fDotDir;          // directory of GraphViz's dot binary
   Int_t          fFoundDot;        // whether dot is accessible (-1 dunno, 1 yes, 0 no)
   TString        fCounter;         // counter string
   TString        fCounterFormat;   // counter printf-like format
   TString        fClassFilter;     // filter used for buidling known classes
   TString        fProductName;     // name of the product to document
   TString        fMacroPath;       // path for macros run via the Begin/End Macro directive
   TString        fModuleDocPath;   // path to check for module documentation
   THashList      fClasses;         // known classes
   THashList      fModules;         // known modules
   std::map<TClass*,std::string> fGuessedDeclFileNames; // names of additional decl file names
   std::map<TClass*,std::string> fGuessedImplFileNames; // names of additional impl file names
   LibDep_t       fSetLibDeps;      // Library dependencies

   void            CreateListOfTypes();
   void            CreateListOfClasses(const char* filter);
   void            MakeClass(void* cdi, Bool_t force=kFALSE);

public:
   THtml();
   virtual      ~THtml();
   void          Convert(const char *filename, const char *title, 
                         const char *dirname = "", const char *relpath="../");
   Bool_t        CopyFileFromEtcDir(const char* filename) const;
   void          CreateHierarchy();
   virtual void  CreateJavascript() const;
   virtual void  CreateStyleSheet() const;
   virtual TClass *GetClass(const char *name);
   const char*   GetCounter() const { return fCounter; }
   const char*   GetCounterFormat() const { return fCounterFormat; }
   virtual const char *GetDeclFileName(TClass* cl) const;
   void          GetDerivedClasses(TClass* cl, std::set<TClass*>& derived) const;
   const char*   GetDotDir() const { return fDotDir; }
   virtual const char *GetImplFileName(TClass* cl) const;
   virtual const char* GetEtcDir() const;
   virtual const char* GetFileName(const char *filename);
   virtual void  GetHtmlFileName(TClass *classPtr, TString& filename);
   virtual const char* GetHtmlFileName(const char* classname);
   LibDep_t&     GetLibraryDependencies() { return fSetLibDeps; }
   const TList*  GetListOfModules() const { return &fModules; }
   const TList*  GetListOfClasses() const { return &fClasses; }
   const TString&GetMacroPath() const { return fMacroPath; }
   const TString&GetModuleDocPath() const { return fModuleDocPath; }
   virtual void  GetModuleName(TString& module, const char* filename) const;
   virtual void  GetModuleNameForClass(TString& module, TClass* cl) const;
   const char   *GetOutputDir() const { return fOutputDir; }
   const char   *GetProductName() const { return fProductName; }
   const char   *GetSourceDir() const { return fSourceDir; }
   virtual void  GetSourceFileName(TString& filename);
   const char   *GetXwho() const { return fXwho.Data(); }
   Bool_t        HaveDot();
   static Bool_t IsNamespace(const TClass*cl);
   void          MakeAll(Bool_t force=kFALSE, const char *filter="*");
   void          MakeClass(const char *className, Bool_t force=kFALSE);
   void          MakeIndex(const char *filter="*");
   void          MakeTree(const char *className, Bool_t force=kFALSE);
   void          ReplaceSpecialChars(std::ostream&, const char*) {
      Error("ReplaceSpecialChars",
            "Removed, call TDocOutput::ReplaceSpecialChars() instead!"); }
   void          SetCounterFormat(const char* format) { fCounterFormat = format; }
   void          SetDeclFileName(TClass* cl, const char* filename);
   void          SetEscape(char /*esc*/ ='\\') {} // for backward comp
   void          SetFoundDot(Bool_t found = kTRUE) { fFoundDot = found; }
   void          SetImplFileName(TClass* cl, const char* filename);
   void          SetMacroPath(const char* path) {fMacroPath = path;}
   void          AddMacroPath(const char* path);
   void          SetSourcePrefix(const char *prefix);
   void          SetSourceDir(const char *dir);
   void          SetOutputDir(const char *dir) { fOutputDir = dir; }
   void          SetProductName(const char* product) { fProductName = product; }
   void          SetDotDir(const char* dir) { fDotDir = dir; fFoundDot = -1; }
   void          SetXwho(const char *xwho) { fXwho = xwho; }
   ClassDef(THtml,0)  //Convert class(es) into HTML file(s)
};

R__EXTERN THtml *gHtml;

#endif
