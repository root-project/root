// @(#)root/html:$Name:  $:$Id: THtml.h,v 1.14 2006/04/07 13:17:51 rdm Exp $
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

class TClass;
class TVirtualPad;
class TPaveText;

class THtml : public TObject {
protected:
    TString      fXwho;            // by default http://xwho.cern.ch/WHO/people?
  const char    *fSourcePrefix;    // prefix to relative source path
  const char    *fSourceDir;       // source path
  const char    *fOutputDir;       // output directory
        char    *fLine;            // current line
        Int_t    fLen;             // maximum line length
        char    *fCounter;         // counter string
        Bool_t   fEscFlag;         // Flag to mark the symbol must be written "as is"
        char     fEsc;             // The special symbol ("backslash" by default) to mark "the next symbol should not be converted
        Int_t    fHierarchyLines;  // counter for no. lines in hierarchy
        Int_t    fNumberOfClasses; // Number of known classes
  const char   **fClassNames;      // Names of known classes
        Int_t    fMaxLenClassName; // Maximum length of class names
        Int_t    fNumberOfFileNames;// Number of names of files for known classes
        char   **fFileNames;       // Names of files for known classes
  std::list<std::string> fModules; // Names of modules
    
        enum ETraverse {
          kUp, kDown, kBoth        // direction to traverse class tree in ClassHtmlTree()
        };

        void    Class2Html(TClass *classPtr, Bool_t force=kFALSE);
        void    ClassDescription(ofstream &out, TClass *classPtr, Bool_t &flag);
        void    ClassHtmlTree(ofstream &out, TClass *classPtr, ETraverse dir=kBoth, int depth=1);
        void    ClassTree(TVirtualPad *canvas, TClass *classPtr, Bool_t force=kFALSE);
        Bool_t  CopyHtmlFile(const char *sourceName, const char *destName="");
        void    CreateIndex(const char **classNames, Int_t numberOfClasses);
        void    CreateIndexByTopic(char **filenames, Int_t numberOfNames, Int_t maxLen);
        void    CreateHierarchy(const char **classNames, Int_t numberOfClasses);
        void    CreateListOfTypes();
        void    CreateListOfClasses(const char* filter);
        void    CreateStyleSheet();
        void    DescendHierarchy(ofstream &out, TClass* basePtr, 
                  const char **classNames, Int_t numberOfClasses, 
                  Int_t maxLines=0, Int_t depth=1);
        void    ExpandKeywords(ofstream &out, char *text, TClass *ptr2class, Bool_t &flag, const char *dir="");
        void    ExpandPpLine(ofstream &out, char *line);
      TClass   *GetClass(const char *name, Bool_t load=kFALSE);
  const char   *GetFileName(const char *filename);
        char   *GetSourceFileName(const char *filename);
        char   *GetHtmlFileName(TClass *classPtr);
        Bool_t  IsModified(TClass *classPtr, const Int_t type);
 static Bool_t  IsName(UChar_t c);
 static Bool_t  IsWord(UChar_t c);
        void    NameSpace2FileName(char *name);
        void    ReplaceSpecialChars(ofstream &out, const char c);
        void    ReplaceSpecialChars(ofstream &out, const char *string);
        void    SortNames(const char **strings, Int_t num, Bool_t type=0);
        char   *StrDup(const char *s1, Int_t n = 1);

   friend Int_t CaseSensitiveSort(const void *name1, const void *name2);
   friend Int_t CaseInsensitiveSort(const void *name1, const void *name2);

public:
                 THtml();
       virtual   ~THtml();
          void   Convert(const char *filename, const char *title, const char *dirname = "");
    const char  *GetSourceDir()  { return fSourceDir; }
    const char  *GetOutputDir()  { return fOutputDir; }
    const char  *GetXwho() const { return fXwho.Data(); }
          void   MakeAll(Bool_t force=kFALSE, const char *filter="*");
          void   MakeClass(const char *className, Bool_t force=kFALSE);
          void   MakeIndex(const char *filter="*");
          void   MakeTree(const char *className, Bool_t force=kFALSE);
          void   SetEscape(char esc='\\') { fEsc = esc; }
          void   SetSourcePrefix(const char *prefix) { fSourcePrefix = prefix; }
          void   SetSourceDir(const char *dir) { fSourceDir = dir; }
          void   SetOutputDir(const char *dir) { fOutputDir = dir; }
          void   SetXwho(const char *xwho) { fXwho = xwho; }
   virtual void  WriteHtmlHeader(ofstream & out, const char *title, TClass *cls=0);
   virtual void  WriteHtmlFooter(ofstream &out, const char *dir="", const char *lastUpdate="",
                                 const char *author="", const char *copyright="");

   ClassDef(THtml,0)  //Convert class(es) into HTML file(s)
};

R__EXTERN THtml *gHtml;

#endif
