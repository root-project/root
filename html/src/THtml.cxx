// @(#)root/html:$Name:  $:$Id: THtml.cxx,v 1.105 2006/07/30 11:20:00 rdm Exp $
// Author: Nenad Buncic (18/10/95), Axel Naumann <mailto:axel@fnal.gov> (09/28/01)

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TBaseClass.h"
#include "TVirtualPad.h"
#include "TClass.h"
#include "TClassTable.h"
#include "TDataMember.h"
#include "TDataType.h"
#include "TGlobal.h"
#include "TDatime.h"
#include "TEnv.h"
#include "TError.h"
#include "THtml.h"
#include "TMethod.h"
#include "TMethodArg.h"
#include "TSystem.h"
#include "TObjString.h"
#include "TInterpreter.h"
#include "TRegexp.h"
#include "Riostream.h"
#include "TPluginManager.h"
#include "TVirtualUtilPad.h"
#include "TPaveText.h"
#include "TClassEdit.h"

#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <list>
#include <vector>
#include <algorithm>

THtml *gHtml = 0;

const Int_t kSpaceNum = 1;
const char *formatStr = "%12s %5s %s";

enum ESortType { kCaseInsensitive, kCaseSensitive };
enum EFileType { kSource, kInclude, kTree };
std::set<std::string>  THtml::fgKeywords;

////////////////////////////////////////////////////////////////////////////////
//
// The HyperText Markup Language (HTML) is a simple data format used to
// create hypertext documents that are portable from one platform to another.
// HTML documents are SGML documents with generic semantics that are
// appropriate for representing information from a wide range of domains.
//
// The THtml class is designed to provide an easy way for converting ROOT
// classes, and files as well, into HTML documents. Here are the few rules and
// suggestions for a configuration, coding and usage.
//
//
// Configuration:
// -------------
//
// (i)   Input files
//
// Define Root.Html.SourceDir to point to directories containing .cxx and
// .h files ( see: TEnv ) of the classes you want to document. Better yet,
// specify separate Unix.*.Root.Html.SourceDir and WinNT.*.Root.Html.SourceDir.
// Root.Html.SourcePrefix can hold an additional (relative) path to help THtml
// find the source files. Its default value is "". Root's class documentation
// files can be linked in if Root.Html.Root is set to Root's class
// documentation root. It defaults to "".
//
// Examples:
//   Unix.*.Root.Html.SourceDir:  .:src:include
//   WinNT.*.Root.Html.SourceDir: .;src;include
//   Root.Html.SourcePrefix:
//   Root.Html.Root:              http://root.cern.ch/root/html
//
//
// (ii)  Output directory
//
// The output directory can be specified using the Root.Html.OutputDir
// environment variable ( default value: "htmldoc" ). If it doesn't exist, it
// will be created.
//
// Examples (with defaults given):
//   Root.Html.OutputDir:         htmldoc
//
//
// (iii) Class documentation
//
// The class documentation has to appear in the header file containing the
// class, right in front of its declaration. It is introduced by a string
// defined by Root.Html.Description. See below, section "Coding", for
// further details.
//
// Examples (with defaults given):
//   Root.Html.Description:       //____________________
//
//
// (iv)  Source file information
//
// During the conversion, THtml will look for the certain number of user
// defined strings ("tags") in the source file, which have to appear right in
// front of e.g. the author's name, copyright notice, etc. These tags can be
// defined with the following environment variables: Root.Html.Author,
// Root.Html.LastUpdate and Root.Html.Copyright.
//
// If the LastUpdate tag is not found, the current date and time are given.
// This makes sense if one uses THtml's default option force=kFALSE, in which
// case THtml generates documentation only for changed classes.
//
// Authors can be a comma separated list of author entries. Each entry has
// one of the following two formats:
//  * "Name (non-alpha)". THtml will generate an HTML link for Name, taking
//    the Root.Html.XWho environment variable (defaults to
//    "http://consult.cern.ch/xwho/people?") and adding all parts of the name
//    with spaces replaces by '+'. Non-Alphas are printed out behind Name.
//
//    Example: "// Author: Enrico Fermi" appears in the source file. THtml
//    will generate the link
//    "http://consult.cern.ch/xwho/people?Enrico+Fermi". This works well for
//    people at CERN.
//
//  * "Name <link> Info" THtml will generate a HTML link for Name as specified
//    by "link" and print Info behind Name.
//
//    Example: "// Author: Enrico Fermi <http://www.enricos-home.it>" or
//    "// Author: Enrico Fermi <mailto:enrico@fnal.gov>" in the
//    source file. That's world compatible.
//
// Examples (with defaults given):
//       Root.Html.Author:     // Author:
//       Root.Html.LastUpdate: // @(#)
//       Root.Html.Copyright:  * Copyright
//       Root.Html.XWho:       http://consult.cern.ch/xwho/people?
//
//
// (v)   Style
//
// THtml generates a default header and footer for all pages. You can
// specify your own versions with the environment variables Root.Html.Header
// and Root.Html.Footer. Both variables default to "", using the standard Root
// versions. If set the parameter to your file and append a "+", THtml will
// write both versions (user and root) to a file, for the header in the order
// 1st root, 2nd user, and for the footer 1st user, 2nd root (the root
// versions containing "<html>" and </html> tags, resp).
//
// If you want to replace root's header you have to write a file containing
// all HTML elements necessary starting with the <DOCTYPE> tag and ending with
// (and including) the <BODY> tag. If you add your header it will be added
// directly after Root's <BODY> tag. Any occurrence of the string "%TITLE%"
// (without the quotation marks) in the user's header file will be replaced by
// a sensible, automatically generated title. If the header is generated for a
// class, occurrences of %CLASS% will be replaced by the current class's name,
// %SRCFILE% and %INCFILE% by the name of the source and header file, resp.
// (as given by TClass::GetImplFileName(), TClass::GetDeclFileName()).
// If the header is not generated for a class, they will be replaced by "".
//
// Root's footer starts with the tag "<!--SIGNATURE-->". It includes the
// author(s), last update, copyright, the links to the Root home page, to the
// user home page, to the index file (ClassIndex.html), to the top of the page
// and "this page is automatically generated" infomation. It ends with the
// tags "</body></html>". If you want to replace it, THtml will search for some
// tags in your footer: Occurrences of the strings "%AUTHOR%", "%UPDATE%", and
// "%COPYRIGHT%" (without the quotation marks) are replaced by their
// corresponding values before writing the html file. The %AUTHOR% tag will be
// replaced by the exact string that follows Root.Html.Author, no link
// generation will occur.
//
//
// (vi)  Miscellaneous
//
// Additional parameters can be set by Root.Html.Homepage (address of the
// user's home page), Root.Html.SearchEngine (search engine for the class
// documentation) and Root.Html.Search (search URL). All default to "".
//
// Examples:
//       Root.Html.Homepage:     http://www.enricos-home.it
//       Root.Html.SearchEngine: http://root.cern.ch/root/Search.phtml
//       Root.Html.Search:       http://www.google.com/search?q=%s+site%3Aroot.cern.ch%2Froot%2Fhtml
//
//
// (vii) HTML Charset
//
// HTML 4.01 transitional recommends the specification of the charset in the
// content type meta tag, see e.g. BEGIN_HTML<a href="http://www.w3.org/TR/REC-html40/charset.html">http://www.w3.org/TR/REC-html40/charset.html</a>END_HTML
// THtml generates it for the HTML output files. It defaults to ISO-8859-1, and
// can be changed using Root.Html.Charset.
//
// Example:
//       Root.Html.Charset:      EUC-JP
//
//
//
//
// Coding rules:
// ------------
//
// A class description block, which must be placed before the first
// member function, has a following form:
//
//       ////////////////////////////////////////////////////////////////
//       //                                                            //
//       // TMyClass                                                   //
//       //                                                            //
//       // This is the description block.                             //
//       //                                                            //
//       ////////////////////////////////////////////////////////////////
//
// The environment variable Root.Html.Description ( see: TEnv ) contents
// the delimiter string ( default value: //_________________ ). It means
// that you can also write your class description block like this:
//
//       //_____________________________________________________________
//       // A description of the class starts with the line above, and
//       // will take place here !
//       //
//
// Note that EVERYTHING until the first non-commented line is considered
// as a valid class description block.
//
// A member function description block starts immediately after '{'
// and looks like this:
//
//       void TWorld::HelloWorldFunc(string *text)
//       {
//          // This is an example of description for the
//          // TWorld member function
//
//          helloWorld.Print( text );
//       }
//
// Like in a class description block, EVERYTHING until the first
// non-commented line is considered as a valid member function
// description block.
//
//   ==> The "Begin_Html" and "End_Html" special keywords <=========
//       --------------------------------------------
// You can insert pure html code in your comment lines. During the
// generation of the documentation, this code will be inserted as is
// in the html file.
// Pure html code must be inserted between the keywords "Begin_Html"
// and "End_Html" starting/finishing anywhere in the comment lines.
// Examples of pure html code are given in many Root classes.
// See for example the classes TDataMember and TMinuit.
//
//   ==> The escape character
//       --------------------
// Outside blocks starting with "Begin_Html" and finishing with "End_Html"
// one can prevent the automatic translation of symbols like "<" and ">"
// to "&lt;" and "&gt;" by using the escape character in front.
// The default escape character is backslash and can be changed
// via the member function SetEscape.
//
//   ==> The ClassIndex
//       --------------
// All classes to be documented will have an entry in the ClassIndex.html,
// showing their name with a link to their documentation page and a miniature
// description. This discription for e.g. the class MyClass has to be given
// in MyClass's header as a comment right after ClassDef( MyClass, n ).
//
//
//
// Usage:
// -----
//
//     Root> THtml html;                // create a THtml object
//     Root> html.MakeAll()             // invoke a make for all classes
//     Root> html.MakeClass("TMyClass") // create a HTML files for that class only
//     Root> html.MakeIndex()           // creates an index files only
//     Root> html.MakeTree("TMyClass")  // creates an inheritance tree for a class
//
//     Root> html.Convert( hist1.mac, "Histogram example" )
//
//
// Environment variables:
// ---------------------
//
//   Root.Html.OutputDir    (default: htmldoc)
//   Root.Html.SourceDir    (default: .:src/:include/)
//   Root.Html.Author       (default: // Author:) - start tag for authors
//   Root.Html.LastUpdate   (default: // @(#)) - start tag for last update
//   Root.Html.Copyright    (default:  * Copyright) - start tag for copyright notice
//   Root.Html.Description  (default: //____________________ ) - start tag for class descr
//   Root.Html.HomePage     (default: ) - URL to the user defined home page
//   Root.Html.Header       (default: ) - location of user defined header
//   Root.Html.Footer       (default: ) - location of user defined footer
//   Root.Html.Root         (default: ) - URL of Root's class documentation
//   Root.Html.SearchEngine (default: ) - link to the search engine
//   Root.Html.XWho         (default: http://consult.cern.ch/xwho/people?) - URL stem of CERN's xWho system
//   Root.Html.Charset      (default: ISO-8859-1) - HTML character set
//
////////////////////////////////////////////////////////////////////////////////

ClassImp(THtml)
//______________________________________________________________________________
THtml::THtml(): fCurrentClass(0), fDocContext(kIgnore), fParseContext(kCode), 
   fHierarchyLines(0), fNumberOfClasses(0), fClassNames(0), fNumberOfFileNames(0), fFileNames(0)
{
   // Create a THtml object.
   // In case output directory does not exist an error
   // will be printed and gHtml stays 0 also zombie bit will be set.

   fEscFlag = kFALSE;
   fClassNames = 0;
   fFileNames = 0;
   SetEscape();

   // get prefix for source directory
   fSourcePrefix = gEnv->GetValue("Root.Html.SourcePrefix", "");

   // check for source directory
   fSourceDir = gEnv->GetValue("Root.Html.SourceDir", "./:src/:include/");

   // check for output directory
   fOutputDir = gEnv->GetValue("Root.Html.OutputDir", "htmldoc");

   fXwho =
       gEnv->GetValue("Root.Html.XWho",
                      "http://consult.cern.ch/xwho/people?");

   Int_t st;
   Long64_t sSize;
   Long_t sId, sFlags, sModtime;
   if ((st =
        gSystem->GetPathInfo(fOutputDir, &sId, &sSize, &sFlags, &sModtime))
       || !(sFlags & 2)) {
      if (st == 0) {
         Error("THtml", "output directory %s is an existing file",
               fOutputDir.Data());
         MakeZombie();
         return;
      }
      // Try creating directory
      if (gSystem->MakeDirectory(fOutputDir) == -1) {
         Error("THtml", "output directory %s does not exist", fOutputDir.Data());
         MakeZombie();
         return;
      }
   }
   // insert html object in the list of special ROOT objects
   if (!gHtml) {
      gHtml = this;
      gROOT->GetListOfSpecials()->Add(gHtml);
   }

   if (fgKeywords.empty()) {
      fgKeywords.insert("asm");
      fgKeywords.insert("auto");
      fgKeywords.insert("bool");
      fgKeywords.insert("break");
      fgKeywords.insert("case");
      fgKeywords.insert("catch");
      fgKeywords.insert("char");
      fgKeywords.insert("class");
      fgKeywords.insert("const");
      fgKeywords.insert("const_cast");
      fgKeywords.insert("continue");
      fgKeywords.insert("default");
      fgKeywords.insert("delete");
      fgKeywords.insert("do");
      fgKeywords.insert("double");
      fgKeywords.insert("dynamic_cast");
      fgKeywords.insert("else");
      fgKeywords.insert("enum");
      fgKeywords.insert("explicit");
      fgKeywords.insert("export");
      fgKeywords.insert("extern");
      fgKeywords.insert("false");
      fgKeywords.insert("float");
      fgKeywords.insert("for");
      fgKeywords.insert("friend");
      fgKeywords.insert("goto");
      fgKeywords.insert("if");
      fgKeywords.insert("inline");
      fgKeywords.insert("int");
      fgKeywords.insert("long");
      fgKeywords.insert("mutable");
      fgKeywords.insert("namespace");
      fgKeywords.insert("new");
      fgKeywords.insert("operator");
      fgKeywords.insert("private");
      fgKeywords.insert("protected");
      fgKeywords.insert("public");
      fgKeywords.insert("register");
      fgKeywords.insert("reinterpret_cast");
      fgKeywords.insert("return");
      fgKeywords.insert("short");
      fgKeywords.insert("signed");
      fgKeywords.insert("sizeof");
      fgKeywords.insert("static");
      fgKeywords.insert("static_cast");
      fgKeywords.insert("struct");
      fgKeywords.insert("switch");
      fgKeywords.insert("template");
      fgKeywords.insert("this");
      fgKeywords.insert("throw");
      fgKeywords.insert("true");
      fgKeywords.insert("try");
      fgKeywords.insert("typedef");
      fgKeywords.insert("typeid");
      fgKeywords.insert("typename");
      fgKeywords.insert("union");
      fgKeywords.insert("unsigned");
      fgKeywords.insert("using");
      fgKeywords.insert("virtual");
      fgKeywords.insert("void");
      fgKeywords.insert("volatile");
      fgKeywords.insert("wchar_t");
      fgKeywords.insert("while");
   }
}


//______________________________________________________________________________
THtml::~THtml()
{
// Default destructor

   delete []fClassNames;
   delete []fFileNames;

   if (gHtml == this) {
      gROOT->GetListOfSpecials()->Remove(gHtml);
      gHtml = 0;
   }
}

//______________________________________________________________________________
bool IsNamespace(TClass*cl)
{
   // Check whether cl is a namespace
   return (cl->Property() & kIsNamespace);
}

//______________________________________________________________________________
int CaseSensitiveSort(const void *name1, const void *name2)
{
// Friend function for sorting strings, case sensitive
//
//
// Input: name1 - pointer to the first string
//        name2 - pointer to the second string
//
//  NOTE: This function compares its arguments and returns an integer less
//        than, equal to, or greater than zero, depending on whether name1
//        is lexicographically less than, equal to, or greater than name2.
//
//

   return (strcmp(*((char **) name1), *((char **) name2)));
}


//______________________________________________________________________________
int CaseInsensitiveSort(const void *name1, const void *name2)
{
// Friend function for sorting strings, case insensitive
//
//
// Input: name1 - pointer to the first string
//        name2 - pointer to the second string
//
//  NOTE: This function compares its arguments and returns an integer less
//        than, equal to, or greater than zero, depending on whether name1
//        is lexicographically less than, equal to, or greater than name2,
//        but characters are forced to lower-case prior to comparison.
//
//

   return (strcasecmp(*((char **) name1), *((char **) name2)));
}

namespace {
   typedef std::vector<std::string> Words_t;
   typedef Words_t::const_iterator SectionStart_t;
   class TSectionInfo {
   public:
      TSectionInfo(SectionStart_t start, size_t chars, size_t size):
         fStart(start), fChars(chars), fSize(size) {};

         SectionStart_t fStart;
         size_t fChars;
         size_t fSize;
   };
   typedef std::list<TSectionInfo> SectionStarts_t;

   void Sections_BuildIndex(SectionStarts_t& sectionStarts,
      SectionStart_t begin, SectionStart_t end, 
      size_t maxPerSection) 
   {
      // for each assumed section border, check that previous entry's
      // char[selectionChar] differs, else move section start forward

      SectionStart_t cursor = begin;
      if (sectionStarts.empty() || sectionStarts.back().fStart != cursor)
         sectionStarts.push_back(TSectionInfo(cursor, 1, 0));

      SectionStarts_t::iterator prevSection = sectionStarts.end();
      --prevSection;

      while (cursor != end) {
         size_t numLeft = end - cursor;
         size_t assumedNumSections = (numLeft + maxPerSection - 1 ) / maxPerSection;
         size_t step = ((numLeft + assumedNumSections - 1) / assumedNumSections);
         if (!step || step >= numLeft) return;
         cursor += step;
         if (cursor == end) break;

         SectionStart_t addWhichOne = prevSection->fStart;

         size_t selectionChar=1;
         for (; selectionChar <= cursor->length() && addWhichOne == prevSection->fStart; 
            ++selectionChar) {
            SectionStart_t checkPrev = cursor;
            while (--checkPrev != prevSection->fStart 
               && !strncasecmp(checkPrev->c_str(), cursor->c_str(), selectionChar));

            SectionStart_t checkNext = cursor;
            while (++checkNext != end
               && !strncasecmp(checkNext->c_str(), cursor->c_str(), selectionChar));

            // if the previous matching one is closer but not previous section start, take it!
            if (checkPrev != prevSection->fStart)
               if ((cursor - checkPrev) <= (checkNext - cursor))
                  addWhichOne = ++checkPrev;
               else if (checkNext != end
                  && (size_t)(checkNext - cursor) < maxPerSection) {
                  addWhichOne = checkNext;
               }
         }
         if (addWhichOne == prevSection->fStart)
            addWhichOne = cursor;

         selectionChar = 1;
         while (selectionChar <= prevSection->fStart->length() 
            && selectionChar <= addWhichOne->length() 
            && !strncasecmp(prevSection->fStart->c_str(), addWhichOne->c_str(), selectionChar))
            ++selectionChar;

         sectionStarts.push_back(TSectionInfo(addWhichOne, selectionChar, 0));
         cursor = addWhichOne;
         ++prevSection;
      } // while cursor != end
   }

   void Sections_SetSize(SectionStarts_t& sectionStarts, const Words_t &words)
   {
      // Update the length of the sections
      for (SectionStarts_t::iterator iSectionStart = sectionStarts.begin();
         iSectionStart != sectionStarts.end(); ++iSectionStart) {
         SectionStarts_t::iterator next = iSectionStart;
         ++next;
         if (next == sectionStarts.end()) {
            iSectionStart->fSize = (words.end() - iSectionStart->fStart);
            break;
         }
         iSectionStart->fSize = (next->fStart - iSectionStart->fStart);
      }
   }

   void Sections_PostMerge(SectionStarts_t& sectionStarts, const size_t maxPerSection)
   {
      // Merge sections that ended up being too small, up to maxPerSection entries
      for (SectionStarts_t::iterator iSectionStart = sectionStarts.begin();
         iSectionStart != sectionStarts.end();) {
         SectionStarts_t::iterator iNextSectionStart = iSectionStart;
         ++iNextSectionStart;
         if (iNextSectionStart == sectionStarts.end()) break;
         if (iNextSectionStart->fSize + iSectionStart->fSize < maxPerSection) {
            iSectionStart->fSize += iNextSectionStart->fSize;
            sectionStarts.erase(iNextSectionStart);
         } else ++iSectionStart;
      }
   }

   void GetIndexChars(const Words_t& words, UInt_t numSectionsIn, 
      std::vector<std::string> &sectionMarkersOut)
   {
      // Given a list of words (class names, in this case), this function builds an
      // optimal set of about numSectionIn sections (even if almost all words start 
      // with a "T"...), and returns the significant characters for each section start 
      // in sectionMarkersOut.

      const size_t maxPerSection = (words.size() + numSectionsIn - 1)/ numSectionsIn;
      SectionStarts_t sectionStarts;
      Sections_BuildIndex(sectionStarts, words.begin(), words.end(), maxPerSection);
      Sections_SetSize(sectionStarts, words);
      Sections_PostMerge(sectionStarts, maxPerSection);

      // convert to index markers
      sectionMarkersOut.clear();
      sectionMarkersOut.resize(sectionStarts.size());
      size_t idx = 0;
      for (SectionStarts_t::iterator iSectionStart = sectionStarts.begin();
         iSectionStart != sectionStarts.end(); ++iSectionStart)
         sectionMarkersOut[idx++] = 
            iSectionStart->fStart->substr(0, iSectionStart->fChars);
   }

   void GetIndexChars(const char** wordsIn, UInt_t numWordsIn, UInt_t numSectionsIn, 
      std::vector<std::string> &sectionMarkersOut)
   {
      // initialize word vector
      Words_t words(numWordsIn);
      for (UInt_t iWord = 0; iWord < numWordsIn; ++iWord)
         words[iWord] = wordsIn[iWord];
      GetIndexChars(words, numSectionsIn, sectionMarkersOut);
   }

   void GetIndexChars(const std::list<std::string>& wordsIn, UInt_t numSectionsIn, 
      std::vector<std::string> &sectionMarkersOut)
   {
      // initialize word vector
      Words_t words(wordsIn.size());
      size_t idx = 0;
      for (std::list<std::string>::const_iterator iWord = wordsIn.begin(); iWord != wordsIn.end(); ++iWord)
         words[idx++] = *iWord;
      GetIndexChars(words, numSectionsIn, sectionMarkersOut);
   }

   // std::list::sort(with_stricmp_predicate) doesn't work with Solaris CC...
   void sort_strlist_stricmp(std::list<std::string>& l)
   {
      // sort strings ignoring case - easier for humans
      struct posList {
         const char* str;
         std::list<std::string>::const_iterator pos;
      };
      posList* carr = new posList[l.size()];
      size_t idx = 0;
      for (std::list<std::string>::const_iterator iS = l.begin(); iS != l.end(); ++iS) {
         carr[idx].pos = iS;
         carr[idx++].str = iS->c_str();
      }
      qsort(&carr[0].str, idx, sizeof(posList), CaseInsensitiveSort);
      std::list<std::string> lsort;
      for (idx = 0; idx < l.size(); ++idx) {
         lsort.push_back(*carr[idx].pos);
      }
      delete [] carr;
      l.swap(lsort);
   }
}

//______________________________________________________________________________
void THtml::Class2Html(Bool_t force)
{
// Create HTML files for a single class.
//

   const char *tab = "";
   const char *tab2 = "  ";
   const char *tab4 = "    ";
   const char *tab6 = "      ";

   gROOT->GetListOfGlobals(kTRUE);

   // create a filename
   TString filename(fCurrentClass->GetName());
   NameSpace2FileName(filename);

   gSystem->ExpandPathName(fOutputDir);
   gSystem->PrependPathName(fOutputDir, filename);

   filename += ".html";

   if (IsModified(fCurrentClass, kSource) || force) {

      // open class file
      ofstream classFile;
      classFile.open(filename, ios::out);

      if (classFile.good()) {

         Printf(formatStr, "", fCounter.Data(), filename.Data());

         // write a HTML header for the classFile file
         WriteHtmlHeader(classFile, fCurrentClass->GetName(), "", fCurrentClass);

         // show box with lib, include
         // needs to go first to allow title on the left
         const char* lib=fCurrentClass->GetSharedLibs();
         const char* incl=GetDeclFileName(fCurrentClass);
         if (incl) incl=gSystem->BaseName(incl);
         if (lib && strlen(lib)|| incl && strlen(incl)) {
            classFile << "<table class=\"libinfo\"><tr><td>";
            if (lib) {
               char* libDup=StrDup(lib);
               char* libDupSpace=strchr(libDup,' ');
               if (libDupSpace) *libDupSpace=0;
               char* libDupEnd=libDup+strlen(libDup);
               while (libDupEnd!=libDup)
                  if (*(--libDupEnd)=='.') {
                     *libDupEnd=0;
                     break;
                  }
               classFile << "library: "
                         << libDup;
               delete[] libDup;
            }
            if (incl) {
               if (lib)
                  classFile << "<br />";
               classFile << "#include \""
                         << incl << "\"";
            }
            classFile << "</td></tr></table>"
                      << endl;
         }

         // make a link to the description
         classFile << "<!--BEGIN-->" << endl;
         classFile << "<center>" << endl;
         classFile << "<h1>";
         ReplaceSpecialChars(classFile, fCurrentClass->GetName());
         classFile << "</h1>" << endl;
         classFile << "<hr width=\"300\" />" << endl;
         TString currClassNameMangled(fCurrentClass->GetName());
         NameSpace2FileName(currClassNameMangled);
         classFile << "<!--SDL--><em><a href=\"#" << currClassNameMangled;
         if (IsNamespace(fCurrentClass)) {
            classFile << ":description\">namespace description</a>";
         } else {
            classFile << ":description\">class description</a>";
         }

         // make a link to the '.cxx' file
         TString classFileName(fCurrentClass->GetName());
         NameSpace2FileName(classFileName);

         classFile << " - <a href=\"src/" << classFileName <<
             ".h.html\"";
         classFile << ">header file</a>";

         classFile << " - <a href=\"src/" << classFileName <<
             ".cxx.html\"";
         classFile << ">source file</a>";

         if (!IsNamespace(fCurrentClass)) {
            // make a link to the inheritance tree (postscript)
            classFile << " - <a href=\"" << classFileName << "_Tree.pdf\"";
            classFile << ">inheritance tree (.pdf)</a>";
         }
         const char* viewCVSLink = gEnv->GetValue("Root.Html.ViewCVS","");
         const char* headerFileName = GetDeclFileName(fCurrentClass);
         const char* sourceFileName = GetImplFileName(fCurrentClass);
         if (viewCVSLink && viewCVSLink[0] && (headerFileName || sourceFileName )) {
            classFile << "<br />";
            if (headerFileName)
               classFile << "<a href=\"" << viewCVSLink << headerFileName << "\">viewCVS header</a>";
            if (headerFileName && sourceFileName)
               classFile << " - ";
            if (sourceFileName)
               classFile << "<a href=\"" << viewCVSLink << sourceFileName << "\">viewCVS source</a>";
         }

         classFile << "</em>" << endl;
         classFile << "<hr width=\"300\" />" << endl;
         classFile << "</center>" << endl;


         // make a link to the '.h.html' file
         TString headerHtmlFileName = fCurrentClass->GetName();
         NameSpace2FileName(headerHtmlFileName);
         gSystem->PrependPathName("src", headerHtmlFileName);
         headerHtmlFileName += ".h.html";

         classFile << "<h2>class <a name=\"" << fCurrentClass->GetName()
                   << "\" href=\"";
         classFile << headerHtmlFileName << "\">";
         ReplaceSpecialChars(classFile, fCurrentClass->GetName());
         classFile << "</a> ";


         // copy .h file to the Html output directory
         TString declf(GetDeclFileName(fCurrentClass));
         GetSourceFileName(declf);
         if (declf.Length())
            CopyHtmlFile(declf);

         // make a loop on base classes
         Bool_t first = kTRUE;
         TBaseClass *inheritFrom;
         TIter nextBase(fCurrentClass->GetListOfBases());

         while ((inheritFrom = (TBaseClass *) nextBase())) {
            if (first) {
               classFile << ": ";
               first = kFALSE;
            } else
               classFile << ", ";
            classFile << "public ";

            // get a class
            TClass *classInh =
                GetClass((const char *) inheritFrom->GetName());

            TString htmlFile;
            GetHtmlFileName(classInh, htmlFile);

            if (htmlFile.Length()) {
               classFile << "<a href=\"";

               // make a link to the base class
               classFile << htmlFile;
               classFile << "\">";
               ReplaceSpecialChars(classFile, inheritFrom->GetName());
               classFile << "</a>";
            } else
               ReplaceSpecialChars(classFile, inheritFrom->GetName());
         }

         classFile << "</h2>" << endl;


         // create an html inheritance tree
         if (!IsNamespace(fCurrentClass)) ClassHtmlTree(classFile, fCurrentClass);


         // make a loop on member functions
         TMethod *method;
         TIter nextMethod(fCurrentClass->GetListOfMethods());

         Int_t len, maxLen[3];
         len = maxLen[0] = maxLen[1] = maxLen[2] = 0;

         // loop to get a pointers to a method names
         const Int_t nMethods = fCurrentClass->GetNmethods();
         const char **fMethodNames = new const char *[3 * 2 * nMethods];

         Int_t mtype, num[3];
         mtype = num[0] = num[1] = num[2] = 0;

         while ((method = (TMethod *) nextMethod())) {

            if (!strcmp(method->GetName(), "Dictionary") ||
                !strcmp(method->GetName(), "Class_Version") ||
                !strcmp(method->GetName(), "Class_Name") ||
                !strcmp(method->GetName(), "DeclFileName") ||
                !strcmp(method->GetName(), "DeclFileLine") ||
                !strcmp(method->GetName(), "ImplFileName") ||
                !strcmp(method->GetName(), "ImplFileLine")
                )
               continue;


            if (kIsPrivate & method->Property())
               mtype = 0;
            else if (kIsProtected & method->Property())
               mtype = 1;
            else if (kIsPublic & method->Property())
               mtype = 2;

            fMethodNames[mtype * 2 * nMethods + 2 * num[mtype]] =
               method->GetName();

            if (method->GetReturnTypeName())
               len = strlen(method->GetReturnTypeName());
            else
               len = 0;

            if (kIsVirtual & method->Property())
               len += 8;
            if (kIsStatic & method->Property())
               len += 7;

            maxLen[mtype] = maxLen[mtype] > len ? maxLen[mtype] : len;

            const char *type = strrchr(method->GetReturnTypeName(), ' ');
            if (!type)
               type = method->GetReturnTypeName();
            else
               type++;

            if (fCurrentClass && !strcmp(type, fCurrentClass->GetName()))
               fMethodNames[mtype * 2 * nMethods + 2 * num[mtype]] =
                  "A00000000";

            // if this is the destructor
            while ('~' ==
                   *fMethodNames[mtype * 2 * nMethods + 2 * num[mtype]])
               fMethodNames[mtype * 2 * nMethods + 2 * num[mtype]] =
                  "A00000001";

            fMethodNames[mtype * 2 * nMethods + 2 * num[mtype] + 1] =
               (char *) method;

            num[mtype]++;
         }

         const char* tab4nbsp="&nbsp;&nbsp;&nbsp;&nbsp;";
         if (fCurrentClass->Property() & kIsAbstract)
            classFile << "&nbsp;<br /><b>"
                      << tab4nbsp << "This is an abstract class, constructors will not be documented.<br />" << endl
                      << tab4nbsp << "Look at the <a href=\""
                      << GetFileName((const char *) GetDeclFileName(fCurrentClass))
                      << "\">header</a> to check for available constructors.</b><br />" << endl;

         classFile << "<pre>" << endl;

         Int_t i, j;

         if (IsNamespace(fCurrentClass)) {
            j = 2;
         } else {
            j = 0;
         }
         for (; j < 3; j++) {
            if (num[j]) {
                 qsort(fMethodNames + j * 2 * nMethods, num[j],
                        2 * sizeof(fMethodNames), CaseInsensitiveSort);

               const char *ftitle = 0;
               switch (j) {
               case 0:
                  ftitle = "private:";
                  break;
               case 1:
                  ftitle = "protected:";
                  break;
               case 2:
                  ftitle = "public:";
                  break;
               }
               if (j)
                  classFile << endl;
               classFile << tab4 << "<b>" << ftitle << "</b><br />" << endl;

               TString strClassNameNoScope(fCurrentClass->GetName());
               
               UInt_t templateNest = 0;
               Ssiz_t posLastScope = strClassNameNoScope.Length()-1;
               for (;posLastScope && (templateNest || strClassNameNoScope[posLastScope] != ':'); --posLastScope)
                  if (strClassNameNoScope[posLastScope] == '>') ++templateNest;
                  else if (strClassNameNoScope[posLastScope] == '<') --templateNest;
                  else if (!strncmp(strClassNameNoScope.Data()+posLastScope,"operator", 8) && templateNest==1) 
                     --templateNest;
               if (strClassNameNoScope[posLastScope] == ':' 
                   && strClassNameNoScope[posLastScope-1] == ':')
                  strClassNameNoScope.Remove(0, posLastScope+1);

               for (i = 0; i < num[j]; i++) {
                  method =
                     (TMethod *) fMethodNames[j * 2 * nMethods + 2 * i +
                                             1];

                  if (method) {
                     Int_t w = 0;
                     Bool_t isctor=false;
                     Bool_t isdtor=false;
                     if (method->GetReturnTypeName())
                        len = strlen(method->GetReturnTypeName());
                     else
                        len = 0;
                     if (!strcmp(method->GetName(),strClassNameNoScope.Data()))
                        // it's a c'tor - Cint stores the class name as return type
                        isctor=true;
                     if (!isctor && method->GetName()[0] == '~' && !strcmp(method->GetName()+1,strClassNameNoScope.Data()))
                        // it's a d'tor - Cint stores "void" as return type
                        isdtor=true;
                     if (isctor || isdtor)
                        len=0;

                     if (kIsVirtual & method->Property())
                        len += 8;
                     if (kIsStatic & method->Property())
                        len += 7;

                     classFile << tab6;
                     for (w = 0; w < (maxLen[j] - len); w++)
                        classFile << " ";

                     if (kIsVirtual & method->Property())
                        if (!isdtor)
                           classFile << "virtual ";
                        else
                           classFile << " virtual";

                     if (kIsStatic & method->Property())
                        classFile << "static ";

                     if (!isctor && !isdtor)
                        ExpandKeywords(classFile, method->GetReturnTypeName());

                     TString mangled(fCurrentClass->GetName());
                     NameSpace2FileName(mangled);
                     classFile << " " << tab << "<!--BOLD-->";
                     classFile << "<a href=\"#" << mangled;
                     classFile << ":";
                     mangled = method->GetName();
                     NameSpace2FileName(mangled);
                     classFile << mangled << "\">";
                     ReplaceSpecialChars(classFile, method->GetName());
                     classFile << "</a><!--PLAIN-->";

                     ExpandKeywords(classFile, method->GetSignature());
                     classFile << endl;
                  }
               }
            }
         }

         delete[]fMethodNames;

         classFile << "</pre>" << endl;

         // make a loop on data members
         first = kFALSE;
         TDataMember *member;
         TIter nextMember(fCurrentClass->GetListOfDataMembers());


         Int_t len1, len2, maxLen1[3], maxLen2[3];
         len1 = len2 = maxLen1[0] = maxLen1[1] = maxLen1[2] = 0;
         maxLen2[0] = maxLen2[1] = maxLen2[2] = 0;
         mtype = num[0] = num[1] = num[2] = 0;

         Int_t ndata = fCurrentClass->GetNdata();

         // if data member exist
         if (ndata) {
            TDataMember **memberArray = new TDataMember *[3 * ndata];

            if (memberArray) {
               while ((member = (TDataMember *) nextMember())) {
                  if (!strcmp(member->GetName(), "fgIsA")
                      )
                     continue;

                  if (kIsPrivate & member->Property())
                     mtype = 0;
                  else if (kIsProtected & member->Property())
                     mtype = 1;
                  else if (kIsPublic & member->Property())
                     mtype = 2;

                  memberArray[mtype * ndata + num[mtype]] = member;
                  num[mtype]++;

                  if (member->GetFullTypeName())
                     len1 = strlen((char *) member->GetFullTypeName());
                  else
                     len1 = 0;
                  if (member->GetName())
                     len2 = strlen(member->GetName());
                  else
                     len2 = 0;

                  if (kIsStatic & member->Property())
                     len1 += 7;

                  // Take in account the room the array index will occupy

                  Int_t dim = member->GetArrayDim();
                  Int_t indx = 0;
                  Int_t maxidx;
                  while (indx < dim) {
                     maxidx = member->GetMaxIndex(indx);
                     if (maxidx <= 0)
                        break;
                     else
                        len2 += (Int_t)TMath::Log10(maxidx) + 3;
                     indx++;
                  }
                  maxLen1[mtype] =
                      maxLen1[mtype] > len1 ? maxLen1[mtype] : len1;
                  maxLen2[mtype] =
                      maxLen2[mtype] > len2 ? maxLen2[mtype] : len2;
               }

               classFile << endl;
               classFile << "<h3>" << tab2 << "<a name=\"";
               classFile << fCurrentClass->GetName();
               classFile << ":Data_Members\">Data Members</a></h3>" <<
                   endl;
               classFile << "<pre>" << endl;

               for (j = 0; j < 3; j++) {
                  if (num[j]) {
                     const char *ftitle = 0;
                     switch (j) {
                     case 0:
                        ftitle = "private:";
                        break;
                     case 1:
                        ftitle = "protected:";
                        break;
                     case 2:
                        ftitle = "public:";
                        break;
                     }
                     if (j)
                        classFile << endl;
                     classFile << tab4 << "<b>" << ftitle << "</b><br />" <<
                         endl;

                     for (i = 0; i < num[j]; i++) {
                        Int_t w = 0;
                        member = memberArray[j * ndata + i];

                        classFile << tab6;
                        if (member->GetFullTypeName())
                           len1 = strlen(member->GetFullTypeName());
                        else
                           len1 = 0;

                        if (kIsStatic & member->Property())
                           len1 += 7;

                        for (w = 0; w < (maxLen1[j] - len1); w++)
                           classFile << " ";

                        if (kIsStatic & member->Property())
                           classFile << "static ";

                        ExpandKeywords(classFile, member->GetFullTypeName());

                        classFile << " " << tab << "<!--BOLD-->";
                        classFile << "<a name=\"" << fCurrentClass->
                            GetName() << ":";
                        classFile << member->GetName();
                        classFile << "\">" << member->GetName();

                        // Add the dimensions to "array" members

                        Int_t dim = member->GetArrayDim();
                        Int_t indx = 0;
                        Int_t indxlen = 0;
                        while (indx < dim) {
                           if (member->GetMaxIndex(indx) <= 0)
                              break;
                           classFile << "[" << member->
                               GetMaxIndex(indx) << "]";
                           // Take in account the room this index will occupy
                           indxlen +=
                               Int_t(TMath::
                                     Log10(member->GetMaxIndex(indx))) + 3;
                           indx++;
                        }

                        classFile << "</a><!--PLAIN--> ";

                        len2 = 0;
                        if (member->GetName())
                           len2 = strlen(member->GetName()) + indxlen;

                        for (w = 0; w < (maxLen2[j] - len2); w++)
                           classFile << " ";
                        classFile << " " << tab;

                        classFile << "<i><a name=\"Title:";
                        classFile << member->GetName();

                        classFile << "\">";

                        ReplaceSpecialChars(classFile, member->GetTitle());
                        classFile << "</a></i>" << endl;
                     }
                  }
               }
               classFile << "</pre>" << endl;
               delete[]memberArray;
            }
         }

         classFile << "<!--END-->" << endl;

         // process a '.cxx' file
         ClassDescription(classFile);


         // close a file
         classFile.close();

      } else
         Error("Make", "Can't open file '%s' !", filename.Data());
   } else
      Printf(formatStr, "-no change-", fCounter.Data(), filename.Data());
}

//______________________________________________________________________________
void THtml::CreateSourceOutputStream(std::ofstream& out, const char* extension, 
                                     TString& sourceHtmlFileName) 
{
   // Open a Class.cxx.html file, where Class is defined by classPtr, and .cxx.html by extension
   // It's created in fOutputDir/src. If successful, the HTML header is written to out.

   gSystem->ExpandPathName(fOutputDir);
   TString sourceHtmlDir("src");
   gSystem->PrependPathName(fOutputDir, sourceHtmlDir);
   // create directory if necessary
   if (gSystem->AccessPathName(sourceHtmlDir))
      gSystem->MakeDirectory(sourceHtmlDir);
   sourceHtmlFileName = fCurrentClass->GetName();
   NameSpace2FileName(sourceHtmlFileName);
   gSystem->PrependPathName(sourceHtmlDir, sourceHtmlFileName);
   sourceHtmlFileName += extension;
   out.open(sourceHtmlFileName);
   if (!out) {
      Warning("LocateMethodsInSource", "Can't open beautified source file '%s' for writing!", 
         sourceHtmlFileName.Data());
      sourceHtmlFileName.Remove(0);
      return;
   }

   // write a HTML header
   TString title(fCurrentClass->GetName());
   title += " - source file";
   WriteHtmlHeader(out, title, "../", fCurrentClass);
   out << "<pre class=\"code\">" << std::endl;
}

//______________________________________________________________________________
void THtml::AnchorFromLine(TString& anchor) {
   // Create an anchor from the given line, by hashing it and
   // convertig the hash into a custom base64 string.

   const char base64String[65] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_.";

   // use hash of line instead of e.g. line number.
   // advantages: more stable (lines can move around, we still find them back),
   // no need for keeping a line number context
   UInt_t hash = ::Hash(fLineStripped);
   anchor.Remove(0);
   while (hash) {
      anchor += base64String[hash % 64];
      hash /= 64;
   }
}

//______________________________________________________________________________
void THtml::BeautifyLine(std::ostream &sOut)
{
   // Put colors around tags, create links, escape characters.
   // In short: make a nice HTML page out of C++ code, and put it into srcOut.
   // Create an anchor at the beginning of the line, and put its name into
   // anchor, if set.

   enum EBeautifyContext {
      kNothingSpecialMoveOn,
      kCommentC, /* */
      kCommentCXX, //
      kPreProc,
      kDontTouch,
      kNumBeautifyContexts
   };
   EBeautifyContext context = kNothingSpecialMoveOn;


   switch (fParseContext) {
      case kCode:
         context = kNothingSpecialMoveOn;
         if (fLineStripped.Length() && fLineStripped[0] == '#') {
            context = kPreProc;
            sOut << "<span class=\"cpp\">";
            ExpandPpLine(sOut);
            sOut << "</span>" << std::endl;
            context = kNothingSpecialMoveOn;
            return;
         }
         break;
      case kBeginEndHtml:
      case kBeginEndHtmlInCComment:
         context = kDontTouch;
         break;
      case kCComment:
         context = kCommentC;
         break;
      default: ;
   }

   if (context == kDontTouch 
      || kNPOS != fLine.Index("End_Html", 0, TString::kIgnoreCase)
      && kNPOS == fLine.Index("\"End_Html", 0, TString::kIgnoreCase)) {
      ReplaceSpecialChars(sOut, fLine);
      sOut << std::endl;
      return;
   }

   TSubString stripSubExpanded = fLineExpanded.Strip(TString::kBoth);
   TString lineExpandedDotDot(stripSubExpanded);

   // adjust relative path
   lineExpandedDotDot.ReplaceAll("=\"./", "=\"../");
   if (stripSubExpanded.Start() > 0)
      sOut << fLineExpanded(0,stripSubExpanded.Start());
   for (Int_t i = 0; i < lineExpandedDotDot.Length(); ++i)
      switch (lineExpandedDotDot[i]) {
         case '/':
            if (lineExpandedDotDot.Length() > i + 1)
               if (lineExpandedDotDot[i + 1] == '/') {
                  if (context == kPreProc) {
                     // close preproc span
                     sOut << "</span>";
                     context = kNothingSpecialMoveOn;
                  }
                  sOut << lineExpandedDotDot.Data() + i;
                  i = lineExpandedDotDot.Length();
               } else if (lineExpandedDotDot[i + 1] == '*') {
                  if (context == kPreProc) {
                     // close preproc span
                     sOut << "</span>";
                     context = kNothingSpecialMoveOn;
                  }
                  if (context == kNothingSpecialMoveOn || context == kCommentC) {
                     context = kCommentC;
                     fParseContext = kCComment;
                     Ssiz_t posEndComment = lineExpandedDotDot.Index("*/", i);
                     if (posEndComment == kNPOS)
                        posEndComment = lineExpandedDotDot.Length();
                     TString comment(lineExpandedDotDot(i, posEndComment - i));
                     sOut << comment;
                     // leave "*/" fot next iteration
                     i = posEndComment - 1;
                  } else
                  sOut << "/";
               } else
                  sOut << "/";
            else
               sOut << "/";
            break;
         case '*':
            if (lineExpandedDotDot.Length() > i + 1 &&
               lineExpandedDotDot[i + 1] == '/' &&
               (context == kCommentC ||
               /*can happen if CPP comment inside C comment: */
               context == kNothingSpecialMoveOn)) {
                  sOut << "*/";
                  context = kNothingSpecialMoveOn;
                  fParseContext = kCode;
                  i += 1;
            }
            else sOut << "*";
            break;
         default:
            sOut << lineExpandedDotDot[i];
   }
   sOut << std::endl;
}

//______________________________________________________________________________
TMethod* THtml::LocateMethodInCurrentLine(Ssiz_t &posMethodName, TString& ret, TString& name, TString& params,
                             std::ostream &srcOut, TString &anchor, std::ifstream& sourceFile, Bool_t allowPureVirtual)
{
   // Search for a method starting at posMethodName, and return its return type, 
   // its name, and its arguments. If the end of arguments is not found in the 
   // current line, get a new line from sourceFile, beautify it to srcOut, creating
   // an anchor as necessary. When this function returns, posMethodName points to the
   // end of the function declaration, i.e. right after the arguments' closing bracket.
   // If posMethodName == kNPOS, we look for the first matching method in fMethodNames.

   if (posMethodName == kNPOS) {
      name.Remove(0);
      TMethod * meth = 0;
      Ssiz_t posBlock = fLine.Index('{');
      if (posBlock == kNPOS) 
         posBlock = fLine.Length();
      for (MethodNames_t::iterator iMethodName = fMethodNames.begin();
         !name.Length() && iMethodName != fMethodNames.end(); ++iMethodName) {
         TString lookFor(iMethodName->first);
         lookFor += "(";
         posMethodName = fLine.Index(lookFor);
         if (posMethodName != kNPOS && posMethodName < posBlock 
            && (posMethodName == 0 || !IsWord(fLine[posMethodName - 1]))) {
            meth = LocateMethodInCurrentLine(posMethodName, ret, name, params, srcOut, 
               anchor, sourceFile, allowPureVirtual);
            if (name.Length())
               return meth;
         }
      }
      return 0;
   }

   name = fLine(posMethodName, fLine.Length() - posMethodName);

   // extract return type
   ret = fLine(0, posMethodName);
   if (ret.Length()) {
      while (ret.Length() && (IsName(ret[ret.Length() - 1]) || ret[ret.Length()-1] == ':'))
         ret.Remove(ret.Length() - 1, 1);
      ret = ret.Strip(TString::kBoth);
      Bool_t didSomething = kTRUE;
      while (didSomething) {
         didSomething = kFALSE;
         if (ret.BeginsWith("inline ")) {
            didSomething = kTRUE;
            ret.Remove(0, 7);
         }
         if (ret.BeginsWith("static ")) {
            didSomething = kTRUE;
            ret.Remove(0, 7);
         }
         if (ret.BeginsWith("virtual ")) {
            didSomething = kTRUE;
            ret.Remove(0, 8);
         }
      } // while replacing static, virtual, inline
      ret = ret.Strip(TString::kBoth);
   }

   // extract parameters
   Ssiz_t posParam = name.First('(');
   if (posParam == kNPOS || 
      // no strange return types, please
      ret.Contains("{") || ret.Contains("}") || ret.Contains("(") || ret.Contains(")")) {
      ret.Remove(0);
      name.Remove(0);
      params.Remove(0);
      return 0;
   }

   if (name.BeginsWith("operator")) {
      // op () (...)
      Ssiz_t checkOpBracketParam = posParam + 1;
      while (isspace(name[checkOpBracketParam])) 
         ++checkOpBracketParam;
      if (name[checkOpBracketParam] == ')') {
         ++checkOpBracketParam;
         while (isspace(name[checkOpBracketParam]))
            ++checkOpBracketParam;
         if (name[checkOpBracketParam] == '(')
            posParam = checkOpBracketParam;
      }
   } // check for op () (...)

   if (posParam == kNPOS) {
      ret.Remove(0);
      name.Remove(0);
      params.Remove(0);
      return 0;
   }

   params = name(posParam, name.Length() - posParam);
   name.Remove(posParam);

   MethodNames_t::const_iterator iMethodName = fMethodNames.find(name.Data());
   if (iMethodName == fMethodNames.end() || iMethodName->second <= 0) {
      ret.Remove(0);
      name.Remove(0);
      params.Remove(0);
      return 0;
   }

   // find end of param
   Ssiz_t posParamEnd = 1;
   Int_t bracketLevel = 1;
   while (bracketLevel) {
      const char* paramEnd = strpbrk(params.Data() + posParamEnd, ")(\"'");
      if (!paramEnd) {
         // func with params over multiple lines
         // gotta write out this line before it gets lost
         if (!anchor.Length()) {
            // request an anchor, just in case...
            AnchorFromLine(anchor);
            if (srcOut)
               srcOut << "<a name=\"" << anchor << "\"></a>";
         }
         BeautifyLine(srcOut);

         fLine.ReadLine(sourceFile, kFALSE);
         if (sourceFile.eof()) {
            Error("LocateMethodInCurrentLine", 
               "Cannot find end of signature for function %s!",
               name.Data());
            break;
         }

         // replace class names etc
         fLineExpanded = fLine;
         ExpandKeywords(fLineExpanded);
         posParamEnd = params.Length();
         params += fLine;
      } else
         posParamEnd = paramEnd - params.Data();
      switch (params[posParamEnd]) {
         case '(': ++bracketLevel; ++posParamEnd; break;
         case ')': --bracketLevel; ++posParamEnd; break;
         case '"': // skip ")"
            ++posParamEnd;
            while (params.Length() > posParamEnd && params[posParamEnd] != '"') {
               // skip '\"'
               if (params[posParamEnd] == '\\') ++posParamEnd;
               ++posParamEnd;
            }
            if (params.Length() <= posParamEnd) {
               // something is seriously wrong - skip :-/
               ret.Remove(0);
               name.Remove(0);
               params.Remove(0);
               return 0;
            }
            ++posParamEnd; // skip trailing '"'
            break;
         case '\'': // skip ')'
            ++posParamEnd;
            if (params[posParamEnd] == '\\') ++posParamEnd;
            posParamEnd += 2;
            break;
         default:
            ++posParamEnd;
      }
   } // while bracketlevel, i.e. (...(..)...)
   Ssiz_t posBlock     = params.Index('{', posParamEnd);
   Ssiz_t posSemicolon = params.Index(';', posParamEnd);
   Ssiz_t posPureVirt  = params.Index('=', posParamEnd);
   if (posSemicolon != kNPOS)
      if ((posBlock == kNPOS || (posSemicolon < posBlock)) &&
         (posPureVirt == kNPOS || !allowPureVirtual)
         && !allowPureVirtual) // allow any "func();" if pv is allowed
         params.Remove(0);

   if (params.Length())
      params.Remove(posParamEnd);

   if (!params.Length()) {
      ret.Remove(0);
      name.Remove(0);
      return 0;
   }
   // update posMethodName to point behind the method
   posMethodName = posParam + posParamEnd;
   if (fCurrentClass) 
      return fCurrentClass->GetMethodAny(name);

   return 0;
}

//______________________________________________________________________________
void THtml::WriteMethod(std::ostream & out, TString& ret, 
                        TString& name, TString& params,
                        const char* filename, TString& anchor,
                        TString& comment, TString& codeOneLiner)
{
   // Write method name with return type ret and parameters param to out.
   // Build a link using file and anchor. Cooment it with comment, and
   // show the code codeOneLiner (set if the func consists of only one line
   // of code, immediately surrounded by "{","}"). Also updates fMethodNames's
   // count of method names.

   ExpandKeywords(ret);
   ExpandKeywords(params);
   out << "<div class=\"funcdoc\"><span class=\"funcname\">";
   out << ret << " <a name=\"";
   TString mangled(fCurrentClass->GetName());
   NameSpace2FileName(mangled);
   out << mangled << ":";
   mangled = name;
   NameSpace2FileName(mangled);
   out << mangled << "\" href=\"src/" << filename;
   if (anchor.Length())
      out << "#" << anchor;
   out << "\">";
   ReplaceSpecialChars(out, name);
   out << "</a>" << params << "</span><br />" << std::endl;

   out << "<pre>" << comment << "</pre>" << std::endl;

   if (codeOneLiner.Length()) {
      out << std::endl << "<div class=\"inlinecode\"><code class=\"inlinecode\">" 
          << codeOneLiner << "</code></div>" << std::endl;
      codeOneLiner.Remove(0);
   }
   out << "</div>" << std::endl;

   MethodNames_t::iterator iMethodName = fMethodNames.find(name.Data());
   if (iMethodName != fMethodNames.end()) {
      --(iMethodName->second);
      if (iMethodName->second <= 0)
         fMethodNames.erase(iMethodName);
   }

   ret.Remove(0);
   name.Remove(0);
   params.Remove(0);
   anchor.Remove(0);
   comment.Remove(0);

   fDocContext = kIgnore;
}


//______________________________________________________________________________
Bool_t THtml::ExtractComments(const TString &lineExpandedStripped, 
                              Bool_t &foundClassDescription,
                              const char* classDescrTag, 
                              TString& comment) {
// Extracts comments from the current line into comment, 
// updating whether a class description was found (foundClassDescription).
// Returns kTRUE if comment found.

   //if (fParseContext != kCComment && fParseContext != kBeginEndHtml &&
   //   fParseContext != kBeginEndHtmlInCComment) 
   //   return kFALSE;
   if (fParseContext != kBeginEndHtml &&
      fParseContext != kBeginEndHtmlInCComment &&
      fParseContext != kCComment && 
      !lineExpandedStripped.BeginsWith("<span class=\"comment\">"))
      return kFALSE;

   TString commentLine(lineExpandedStripped);
   if (commentLine.BeginsWith("<span class=\"comment\">/")) {
      if (commentLine[23] == '/')
         if (!commentLine.EndsWith("</span>"))
            return kFALSE;
         else 
            commentLine.Remove(commentLine.Length()-7, 7);
      commentLine.Remove(0,22);
   }

   // remove repeating characters from the end of the line
   if (!foundClassDescription && commentLine.Length() > 3) {
      TString lineAllOneChar(commentLine);

      Ssiz_t len = lineAllOneChar.Length();
      Char_t c = lineAllOneChar[len - 1];
      // also a class doc signature: line consists of same char
      if (c == lineAllOneChar[len - 2] && c == lineAllOneChar[len - 3]) {
         TString lineAllOneCharStripped(lineAllOneChar.Strip(TString::kTrailing, c));
         if (lineAllOneCharStripped.BeginsWith("//") || lineAllOneCharStripped.BeginsWith("/*"))
            lineAllOneCharStripped.Remove(0, 2);
         lineAllOneCharStripped.Strip(TString::kBoth);
         if (!lineAllOneCharStripped.Length())
            commentLine.Remove(0);
      }
   }

   // look for start tag of class description
   if (!foundClassDescription && !comment.Length() && fDocContext == kIgnore &&
      (!commentLine.Length() || classDescrTag && lineExpandedStripped.Contains(classDescrTag))) {
      fDocContext = kDocClass;
      foundClassDescription = kTRUE;
   }

   // remove leading and trailing chars if non-word and identical, e.g.
   // * some doc *, or // some doc //
   while (fParseContext != kBeginEndHtml && fParseContext != kBeginEndHtmlInCComment && 
      commentLine.Length() > 2 &&
      commentLine[0] == commentLine[commentLine.Length() - 1] &&
      (commentLine[0] == '/' || commentLine[0] == '*')) {
      commentLine = commentLine.Strip(TString::kBoth, commentLine[0]);
   }

   // remove leading /*, //
   if (commentLine.Length()>1 && commentLine[0] == '/' && 
      (commentLine[1] == '/' || commentLine[1] == '*'))
      commentLine.Remove(0, 2);
   // remove trailing */
   if (commentLine.Length()>1 && commentLine[commentLine.Length() - 2] == '*' && 
      commentLine[commentLine.Length() - 1] == '/')
      commentLine.Remove(commentLine.Length()-2);

   comment += commentLine + "\n";

   return kTRUE;
} 

      
//______________________________________________________________________________
void THtml::LocateMethods(std::ofstream & out, const char* filename,
                          Bool_t lookForSourceInfo /*= kTRUE*/, 
                          Bool_t useDocxxStyle /*= kFALSE*/, 
                          Bool_t lookForClassDescr /*= kTRUE*/,
                          Bool_t allowPureVirtual /*= kFALSE*/,
                          const char* methodPattern /*= 0*/, 
                          const char* sourceExt /*= 0 */)
{
   // Collect methods from the source or header file called filename.
   // It generates a beautified version of the source file on the fly;
   // the output file is given by the fCurrentClass's name, and sourceExt.
   // Documentation is extracted to out.
   //   lookForSourceInfo: if set, author, lastUpdate, and copyright are 
   //     extracted (i.e. the values contained in fSourceInfo)
   //   useDocxxStyle: if set, documentation can be in front of the method
   //     name, not only inside the method. Useful doc Doc++/Doxygen style,
   //     and inline methods.
   //   lookForClassDescr: if set, the first line matching the class description 
   //     rules is assumed to be the class description for fCurrentClass; the 
   //     description is written to out.
   //   methodPattern: if set, methods have to be prepended by this tag. Usually
   //     the class name + "::". In header files, looking for in-place function
   //     definitions, this should be 0. In that case, only functions in 
   //     fMethodNames are searched for.

   TString sourceFileName(filename);
   GetSourceFileName(sourceFileName);
   if (!sourceFileName.Length()) {
      Error("LocateMethods", "Can't find source file '%s' for class %s!", 
         GetImplFileName(fCurrentClass), fCurrentClass->GetName());
      return;
   }
   ifstream sourceFile(sourceFileName.Data());
   if (!sourceFile || !sourceFile.good()) {
      Error("LocateMethods", "Can't open file '%s' for reading!", sourceFileName.Data());
      return;
   }

   // get environment variables
   const char *sourceInfoTags[kNumSourceInfos];
   sourceInfoTags[kInfoLastUpdate] = gEnv->GetValue("Root.Html.LastUpdate", "// @(#)");
   sourceInfoTags[kInfoAuthor]     = gEnv->GetValue("Root.Html.Author", "// Author:");
   sourceInfoTags[kInfoCopyright]  = gEnv->GetValue("Root.Html.Copyright", " * Copyright");

   const char *descriptionStr =
       gEnv->GetValue("Root.Html.Description", "//____________________");

   Bool_t foundClassDescription = !lookForClassDescr;

   TString pattern(methodPattern);

   TString prevComment;
   TString codeOneLiner;
   TString methodRet;
   TString methodName;
   TString methodParam;
   TString anchor;

   Bool_t wroteMethodNowWaitingForOpenBlock = kFALSE;

   ofstream srcHtmlOut;
   TString srcHtmlOutName;
   if (sourceExt && sourceExt[0])
      CreateSourceOutputStream(srcHtmlOut, sourceExt, srcHtmlOutName);
   else {
      sourceExt = 0;
      srcHtmlOutName = fCurrentClass->GetName();
      NameSpace2FileName(srcHtmlOutName);
      gSystem->PrependPathName("src", srcHtmlOutName);
      srcHtmlOutName += ".h.html";
   }

   fParseContext = kCode;
   fDocContext = kIgnore;
   fLineNo = 0;

   while (!sourceFile.eof()) {
      Bool_t needAnchor = kFALSE;

      ++fLineNo; // we count fortrany

      fLine.ReadLine(sourceFile, kFALSE);
      if (sourceFile.eof()) break;

      // replace class names etc
      fLineExpanded = fLine;
      ExpandKeywords(fLineExpanded);
      fLineStripped = fLine.Strip(TString::kBoth);

      // remove leading and trailing spaces
      TString lineExpandedStripped(fLineExpanded.Strip(TString::kBoth));

      if (!ExtractComments(lineExpandedStripped, foundClassDescription, 
                           descriptionStr, prevComment)) {
         // not a commented line

         // write previous method
         if (methodName.Length() && !wroteMethodNowWaitingForOpenBlock) {
            WriteMethod(out, methodRet, methodName, methodParam, 
               gSystem->BaseName(srcHtmlOutName), anchor, 
               prevComment, codeOneLiner);
         } else if (fDocContext == kDocClass) {
            // write class description
            out << "<pre>" << prevComment << "</pre></div>" << std::endl;
            prevComment.Remove(0);
            fDocContext = kIgnore;
         }

         if (!wroteMethodNowWaitingForOpenBlock) {
            // check for method
            Ssiz_t posPattern = pattern.Length() ? fLine.Index(pattern) : kNPOS;
            if (posPattern != kNPOS || !pattern.Length()) {
               posPattern += pattern.Length();
               LocateMethodInCurrentLine(posPattern, methodRet, methodName, 
                  methodParam, srcHtmlOut, anchor, sourceFile, allowPureVirtual);
               if (methodName.Length()) {
                  fDocContext = kDocFunc;
                  needAnchor = !anchor.Length();
                  if (!useDocxxStyle)
                     prevComment.Remove(0);
                  codeOneLiner.Remove(0);

                  wroteMethodNowWaitingForOpenBlock = fLine.Index("{", posPattern) == kNPOS;
                  wroteMethodNowWaitingForOpenBlock &= fLine.Index(";", posPattern) == kNPOS;
               } else 
                  prevComment.Remove(0);
            } // pattern matches - could be a method
            else 
               prevComment.Remove(0);
         } else {
            wroteMethodNowWaitingForOpenBlock &= fLine.Index("{") == kNPOS;
            wroteMethodNowWaitingForOpenBlock &= fLine.Index(";") == kNPOS;
         } // if !wroteMethodNowWaitingForOpenBlock

         if (methodName.Length() && !wroteMethodNowWaitingForOpenBlock) {
            // make sure we don't have more '{' in lineExpandedStripped than in fLine
            if (!codeOneLiner.Length() &&
                lineExpandedStripped.CountChar('{') == 1 && 
                lineExpandedStripped.CountChar('}') == 1) {
               // a one-liner
               codeOneLiner = lineExpandedStripped;
               codeOneLiner.Remove(0, codeOneLiner.Index('{'));
               codeOneLiner.Remove(codeOneLiner.Index('}') + 1);
            }
         } // if method name and '{'
      } // if comment

      // check for last update,...
      Ssiz_t posTag = kNPOS;
      if (lookForSourceInfo)
         for (Int_t si = 0; si < (Int_t) kNumSourceInfos; ++si)
            if (!fSourceInfo[si].Length() && (posTag = fLine.Index(sourceInfoTags[si])) != kNPOS)
               fSourceInfo[si] = fLine(posTag + strlen(sourceInfoTags[si]), fLine.Length() - posTag);

      if (needAnchor || fExtraLinesWithAnchor.find(fLineNo) != fExtraLinesWithAnchor.end()) {
         AnchorFromLine(anchor);
         if (sourceExt)
            srcHtmlOut << "<a name=\"" << anchor << "\"></a>";
      }
      // else anchor.Remove(0); - NO! WriteMethod will need it later!

      // write to .cxx.html
      if (sourceExt)
         BeautifyLine(srcHtmlOut);
      else if (needAnchor)
         fExtraLinesWithAnchor.insert(fLineNo);
   } // while !sourceFile.eof()

   // deal with last func
   if (methodName.Length()) {
      WriteMethod(out, methodRet, methodName, methodParam, 
         gSystem->BaseName(srcHtmlOutName), anchor, 
         prevComment, codeOneLiner);
   } else if (fDocContext == kDocClass) {
      // write class description
      out << prevComment << "</div>" << std::endl;
   } else if (!foundClassDescription && lookForClassDescr)
      out << "</div>" << std::endl;

   srcHtmlOut << "</pre>" << std::endl;
   WriteHtmlFooter(srcHtmlOut, "../");

   fParseContext = kCode;
   fDocContext = kIgnore;
}

//______________________________________________________________________________
void THtml::LocateMethodsInSource(ofstream & out)
{
   // Given fCurrentClass, look for methods in its source file, 
   // and extract documentation to out, while beautifying the source 
   // file in parallel.

   // for Doc++ style
   const char* docxxEnv = gEnv->GetValue("Root.Html.DescriptionStyle", "");
   Bool_t useDocxxStyle = (strcmp(docxxEnv, "Doc++") == 0);

   TString pattern(fCurrentClass->GetName());
   // take unscoped version
   Ssiz_t posLastScope = kNPOS;
   while ((posLastScope = pattern.Index("::")) != kNPOS)
      pattern.Remove(0, posLastScope + 1);
   pattern += "::";
   
   const char* implFileName = GetImplFileName(fCurrentClass);
   if (implFileName && implFileName[0])
      LocateMethods(out, implFileName, kTRUE, useDocxxStyle, kTRUE, 
         kFALSE, pattern, ".cxx.html");
   else out << "</div>" << endl; // close class descr div
}

//______________________________________________________________________________
void THtml::LocateMethodsInHeaderInline(ofstream & out)
{
   // Given fCurrentClass, look for methods in its header file, 
   // and extract documentation to out.

   // for inline methods, always allow doc before func
   Bool_t useDocxxStyle = kTRUE; 

   TString pattern(fCurrentClass->GetName());
   // take unscoped version
   Ssiz_t posLastScope = kNPOS;
   while ((posLastScope = pattern.Index("::")) != kNPOS)
      pattern.Remove(0, posLastScope + 1);
   pattern += "::";
   
   const char* declFileName = GetDeclFileName(fCurrentClass);
   if (declFileName && declFileName[0])
      LocateMethods(out, declFileName, kFALSE, useDocxxStyle, kFALSE, 
         kFALSE, pattern, 0);
}

//______________________________________________________________________________
void THtml::LocateMethodsInHeaderClassDecl(ofstream & out)
{
   // Given fCurrentClass, look for methods in its header file's
   // class declaration block, and extract documentation to out,
   // while beautifying the header file in parallel.

   const char* declFileName = GetDeclFileName(fCurrentClass);
   if (declFileName && declFileName[0])
      LocateMethods(out, declFileName, kFALSE, kTRUE, kFALSE, kTRUE, 0, ".h.html");
}

//______________________________________________________________________________
void THtml::ClassDescription(ofstream & out)
{
// This function builds the description of the class
//
//
// Input: out      - output file stream
//


   // Class Description Title
   out << "<hr />" << endl;
   out << "<!--DESCRIPTION-->";
   out << "<div class=\"classdescr\">";
   out << "<h2><a name=\"" << fCurrentClass->GetName();
   out << ":description\">Class Description</a></h2>" << endl;

   // create an array of method names
   TMethod *method;
   TIter nextMethod(fCurrentClass->GetListOfMethods());
   fMethodNames.clear();
   while ((method = (TMethod *) nextMethod())) {
      ++fMethodNames[method->GetName()];
   }

   for (Int_t si = 0; si < (Int_t) kNumSourceInfos; ++si)
      fSourceInfo[si].Remove(0);

   LocateMethodsInSource(out);
   LocateMethodsInHeaderInline(out);
   LocateMethodsInHeaderClassDecl(out);

   // write classFile footer
   TDatime date;
   if (!fSourceInfo[kInfoLastUpdate].Length())
      fSourceInfo[kInfoLastUpdate] = date.AsString();
   WriteHtmlFooter(out, "", fSourceInfo[kInfoLastUpdate],
      fSourceInfo[kInfoAuthor], fSourceInfo[kInfoCopyright]);
}

//______________________________________________________________________________
void THtml::ClassHtmlTree(ofstream & out, TClass * classPtr,
                          ETraverse dir, int depth)
{
// This function builds the class tree for one class in HTML
// (inherited and succeeding classes, called recursively)
//
//
// Input: out      - output file stream
//        classPtr - pointer to the class
//        dir      - direction to traverse tree: up, down or both
//

   if (dir == kBoth) {
      out << "<!--INHERITANCE TREE-->" << endl;

      // draw class tree into nested tables recursively
      out << "<table><tr><td width=\"10%\"></td><td width=\"70%\">Inheritance Chart:</td></tr>";
      out << "<tr class=\"inhtree\"><td width=\"10%\"></td><td width=\"70%\">";

      out << "<table class=\"inhtree\" width=\"100%\"><tr><td>" << endl;
      out << "<table width=\"100%\" border=\"0\" ";
      out << "cellpadding =\"0\" cellspacing=\"2\"><tr>" << endl;
   } else {
      out << "<table><tr>";
   }

   ////////////////////////////////////////////////////////
   // Loop up to mother classes
   if (dir == kUp || dir == kBoth) {

      // make a loop on base classes
      TBaseClass *inheritFrom;
      TIter nextBase(classPtr->GetListOfBases());

      UInt_t bgcolor=255-depth*8;
      Bool_t first = kTRUE;
      while ((inheritFrom = (TBaseClass *) nextBase())) {

         if (first) {
            out << "<td><table><tr>" << endl;
            first = kFALSE;
         } else
            out << "</tr><tr>" << endl;
         out << "<td bgcolor=\""
            << Form("#%02x%02x%02x", bgcolor, bgcolor, bgcolor)
            << "\" align=\"right\">" << endl;
         // get a class
         TClass *classInh = GetClass((const char *) inheritFrom->GetName());
         if (classInh)
            ClassHtmlTree(out, classInh, kUp, depth+1);
         else
            out << "<tt>"
                << (const char *) inheritFrom->GetName()
                << "</tt>";
         out << "</td>"<< endl;
      }
      if (!first) {
         out << "</tr></table></td>" << endl; // put it in additional row in table
         out << "<td>&lt;-</td>";
      }
   }

   out << "<td>" << endl; // put it in additional row in table
   ////////////////////////////////////////////////////////
   // Output Class Name

   const char *className = classPtr->GetName();
   TString htmlFile;
   GetHtmlFileName(classPtr, htmlFile);

   if (dir == kUp) {
      if (htmlFile) {
         out << "<center><tt><a name=\"" << className;
         out << "\" href=\"" << htmlFile << "\">";
         ReplaceSpecialChars(out, className);
         out << "</a></tt></center>" << endl;
      } else
         ReplaceSpecialChars(out, className);
   }

   if (dir == kBoth) {
      if (htmlFile.Length()) {
         out << "<center><big><b><tt><a name=\"" << className;
         out << "\" href=\"" << htmlFile << "\">";
         ReplaceSpecialChars(out, className);
         out << "</a></tt></b></big></center>" << endl;
      } else
         ReplaceSpecialChars(out, className);
   }

   out << "</td>" << endl; // put it in additional row in table

   ////////////////////////////////////////////////////////
   // Loop down to child classes

   if (dir == kDown || dir == kBoth) {

      // 1. make a list of class names
      // 2. use DescendHierarchy

      out << "<td><table><tr>" << endl;
      fHierarchyLines = 0;
      DescendHierarchy(out,classPtr,fClassNames,fNumberOfClasses,10);

      out << "</tr></table>";
      if (dir==kBoth && fHierarchyLines>=10)
         out << "</td><td align=\"left\">&nbsp;<a href=\"ClassHierarchy.html\">[more...]</a>";
      out<<"</td>" << endl;

      // free allocated memory
   }

   out << "</tr></table>" << endl;
   if (dir == kBoth)
      out << "</td></tr></table></td></tr></table>"<<endl;
}


//______________________________________________________________________________
void THtml::ClassTree(TVirtualPad * psCanvas, TClass * classPtr,
                      Bool_t force)
{
// It makes a graphical class tree
//
//
// Input: psCanvas - pointer to the current canvas
//        classPtr - pointer to the class
//

   if (psCanvas && classPtr) {
      TString filename(classPtr->GetName());
      NameSpace2FileName(filename);

      gSystem->ExpandPathName(fOutputDir);
      gSystem->PrependPathName(fOutputDir, filename);


      filename += "_Tree.pdf";

      if (IsModified(classPtr, kTree) || force) {
         // TCanvas already prints pdf being saved
         // Printf(formatStr, "", "", filename);
         classPtr->Draw("same");
         psCanvas->SaveAs(filename);
      } else
         Printf(formatStr, "-no change-", "", filename.Data());
   }
}


//______________________________________________________________________________
void THtml::Convert(const char *filename, const char *title,
                    const char *dirname)
{
// It converts a single text file to HTML
//
//
// Input: filename - name of the file to convert
//        title    - title which will be placed at the top of the HTML file
//        dirname  - optional parameter, if it's not specified, output will
//                   be placed in html/examples directory.
//
//  NOTE: Output file name is the same as filename, but with extension .html
//

   gROOT->GetListOfGlobals(kTRUE);        // force update of this list
   CreateListOfClasses("*");

   const char *dir;
   Bool_t isCommentedLine = kFALSE;

   // if it's not defined, make the "examples" as a default directory
   if (!*dirname) {
      gSystem->ExpandPathName(fOutputDir);
      dir = gSystem->ConcatFileName(fOutputDir, "examples");

      // create directory if necessary
      if (gSystem->AccessPathName(dir))
         gSystem->MakeDirectory(dir);
   } else
      dir = dirname;


   // find a file
   char *realFilename =
       gSystem->Which(fSourceDir, filename, kReadPermission);

   if (realFilename) {

      // open source file
      ifstream sourceFile;
      sourceFile.open(realFilename, ios::in);

      delete[]realFilename;
      realFilename = 0;

      if (sourceFile.good()) {

         // open temp file with extension '.html'
         if (!gSystem->AccessPathName(dir)) {
            char *tmp1 =
                gSystem->ConcatFileName(dir, GetFileName(filename));
            char *htmlFilename = StrDup(tmp1, 16);
            strcat(htmlFilename, ".html");

            if (tmp1)
               delete[]tmp1;
            tmp1 = 0;

            ofstream tempFile;
            tempFile.open(htmlFilename, ios::out);

            if (tempFile.good()) {

               Printf("Convert: %s", htmlFilename);

               // write a HTML header
               WriteHtmlHeader(tempFile, title);

               tempFile << "<h1>" << title << "</h1>" << endl;
               tempFile << "<pre>" << endl;

               while (!sourceFile.eof()) {
                  fLine.ReadLine(sourceFile, kFALSE);
                  if (sourceFile.eof())
                     break;


                  // remove leading spaces
                  fLine = fLine.Strip(TString::kBoth);

                  // check for a commented line
                  isCommentedLine = fLine.BeginsWith("//");

                  // write to a '.html' file
                  if (isCommentedLine)
                     tempFile << "<b>";
                  ExpandKeywords(fLine);
                  fLine.ReplaceAll("=\"./", "=\"../"); // adjust rel path
                  tempFile << fLine;
                  if (isCommentedLine)
                     tempFile << "</b>";
                  tempFile << endl;
               }
               tempFile << "</pre>" << endl;


               // write a HTML footer
               WriteHtmlFooter(tempFile, "../");


               // close a temp file
               tempFile.close();

            } else
               Error("Convert", "Can't open file '%s' !", htmlFilename);

            // close a source file
            sourceFile.close();
            if (htmlFilename)
               delete[]htmlFilename;
            htmlFilename = 0;
         } else
            Error("Convert",
                  "Directory '%s' doesn't exist, or it's write protected !",
                  dir);
      } else
         Error("Convert", "Can't open file '%s' !", realFilename);
   } else
      Error("Convert", "Can't find file '%s' !", filename);
}


//______________________________________________________________________________
Bool_t THtml::CopyHtmlFile(const char *sourceName, const char *destName)
{
// Copy file to HTML directory
//
//
//  Input: sourceName - source file name
//         destName   - optional destination name, if not
//                      specified it would be the same
//                      as the source file name
//
// Output: TRUE if file is successfully copied, or
//         FALSE if it's not
//
//
//   NOTE: The destination directory is always fOutputDir
//

   Bool_t ret = kFALSE;
   Int_t check = 0;

   // source file name
   char *tmp1 = gSystem->Which(fSourceDir, sourceName, kReadPermission);
   char *sourceFile = StrDup(tmp1, 16);

   if (tmp1)
      delete[]tmp1;
   tmp1 = 0;

   if (sourceFile) {

      // destination file name
      char *tmpstr = 0;
      if (!*destName)
         tmpstr = StrDup(GetFileName(sourceFile), 16);
      else
         tmpstr = StrDup(GetFileName(destName), 16);
      destName = tmpstr;

      gSystem->ExpandPathName(fOutputDir);
      tmp1 = gSystem->ConcatFileName(fOutputDir, destName);
      char *filename = StrDup(tmp1, 16);

      if (tmp1)
         delete[]tmp1;
      tmp1 = 0;

      // Get info about a file
      Long64_t sSize, dSize;
      Long_t sId, sFlags, sModtime;
      Long_t dId, dFlags, dModtime;
      sModtime = 0;
      dModtime = 0;
      if (!(check =
           gSystem->GetPathInfo(sourceFile, &sId, &sSize, &sFlags,
                                &sModtime)))
         check = gSystem->GetPathInfo(filename, &dId, &dSize, &dFlags,
                                  &dModtime);

      if ((sModtime != dModtime) || check)
         gSystem->CopyFile(sourceFile, filename, kTRUE);

      delete[]filename;
      delete[]tmpstr;
      delete[]sourceFile;
   } else
      Error("Copy", "Can't copy file '%s' to '%s' directory !", sourceName,
            fOutputDir.Data());

   return (ret);
}



//______________________________________________________________________________
void THtml::CreateIndex(const char **classNames, Int_t numberOfClasses)
{
// Create an index
//
//
// Input: classNames      - pointer to an array of class names
//        numberOfClasses - number of elements
//

   Int_t i = 0;

   gSystem->ExpandPathName(fOutputDir);
   char *tmp1 = gSystem->ConcatFileName(fOutputDir, "ClassIndex.html");
   char *filename = StrDup(tmp1);

   if (tmp1)
      delete[]tmp1;
   tmp1 = 0;

   // create CSS file, we need it
   CreateStyleSheet();

   // open indexFile file
   ofstream indexFile;
   indexFile.open(filename, ios::out);

   if (indexFile.good()) {

      Printf(formatStr, "", fCounter.Data(), filename);

      // write indexFile header
      WriteHtmlHeader(indexFile, "Class Index");

      indexFile << "<h1>Index</h1>" << endl;

      if (fModules.size()) {
         indexFile << "<div id=\"indxModules\"><h4>Modules</h4>" << endl;
         // find index chars
         sort_strlist_stricmp(fModules);
         for (std::list<std::string>::iterator iModule = fModules.begin(); 
            iModule != fModules.end(); ++iModule) {
            indexFile << "<a href=\"" << *iModule << "_Index.html\">" << *iModule << "</a>" << endl;
         }
         indexFile << "</div><br />" << endl;
      }

      std::vector<std::string> indexChars;
      if (numberOfClasses > 10) {
         indexFile << "<div id=\"indxShortX\"><h4>Jump to</h4>" << endl;
         // find index chars
         GetIndexChars(classNames, (Int_t)numberOfClasses, 50 /*sections*/, indexChars);
         for (UInt_t iIdxEntry = 0; iIdxEntry < indexChars.size(); ++iIdxEntry) {
            indexFile << "<a href=\"#idx" << iIdxEntry << "\">" << indexChars[iIdxEntry] 
                      << "</a>" << endl;
         }
         indexFile << "</div><br />" << endl;
      }

      // check for a search engine
      const char *searchEngine =
          gEnv->GetValue("Root.Html.SearchEngine", "");

      // if exists ...
      if (*searchEngine) {

         // create link to search engine page
         indexFile << "<h2><a href=\"" << searchEngine
             << "\">Search the Class Reference Guide</a></h2>" << endl;

      } else {
         const char *searchCmd =
             gEnv->GetValue("Root.Html.Search", "");

         //e.g. searchCmd = "http://www.google.com/search?q=%s+site%3Aroot.cern.ch%2Froot%2Fhtml";
         // if exists ...
         if (*searchCmd) {
            // create link to search engine page
            indexFile << "<script language=\"javascript\">" << endl
               << "function onSearch() {" << endl
               << "var s='" << searchCmd <<"';" << endl
               << "window.location.href=s.replace(/%s/ig,escape(document.searchform.t.value));" << endl
               << "return false;}" << endl
               << "</script><form action=\"javascript:onSearch();\" id=\"searchform\" name=\"searchform\" onsubmit=\"return onSearch()\">" << endl
               << "<input name=\"t\" value=\"Search documentation...\"  onfocus=\"if (document.searchform.t.value=='Search documentation...') document.searchform.t.value='';\"></input>" << endl
               << "<button type=\"submit\">Search</button></form>" << endl;
         }
      }

      indexFile << "<ul id=\"indx\">" << endl;

      // loop on all classes
      UInt_t currentIndexEntry = 0;
      for (i = 0; i < numberOfClasses; i++) {
         // get class
         fCurrentClass = GetClass((const char *) classNames[i]);
         if (fCurrentClass == 0) {
            Warning("THtml::CreateIndex", "skipping class %s\n", classNames[i]);
            continue;
         }

         indexFile << "<li class=\"idxl" << i%2 << "\"><tt>";
         if (currentIndexEntry < indexChars.size()
            && !strncmp(indexChars[currentIndexEntry].c_str(), classNames[i], 
                        indexChars[currentIndexEntry].length()))
            indexFile << "<a name=\"idx" << currentIndexEntry++ << "\"></a>" << endl;

         TString htmlFile;
         GetHtmlFileName(fCurrentClass, htmlFile);
         if (htmlFile.Length()) {
            indexFile << "<a name=\"";
            indexFile << classNames[i];
            indexFile << "\" href=\"";
            indexFile << htmlFile;
            indexFile << "\">";
            ReplaceSpecialChars(indexFile, classNames[i]);
            indexFile << "</a>";
         } else
            ReplaceSpecialChars(indexFile, classNames[i]);


         // write title
         indexFile << "</tt>";

         indexFile << "<a name=\"Title:";
         indexFile << fCurrentClass->GetName();
         indexFile << "\"></a>";
         ReplaceSpecialChars(indexFile, fCurrentClass->GetTitle());
         indexFile << "</li>" << endl;
      }

      indexFile << "</ul>" << endl;

      // write indexFile footer
      TDatime date;
      WriteHtmlFooter(indexFile, "", date.AsString());


      // close file
      indexFile.close();

   } else
      Error("MakeIndex", "Can't open file '%s' !", filename);

   if (filename)
      delete[]filename;
   fCurrentClass = 0;
}


//______________________________________________________________________________
void THtml::CreateIndexByTopic(char **fileNames, Int_t numberOfNames)
{
// It creates several index files
//
//
// Input: fileNames     - pointer to an array of file names
//        numberOfNames - number of elements in the fileNames array
//        maxLen        - maximum length of a single name
//

   ofstream outputFile;
   char *filename = 0;
   Int_t i;
   UInt_t currentIndexEntry = 0;
   Int_t firstIdxEntry = 0;
   std::vector<std::string> indexChars;
   fModules.clear();


   for (i = 0; i < numberOfNames; i++) {
      if (!filename) {

         // create a filename
         gSystem->ExpandPathName(fOutputDir);
         char *tmp1 = gSystem->ConcatFileName(fOutputDir, fileNames[i]);
         filename = StrDup(tmp1, 16);

         if (tmp1)
            delete[]tmp1;
         tmp1 = 0;

         // look for first _ in basename
         char *underlinePtr = strrchr(filename, '/');
         if (!underlinePtr)
            underlinePtr=strchr(filename,'_');
            else underlinePtr=strchr(underlinePtr,'_');
         *underlinePtr = 0;

         char modulename[1024];
         strcpy(modulename, GetFileName(filename));

         char htmltitle[1024];
         strcpy(htmltitle, "Index of ");
         strcat(htmltitle, modulename);
         strcat(htmltitle, " classes");

         strcat(filename, "_Index.html");

         // open a file
         outputFile.open(filename, ios::out);

         // check if it's OK
         if (outputFile.good()) {
            fModules.push_back(modulename);
            Printf(formatStr, "", fCounter.Data(), filename);

            // write outputFile header
            WriteHtmlHeader(outputFile, htmltitle);
            outputFile << "<h2>" << htmltitle << "</h2>" << endl;

            std::list<std::string> classNames;
            // add classes until end of module
            char *classname = strrchr(fileNames[i], '/');
            if (!classname)
               classname=strchr(fileNames[i],'_');
            else classname=strchr(classname,'_');

            for (int j=i; j < numberOfNames;) {
               classNames.push_back(classname + 1);

               // first base name
               // look for first _ in basename
               char *first = strrchr(fileNames[j], '/');
               if (!first)
                  first=strchr(fileNames[j],'_');
                  else first=strchr(first,'_');
               if (first)
                  *first = 0;

               // second base name
               char *second = 0;
               if (j < (numberOfNames - 1)) {
                  second = strrchr(fileNames[j + 1], '/');
                  if (!second)
                     second=strchr(fileNames[j + 1],'_');
                     else second=strchr(second,'_');
                  if (second)
                     *second = 0;
               }
               // check and close the file if necessary
               Bool_t nextDiffers = (!first || !second || strcmp(fileNames[j], fileNames[j + 1]));
               if (first)
                  *first = '_';
               if (second)
                  *second = '_';
               if (nextDiffers) break;

               ++j;
               classname = strrchr(fileNames[j], '/');
               if (!classname)
                  classname=strchr(fileNames[j],'_');
               else classname=strchr(classname,'_');
            }

            if (classNames.size() > 10) {
               outputFile << "<div id=\"indxShortX\"><h4>Jump to</h4>" << endl;
               UInt_t numSections = classNames.size() / 10;
               if (numSections < 10) numSections = 10;
               if (numSections > 50) numSections = 50;
               // find index chars
               GetIndexChars(classNames, numSections, indexChars);
               for (UInt_t iIdxEntry = 0; iIdxEntry < indexChars.size(); ++iIdxEntry) {
                  outputFile << "<a href=\"#idx" << iIdxEntry << "\">" << indexChars[iIdxEntry] 
                             << "</a>" << endl;
               }
               outputFile << "</div><br />" << endl;
            }
            outputFile << "<ul id=\"indx\">" << endl;
            currentIndexEntry = 0;

         } else
            Error("MakeIndex", "Can't open file '%s' !", filename);
         delete[]filename;
         firstIdxEntry = i;
      }
      // get a class
      char *classname = strrchr(fileNames[i], '/');
      if (!classname)
         classname=strchr(fileNames[i],'_');
         else classname=strchr(classname,'_');
      TClass *classPtr =
          GetClass((const char *) classname + 1);
      if (classPtr) {

         // write a classname to an index file
         outputFile << "<li class=\"idxl" << (i-firstIdxEntry)%2 << "\"><tt>";
         if (currentIndexEntry < indexChars.size()
            && !strncmp(indexChars[currentIndexEntry].c_str(), classname + 1, 
                        indexChars[currentIndexEntry].length()))
            outputFile << "<a name=\"idx" << currentIndexEntry++ << "\"></a>" << endl;

         TString htmlFile; 
         GetHtmlFileName(classPtr, htmlFile);

         if (htmlFile.Length()) {
            outputFile << "<a name=\"";
            outputFile << classPtr->GetName();
            outputFile << "\" href=\"";
            outputFile << htmlFile;
            outputFile << "\">";
            ReplaceSpecialChars(outputFile, classPtr->GetName());
            outputFile << "</a>";
         } else
            ReplaceSpecialChars(outputFile, classPtr->GetName());


         // write title
         outputFile << "</tt><a name=\"Title:";
         outputFile << classPtr->GetName();
         outputFile << "\"></a>";
         ReplaceSpecialChars(outputFile, classPtr->GetTitle());
         outputFile << "</li>" << endl;
      } else
         Error("MakeIndex", "Unknown class '%s' !",
               strchr(fileNames[i], '_') + 1);


      // first base name
      // look for first _ in basename
      char *first = strrchr(fileNames[i], '/');
      if (!first)
         first=strchr(fileNames[i],'_');
         else first=strchr(first,'_');
      if (first)
         *first = 0;

      // second base name
      char *second = 0;
      if (i < (numberOfNames - 1)) {
         second = strrchr(fileNames[i + 1], '/');
         if (!second)
            second=strchr(fileNames[i + 1],'_');
            else second=strchr(second,'_');
         if (second)
            *second = 0;
      }
      // check and close the file if necessary
      if (!first || !second || strcmp(fileNames[i], fileNames[i + 1])) {

         if (outputFile.good()) {

            outputFile << "</ul>" << endl;

            // write outputFile footer
            TDatime date;
            WriteHtmlFooter(outputFile, "", date.AsString());

            // close file
            outputFile.close();

            filename = 0;
         } else
            Error("MakeIndex", "Corrupted file '%s' !", filename);
      }

      if (first)
         *first = '_';
      if (second)
         *second = '_';
   }
}

//______________________________________________________________________________
void THtml::CreateHierarchy(const char **classNames, Int_t numberOfClasses)
{
// Create a hierarchical class list
// The algorithm descends from the base classes and branches into
// all derived classes. Mixing classes are displayed several times.
//
// Input: classNames      - pointer to an array of class names
//        numberOfClasses - number of elements
//
   Int_t i=0;

   gSystem->ExpandPathName(fOutputDir);
   char *filename = gSystem->ConcatFileName(fOutputDir, "ClassHierarchy.html");

   // open out file
   ofstream out;
   out.open(filename, ios::out);

   if (out.good()) {

      Printf(formatStr, "", fCounter.Data(), filename);

      // write out header
      WriteHtmlHeader(out, "Class Hierarchy");
      out << "<h1>Class Hierarchy</h1>" << endl;

      // check for a search engine
      const char *searchEngine =
          gEnv->GetValue("Root.Html.SearchEngine", "");

      // if exists ...
      if (*searchEngine) {

         // create link to search engine page
         out << "<h2><a href=\"" << searchEngine
             << "\">Search the Class Reference Guide</a></h2>" << endl;
      }

      // loop on all classes
      for (i = 0; i < numberOfClasses; i++) {

         // get class
         TClass *basePtr = GetClass((const char *) classNames[i]);
         if (basePtr == 0) {
            Warning("THtml::CreateHierarchy", "skipping class %s\n", classNames[i]);
            continue;
         }

         // Find basic base classes
         TList *bases = basePtr->GetListOfBases();
         if (bases && bases->IsEmpty()){

            out << "<hr />" << endl;

            out << "<table><tr><td><ul><li><tt>";
            TString htmlFile;
            GetHtmlFileName(basePtr, htmlFile);
            if (htmlFile.Length()) {
               out << "<a name=\"";
               out << classNames[i];
               out << "\" href=\"";
               out << htmlFile;
               out << "\">";
               ReplaceSpecialChars(out, classNames[i]);
               out << "</a>";
            } else {
               ReplaceSpecialChars(out, classNames[i]);
            }

            // find derived classes
            out << "</tt></li></ul></td>";
            fHierarchyLines = 0;
            DescendHierarchy(out,basePtr,classNames,numberOfClasses);

            out << "</tr></table>" << endl;
         }
      }

      // write out footer
      TDatime date;
      WriteHtmlFooter(out, "", date.AsString());

      // close file
      out.close();

   } else
      Error("CreateHierarchy", "Can't open file '%s' !", filename);

   if (filename)
      delete[]filename;
}

//______________________________________________________________________________
void THtml::DescendHierarchy(ofstream & out, TClass* basePtr,
  const char **classNames, Int_t numberOfClasses, Int_t maxLines, Int_t depth)
{
// Descend hierarchy recursively
// loop over all classes and look for classes with base class basePtr

   if (maxLines)
      if (fHierarchyLines >= maxLines) {
         out << "<td></td>" << endl;
         return;
      }

   Int_t numClasses=0;
   for (Int_t j = 0; j < numberOfClasses && (!maxLines || fHierarchyLines<maxLines); j++) {

      TClass *classPtr = GetClass((const char *) classNames[j]);
      if (!classPtr) continue;

      // find base classes with same name as basePtr
      TList* bases=classPtr->GetListOfBases();
      if (!bases) continue;

      TBaseClass *inheritFrom=(TBaseClass*)bases->FindObject(basePtr->GetName());
      if (!inheritFrom) continue;

      if (!numClasses)
         out << "<td>&lt;-</td><td><table><tr>" << endl;
      else
         out << "</tr><tr>"<<endl;
      fHierarchyLines++;
      numClasses++;
      UInt_t bgcolor=255-depth*8;
      out << "<td bgcolor=\""
          << Form("#%02x%02x%02x", bgcolor, bgcolor, bgcolor)
          << "\">";
      out << "<table><tr><td>" << endl;

      TString htmlFile;
      GetHtmlFileName(classPtr, htmlFile);
      if (htmlFile.Length()) {
         out << "<center><tt><a name=\"";
         out << classNames[j];
         out << "\" href=\"";
         out << htmlFile;
         out << "\">";
         ReplaceSpecialChars(out, classNames[j]);
         out << "</a></tt></center>";
      } else {
         ReplaceSpecialChars(out, classNames[j]);
      }
      // write title
      // commented out for now because it reduces overview
      /*
        len = strlen(classNames[i]);
        for (Int_t w = 0; w < (maxLen - len + 2); w++)
        out << ".";
        out << " ";

        out << "<a name=\"Title:";
        out << classPtr->GetName();
        out << "\">";
        ReplaceSpecialChars(out, classPtr->GetTitle());
        out << "</a></tt>" << endl;
      */

      out << "</td>" << endl;
      DescendHierarchy(out,classPtr,classNames,numberOfClasses,maxLines, depth+1);
      out << "</tr></table></td>" << endl;

   }  // loop over all classes
   if (numClasses)
      out << "</tr></table></td>" << endl;
   else
      out << "<td></td>" << endl;
}


//______________________________________________________________________________
void THtml::CreateListOfClasses(const char* filter)
{
// Create the list of all known classes

   // get total number of classes
   Int_t totalNumberOfClasses = gClassTable->Classes();

   // allocate memory
   if (fClassNames) delete [] fClassNames;
   if (fFileNames) delete [] fFileNames;
   fClassNames = new const char *[totalNumberOfClasses];
   fFileNames = new char *[totalNumberOfClasses];

   // start from begining
   gClassTable->Init();

   // get class names
   fNumberOfClasses = 0;
   fNumberOfFileNames = 0;

   TString reg = filter;
   TRegexp re(reg, kTRUE);

   for (Int_t i = 0; i < totalNumberOfClasses; i++) {

      // get class name
      const char *cname = gClassTable->Next();
      TString s = cname;
      if (filter && filter[0] && strcmp(filter,"*") && s.Index(re) == kNPOS)
         continue;

      // This is a hack for until after Cint and Reflex are one.
      if (strstr(cname, "__gnu_cxx::")) continue;

      // get class & filename - use TROOT::GetClass, as we also
      // want those classes without decl file name!
      TClass *classPtr = gROOT->GetClass((const char *) cname, kTRUE);
      if (!classPtr) continue;

      TString srcGuess;
      TString hdrGuess;
      const char *impname=GetImplFileName(classPtr);
      if (!impname || !impname[0]) {
         impname = GetDeclFileName(classPtr);
         if (impname && !impname[0]) {
            // no impl, no decl - might be a cintex dict
            // use namespace to decrypt path.
            TString impnameString(cname);
            TObjArray* arrScopes = impnameString.Tokenize("::");

            // for A::B::C, we assume B to be the module, 
            // b/inc/B/C.h the header, and b/src/C.cxx the source.
            TIter iScope(arrScopes, kIterBackward);
            TObjString *osFile   = (TObjString*)iScope();
            TObjString *osModule = 0;
            if (osFile) osModule = (TObjString*)iScope();

            if (osModule) {
               hdrGuess = osModule->String();
               hdrGuess.ToLower();
               hdrGuess += "/inc/";
               hdrGuess += osModule->String();
               hdrGuess += "/";
               hdrGuess += osFile->String();
               hdrGuess += ".h";
               char* realFile = gSystem->Which(fSourceDir, hdrGuess, kReadPermission);
               if (realFile) {
                  delete realFile;
                  fGuessedDeclFileNames[classPtr] = hdrGuess.Data();
                  impname = hdrGuess.Data();
                  
                  // only check for source if we've found the header!
                  srcGuess = osModule->String();
                  srcGuess.ToLower();
                  srcGuess += "/src/";
                  srcGuess += osFile->String();
                  srcGuess += ".cxx";
                  realFile = gSystem->Which(fSourceDir, srcGuess, kReadPermission);
                  if (realFile) {
                     delete realFile;
                     fGuessedImplFileNames[classPtr] = srcGuess.Data();
                     impname = srcGuess.Data();
                  }
               }
            }
            delete arrScopes;
         }
      }

      if (!impname || !impname[0]) {
         cout << "WARNING class " << cname <<
            " has no implementation file name !" << endl;
         continue;
      }
      if (strstr(impname,"prec_stl/")) continue;
      if (strstr(cname, "ROOT::") && !strstr(cname,"Math::")
          && !strstr(cname,"Reflex::") && !strstr(cname,"Cintex::"))
         continue;

      fClassNames[fNumberOfClasses] = cname;
      
      fFileNames[fNumberOfFileNames] = StrDup(impname, strlen(fClassNames[fNumberOfClasses])+2);
      char* posSlash = strchr(fFileNames[fNumberOfFileNames], '/');
      
      char *srcdir = 0;
      if (posSlash) {
         // for new ROOT install the impl file name has the form: base/src/TROOT.cxx
         srcdir = strstr(posSlash, "/src/");
         
         // if impl is unset, check for decl and see if it matches
         // format "base/inc/TROOT.h" - in which case it's not a USER
         // class, but a BASE class.
         if (!srcdir) srcdir=strstr(posSlash, "/inc/");
      } else srcdir = 0;
      if (srcdir && srcdir == posSlash) {
         strcpy(srcdir, "_");
         for (char *t = fFileNames[fNumberOfFileNames];
              (t[0] = toupper(t[0])); t++);
         strcat(srcdir, fClassNames[fNumberOfClasses]);
      } else {
         if (posSlash && !strncmp(posSlash,"/Math/GenVector/", 16))
            strcpy(fFileNames[fNumberOfFileNames], "MATHCORE_");
         else if (posSlash && !strncmp(posSlash,"/Math/Matrix", 12))
            strcpy(fFileNames[fNumberOfFileNames], "SMATRIX_");
         else
            strcpy(fFileNames[fNumberOfFileNames], "USER_");
         strcat(fFileNames[fNumberOfFileNames], fClassNames[fNumberOfClasses]);
      }
      fNumberOfFileNames++;
      fNumberOfClasses++;
   }

   // quick sort
   SortNames(fClassNames, fNumberOfClasses, kCaseInsensitive);
   SortNames((const char **) fFileNames, fNumberOfFileNames);

}


//______________________________________________________________________________
void THtml::CreateListOfTypes()
{
// Create list of all data types

   // open file
   ofstream typesList;

   gSystem->ExpandPathName(fOutputDir);
   char *outFile = gSystem->ConcatFileName(fOutputDir, "ListOfTypes.html");
   typesList.open(outFile, ios::out);


   if (typesList.good()) {
      Printf(formatStr, "", "", outFile);

      // write typesList header
      WriteHtmlHeader(typesList, "List of data types");
      typesList << "<h2> List of data types </h2>" << endl;

      typesList << "<dl><dd>" << endl;

      // make loop on data types
      TDataType *type;
      TIter nextType(gROOT->GetListOfTypes());

      std::list<std::string> typeNames;
      while ((type = (TDataType *) nextType()))
         // no templates ('<' and '>'), no idea why the '(' is in here...
         if (*type->GetTitle() && !strchr(type->GetName(), '(')
             && !( strchr(type->GetName(), '<') && strchr(type->GetName(),'>'))
             && type->GetName())
               typeNames.push_back(type->GetName());
      sort_strlist_stricmp(typeNames);

      std::vector<std::string> indexChars;
      if (typeNames.size() > 10) {
         typesList << "<div id=\"indxShortX\"><h4>Jump to</h4>" << endl;
         // find index chars
         GetIndexChars(typeNames, 10 /*sections*/, indexChars);
         for (UInt_t iIdxEntry = 0; iIdxEntry < indexChars.size(); ++iIdxEntry) {
            typesList << "<a href=\"#idx" << iIdxEntry << "\">" << indexChars[iIdxEntry] 
                      << "</a>" << endl;
         }
         typesList << "</div><br />" << endl;
      }

      typesList << "<ul id=\"indx\">" << endl;

      nextType.Reset();
      int idx = 0;
      UInt_t currentIndexEntry = 0;

      for (std::list<std::string>::iterator iTypeName = typeNames.begin(); 
         iTypeName != typeNames.end(); ++iTypeName) {
         TDataType* type = gROOT->GetType(iTypeName->c_str(), kFALSE);
         typesList << "<li class=\"idxl" << idx%2 << "\">";
         if (currentIndexEntry < indexChars.size()
            && !strncmp(indexChars[currentIndexEntry].c_str(), iTypeName->c_str(), 
                        indexChars[currentIndexEntry].length()))
            typesList << "<a name=\"idx" << currentIndexEntry++ << "\"></a>" << endl;
         typesList << "<a name=\"";
         ReplaceSpecialChars(typesList, iTypeName->c_str());
         typesList << "\"><tt>";
         ReplaceSpecialChars(typesList, iTypeName->c_str());
         typesList << "</tt></a>";
         typesList << "<a name=\"Title:";
         ReplaceSpecialChars(typesList, type->GetTitle());
         typesList << "\"></a>";
         ReplaceSpecialChars(typesList, type->GetTitle());
         typesList << "</li>" << endl;
         ++idx;
      }
      typesList << "</ul>" << endl;

      // write typesList footer
      TDatime date;
      WriteHtmlFooter(typesList, "", date.AsString());

      // close file
      typesList.close();

   } else
      Error("Make", "Can't open file '%s' !", outFile);

   if (outFile)
      delete[]outFile;
}


//______________________________________________________________________________
void THtml::CreateStyleSheet() {
   // Write the default ROOT style sheet.

   // open file
   ofstream styleSheet;

   gSystem->ExpandPathName(fOutputDir);
   char *outFile = gSystem->ConcatFileName(fOutputDir, "ROOT.css");
   styleSheet.open(outFile, ios::out);
   if (styleSheet.good()) {
      styleSheet 
         << "a {" << std::endl
         << "   text-decoration: none;" << std::endl
         << "   font-weight: bolder;" << std::endl
         << "}" << std::endl
         << "a:link {" << std::endl
         << "   color: #0000ff;" << std::endl
         << "   text-decoration: none;" << std::endl
         << "}" << std::endl
         << "a:visited {" << std::endl
         << "/*   color: #551a8b;*/" << std::endl
         << "   color: #5500cc;" << std::endl
         << "}" << std::endl
         << "a:active {" << std::endl
         << "   color: #551a8b;" << std::endl
         << "   border: dotted 1px #0000ff;" << std::endl
         << "}" << std::endl
         << "a:hover {" << std::endl
         << "   background: #eeeeff;" << std::endl
         << "}" << std::endl
         << "" << std::endl
         << "#indx {" << std::endl
         << "  list-style:   none;" << std::endl
         << "  padding-left: 0em;" << std::endl
         << "  margin-left:  0em;" << std::endl
         << "}" << std::endl
         << "#indx li {" << std::endl
         << "  margin-top:    1px;" << std::endl
         << "  margin-bottom: 1px;" << std::endl
         << "  margin-left:   0px;" << std::endl
         << "  padding-left:  2em;" << std::endl
         << "  padding-bottom: 0.3em;" << std::endl
         << "  border-top:    0px hidden #afafaf;" << std::endl
         << "  border-left:   0px hidden #afafaf;" << std::endl
         << "  border-bottom: 1px hidden #ffffff;" << std::endl
         << "  border-right:  1px hidden #ffffff;" << std::endl
         << "}" << std::endl
         << "#indx li:hover {" << std::endl
         << "  border-top:    1px solid #afafaf;" << std::endl
         << "  border-left:   1px solid #afafaf;" << std::endl
         << "  border-bottom: 0px solid #ffffff;" << std::endl
         << "  border-right:  0px solid #ffffff;" << std::endl
         << "}" << std::endl
         << "#indx li.idxl0 {" << std::endl
         << "  background-color: #e7e7ff;" << std::endl
         << "}" << std::endl
         << "#indx a {" << std::endl
         << "  font-weight: bold;" << std::endl
         << "  display:     block;" << std::endl
         << "  margin-left: -1em;" << std::endl
         << "}" << std::endl
         << "#indxShortX {" << std::endl
         << "  border: 3px solid gray;" << std::endl
         << "  padding: 8pt;" << std::endl
         << "  margin-left: 2em;" << std::endl
         << "}" << std::endl
         << "#indxShortX h4 {" << std::endl
         << "  margin-top: 0em;" << std::endl
         << "  margin-bottom: 0.5em;" << std::endl
         << "}" << std::endl
         << "#indxShortX a {" << std::endl
         << "  margin-right: 0.25em;" << std::endl
         << "  margin-left: 0.25em;" << std::endl
         << "}" << std::endl
         << "#indxModules {" << std::endl
         << "  border: 3px solid gray;" << std::endl
         << "  padding: 8pt;" << std::endl
         << "  margin-left: 2em;" << std::endl
         << "}" << std::endl
         << "#indxModules h4 {" << std::endl
         << "  margin-top: 0em;" << std::endl
         << "  margin-bottom: 0.5em;" << std::endl
         << "}" << std::endl
         << "#indxModules a {" << std::endl
         << "  margin-right: 0.25em;" << std::endl
         << "  margin-left: 0.25em;" << std::endl
         << "}" << std::endl
         << "#searchform {" << std::endl
         << "  margin-left: 2em;" << std::endl
         << "}" << std::endl
         << "" << std::endl
         << "div.funcdoc {" << std::endl
         << "   width: 100%;" << std::endl
         << "   border-bottom: solid 3px #cccccc;" << std::endl
         << "   border-left: solid 1px #cccccc;" << std::endl
         << "   margin-bottom: 1em;" << std::endl
         << "   margin-left: 0.3em;" << std::endl
         << "   padding-left: 1em;" << std::endl
         << "   background-color: White;" << std::endl
         << "}" << std::endl
         << "span.funcname {" << std::endl
         << "   margin-left: -0.7em;" << std::endl
         << "   /*border-bottom: solid 1px #cccccc;*/" << std::endl
         << "   font-weight: bolder;" << std::endl
         << "}" << std::endl
         << "" << std::endl
         << "span.comment {" << std::endl
         << "   background-color: #eeeeee;" << std::endl
         << "   color: Green;" << std::endl
         << "   font-weight: normal;" << std::endl
         << "}" << std::endl
         << "span.keyword {" << std::endl
         << "   color: Black;" << std::endl
         << "   font-weight: normal;" << std::endl
         << "}" << std::endl
         << "span.cpp {" << std::endl
         << "	 color: Gray;" << std::endl
         << "   font-weight: normal;" << std::endl
         << "}" << std::endl
         << "span.string {" << std::endl
         << "	 color: Teal;" << std::endl
         << "   font-weight: normal;" << std::endl
         << "}" << std::endl
         << "pre.code {" << std::endl
         << "   font-weight: bolder;" << std::endl
         << "}" << std::endl
         << "div.classdescr {" << std::endl
         << "	width: 100%;" << std::endl
         << "	margin-left: 0.3em;" << std::endl
         << "	padding-left: 1em;" << std::endl
         << "	margin-bottom: 2em;" << std::endl
         << "	/*border-bottom: solid 1px Black;*/" << std::endl
         << "    border-bottom: solid 3px #cccccc;" << std::endl
         << "    border-left: solid 1px #cccccc;" << std::endl
         << "    background-color: White;" << std::endl
         << "}" << std::endl
         << "div.inlinecode {" << std::endl
         << "	margin-bottom: 0.5em;" << std::endl
         << "	padding: 0.5em;" << std::endl
         << "}" << std::endl
         << "code.inlinecode {" << std::endl
         << "	padding: 0.5em;" << std::endl
         << "	border: solid 1px #ffff77;" << std::endl
         << "   background-color: #ffffdd;" << std::endl
         << "}" << std::endl
         << "body {" << std::endl
         << "	background-color: #fcfcfc;" << std::endl
         << "}" << std::endl
         << "table.inhtree {" << std::endl
         << "	background-color: White;" << std::endl
         << "	border: solid 1px Black;" << std::endl
         << "	width: 100%;" << std::endl
         << "}" << std::endl
         << "table.libinfo " << std::endl
         << "{" << std::endl
         << "	background-color: White;" << std::endl
         << "	padding: 2px;" << std::endl
         << "	border: solid 1px Gray; " << std::endl
         << "	float: right;" << std::endl
         << "}" << std::endl;
   }      
   delete outFile;
}


//______________________________________________________________________________
void THtml::ExpandKeywords(ostream & out, const char *text)
{
   // Expand keywords in text, writing to out.
   TString str(text);
   ExpandKeywords(str);
   out << str;
}

//______________________________________________________________________________
void THtml::ExpandKeywords(TString& keyword)
{
   // Find keywords in keyword and create URLs around them. Escape characters with a 
   // special meaning for HTML. Protect "Begin_Html"/"End_Html" pairs, and set the
   // parsing context. Evaluate sequences like a::b->c.

   static Bool_t pre_is_open = kFALSE;
   // we set parse context to kCComment even for CPP comment, and note it here:
   Bool_t commentIsCPP = kFALSE;
   TClass* currentType = 0;

   enum {
      kNada,
      kMember,
      kScope,
      kNumAccesses
   } scoping = kNada;

   Ssiz_t i;
   for (i = 0; i < keyword.Length(); ++i) {
      if (!currentType)
         scoping = kNada;

      // skip until start of the word
      if (fParseContext == kCode || fParseContext == kCComment) {
         if (!strncmp(keyword.Data() + i, "::", 2)) {
            scoping = kScope;
            i += 2;
         } else if (!strncmp(keyword.Data() + i, "->", 2)) {
            scoping = kMember;
            i += 2;
         } else if (keyword[i] == '.') {
            scoping = kMember;
            ++i;
         } else currentType = 0;
         if (i >= keyword.Length()) 
            break;
      } else currentType = 0;

      if (!IsWord(keyword[i])){
         if (fParseContext != kBeginEndHtml && fParseContext != kBeginEndHtmlInCComment) {
            Bool_t closeString = !fEscFlag && fParseContext == kString && 
               ( keyword[i] == '"' || keyword[i] == '\'');
            if (!fEscFlag)
               if (fParseContext == kCode || fParseContext == kCComment)
                  if (keyword.Length() > i + 1 && keyword[i] == '"' || 
                     keyword[i] == '\'' && (
                        // 'a'
                        keyword.Length() > i + 2 && keyword[i + 2] == '\'' ||
                        // '\a'
                        keyword.Length() > i + 3 && keyword[i + 1] == '\'' && keyword[i + 3] == '\'')) {
                     keyword.Insert(i, "<span class=\"string\">");
                     i += 21;
                     fParseContext = kString;
                     currentType = 0;
                  } else if (fParseContext != kCComment && 
                     keyword.Length() > i + 1 && keyword[i] == '/' && 
                     (keyword[i+1] == '/' || keyword[i+1] == '*')) {
                     fParseContext = kCComment;
                     commentIsCPP = keyword[i+1] == '/';
                     currentType = 0;
                     keyword.Insert(i, "<span class=\"comment\">");
                     i += 23;
                  } else if (fParseContext == kCComment && !commentIsCPP
                     && keyword.Length() > i + 1 
                     && keyword[i] == '*' && keyword[i+1] == '/') {
                     fParseContext = kCode;
                     currentType = 0;
                     keyword.Insert(i + 2, "</span>");
                     i += 9;
                  }

            ReplaceSpecialChars(keyword, i);
            if (closeString) {
               keyword.Insert(i, "</span>");
               i += 7;
               fParseContext = kCode;
               currentType = 0;
            }
            --i; // i already moved by ReplaceSpecialChar
         } else
            // protect html code from special chars
            if ((unsigned char)keyword[i]>31)
               if (keyword[i] == '<'){
                  if (!strncasecmp(keyword.Data() + i, "<pre>", 5)){
                     if (pre_is_open) {
                        keyword.Remove(i, 4);
                        continue;
                     } else {
                        pre_is_open = kTRUE;
                        i += 3;
                     }
                     currentType = 0;
                  } else 
                     if (!strncasecmp(keyword,"</pre>", 6)) {
                        if (!pre_is_open) {
                           keyword.Remove(i, 5);
                           continue;
                        } else {
                           pre_is_open = kFALSE;
                           i += 4;
                        }
                        currentType = 0;
                     }
               } // i = '<'
         continue;
      } // not a word

      // get end of the word
      Ssiz_t endWord = i;
      while (endWord < keyword.Length() && IsName(keyword[endWord]))
         endWord++;

      if (fParseContext != kCode && fParseContext != kCComment && 
         fParseContext != kBeginEndHtml && fParseContext != kBeginEndHtmlInCComment) {
         // don't replace in strings, cpp, etc
         i = endWord - 1;
         continue;
      }

      TString word(keyword(i, endWord - i));

      // check if this is a HTML block
      if (fParseContext == kBeginEndHtml || fParseContext == kBeginEndHtmlInCComment) {
         if (!word.CompareTo("end_html", TString::kIgnoreCase) && 
            (i == 0 || keyword[i - 1] != '\"')) {
            if (fParseContext == kBeginEndHtmlInCComment)
               commentIsCPP = kFALSE;
            else {
               commentIsCPP = kTRUE;
               // special case; we're skipping the "//" recognition inside BeginEndHtml,
               // but here, we need it so this line is added to the doc
               if (keyword.BeginsWith("//")) {
                  keyword.Prepend("<span class=\"comment\">");
                  i += 22;
               }
            }
            fParseContext = kCComment;
            pre_is_open = kTRUE;
            keyword.Replace(i, word.Length(), "<pre>");
            i += 4;
         }
         // we're in a begin/end_html block, just keep what we have
         currentType = 0;
         continue;
      }
      if (fParseContext == kCComment 
         && !word.CompareTo("begin_html", TString::kIgnoreCase)
         && (i == 0 || keyword[i - 1] != '\"')) {
         if (commentIsCPP)
            fParseContext = kBeginEndHtml;
         else
            fParseContext = kBeginEndHtmlInCComment;
         pre_is_open = kFALSE;
         keyword.Replace(i, word.Length(), "</pre>");
         i += 5;
         currentType = 0;
         continue;
      }

      // don't replace keywords in comments
      if (fParseContext == kCode && 
         fgKeywords.find(word.Data()) != fgKeywords.end()) {
         keyword.Insert(i, "<span class=\"keyword\">");
         i += 22 + word.Length();
         keyword.Insert(i, "</span>");
         i += 7 - 1; // -1 for ++i
         currentType = 0;
         continue;
      }

      // generic layout:
      // A::B::C::member[arr]->othermember
      // we iterate through this, first scope is A, and currentType will be set toA,
      // next we see ::B, "::" signals to use currentType,...

      TDataType* subType = 0;
      TClass* subClass = 0;
      TDataMember *datamem = 0;
      TMethod *meth = 0;
      TClass* lookupScope = currentType;
      const char* globalTypeName = 0;

      if (!lookupScope)
         lookupScope = fCurrentClass;

      if (scoping == kNada) {
         subType = gROOT->GetType(word);
         if (!subType)
            subClass = GetClass(word);
         if (!subType && !subClass) {
            TGlobal *global = gROOT->GetGlobal(word);
            if (global) {
               // cannot doc globals; take at least their type...
               globalTypeName = global->GetTypeName();
               subClass = GetClass(globalTypeName);
               if (!subClass)
                  subType = gROOT->GetType(globalTypeName);
               else // hack to prevent current THtml obj from showing up - we only want gHtml
                  if (subClass == THtml::Class() && word != "gHtml")
                     subClass = 0;
            }
         }
         if (!subType && !subClass) {
            // too bad - cannot doc yet...
            //TFunction *globFunc = gROOT->GetGlobalFunctionWithPrototype(word);
            //globFunc = 0;
         }
      }

      if (lookupScope && !subType && !subClass) {
         if (scoping == kScope) {
            TString subClassName(lookupScope->GetName());
            subClassName += "::";
            subClassName += word;
            subClass = GetClass(subClassName);
         }
         if (!subClass)
            datamem = lookupScope->GetDataMember(word);
         if (!subClass && !datamem)
            meth = lookupScope->GetMethodAllAny(word);
      }

      TString link;
      if (subType) {
         link = "./ListOfTypes.html";
         link += "#";
         TString mangledWord;
         if (!globalTypeName)
            mangledWord = word;
         else 
            mangledWord = globalTypeName;
         NameSpace2FileName(mangledWord);
         link += mangledWord;
         currentType = 0;
      } else if (subClass) {
         GetHtmlFileName(subClass, link);
         if (!link.BeginsWith("http://") && !link.BeginsWith("https://"))
            link.Prepend("./");
         currentType = subClass;
      } else if (datamem || meth) {
         GetHtmlFileName(lookupScope, link);
         if (!link.BeginsWith("http://") && !link.BeginsWith("https://"))
            link.Prepend("./");
         link += "#";
         TString mangledName(lookupScope->GetName());
         NameSpace2FileName(mangledName);
         link += mangledName;
         link += ":";
         if (datamem) {
            mangledName = datamem->GetName();
            if (datamem->GetDataType())
               currentType = 0;
            else
               currentType = GetClass(datamem->GetTypeName());
         } else {
            mangledName = meth->GetName();
            const char* retTypeName = meth->GetReturnTypeName();
            if (retTypeName)
               if (gROOT->GetType(retTypeName))
                  currentType = 0;
               else
                  currentType = GetClass(retTypeName);
         }
         NameSpace2FileName(mangledName);
         link += mangledName;
      } else
         currentType = 0;

      if (link.Length()) {
         link.Prepend("<a href=\"");
         link += "\">";
         keyword.Insert(i, link);
         i += link.Length();
      }
      TString mangledWord(word);
      Ssiz_t posReplace = 0;
      ReplaceSpecialChars(mangledWord, posReplace);
      keyword.Replace(i, word.Length(), mangledWord);
      i += mangledWord.Length();
      if (link.Length()) {
         keyword.Insert(i, "</a>");
         i += 4;
      }

      --i; // due to ++i
   } // while i < keyword.Length()

   // clean up, no CPP comment across lines
   if (commentIsCPP) {
      keyword += "</span>";
      i += 7;
      if (fParseContext == kCComment) // and not BeginEndHtml
         fParseContext = kCode;
      currentType = 0;
   }

   // clean up, no strings across lines
   if (fParseContext == kString) {
      keyword += "</span>";
      i += 7;
      fParseContext = kCode;
      currentType = 0;
   }
}


//______________________________________________________________________________
void THtml::ExpandPpLine(ostream & out)
{
// Expand preprocessor statements
//
//
// Input: out  - output file stream
//
//  NOTE: Looks for the #include statements and
//        creates link to the corresponding file
//        if such file exists
//

   const char *ptr;
   const char *ptrStart;
   const char *ptrEnd;
   char *fileName;

   Bool_t linkExist = kFALSE;

   ptrEnd = strstr(fLine.Data(), "include");
   if (ptrEnd) {
      ptrEnd += 7;
      if ((ptrStart = strpbrk(ptrEnd, "<\""))) {
         ptrStart++;
         ptrEnd = strpbrk(ptrStart, ">\"");
         if (ptrEnd) {
            Int_t len = ptrEnd - ptrStart;
            fileName = new char[len + 1];
            strncpy(fileName, ptrStart, len);
            fileName[len]=0;
            char *tmpstr =
                gSystem->Which(fSourceDir, fileName, kReadPermission);
            if (tmpstr) {
               char *realFileName = StrDup(tmpstr);

               if (realFileName) {
                  CopyHtmlFile(realFileName);

                  ptr = fLine.Data();
                  while (ptr < ptrStart)
                     ReplaceSpecialChars(out, *ptr++);
                  out << "<a href=\"../" << GetFileName(realFileName) <<
                      "\">";
                  out << fileName << "</a>";
                  out << ptrEnd;

                  linkExist = kTRUE;
               }
               if (realFileName)
                  delete[]realFileName;
               if (fileName)
                  delete[]fileName;
               delete[]tmpstr;
            }
         }
      }
   }

   if (!linkExist)
      ReplaceSpecialChars(out, fLine);
}

//______________________________________________________________________________
const char *THtml::GetFileName(const char *filename)
{
// It discards any directory information inside filename
//
//
//  Input: filename - pointer to the file name
//
// Output: pointer to the string containing just a file name
//         without any other directory information, i.e.
//         '/usr/root/test.dat' will return 'test.dat'
//

   return gSystem->BaseName(filename);
}

//______________________________________________________________________________
void THtml::GetSourceFileName(TString& filename)
{
   // Find the source file. If filename contains a path it will be used
   // together with the possible source prefix. If not found we try
   // old algorithm, by stripping off the path and trying to find it in the
   // specified source search path.

   TString found(filename);

   if (strchr(filename, '/') 
#ifdef WIN32
   || strchr(filename, '\\')
#endif
   ){
      TString found(fSourcePrefix);
      if (found.Length())
         gSystem->PrependPathName(found, filename);
      gSystem->FindFile(fSourceDir, filename, kReadPermission);
      if (filename.Length())
         return;
   }

   filename = GetFileName(filename);
   if (filename.Length())
      gSystem->FindFile(fSourceDir, filename, kReadPermission);
}

//______________________________________________________________________________
void THtml::GetHtmlFileName(TClass * classPtr, TString& filename)
{
// Return real HTML filename
//
//
//  Input: classPtr - pointer to a class
//         filename - string containing a full name
//         of the corresponding HTML file after the function returns. 
//

   filename.Remove(0);
   if (!classPtr) return;

   const char* cFilename = GetImplFileName(classPtr);
   if (!cFilename || !cFilename[0])
      cFilename = GetDeclFileName(classPtr);

   // classes without Impl/DeclFileName don't have docs,
   // and classes without docs don't have output file names
   if (!cFilename || !cFilename[0])
      return;

   // this should be a prefix
   TString varName("Root.Html.");

   const char *colon = strchr(cFilename, ':');
   if (colon)
      varName += TString(cFilename, colon - cFilename);
   else
      varName += "Root";

   filename = cFilename;
   TString htmlFileName;
   if (!filename.Length() ||
       !gSystem->FindFile(fSourceDir, filename, kReadPermission))
      htmlFileName = gEnv->GetValue(varName, "");
   else
      htmlFileName = "./";

   if (htmlFileName.Length()) {
      filename = htmlFileName;
      TString className(classPtr->GetName());
      NameSpace2FileName(className);
      gSystem->PrependPathName(filename, className);
      filename = className;
      filename.ReplaceAll("\\", "/");
      filename += ".html";
   } else filename.Remove(0);
}

//______________________________________________________________________________
TClass *THtml::GetClass(const char *name1, Bool_t load)
{
//*-*-*-*-*Return pointer to class with name*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*      =================================
   if(!name1) return 0;
   // no doc for internal classes
   if (strstr(name1,"ROOT::")==name1) {
      Bool_t ret = kTRUE;
      if (strstr(name1,"Math::"))   ret = kFALSE;
      if (strstr(name1,"Reflex::")) ret = kFALSE;
      if (strstr(name1,"Cintex::")) ret = kFALSE;
      if (ret) return 0;
   }

   Int_t n = strlen(name1);
   if (!n) return 0;
   char *name = new char[n + 1];
   strcpy(name, name1);
   char *t = name + n - 1;
   while (*t == ' ') {
      *t = 0;
      if (t == name)
         break;
      t--;
   }
   t = name;
   while (*t == ' ')
      t++;

   TClass *cl=gROOT->GetClass(t, load);
   // hack to get rid of prec_stl types
   // TClassEdit checks are far too slow...
   if (cl && GetDeclFileName(cl) &&
       strstr(GetDeclFileName(cl),"prec_stl/"))
      cl = 0;   
   delete [] name;
   if (cl && GetDeclFileName(cl) && GetDeclFileName(cl)[0])
      return cl;
   return 0;
}

//______________________________________________________________________________
const char* THtml::GetDeclFileName(TClass * cl) const
{
   // Return declaration file name
   std::map<TClass*,std::string>::const_iterator iClDecl = fGuessedDeclFileNames.find(cl);
   if (iClDecl == fGuessedDeclFileNames.end()) return cl->GetDeclFileName();
   return iClDecl->second.c_str();
}

//______________________________________________________________________________
const char* THtml::GetImplFileName(TClass * cl) const
{
   // Return implementation file name
   std::map<TClass*,std::string>::const_iterator iClImpl = fGuessedImplFileNames.find(cl);
   if (iClImpl == fGuessedImplFileNames.end()) return cl->GetImplFileName();
   return iClImpl->second.c_str();
}

//______________________________________________________________________________
Bool_t THtml::IsModified(TClass * classPtr, const Int_t type)
{
// Check if file is modified
//
//
//  Input: classPtr - pointer to the class
//         type     - file type to compare with
//                    values: kSource, kInclude, kTree
//
// Output: TRUE     - if file is modified since last time
//         FALSE    - if file is up to date
//

   Bool_t ret = kTRUE;

   TString sourceFile;
   TString classname(classPtr->GetName());
   TString filename;
   TString dir;

   switch (type) {
   case kSource:
      if (classPtr->GetImplFileLine()) {
         sourceFile = GetDeclFileName(classPtr);
         GetSourceFileName(sourceFile);
      } else {
         sourceFile = GetDeclFileName(classPtr);
         GetSourceFileName(sourceFile);
      }
      dir = "src";
      gSystem->ExpandPathName(fOutputDir);
      gSystem->PrependPathName(fOutputDir, dir);
      filename = classname;
      NameSpace2FileName(filename);
      gSystem->PrependPathName(dir, filename);
      filename += ".cxx.html";
      break;

   case kInclude:
      filename = GetDeclFileName(classPtr);
      sourceFile = filename;
      GetSourceFileName(sourceFile);
      filename = GetFileName(filename);
      gSystem->ExpandPathName(fOutputDir);
      gSystem->PrependPathName(fOutputDir, filename);
      break;

   case kTree:
      sourceFile = GetDeclFileName(classPtr);
      GetSourceFileName(sourceFile);
      NameSpace2FileName(classname);
      gSystem->ExpandPathName(fOutputDir);
      gSystem->PrependPathName(fOutputDir, classname);
      filename = classname;
      filename += "_Tree.pdf";
      break;

   default:
      Error("IsModified", "Unknown file type !");
   }

   // Get info about a file
   Long64_t sSize, dSize;
   Long_t sId, sFlags, sModtime;
   Long_t dId, dFlags, dModtime;

   if (!(gSystem->GetPathInfo(sourceFile, &sId, &sSize, &sFlags, &sModtime)))
      if (!(gSystem->GetPathInfo(filename, &dId, &dSize, &dFlags, &dModtime)))
         ret = (sModtime > dModtime) ? kTRUE : kFALSE;

   return (ret);
}


//______________________________________________________________________________
Bool_t THtml::IsName(UChar_t c)
{
// Check if c is a valid C++ name character
//
//
//  Input: c - a single character
//
// Output: TRUE if c is a valid C++ name character
//         and FALSE if it's not.
//
//   NOTE: Valid name characters are [a..zA..Z0..9_~],
//

   Bool_t ret = kFALSE;

   if (isalnum(c) || c == '_' || c == '~')
      ret = kTRUE;

   return ret;
}


//______________________________________________________________________________
Bool_t THtml::IsWord(UChar_t c)
{
// Check if c is a valid first character for C++ name
//
//
//  Input: c - a single character
//
// Output: TRUE if c is a valid first character for C++ name,
//         and FALSE if it's not.
//
//   NOTE: Valid first characters are [a..zA..Z_~]
//

   Bool_t ret = kFALSE;

   if (isalpha(c) || c == '_' || c == '~')
      ret = kTRUE;

   return ret;
}


//______________________________________________________________________________
void THtml::MakeAll(Bool_t force, const char *filter)
{
// Produce documentation for all the classes specified in the filter (by default "*")
// To process all classes having a name starting with XX, do:
//        html.MakeAll(kFALSE,"XX*");
// If force=kFALSE (default), only the classes that have been modified since
// the previous call to this function will be generated.
// If force=kTRUE, all classes passing the filter will be processed.
//

   Int_t i;

   TString reg = filter;
   TRegexp re(reg, kTRUE);

   MakeIndex(filter);

   // CreateListOfClasses(filter); already done by MakeIndex
   for (i = 0; i < fNumberOfClasses; i++) {
      fCounter.Form("%5d", fNumberOfClasses - i);
      MakeClass((char *) fClassNames[i], force);
   }

   fCounter.Remove(0);
}


//______________________________________________________________________________
void THtml::MakeClass(const char *className, Bool_t force)
{
// Make HTML files for a single class
//
//
// Input: className - name of the class to process
//

   if (!fClassNames) CreateListOfClasses("*"); // calls gROOT->GetClass(...,true) for each available class
   fCurrentClass = GetClass(className);

   if (fCurrentClass) {
      TString htmlFile;
      GetHtmlFileName(fCurrentClass, htmlFile);
      if (htmlFile.Length()
          && (htmlFile.BeginsWith("http://")
              || htmlFile.BeginsWith("https://")
              || gSystem->IsAbsoluteFileName(htmlFile))
          ) {
         htmlFile.Remove(0);
         //printf("CASE skipped, class=%s, htmlFile=%s\n",className,htmlFile);
      }
      if (htmlFile.Length()) {
         Class2Html(force);
         MakeTree(className, force);
      } else
         Printf(formatStr, "-skipped-", fCounter.Data(), className);
   } else
      if (!TClassEdit::IsStdClass(className)) // stl classes won't be available, so no warning
         Error("MakeClass", "Unknown class '%s' !", className);

}


//______________________________________________________________________________
void THtml::MakeIndex(const char *filter)
{
   // It makes an index files
   // by default makes an index of all classes (if filter="*")
   // To generate an index for all classes starting with "XX", do
   //    html.MakeIndex("XX*");

   CreateListOfClasses(filter);
   CreateListOfTypes();

   // create an index
   CreateIndexByTopic(fFileNames, fNumberOfFileNames);
   CreateIndex(fClassNames, fNumberOfClasses);

   // create a class hierarchy
   CreateHierarchy(fClassNames, fNumberOfClasses);
}


//______________________________________________________________________________
void THtml::MakeTree(const char *className, Bool_t force)
{
// Make an inheritance tree
//
//
// Input: className - name of the class to process
//

   // create canvas & set fill color
   TVirtualPad *psCanvas = 0;

   //The pad utility manager is required (a plugin)
   TVirtualUtilPad *util = (TVirtualUtilPad*)gROOT->GetListOfSpecials()->FindObject("R__TVirtualUtilPad");
   if (!util) {
      TPluginHandler *h;
      if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualUtilPad"))) {
         if (h->LoadPlugin() == -1)
            return;
         h->ExecPlugin(0);
         util = (TVirtualUtilPad*)gROOT->GetListOfSpecials()->FindObject("R__TVirtualUtilPad");
      }
   }
   util->MakeCanvas("","psCanvas",0,0,1000,1200);

   psCanvas = gPad->GetVirtCanvas();

   TClass *classPtr = GetClass(className);

   if (classPtr) {

      TString htmlFile;
      GetHtmlFileName(classPtr, htmlFile);
      if (htmlFile.Length()
          && (htmlFile.BeginsWith("http://")
              || htmlFile.BeginsWith("https://")
              || gSystem->IsAbsoluteFileName(htmlFile))
          ) {
         htmlFile.Remove(0);
      }
      if (htmlFile.Length()) {
         // make a class tree
         ClassTree(psCanvas, classPtr, force);
         htmlFile.Remove(0);
      } else
         Printf(formatStr, "-skipped-", "", className);

   } else
      Error("MakeTree", "Unknown class '%s' !", className);

   // close canvas
   psCanvas->Close();
   delete psCanvas;

}


//______________________________________________________________________________
void THtml::ReplaceSpecialChars(ostream & out, const char c)
{
// Replace ampersand, less-than and greater-than character, writing to out.
//
//
// Input: out - output file stream
//        c   - single character
//
   static TString s;
   s = c;
   Ssiz_t pos = 0;
   ReplaceSpecialChars(s, pos);
   out << s.Data();
}

//______________________________________________________________________________
void THtml::ReplaceSpecialChars(TString& text, Ssiz_t &pos)
{
// Replace ampersand, less-than and greater-than character
//
//
// Input: text - text where replacement will happen,
//        pos  - index of char to be replaced; will point to next char to be 
//               replaced when function returns
//

   const char c = text[pos];
   if (fEscFlag) {
      fEscFlag = kFALSE;
      ++pos;
      return;
   } else if (c == fEsc) {
      // text.Remove(pos, 1); - NO! we want to keep it nevertheless!
      fEscFlag = kTRUE;
      return;
   }
   switch (c) {
      case '<':
         text.Replace(pos, 1, "&lt;");
         pos += 3;
         break;
      case '&':
         text.Replace(pos, 1, "&amp;");
         pos += 4;
         break;
      case '>':
         text.Replace(pos, 1, "&gt;");
         pos += 3;
         break;
   }
   ++pos;
}


//______________________________________________________________________________
void THtml::ReplaceSpecialChars(ostream & out, const char *string)
{
// Replace ampersand, less-than and greater-than characters, writing to out
//
//
// Input: out    - output file stream
//        string - pointer to an array of characters
//

   while (string && *string) {
      ReplaceSpecialChars(out, *string);
      string++;
   }
}


//______________________________________________________________________________
void THtml::SetDeclFileName(TClass* cl, const char* filename)
{
   // Explicitely set a decl file name for TClass cl.
   fGuessedDeclFileNames[cl] = filename;
}

//______________________________________________________________________________
void THtml::SetImplFileName(TClass* cl, const char* filename)
{
   // Explicitely set a impl file name for TClass cl.
   fGuessedImplFileNames[cl] = filename;
}


//______________________________________________________________________________
void THtml::SortNames(const char **strings, Int_t num, Bool_t type)
{
// Sort strings
//
//
// Input: strings - pointer to an array of strings
//        type    - sort type
//                  values : kCaseInsensitive, kCaseSensitive
//                  default: kCaseInsensitive
//

   if (type == kCaseSensitive)
      qsort(strings, num, sizeof(strings), CaseSensitiveSort);
   else
      qsort(strings, num, sizeof(strings), CaseInsensitiveSort);
}


//______________________________________________________________________________
char *THtml::StrDup(const char *s1, Int_t n)
{
// Returns a pointer to a new string which is a duplicate
// of the string to which 's1' points.  The space for the
// new string is obtained using the 'new' operator. The new
// string has the length of 'strlen(s1) + n'.


   char *str = 0;

   if (s1) {
      if (n < 0)
         n = 0;
      str = new char[strlen(s1) + n + 1];
      if (str)
         strcpy(str, s1);
   }

   return (str);
}

//______________________________________________________________________________
void THtml::WriteHtmlHeader(ofstream & out, const char *title, 
                            const char* dir /*=""*/, TClass *cls/*=0*/)
{
// Write HTML header
//
//
// Input: out   - output file stream
//        title - title for the HTML page
//        cls   - current class
//        dir   - relative directory to reach the top 
//                ("" for html doc, "../" for src/*cxx.html etc)
//
// evaluates the Root.Html.Header setting:
// * if not set, the standard header is written. (ROOT)
// * if set, and ends with a "+", the standard header is written and this file included afterwards. (ROOT, USER)
// * if set but doesn't end on "+" the file specified will be written instead of the standard header (USER)
//
// Any occurrence of "%TITLE%" (without the quotation marks) in the user provided header file
// will be replaced by the value of this method's parameter "title" before written to the output file.
// %CLASS% is replaced by the class name ("" if not a class), %INCFILE% by the header file name as
// given by TClass::GetDeclFileName() and %SRCFILE% by the source file name as given by
// TClass::GetImplFileName() (both "" if not a class).

   const char *addHeader = gEnv->GetValue("Root.Html.Header", "");
   const char *charset = gEnv->GetValue("Root.Html.Charset", "ISO-8859-1");

   // standard header output if Root.Html.Header is not set, or it's set and it ends with a "+".
   if (addHeader
       && (strlen(addHeader) == 0
           || addHeader[strlen(addHeader) - 1] == '+')) {
      TDatime date;
      out << "<?xml version=\"1.0\"?>" << endl;
      out << "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\"" << endl
          << "\"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">" << endl;
      out << "<html xmlns=\"http://www.w3.org/1999/xhtml\" xml:lang=\"en\" lang=\"en\">" << endl;
      out << "<!--                                             -->" <<
          endl;
      out << "<!-- Author: ROOT team (rootdev@pcroot.cern.ch)  -->" <<
          endl;
      out << "<!--                                             -->" <<
          endl;
      out << "<!--   Date: " << date.AsString() << "            -->" << endl;
      out << "<!--                                             -->" <<
          endl;
      out << "<head>" << endl;
      out << "<meta http-equiv=\"Content-Type\" content=\"text/html; charset=" <<
          charset << "\" />" <<
          endl;
      out << "<title>";
      ReplaceSpecialChars(out, title);
      out << "</title>" << endl;
      out << "<meta name=\"rating\" content=\"General\" />" << endl;
      out << "<meta name=\"objecttype\" content=\"Manual\" />" << endl;
      out <<
          "<meta name=\"keywords\" content=\"software development, oo, object oriented, ";
      out << "unix, x11, windows, c++, html, rene brun, fons rademakers, cern\" />"
          << endl;
      out <<
          "<meta name=\"description\" content=\"ROOT - An Object Oriented Framework For Large Scale Data Analysis.\" />"
          << endl;
      out << "<link rel=\"stylesheet\" href=\"" << dir << "ROOT.css\" type=\"text/css\" id=\"ROOTstyle\" />" << endl;
      out << "</head>" << endl;

      out << "<body>" << endl;
   };
   // do we have an additional header?
   if (addHeader && strlen(addHeader) > 0) {
      ifstream addHeaderFile;
      char *addHeaderTmp = StrDup(addHeader);
      if (addHeaderTmp[strlen(addHeaderTmp) - 1] == '+')
         addHeaderTmp[strlen(addHeaderTmp) - 1] = 0;
      addHeaderFile.open(addHeaderTmp, ios::in);

      if (addHeaderFile.good()) {
         while (!addHeaderFile.eof()) {

            fLine.ReadLine(addHeaderFile, kFALSE);
            if (addHeaderFile.eof())
               break;

            if (fLine) {
               TString txt(fLine);
               txt.ReplaceAll("%TITLE%", title);
               txt.ReplaceAll("%CLASS%", cls?cls->GetName():"");
               txt.ReplaceAll("%INCFILE%", cls?GetDeclFileName(cls):"");
               txt.ReplaceAll("%SRCFILE%", cls?GetImplFileName(cls):"");
               out << txt << endl;
            }
         }
      } else
         Warning("THtml::WriteHtmlHeader",
                 "Can't open user html header file %s\n", addHeaderTmp);


      if (addHeaderTmp)
         delete[]addHeaderTmp;
   }

   out << "<a name=\"TopOfPage\"></a>" << endl;
}


//______________________________________________________________________________
void THtml::WriteHtmlFooter(ofstream & out, const char *dir,
                            const char *lastUpdate, const char *author,
                            const char *copyright)
{
// Write HTML footer
//
//
// Input: out        - output file stream
//        dir        - usually equal to "" or "../", depends of
//                     current file directory position, i.e. if
//                     file is in the fOutputDir, then dir will be ""
//        lastUpdate - last update string
//        author     - author's name
//        copyright  - copyright note
//
// Allows optional user provided footer to be written. Root.Html.Footer holds the file name for this footer.
// For details see THtml::WriteHtmlHeader (here, the "+" means the user's footer is written in front of Root's!)
// Occurences of %AUTHOR%, %UPDATE% and %COPYRIGHT% in the user's file are replaced by their corresponding
// values (author, lastUpdate and copyright) before written to out.

   out << endl;

   const char* templateSITags[kNumSourceInfos] = { "%UPDATE%", "%AUTHOR%", "%COPYRIGHT%"};
   const char *addFooter = gEnv->GetValue("Root.Html.Footer", "");
   // standard footer output if Root.Html.Footer is not set, or it's set and it ends with a "+".
   // do we have an additional footer?
   if (addFooter && strlen(addFooter) > 0) {
      ifstream addFooterFile;
      char *addFooterTmp = StrDup(addFooter);
      if (addFooterTmp[strlen(addFooterTmp) - 1] == '+')
         addFooterTmp[strlen(addFooterTmp) - 1] = 0;
      addFooterFile.open(addFooterTmp, ios::in);

      if (addFooterFile.good()) {
         while (!addFooterFile.eof()) {

            fLine.ReadLine(addFooterFile, kFALSE);
            if (addFooterFile.eof())
               break;

            if (fLine) {
               for (Int_t siTag = 0; siTag < (Int_t) kNumSourceInfos; ++siTag) {
                  Ssiz_t siPos = fLine.Index(templateSITags[siTag]);
                  if (siPos != kNPOS)
                     fLine.Replace(siPos, strlen(templateSITags[siTag]), fSourceInfo[siTag]);
               }
               out << fLine;
            }
         }
      } else
         Warning("THtml::WriteHtmlFooter",
                 "Can't open user html footer file %s\n", addFooterTmp);


      if (addFooterTmp)
         delete[]addFooterTmp;
   }

   if (addFooter
       && (strlen(addFooter) == 0
           || addFooter[strlen(addFooter) - 1] == '+')) {
      if (*author || *lastUpdate || *copyright)
         out << "<br />" << endl;

      out << "<!--SIGNATURE-->" << endl;

      // get the author( s )
      if (*author) {

         out << "<em>Author: ";

         char *auth = StrDup(author);

         char *name = strtok(auth, ",");

         Bool_t firstAuthor = kTRUE;

         /* now we have two options. name is a comma separated list of tokens
            either in the format
            (i) "FirstName LastName " or
            (ii) "FirstName LastName <link> "
            The first one generates an XWho link (CERN compatible),
            the second a http link (WORLD compatible), e.g.
            <mailto:user@host.bla> or <http://www.host.bla/page>.
          */

         do {
            char *ptr = name;
            // do we have a link for the current name?
            char *cLink = 0;

            // remove leading spaces
            while (*ptr && isspace((UChar_t)*ptr))
               ptr++;

            if (!firstAuthor)
               out << ", ";

            if (!strncmp(ptr, "Nicolas", 7)) {
               out << "<a href=http://pcbrun.cern.ch/nicolas/index.html";
               ptr += 12;
            } else {
               cLink = strchr(ptr, '<');        // look for link start tag
               if (cLink) {
                  out << "<a href=\"";
                  ptr = cLink-1;
                  for (cLink++; *cLink != 0 && *cLink != '>'; cLink++)
                     if (*cLink != ' ')
                        out << *cLink;
               } else {
                  out << "<a href=\"" << GetXwho();
                  while (*ptr && !cLink) {
                     // Valery's specific case
                     if (!strncmp(ptr, "Valery", 6)) {
                        out << "Valeri";
                        ptr += 6;
                     } else if (!strncmp(ptr, "Fine", 4)) {
                        out << "Faine";
                        ptr += 4;
                     }
                     while (*ptr && !isspace((UChar_t)*ptr))
                        out << *ptr++;

                     if (isspace((UChar_t)*ptr)) {
                        while (*ptr && isspace((UChar_t)*ptr))
                           ptr++;
                        if (isalpha(*ptr))
                           out << '+';
                        else
                           break;
                     } else
                        break;
                  }
               }
            }
            out << "\">";

            char *cCurrentPos = name;
            // remove blanks in front of and behind the name
            while (*cCurrentPos == ' ')
               cCurrentPos++;
            Bool_t bBlank = kFALSE;
            for (; cCurrentPos != ptr && *cCurrentPos != 0; cCurrentPos++) {
               if (*cCurrentPos != ' ') {
                  if (bBlank) {
                     out << ' ';
                     bBlank = kFALSE;
                  }
                  out << *cCurrentPos;
               } else
                  bBlank = kTRUE;
            }
            out << "</a>";
            while (ptr && *ptr==' ') ptr++;
            if (ptr && *ptr=='<') {
               // skip link
               while (*ptr && *ptr!='>') ptr++;
               if (ptr && *ptr=='>') ptr++;
            }
            while (ptr && *ptr==' ') ptr++;
            if (ptr && *ptr)
               out << ' ' << ptr;

            firstAuthor = kFALSE;
            name += strlen(name) + 1;

         } while ((name - auth) < (int) strlen(author)
                  && (name = strtok(name, ",")));
         out << "</em><br />" << endl;
         delete[]auth;
      }

      if (*lastUpdate)
         out << "<em>Last update: " << lastUpdate << "</em><br />" << endl;
      if (*copyright)
         out << "<em>Copyright " << copyright << "</em><br />" << endl;


      // this is a menu
      out << "<br />" << endl;
      out << "<hr />" << endl;
      out << "<center>" << endl;
      out << "<address>" << endl;

      // link to the ROOT home page
      out <<
          "<a href=\"http://root.cern.ch/root/Welcome.html\">ROOT page</a> - ";

      // link to the user home page( if exist )
      const char *userHomePage = gEnv->GetValue("Root.Html.HomePage", "");
      if (*userHomePage) {
         out << "<a href=\"";
         if (*dir) {
            if (strncmp(userHomePage, "http://", 7)
                && strncmp(userHomePage, "https://", 8)
                && !gSystem->IsAbsoluteFileName(userHomePage))
               out << dir;
         }
         out << userHomePage;
         out << "\">Home page</a> - ";
      }
      // link to the index file
      out << "<a href=\"";
      if (*dir)
         out << dir;
      out << "ClassIndex.html\">Class index</a> - ";

      // link to the hierarchy file
      out << "<a href=\"";
      if (*dir)
         out << dir;
      out << "ClassHierarchy.html\">Class Hierarchy</a> - ";

      // link to the top of the page
      out << "<a href=\"#TopOfPage\">Top of the page</a><br />" << endl;
      out << "</address>" << endl;

      out << "</center>" << endl;

      out << "<hr />" << endl;
      out << "<em>" << endl;
      out << "This page has been automatically generated. If you have any comments or suggestions ";
      out <<
          "about the page layout send a mail to <a href=\"mailto:rootdev@root.cern.ch\">ROOT support</a>, or ";
      out <<
          "contact <a href=\"mailto:rootdev@root.cern.ch\">the developers</a> with any questions or problems regarding ROOT."
          << endl;
      out << "</em>" << endl;
      out << "</body>" << endl;
      out << "</html>" << endl;
   }
}

//______________________________________________________________________________
void THtml::NameSpace2FileName(TString& name)
{
   // Replace "::" in name by "__"
   // Replace "<", ">", " ", ",", "~", "=" in name by "_"
   const char* replaceWhat = ":<> ,~=";
   for (Ssiz_t i=0; i < name.Length(); ++i)
      if (strchr(replaceWhat, name[i])) 
         name[i] = '_';
}
