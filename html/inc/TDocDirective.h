// @(#)root/html:$Id$
// Author: Axel Naumann 2007-01-25

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDocDirective
#define ROOT_TDocDirective

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TDocDirective                                                          //
//                                                                        //
// Special treatment of comments, like HTML source, a macro, or latex.    //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#include "TNamed.h"


class TClass;
class TDocParser;
class TDocOutput;
class THtml;
class TLatex;
class TMacro;
class TVirtualPad;

class TDocDirective: public TNamed {
protected:
   TDocParser* fDocParser;  // parser invoking this handler
   THtml*      fHtml;       // parser's THtml object
   TDocOutput* fDocOutput;  // parser invoking this handler
   TString     fParameters; // parameters to the directive
   Int_t       fCounter;    // counter to generate unique names, -1 to ignore

   virtual void AddParameter(const TString& /*name*/, const char* /*value*/ = 0) {}

   TDocDirective() {}
   TDocDirective(const char* name):
      TNamed(name, ""), fDocParser(0), fHtml(0), fDocOutput(0), fCounter(-1) {};
   virtual ~TDocDirective() {}

   const char* GetName() const { return TNamed::GetName(); }
   void GetName(TString& name) const;
   TDocParser* GetDocParser() const { return fDocParser; }
   TDocOutput* GetDocOutput() const { return fDocOutput; }
   THtml*      GetHtml() const { return fHtml; }
   const char* GetOutputDir() const;

   void SetParser(TDocParser* parser);
   void SetParameters(const char* params);
   void SetTag(const char* tag) { SetTitle(tag); }
   void SetCounter(Int_t count) { fCounter = count; }
   virtual void DeleteOutputFiles(const char* ext) const;

public:
   // get the tag ending this directive
   virtual const char* GetEndTag() const = 0;

   // add a line to the directive's text
   virtual void AddLine(const TSubString& line) = 0;

   // retrieve the result (replacement) of the directive; return false if invalid
   virtual Bool_t GetResult(TString& result) = 0;

   // Delete output for the parser's current class or module.
   virtual void DeleteOutput() const {}

   friend class TDocParser;

   ClassDef(TDocDirective, 0); // THtml directive handler
};

class TDocHtmlDirective: public TDocDirective {
private:
   TString fText;     // HTML text to be kept
   Bool_t  fVerbatim; // whether we are in a <pre></pre> block
public:
   TDocHtmlDirective(): TDocDirective("HTML"), fVerbatim(kFALSE) {}
   virtual ~TDocHtmlDirective() {}

   virtual void AddLine(const TSubString& line);
   virtual const char* GetEndTag() const { return "end_html"; }
   virtual Bool_t GetResult(TString& result);

   ClassDef(TDocHtmlDirective, 0); // Handler for "Begin_Html"/"End_Html" for raw HTML in documentation comments
};

class TDocMacroDirective: public TDocDirective {
private:
   TMacro* fMacro;         // macro to be executed
   Bool_t  fNeedGraphics;  // if set, we cannot switch to batch mode
   Bool_t  fShowSource;    // whether a source tab should be created
   Bool_t  fIsFilename;    // whether the directive is a failename to be executed

   virtual void AddParameter(const TString& name, const char* value = 0);
   TString CreateSubprocessInputFile();

public:
   TDocMacroDirective():
      TDocDirective("MACRO"), fMacro(0), fNeedGraphics(kFALSE),
      fShowSource(kFALSE), fIsFilename(kTRUE) {};
   virtual ~TDocMacroDirective();

   virtual void AddLine(const TSubString& line);
   virtual const char* GetEndTag() const { return "end_macro"; }
   virtual Bool_t GetResult(TString& result);
   // Delete output for the parser's current class or module.
   virtual void DeleteOutput() const { DeleteOutputFiles(".gif"); }

   static void SubProcess(const TString& what, const TString& out);

   ClassDef(TDocMacroDirective, 0); // Handler for "Begin_Macro"/"End_Macro" for code that is executed and that can generate an image for documentation
};

class TDocLatexDirective: public TDocDirective {
protected:
   TMacro*      fLatex;       // collection of lines
   Int_t        fFontSize;    // fontsize for current latex block, in pixels
   TString      fSeparator;   // column separator, often "="
   Bool_t       fSepIsRegexp; // whether fSeparator is a regexp expression
   TString      fAlignment;   // column alignment: 'l' for justify left, 'c' for center, 'r' for right
   TVirtualPad* fBBCanvas;    // canvas for bounding box determination

   virtual void    CreateLatex(const char* filename);
   virtual void    AddParameter(const TString& name, const char* value = 0);
   virtual void GetBoundingBox(TLatex& latex, const char* text, Float_t& width, Float_t& height);

public:
   TDocLatexDirective():
      TDocDirective("LATEX"), fLatex(0), fFontSize(16),
      fSepIsRegexp(kFALSE), fBBCanvas(0) {};
   virtual ~TDocLatexDirective();

   virtual void AddLine(const TSubString& line);
   virtual const char* GetEndTag() const {return "end_latex";}

   const char* GetAlignment() const {return fAlignment;}
   const char* GetSeparator() const {return fSeparator;}
   Bool_t SeparatorIsRegexp() const {return fSepIsRegexp;}
   Int_t  GetFontSize() const {return fFontSize;}
   TList* GetListOfLines() const;

   virtual Bool_t GetResult(TString& result);
   // Delete output for the parser's current class or module.
   virtual void DeleteOutput() const { DeleteOutputFiles(".gif"); }

   ClassDef(TDocLatexDirective, 0); // Handler for "Begin_Latex"/"End_Latex" to generate an image from latex
};

#endif // ROOT_TDocDirective
