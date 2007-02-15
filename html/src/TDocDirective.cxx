#include "TDocDirective.h"

#include "TClass.h"
#include "TDocOutput.h"
#include "TDocParser.h"
#include "THtml.h"
#include "TInterpreter.h"
#include "TLatex.h"
#include "TMacro.h"
#include "TObjString.h"
#include "TPRegexp.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TSystem.h"
#include "TVirtualPad.h"
#include <fstream>

ClassImp(TDocDirective);

//______________________________________________________________________________
void TDocDirective::GetName(TString& name) const
{
   // Get the full name, based on fName, fTitle, fDocParser's tag.

   name = fName;
   if (fDocParser && fDocParser->GetCurrentClass()) {
      name += "_";
      name += fDocParser->GetCurrentClass()->GetName();
   }
   if (GetTitle() && strlen(GetTitle())) {
      name += "_";
      name += GetTitle();
   }
   if (fCounter != -1) {
      name += "_";
      name += fCounter;
   }
}

//______________________________________________________________________________
const char* TDocDirective::GetOutputDir() const
{
   // Get the directory for documentation output.

   return fHtml ? fHtml->GetOutputDir() : 0;
}

//______________________________________________________________________________
void TDocDirective::SetParameters(const char* params)
{
   // Given a string containing parameters in params,
   // we call AddParameter() for each of them.
   // This function splits the parameter names and
   // extracts their values if they are given.
   // Parameters are separated by ",", values are
   // separated from parameter names by "=".
   // params being
   //    a = "a, b, c", b='d,e'
   // will issue two calls to AddParameter(), one for 
   // a with value "a, b, c" and one for b with value
   // "d,e" (each without the quotation marks).

   fParameters = params; 

   if (!fParameters.Length())
      return;

   TString param;
   Ssiz_t pos = 0;
   while (fParameters.Tokenize(param, pos, ",")) {
      param = param.Strip(TString::kBoth);
      if (!param.Length())
         continue;

      Ssiz_t posAssign = param.Index('=');
      if (posAssign != kNPOS) {
         TString value(param(posAssign + 1, param.Length()));
         value = value.Strip(TString::kBoth);
         if (value[0] == '\'')
            value = value.Strip(TString::kBoth, '\'');
         else if (value[0] == '"')
            value = value.Strip(TString::kBoth, '"');
         param.Remove(posAssign, param.Length());
         param = param.Strip(TString::kBoth);
         AddParameter(param, value);
      } else {
         param = param.Strip(TString::kBoth);
         AddParameter(param, 0);
      }
   }
}

//______________________________________________________________________________
void TDocDirective::SetParser(TDocParser* parser)
{ 
   // Set the parser, and fDocOutput, fHtml from that
   fDocParser    = parser;
   fDocOutput = parser ? parser->GetDocOutput() : 0;
   fHtml      = fDocOutput? fDocOutput->GetHtml() : 0;
}

ClassImp(TDocHtmlDirective);

//______________________________________________________________________________
void TDocHtmlDirective::AddLine(const TSubString& line)
{
   // Add a line of HTML

   if (line.Start() == -1) return;

   TSubString iLine(line);
   Ssiz_t posPre = iLine.String().Index("pre>", line.Start(), TString::kIgnoreCase);
   if (posPre == kNPOS)
      fText += line;
   else {
      // remove <pre> in fVerbatim environments, and 
      // </pre> in !fVerbatim environments.
      while (posPre != kNPOS && posPre > 0) {
         Bool_t isOpen = line[posPre - 1 - line.Start()] == '<';
         if (fVerbatim) {
            if (isOpen) {
               // skip
               fText += iLine.String()(iLine.Start(), posPre - 2 - iLine.Start());
            } else {
               // write it out
               fText += iLine.String()(iLine.Start(), posPre + 4 - iLine.Start());
               fVerbatim = kFALSE;
            }
         } else {
            if (!isOpen) {
               // skip
               fText += iLine.String()(iLine.Start(), posPre - 1 - iLine.Start());
            } else {
               // write it out
               fText += iLine.String()(iLine.Start(), posPre + 4 - iLine.Start());
               fVerbatim = kTRUE;
            }
         }

         iLine = iLine.String()(posPre + 4, iLine.Length());
         posPre = iLine.String().Index("pre>", posPre + 4, TString::kIgnoreCase);
      }

      fText += iLine;
   }
   fText += "\n";
}

//______________________________________________________________________________
Bool_t TDocHtmlDirective::GetResult(TString& result)
{
   // Set result to the HTML code that was passed in via AddLine().
   // Prepend a closing </pre>, append an opening <pre>

   result = "</pre><!-- TDocHtmlDirective start -->";
   result += fText + "<!-- TDocHtmlDirective end --><pre>";
   return kTRUE;
}



ClassImp(TDocMacroDirective);

//______________________________________________________________________________
TDocMacroDirective::~TDocMacroDirective()
{
   // Destructor
   delete fMacro;
}

//______________________________________________________________________________
void TDocMacroDirective::AddLine(const TSubString& line)
{
   // Add a macro line

   if (!fMacro) {
      TString name;
      GetName(name);
      fMacro = new TMacro(name);
   }

   TString sLine(line);
   fMacro->AddLine(sLine);
   fIsFilename &= !sLine.Contains('{');
}

//______________________________________________________________________________
Bool_t TDocMacroDirective::GetResult(TString& result)
{
   // Get the result (i.e. an HTML img tag) for the macro invocation.
   // If fShowSource is set, a second tab will be created which shows
   // the source.

   if (!fMacro)
      return kFALSE;

   if (!fMacro->GetListOfLines() 
      || !fMacro->GetListOfLines()->First()) {
      Warning("GetResult", "Empty directive found!");
      return kTRUE;
   }

   if (gDebug > 3)
      Info("HandleDirective_Macro", "executing macro \"%s\" with %d lines.",
         fMacro->GetName(), fMacro->GetListOfLines() ? fMacro->GetListOfLines()->GetEntries() + 1 : 0);

   Bool_t wasBatch = gROOT->IsBatch();
   if (!wasBatch && !fNeedGraphics)
      gROOT->SetBatch();

   TVirtualPad* padSave = gPad;

   Int_t error = TInterpreter::kNoError;
   Long_t ret = 0;
   if (fIsFilename) {
      TString filename;
      TIter iLine(fMacro->GetListOfLines());
      while (filename.Length() == 0)
         filename = ((TObjString*)iLine())->String().Strip();

      TString pwd;
      if (GetHtml() && GetDocParser() && GetDocParser()->GetCurrentClass()) {
         pwd = GetHtml()->GetImplFileName(GetDocParser()->GetCurrentClass());
         pwd = gSystem->DirName(pwd);
      } else pwd = gSystem->pwd();
      TString macroPath(GetHtml()->GetMacroPath());
      const char* pathDelimiter = 
#ifdef R__WIN32
         ";";
#else
         ":";
#endif
      TObjArray* arrDirs(macroPath.Tokenize(pathDelimiter));
      TIter iDir(arrDirs);
      TObjString* osDir = 0;
      macroPath = "";
      while ((osDir = (TObjString*)iDir())) {
         if (!gSystem->IsAbsoluteFileName(osDir->String()))
            gSystem->PrependPathName(pwd, osDir->String());
         macroPath += osDir->String() + pathDelimiter;
      }

      TString plusplus;
      while (filename.EndsWith("+")) {
         plusplus += '+';
         filename.Remove(filename.Length() - 1);
      }

      TString fileSysName(filename);
      if (!gSystem->FindFile(macroPath, fileSysName)) {
         Error("GetResult", "Cannot find macro '%s' in path '%s'!", filename.Data(), macroPath.Data());
         result = "";
         return kFALSE;
      }

      if (fShowSource) {
         // copy macro into fMacro - before running it, in case the macro blocks its file
         std::ifstream ifMacro(fileSysName);
         fMacro->GetListOfLines()->Delete();
         while (ifMacro) {
            TString line;
            if (!line.ReadLine(ifMacro, kFALSE) || ifMacro.eof())
               break;
            fMacro->AddLine(line);
         }
      }

      fileSysName.Prepend(".x ");
      fileSysName += plusplus;
      ret = gROOT->ProcessLine(fileSysName, &error);
   } else {
      ret = fMacro->Exec(0, &error);
   }
   Int_t sleepCycles = 50; // 50 = 5 seconds
   while (error == TInterpreter::kProcessing && --sleepCycles > 0)
      gSystem->Sleep(100);

   gSystem->ProcessEvents(); // in case ret needs to handle some events first

   if (error != TInterpreter::kNoError)
      Error("HandleDirective_Macro", "Error processing macro %s!", fMacro->GetName());
   else if (ret) {
      const TObject* objRet = (const TObject*)ret;
      try {
         typeid(*objRet).name(); // needed to test whether ret is indeed an object with a vtable!
         objRet = dynamic_cast<const TObject*>(objRet);
      }
      catch (...) {
         objRet = 0;
      }
      if (objRet) {
         TString filename;
         GetName(filename);

         if (objRet->GetName() && strlen(objRet->GetName())) {
            filename += "_";
            filename += objRet->GetName();
         }
         filename.ReplaceAll(" ", "_");

         result = "<img class=\"macro\" alt=\"output of ";
         result += filename;

         GetDocOutput()->NameSpace2FileName(filename);
         TString id(filename);
         filename += ".gif";
         TString basename(filename);

         result += "\" title=\"MACRO\" src=\"";
         result += basename;
         result += "\" />";

         gSystem->PrependPathName(GetOutputDir(), filename);

         if (gDebug > 3)
            Info("HandleDirective_Macro", "Saving returned %s to file %s.",
               objRet->IsA()->GetName(), filename.Data());

         if (fNeedGraphics)
            // to get X11 to sync :-( gVirtualX->Update()/Sync() don't do it
            gSystem->Sleep(100);

         gSystem->ProcessEvents();
         objRet->SaveAs(filename);
         gSystem->ProcessEvents(); // SaveAs triggers an event
         
         if (objRet != gPad)
            delete objRet;

         if (fShowSource) {
            // TODO: we need an accessible version of the source, i.e. visible w/o javascript
            TString tags("</pre><div class=\"tabs\">\n"
               "<a id=\"" + id + "_A0\" class=\"tabsel\" href=\"" + basename + "\" onclick=\"javascript:return SetDiv('" + id + "',0);\">Picture</a>\n"
               "<a id=\"" + id + "_A1\" class=\"tab\" href=\"#\" onclick=\"javascript:return SetDiv('" + id + "',1);\">Source</a>\n"
               "<br /></div><div class=\"tabcontent\">\n"
               "<div id=\"" + id + "_0\" class=\"tabvisible\">" + result + "</div>\n"
               "<div id=\"" + id + "_1\" class=\"tabhidden\"><div class=\"code\"><pre class=\"code\">");
            TIter iLine(fMacro->GetListOfLines());
            TObjString* osLine = 0;
            while ((osLine = (TObjString*) iLine()))
               if (osLine->String().Strip().Length())
                  tags += osLine->String() + "\n";
            if (tags.EndsWith("\n"))
               tags.Remove(tags.Length()-1); // trailing line break
            tags += "</pre></div></div><div class=\"clear\"></div></div><pre>";
            result = tags;
         }
      }
   }

   if (!wasBatch)
      gROOT->SetBatch(kFALSE);
   if (padSave != gPad) {
      delete gPad;
      gPad = padSave;
   }

   // in case ret's or gPad's deletion provoke events that should be handled
   gSystem->ProcessEvents();

   return kTRUE;
}

//______________________________________________________________________________
void TDocMacroDirective::AddParameter(const TString& name, const char* /*value=0*/)
{
   // Setting fNeedGraphics if name is "GUI",
   // setting fShowSource if name is "SOURCE"

   if (!name.CompareTo("gui", TString::kIgnoreCase))
      fNeedGraphics = kTRUE;
   else if (!name.CompareTo("source", TString::kIgnoreCase))
      fShowSource = kTRUE;
   else Warning("AddParameter", "Unknown option %s!", name.Data());
}



namespace {
   Float_t gLinePadding = 10.; //px
   Float_t gColumnPadding = 10.; //px

   class TLatexLine {
   private:
      std::vector<Float_t> fWidths;
      Float_t fHeight;
      TObjArray* fColumns; // of TObjString*

   public:
      TLatexLine(TObjArray* columns = 0): 
         fHeight(0.), fColumns(columns) { if (columns) fWidths.resize(Size());}

      Float_t& Width(UInt_t col) {return fWidths[col];}
      Float_t& Height() {return fHeight;}
      TString* operator[](Int_t column) { 
         if (fColumns && fColumns->GetEntriesFast() > column)
            return &(((TObjString*)fColumns->At(column))->String());
         return 0;
      }
      UInt_t Size() const { return fColumns ? fColumns->GetEntries() : 0; }
      void Delete() { delete fColumns; }
   };
}

//______________________________________________________________________________
//
// Handle a "Begin_Latex"/"End_Latex" directive.
// called as
// "Begin_Latex(fontsize=10, separator='=,', rseparator='=|,', align=lcl)"
// will create and include a TLatex-processed image, with a given fontsize
// in pixels (defaults to 16). If (r)separator is given, the formulas on the 
// following lines will be grouped into columns; a new column starts with 
// (regexp) match of the separator; by default there is only one column.
// separator matches any character, rseparator matches as regexp with one 
// column per pattern match. Only one of separator or rseparator can be given.
// align defines the alignment for each columns; be default, all columns 
// are right aligned. NOTE that the column separator counts as a column itself!
//______________________________________________________________________________


ClassImp(TDocLatexDirective);

//______________________________________________________________________________
TDocLatexDirective::~TDocLatexDirective()
{
   // Destructor
   gSystem->ProcessEvents();
   delete fLatex;
   delete fBBCanvas;
   gSystem->ProcessEvents();
}

//______________________________________________________________________________
void TDocLatexDirective::AddLine(const TSubString& line)
{
   // Add a latex line

   if (line.Length() == 0)
      return;

   if (!fLatex) {
      TString name;
      GetName(name);
      fLatex = new TMacro(name);
   }

   TString sLine(line);
   GetDocParser()->Strip(sLine);
   if (sLine.Length() == 0)
      return;

   fLatex->AddLine(sLine);
}

//______________________________________________________________________________
void TDocLatexDirective::CreateLatex(const char* filename)
{
   // Create a gif file named filename from a latex expression in fLatex.
   // Called when "Begin_Latex"/"End_Latex" is processed.

   if (!fLatex
      || !fLatex->GetListOfLines()
      || !fLatex->GetListOfLines()->First())
      return;

   TVirtualPad* oldPad = gPad;

   Bool_t wasBatch = gROOT->IsBatch();
   if (!wasBatch)
      gROOT->SetBatch();

   const Float_t canvSize = 1200.;
   if (!fBBCanvas)
      // add magic batch vs. gui canvas sizes (4, 28)
      fBBCanvas = (TVirtualPad*)gROOT->ProcessLineFast(
         Form("new TCanvas(\"R__TDocLatexDirective_BBCanvas\",\"fBBCanvas\",%g,%g);", -(canvSize + 4.), canvSize + 28.));
   fBBCanvas->SetBorderMode(0);
   fBBCanvas->SetFillColor(kWhite);

   gSystem->ProcessEvents();

   std::list<TLatexLine> latexLines;
   std::vector<Float_t> maxWidth(20);
   UInt_t numColumns = 0;
   Float_t totalHeight = gLinePadding;

   TLatex latex;
   latex.SetTextFont(43);
   latex.SetTextSize((Float_t)fFontSize);
   latex.SetTextAlign(12);

   // calculate positions
   TIter iLine(fLatex->GetListOfLines());
   TObjString* line = 0;
   TPRegexp regexp;
   if (fSeparator.Length()) {
      if (fSepIsRegexp)
         regexp = TPRegexp(fSeparator);
   } else fSepIsRegexp = kFALSE;

   while ((line = (TObjString*) iLine())) {
      const TString& str = line->String();
      TObjArray* split = 0;
      if (!fSepIsRegexp) {
         split = new TObjArray();
         split->SetOwner();
      }
      if (!fSeparator.Length())
         split->Add(new TObjString(str));
      else {
         if (fSepIsRegexp)
            split = regexp.MatchS(str);
         else {
            Ssiz_t prevStart = 0;
            for (Ssiz_t pos = 0; pos < str.Length(); ++pos) {
               if (fSeparator.Index(str[pos]) != kNPOS) {
                  split->Add(new TObjString(TString(str(prevStart, pos - prevStart))));
                  split->Add(new TObjString(TString(str(pos, 1))));
                  prevStart = pos + 1;
               }
            }
            split->Add(new TObjString(TString(str(prevStart, str.Length() - prevStart))));
         }
      }

      latexLines.push_back(TLatexLine(split));
      if (numColumns < (UInt_t)split->GetEntries())
         numColumns = split->GetEntries();

      Float_t heightLine = -1.;
      for (UInt_t col = 0; col < (UInt_t)split->GetEntries(); ++col) {
         Float_t widthLatex = 0.;
         Float_t heightLatex = 0.;
         TString* strCol = latexLines.back()[col];
         if (strCol)
            GetBoundingBox(latex, *strCol, widthLatex, heightLatex);
         if (heightLine < heightLatex)   heightLine = heightLatex;
         if (maxWidth.size() < col)
            maxWidth.resize(col * 2);
         if (maxWidth[col] < widthLatex)
            maxWidth[col] = widthLatex;
         latexLines.back().Width(col) = widthLatex;
      }
      latexLines.back().Height() = heightLine;
      totalHeight += heightLine + gLinePadding;
   } // while next line

   std::vector<Float_t> posX(numColumns + 1);
   for (UInt_t col = 0; col <= numColumns; ++col) {
      if (col == 0) posX[col] = gColumnPadding;
      else          posX[col] = posX[col - 1] + maxWidth[col - 1] + gColumnPadding;
   }
   Float_t totalWidth = posX[numColumns];

   // draw
   fBBCanvas->Clear();
   fBBCanvas->cd();
   Float_t padSizeX = totalWidth;
   Float_t padSizeY = totalHeight + 8.;
   // add magic batch vs. gui canvas sizes (4, 28) + rounding
   TVirtualPad* padImg = (TVirtualPad*)gROOT->ProcessLineFast(
      Form("new TCanvas(\"R__TDocLatexDirective_padImg\",\"padImg\",-(Int_t)%g,(Int_t)%g);",
           padSizeX + 4.5, padSizeY + 28.5));
   padImg->SetBorderMode(0);
   padImg->SetFillColor(kWhite);
   padImg->cd();

   Float_t posY = 0.;
   for (std::list<TLatexLine>::iterator iLine = latexLines.begin();
      iLine != latexLines.end(); ++iLine) {
      posY += iLine->Height()/2. + gLinePadding;
      for (UInt_t iCol = 0; iCol < iLine->Size(); ++iCol) {
         TString* str = (*iLine)[iCol];
         if (!str) continue;
         char align = 'l';
         if ((UInt_t)fAlignment.Length() > iCol)
            align = fAlignment[(Int_t)iCol];
         Float_t x = posX[iCol];
         switch (align) {
            case 'l': break;
            case 'r': x += maxWidth[iCol] - iLine->Width(iCol); break;
            case 'c': x += 0.5*(maxWidth[iCol] - iLine->Width(iCol)); break;
            default:
               if (iLine == latexLines.begin())
                  Error("CreateLatex", "Invalid alignment character '%c'!", align);
         }
         latex.DrawLatex( x / padSizeX, 1. - posY / padSizeY, str->Data());
      }
      posY += iLine->Height()/2.;
   }

   padImg->Print(filename);

   // delete the latex objects
   for (std::list<TLatexLine>::iterator iLine = latexLines.begin();
      iLine != latexLines.end(); ++iLine) {
      iLine->Delete();
   }

   delete padImg;

   if (!wasBatch)
      gROOT->SetBatch(kFALSE);

   gPad = oldPad;
}

//______________________________________________________________________________
void TDocLatexDirective::GetBoundingBox(TLatex& latex, const char* text, Float_t& width, Float_t& height)
{
   // Determines the bounding box for text as height and width.
   // Assumes that we are in batch mode.

   UInt_t uiWidth = 0;
   UInt_t uiHeight = 0;
   fBBCanvas->cd();
   latex.SetText(0.1, 0.5, text);
   latex.GetBoundingBox(uiWidth, uiHeight);

   width = uiWidth;
   height = uiHeight;
}

//______________________________________________________________________________
TList* TDocLatexDirective::GetListOfLines() const
{
   // Get the list of lines as TObjStrings
   return fLatex ? fLatex->GetListOfLines() : 0;
}

//______________________________________________________________________________
Bool_t TDocLatexDirective::GetResult(TString& result)
{
   // convert fLatex to a gif by creating a TLatex, drawing it on a 
   // temporary canvas, and saving that to a filename in the output
   // directory.

   TString filename;
   GetName(filename);
   filename.ReplaceAll(" ", "_");
   const TString& firstLine = ((TObjString*)fLatex->GetListOfLines()->First())->String();
   TString latexFilename(firstLine);
   for (Ssiz_t namepos = 0; namepos < latexFilename.Length(); ++namepos)
      if (!GetDocParser()->IsWord(latexFilename[namepos])) {
         latexFilename.Remove(namepos, 1);
         --namepos;
      }
   filename += "_";
   filename += latexFilename;

   GetDocOutput()->NameSpace2FileName(filename);
   filename += ".gif";

   TString altText(firstLine);
   GetDocOutput()->ReplaceSpecialChars(altText);
   altText.ReplaceAll("\"", "&quot;");
   result = "<img class=\"latex\" alt=\"";
   result += altText;
   result += "\" title=\"LATEX\" src=\"";
   result += filename;
   result += "\" />";

   gSystem->PrependPathName(GetOutputDir(), filename);

   if (gDebug > 3)
      Info("HandleDirective_Latex", "Writing Latex \"%s\" to file %s.",
           fLatex->GetName(), filename.Data());

   CreateLatex(filename);

   return kTRUE;
}

//______________________________________________________________________________
void TDocLatexDirective::AddParameter(const TString& name, const char* value /*=0*/)
{
   // Parse fParameters, setting fFontSize, fAlignment, and fSeparator

   if (!name.CompareTo("fontsize", TString::kIgnoreCase)) {
      if (!value || !strlen(value))
         Error("AddParameter", "Option \"fontsize\" needs a value!");
      else fFontSize = atol(value);
   } else if (!name.CompareTo("separator", TString::kIgnoreCase)) {
      if (!value || !strlen(value))
         Error("AddParameter", "Option \"separator\" needs a value!");
      else fSeparator = value;
   } else if (!name.CompareTo("align", TString::kIgnoreCase)) {
      if (!value || !strlen(value))
         Error("AddParameter", "Option \"align\" needs a value!");
      else fAlignment = value;
   } else
      Warning("AddParameter", "Unknown option %s!", name.Data());
}
