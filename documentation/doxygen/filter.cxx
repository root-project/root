// @(#)root/test:$name:  $:$id: filter.cxx,v 1.0 exp $
// Author: O.Couet

//
//    filters doc files.
//


#include <stdlib.h>
#include <Riostream.h>
#include <time.h>
#include <TString.h>
#include <TROOT.h>
#include <TError.h>
#include <TRandom.h>
#include <TBenchmark.h>
#include <TSystem.h>
#include <TApplication.h>
#include <TDatime.h>
#include <TFile.h>
#include <TF1.h>
#include "TF2.h"
#include <TF3.h>
#include <TH2.h>
#include <TNtuple.h>
#include <TProfile.h>
#include "TString.h"

#include <TStyle.h>
#include <TCanvas.h>
#include <TColor.h>
#include <TFrame.h>
#include <TPostScript.h>
#include <TPDF.h>
#include <TLine.h>
#include <TMarker.h>
#include <TPolyLine.h>
#include <TLatex.h>
#include <TMathText.h>
#include <TLegend.h>
#include <TEllipse.h>
#include <TCurlyArc.h>
#include <TArc.h>
#include <TPaveText.h>
#include <TPaveStats.h>
#include <TPaveLabel.h>
#include <TGaxis.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TGraphAsymmErrors.h>
#include <TGraphBentErrors.h>
#include <TMultiGraph.h>
#include <TGraph2D.h>
#include <TParallelCoord.h>
#include <TImage.h>
#include <TMath.h>
#include <TSystem.h>


// Auxiliary functions
void    GetClassName();
void    StandardizeKeywords();
void    ExecuteMacro();
void    ExecuteCommand(TString);


// Global variables.
char    gLine[255];  // Current line in the current input file
TString gFileName;   // Input file name
TString gLineString; // Current line (as a TString) in the current input file
TString gClassName;  // Current class name
TString gImageName;  // Current image name
TString gMacroName;  // Current macro name
TString gCwd;        // Current working directory
TString gOutDir;     // Output directory
Bool_t  gHeader;     // True if the input file is a header
Bool_t  gSource;     // True if the input file is a source file
Bool_t  gInClassDef;
Bool_t  gClass;
Int_t   gInMacro;
Int_t   gImageID;
Int_t   gMacroID;


////////////////////////////////////////////////////////////////////////////////
/// Filter ROOT files for Doxygen.

int main(int argc, char *argv[])
{
   // Initialisation

   gFileName   = argv[1];
   gHeader     = kFALSE;
   gSource     = kFALSE;
   gInClassDef = kFALSE;
   gClass      = kFALSE;
   gInMacro    = 0;
   gImageID    = 0;
   gMacroID    = 0;
   if (gFileName.EndsWith(".cxx")) gSource = kTRUE;
   if (gFileName.EndsWith(".h"))   gHeader = kTRUE;
   GetClassName();

   // Retrieve the current working directory
   gCwd = gFileName(0, gFileName.Last('/'));

   // Retrieve the output directory
   gOutDir = gSystem->Getenv("DOXYGEN_OUTPUT_DIRECTORY");
   gOutDir.ReplaceAll("\"","");

   // Open the input file name.
   FILE *f = fopen(gFileName.Data(),"r");

   // File for inline macros.
   FILE *m = 0;

   // File header.
   if (gHeader) {
      while (fgets(gLine,255,f)) {
         gLineString = gLine;

         if (gLineString.BeginsWith("class"))    gInClassDef = kTRUE;
         if (gLineString.Index("ClassDef") >= 0) gInClassDef = kFALSE;

         if (gInClassDef && gLineString.Index("//") >= 0) {
            gLineString.ReplaceAll("//","///<");
         }

         printf("%s",gLineString.Data());
      }
      fclose(f);
      return 0;
   }

   // Source file.
   if (gSource) {
      while (fgets(gLine,255,f)) {
         gLineString = gLine;
         StandardizeKeywords();

         if (gLineString.Index("/*! \\class") >= 0) gClass = kTRUE;

         if (gLineString.Index("begin_html") >= 0) {
            if (!gClass) {
               gLineString = TString::Format("/*! \\class %s\n",gClassName.Data());
               gClass = kTRUE;
            } else {
               gLineString.ReplaceAll("begin_html","");
            }
         }

         if (gLineString.Index("end_html") >= 0) {
            gLineString.ReplaceAll("end_html","");
         }

         if (gLineString.Index("End_Macro") >= 0) {
            gLineString.ReplaceAll("End_Macro","");
            gInMacro = 0;
            if (m) {
               fclose(m);
               m = 0;
               ExecuteCommand(TString::Format("root -l -b -q \"makeimage.C(\\\"%s\\\",\\\"%s\\\",\\\"%s\\\")\""
                                              , TString::Format("%s_%3.3d.C", gClassName.Data(), gMacroID).Data()
                                              , TString::Format("%s_%3.3d.png", gClassName.Data(), gImageID).Data()
                                              , gOutDir.Data()));
               ExecuteCommand(TString::Format("rm %s_%3.3d.C", gClassName.Data(), gMacroID).Data());
            }
         }

         if (gInMacro) {
            if (gInMacro == 1) {
               if (gLineString.EndsWith(".C\n")) {
                  ExecuteMacro();
                  gInMacro++;
               } else {
                  gMacroID++;
                  m = fopen(TString::Format("%s_%3.3d.C", gClassName.Data()
                                                        , gMacroID)
                                                        , "w");
                  if (m) fprintf(m,"%s",gLineString.Data());
                  if (gLineString.BeginsWith("{")) {
                     gLineString.ReplaceAll("{"
                                             , TString::Format("\\include %s_%3.3d.C"
                                             , gClassName.Data()
                                             , gMacroID));
                  }
                  gInMacro++;
               }
            } else {
               if (m) fprintf(m,"%s",gLineString.Data());
               if (gLineString.BeginsWith("}")) {
                  gLineString.ReplaceAll("}"
                                          , TString::Format("\\image html %s_%3.3d.png"
                                          , gClassName.Data()
                                          , gImageID));
               } else {
                  gLineString = "\n";
               }
               gInMacro++;
            }
         }

         if (gLineString.Index("Begin_Macro") >= 0) {
            gImageID++;
            gInMacro++;
            gLineString = "\n";
         }

         printf("%s",gLineString.Data());
      }
      fclose(f);
      return 0;
   }

   // Output anything not header nor source
   while (fgets(gLine,255,f)) {
      gLineString = gLine;
      printf("%s",gLineString.Data());
   }
   fclose(f);
   return 0;
}


////////////////////////////////////////////////////////////////////////////////
/// Retrieve the class name.

void GetClassName()
{
   Int_t i1 = 0;
   Int_t i2 = 0;

   FILE *f = fopen(gFileName.Data(),"r");

   // File header.
   if (gHeader) {
      while (fgets(gLine,255,f)) {
         gLineString = gLine;
         if (gLineString.Index("ClassDef") >= 0) {
            i1 = gLineString.Index("(")+1;
            i2 = gLineString.Index(",")-1;
            gClassName = gLineString(i1,i2-i1+1);
            fclose(f);
            return;
         }
      }
   }

   // Source file.
   if (gSource) {
      while (fgets(gLine,255,f)) {
         gLineString = gLine;
         if (gLineString.Index("ClassImp") >= 0) {
            i1 = gLineString.Index("(")+1;
            i2 = gLineString.Index(")")-1;
            gClassName = gLineString(i1,i2-i1+1);
            fclose(f);
            return;
         }
      }
   }

   fclose(f);

   return;
}


////////////////////////////////////////////////////////////////////////////////
/// Standardize the THTML keywords to ease the parsing.

void StandardizeKeywords()
{
   gLineString.ReplaceAll("End_Html","end_html");
   gLineString.ReplaceAll("End_html","end_html");
   gLineString.ReplaceAll("end_html ","end_html");
   gLineString.ReplaceAll("Begin_Html","begin_html");
   gLineString.ReplaceAll("Begin_html","begin_html");
   gLineString.ReplaceAll("<big>","");
   gLineString.ReplaceAll("</big>","");
}


////////////////////////////////////////////////////////////////////////////////
/// Execute the macro in gLineString and produce the corresponding picture

void ExecuteMacro()
{
   // Name of the next Image to be generated
   gImageName = TString::Format("%s_%3.3d.png", gClassName.Data()
                                              , gImageID);

   // Retrieve the macro to be executed.
   if (gLineString.Index("../../..") >= 0) {
      gLineString.ReplaceAll("../../..","../..");
   } else {
      gLineString.Prepend(TString::Format("%s/../doc/macros/",gCwd.Data()));
   }
   Int_t i1 = gLineString.Last('/')+1;
   Int_t i2 = gLineString.Last('C');
   gMacroName = gLineString(i1,i2-i1+1);

   // Build the ROOT command to be executed.
   gLineString.Prepend(TString::Format("root -l -b -q \"makeimage.C(\\\""));
   Int_t l = gLineString.Length();
   gLineString.Replace(l-2,1,TString::Format("C\\\",\\\"%s\\\",\\\"%s\\\")\"",
                                             gImageName.Data(),
                                             gOutDir.Data()));

   ExecuteCommand(gLineString);

   gLineString = TString::Format("\\include %s\n\\image html %s\n", gMacroName.Data(),
                                                                    gImageName.Data());
}


////////////////////////////////////////////////////////////////////////////////
/// Execute a command making sure stdout will not go in the doxygen file.

void ExecuteCommand(TString command)
{
   int o = dup(fileno(stdout));
   freopen("stdout.dat","a",stdout);
   gSystem->Exec(command.Data());
   dup2(o,fileno(stdout));
   close(o);
}
