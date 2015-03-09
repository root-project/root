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
char    gLine[255];
TString gFileName;
TString gLineString;
TString gClassName;
TString gImageName;
TString gMacroName;
TString gCwd;
Bool_t  gHeader;
Bool_t  gSource;
Bool_t  gInClassDef;
Bool_t  gClass;
Int_t   gInMacro;
Int_t   gImageID;


//______________________________________________________________________________
int main(int argc, char *argv[])
{
   // Filter ROOT files for Doxygen.

   // Initialisation

   gFileName   = argv[1];
   gHeader     = kFALSE;
   gSource     = kFALSE;
   gInClassDef = kFALSE;
   gClass      = kFALSE;
   gInMacro    = 0;
   gImageID    = 0;
   if (gFileName.EndsWith(".cxx")) gSource = kTRUE;
   if (gFileName.EndsWith(".h"))   gHeader = kTRUE;
   GetClassName();
   gCwd = gFileName(0, gFileName.Last('/'));

   // Loop on file.
   FILE *f = fopen(gFileName.Data(),"r");

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
      return 0;
   }

   // Source file.
   if (gSource) {
      while (fgets(gLine,255,f)) {
         gLineString = gLine;
         StandardizeKeywords();

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

         if (gInMacro) {
            if (gInMacro == 1) {
               if (gLineString.EndsWith(".C\n")) {
                  ExecuteMacro();
                  gInMacro++;
               } else {
               }
            } else {
            }
         }

         if (gLineString.Index("Begin_Macro") >= 0) {
            gLineString = "";
            gInMacro++;
         }

         if (gLineString.Index("End_Macro") >= 0) {
            gLineString.ReplaceAll("End_Macro","");
            gInMacro = 0;
         }

         printf("%s",gLineString.Data());
      }
      return 0;
   }

   // Output anything not header nor source
   while (fgets(gLine,255,f)) {
      gLineString = gLine;
      printf("%s",gLineString.Data());
   }

   return 0;
}


//______________________________________________________________________________
void GetClassName()
{
   // Retrieve the class name.

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


//______________________________________________________________________________
void StandardizeKeywords()
{
   // Standardize the THTML keywords to ease the parsing.

   gLineString.ReplaceAll("End_Html","end_html");
   gLineString.ReplaceAll("End_html","end_html");
   gLineString.ReplaceAll("end_html ","end_html");
   gLineString.ReplaceAll("Begin_Html","begin_html");
   gLineString.ReplaceAll("Begin_html","begin_html");
   gLineString.ReplaceAll("<big>","");
   gLineString.ReplaceAll("</big>","");
}


//______________________________________________________________________________
void ExecuteMacro()
{
   // Execute the macro in gLineString and produce the corresponding picture

   // Retrieve the output directory
   TString OutDir = gSystem->Getenv("DOXYGEN_OUTPUT_DIRECTORY");
   OutDir.ReplaceAll("\"","");

   // Name of the next Image to be generated
   gImageName = TString::Format("%s_%3.3d.png", gClassName.Data()
                                              , gImageID++);

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
   gLineString.Replace(l-2,1,TString::Format("C\\\",\\\"%s/html/\\\",\\\"%s\\\")\"",
                                             OutDir.Data(),
                                             gImageName.Data()));

   ExecuteCommand(gLineString);

   gLineString = TString::Format("\\include %s\n\\image html %s\n", gMacroName.Data(),
                                                                    gImageName.Data());
}


//______________________________________________________________________________
void ExecuteCommand(TString command)
{
   // Execute a command making sure stdout will not go in the doxygen file.

   int o = dup(fileno(stdout));
   freopen("stdout.dat","a",stdout);
   gSystem->Exec(command.Data());
   dup2(o,fileno(stdout));
   close(o);
}