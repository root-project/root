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
void GetClassName();

// Global variables.
char    gLine[255];
TString gFileName;
TString gLineString;
TString gClassName;
Bool_t  gHeader;
Bool_t  gSource;
Bool_t  gInClassDef;


//______________________________________________________________________________
int main(int argc, char *argv[])
{
   // Filter ROOT files for Doxygen.

   // Initialisation
   gFileName   = argv[1];
   gHeader     = kFALSE;
   gSource     = kFALSE;
   gInClassDef = kFALSE;
   if (gFileName.EndsWith(".cxx")) gSource = kTRUE;
   if (gFileName.EndsWith(".h"))   gHeader = kTRUE;
   GetClassName();

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
   }

   // Source file.
   if (gSource) {
      while (fgets(gLine,255,f)) {
         gLineString = gLine;

         if (gLineString.Index("Begin_Html") >= 0) {
            gLineString = TString::Format("/*! \\class %s\n",gClassName.Data());
         }

         if (gLineString.Index("Begin_Macro") >= 0) {
         }

         if (gLineString.Index("End_Macro") >= 0) {
         }

         printf("%s",gLineString.Data());
      }
   }

   TString opt1,opt0;
   opt0 = argv[0];
   opt1 = argv[1];
   printf("DEBUG %d : %s - %s %d %d - %s\n",argc,opt0.Data(),opt1.Data(),gSource,gHeader,gClassName.Data());
   fclose(f);

   return 1;
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
