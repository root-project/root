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


//______________________________________________________________________________
int main(int argc, char *argv[])
{
   // prototype of filter... does nothing right now.
   
   FILE *fin;
   int ch;
   
   switch (argc) {
      case 2:
         if ((fin = fopen(argv[1], "r")) == NULL) {
            // First string (%s) is program name (argv[0]).
            // Second string (%s) is name of file that could
            // not be opened (argv[1]).
            (void)fprintf(stderr, "%s: Cannot open input file %s\n", argv[0], argv[1]);
            return(2);
         }
         break;

      case 1:
         fin = stdin;
         break;
   
      default:
         (void)fprintf(stderr, "Usage: %s [file]\n", argv[0]);
         return(2);
   }
   
   while ((ch = getc(fin)) != EOF) (void)putchar(ch);

   fclose(fin);
   return (0);
}
