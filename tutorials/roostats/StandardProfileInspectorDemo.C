/// \file
/// \ingroup tutorial_roostats
/// \notebook -js
/// \brief Standard demo of the ProfileInspector class
/// StandardProfileInspectorDemo
///
/// This is a standard demo that can be used with any ROOT file
/// prepared in the standard way.  You specify:
///  - name for input ROOT file
///  - name of workspace inside ROOT file that holds model and data
///  - name of ModelConfig that specifies details for calculator tools
///  - name of dataset
///
/// With default parameters the macro will attempt to run the
/// standard hist2workspace example and read the ROOT file
/// that it produces.
///
/// The actual heart of the demo is only about 10 lines long.
///
/// The ProfileInspector plots the conditional maximum likelihood estimate
/// of each nuisance parameter in the model vs. the parameter of interest.
/// (aka. profiled value of nuisance parameter vs. parameter of interest)
/// (aka. best fit nuisance parameter with p.o.i fixed vs. parameter of interest)
///
/// \macro_image
/// \macro_output
/// \macro_code
///
/// \author Kyle Cranmer

#include "TFile.h"
#include "TROOT.h"
#include "TCanvas.h"
#include "TList.h"
#include "TMath.h"
#include "TSystem.h"
#include "RooWorkspace.h"
#include "RooAbsData.h"

#include "RooStats/ModelConfig.h"
#include "RooStats/ProfileInspector.h"

using namespace RooFit;
using namespace RooStats;

void StandardProfileInspectorDemo(const char *infile = "", const char *workspaceName = "combined",
                                  const char *modelConfigName = "ModelConfig", const char *dataName = "obsData")
{

   // -------------------------------------------------------
   // First part is just to access a user-defined file
   // or create the standard example file if it doesn't exist

   const char *filename = "";
   if (!strcmp(infile, "")) {
      filename = "results/example_combined_GaussExample_model.root";
      bool fileExist = !gSystem->AccessPathName(filename); // note opposite return code
      // if file does not exists generate with histfactory
      if (!fileExist) {
#ifdef _WIN32
         cout << "HistFactory file cannot be generated on Windows - exit" << endl;
         return;
#endif
         // Normally this would be run on the command line
         cout << "will run standard hist2workspace example" << endl;
         gROOT->ProcessLine(".! prepareHistFactory .");
         gROOT->ProcessLine(".! hist2workspace config/example.xml");
         cout << "\n\n---------------------" << endl;
         cout << "Done creating example input" << endl;
         cout << "---------------------\n\n" << endl;
      }

   } else
      filename = infile;

   // Try to open the file
   TFile *file = TFile::Open(filename);

   // if input file was specified byt not found, quit
   if (!file) {
      cout << "StandardRooStatsDemoMacro: Input file " << filename << " is not found" << endl;
      return;
   }

   // -------------------------------------------------------
   // Tutorial starts here
   // -------------------------------------------------------

   // get the workspace out of the file
   RooWorkspace *w = (RooWorkspace *)file->Get(workspaceName);
   if (!w) {
      cout << "workspace not found" << endl;
      return;
   }

   // get the modelConfig out of the file
   ModelConfig *mc = (ModelConfig *)w->obj(modelConfigName);

   // get the modelConfig out of the file
   RooAbsData *data = w->data(dataName);

   // make sure ingredients are found
   if (!data || !mc) {
      w->Print();
      cout << "data or ModelConfig was not found" << endl;
      return;
   }

   // -----------------------------
   // now use the profile inspector
   ProfileInspector p;
   TList *list = p.GetListOfProfilePlots(*data, mc);

   // now make plots
   TCanvas *c1 = new TCanvas("c1", "ProfileInspectorDemo", 800, 200);
   if (list->GetSize() > 4) {
      double n = list->GetSize();
      int nx = (int)sqrt(n);
      int ny = TMath::CeilNint(n / nx);
      nx = TMath::CeilNint(sqrt(n));
      c1->Divide(ny, nx);
   } else
      c1->Divide(list->GetSize());
   for (int i = 0; i < list->GetSize(); ++i) {
      c1->cd(i + 1);
      list->At(i)->Draw("al");
   }

   cout << endl;
}
