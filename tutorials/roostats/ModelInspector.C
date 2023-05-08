/// \file
/// \ingroup tutorial_roostats
/// RooStats Model Inspector
///
/// Usage:
/// The usage is the same as the StandardXxxDemo.C macros.
/// The macro expects a root file containing a workspace with a ModelConfig and a dataset
///
/// ~~~{.cpp}
/// $ root
/// .L ModelInspector.C+
/// ModelInspector(fileName, workspaceName, modelConfigName, dataSetName);
/// ~~~
///
/// Drag the sliders to adjust the parameters of the model.
/// the min and max range of the sliders are used to define the upper & lower variation
/// the pointer position of the slider is the central blue curve.
///
/// Click the FIT button to
///
/// To Do:
///  - check boxes to specify which nuisance parameters used in making variation
///  - a button to make the profile inspector plots
///  - a check button to use MINOS errors
///  - have fit button show the covariance matrix from the fit
///  - a button to make the log likelihood plots
///  - a dialog to open the desired file
///  - ability to see the signal and background contributions?
///
/// \macro_code
///
///  - Version 1, October 2011
///     - based on tutorial macro by Bertrand Bellenot, Ilka Antcheva
///  - Version 2, November 2011
///     - fixes from Bertrand Bellenot for scrolling window for many parameters
///
/// \author Kyle Cranmer

#include "TGButton.h"
#include "TRootEmbeddedCanvas.h"
#include "TGLayout.h"
#include "TF1.h"
#include "TMath.h"
#include "TSystem.h"
#include "TCanvas.h"
#include "TGTextEntry.h"
#include "TGLabel.h"
#include "TGTripleSlider.h"
#include "RooWorkspace.h"
#include "RooStats/ModelConfig.h"
#include "TFile.h"
#include "RooArgSet.h"
#include "TList.h"
#include "RooAbsPdf.h"
#include "RooRealVar.h"
#include "RooPlot.h"
#include "TGButton.h"
#include <map>
#include "RooFitResult.h"
#include "TROOT.h"
#include <TApplication.h>
#include "RooSimultaneous.h"
#include "RooCategory.h"

enum ETestCommandIdentifiers {
   HId1,
   HId2,
   HId3,
   HCId1,
   HCId2,

   HSId1
};

using namespace std;
using namespace RooFit;
using namespace RooStats;

class ModelInspectorGUI : public TGMainFrame {

private:
   TRootEmbeddedCanvas *fCanvas;
   TGLayoutHints *fLcan;
   TF1 *fFitFcn;
   RooPlot *fPlot;
   RooWorkspace *fWS;
   TFile *fFile;
   ModelConfig *fMC;
   RooAbsData *fData;
   RooFitResult *fFitRes;

   TList fSliderList;
   TList fFrameList;
   vector<RooPlot *> fPlotList;
   map<TGTripleHSlider *, const char *> fSliderMap;
   map<TGTripleHSlider *, TGLabel *> fLabelMap;

   TGButton *fFitButton;
   TGTextButton *fExitButton;

   // BB: a TGCanvas and a vertical frame are needed for using scrollbars
   TGCanvas *fCan;
   TGVerticalFrame *fVFrame;

   TGHorizontalFrame *fHframe0, *fHframe1, *fHframe2;
   TGLayoutHints *fBly, *fBfly1, *fBfly2, *fBfly3;
   TGTripleHSlider *fHslider1;
   TGTextBuffer *fTbh1, *fTbh2, *fTbh3;
   TGCheckButton *fCheck1, *fCheck2;

public:
   ModelInspectorGUI(RooWorkspace *, ModelConfig *, RooAbsData *);
   virtual ~ModelInspectorGUI();

   void CloseWindow();
   void DoText(const char *text);
   void DoSlider();
   void DoSlider(const char *);
   void DoFit();
   void DoExit();
   void HandleButtons();
};


//______________________________________________________________________________
ModelInspectorGUI::ModelInspectorGUI(RooWorkspace *w, ModelConfig *mc, RooAbsData *data)
   : TGMainFrame(gClient->GetRoot(), 100, 100)
{

   RooMsgService::instance().getStream(1).removeTopic(RooFit::NumIntegration);
   fWS = w;
   fMC = mc;
   fData = data;
   RooSimultaneous *simPdf = NULL;
   Int_t numCats = 1;
   if (strcmp(fMC->GetPdf()->ClassName(), "RooSimultaneous") == 0) {
      cout << "Is a simultaneous PDF" << endl;
      simPdf = (RooSimultaneous *)(fMC->GetPdf());
      RooCategory *channelCat = (RooCategory *)(&simPdf->indexCat());
      cout << " with " << channelCat->numTypes() << " categories" << endl;
      numCats = channelCat->numTypes();
   } else {
      cout << "Is not a simultaneous PDF" << endl;
   }
   fFitRes = 0;

   SetCleanup(kDeepCleanup);

   // Create an embedded canvas and add to the main frame, centered in x and y
   // and with 30 pixel margins all around
   fCanvas = new TRootEmbeddedCanvas("Canvas", this, 600, 400);
   fLcan = new TGLayoutHints(kLHintsExpandX | kLHintsExpandY, 10, 10, 10, 10);
   AddFrame(fCanvas, fLcan);
   fPlotList.resize(numCats);
   if (numCats > 1) {
      fCanvas->GetCanvas()->Divide(numCats);
      for (int i = 0; i < numCats; ++i) {
         fCanvas->GetCanvas()->cd(i + 1)->SetBorderMode(0);
         fCanvas->GetCanvas()->cd(i + 1)->SetGrid();
      }
   }

   fHframe0 = new TGHorizontalFrame(this, 0, 0, 0);

   fCheck1 = new TGCheckButton(fHframe0, "&Constrained", HCId1);
   fCheck2 = new TGCheckButton(fHframe0, "&Relative", HCId2);
   fCheck1->SetState(kButtonUp);
   fCheck2->SetState(kButtonUp);
   fCheck1->SetToolTipText("Pointer position constrained to slider sides");
   fCheck2->SetToolTipText("Pointer position relative to slider position");

   fHframe0->Resize(200, 50);

   fHframe2 = new TGHorizontalFrame(this, 0, 0, 0);

   fFitButton = new TGTextButton(fHframe2, "Fit");
   fFitButton->Connect("Clicked()", "ModelInspectorGUI", this, "DoFit()");
   fExitButton = new TGTextButton(fHframe2, "Exit ");
   fExitButton->Connect("Clicked()", "ModelInspectorGUI", this, "DoExit()");

   fCheck1->Connect("Clicked()", "ModelInspectorGUI", this, "HandleButtons()");
   fCheck2->Connect("Clicked()", "ModelInspectorGUI", this, "HandleButtons()");

   fHframe2->Resize(100, 25);

   //--- layout for buttons: top align, equally expand horizontally
   fBly = new TGLayoutHints(kLHintsTop | kLHintsExpandX, 5, 5, 5, 5);

   //--- layout for the frame: place at bottom, right aligned
   fBfly1 = new TGLayoutHints(kLHintsTop | kLHintsCenterX, 5, 5, 5, 5);
   fBfly2 = new TGLayoutHints(kLHintsTop | kLHintsLeft, 5, 5, 5, 5);
   fBfly3 = new TGLayoutHints(kLHintsTop | kLHintsRight, 5, 5, 5, 5);

   fHframe2->AddFrame(fFitButton, fBfly2);
   fHframe2->AddFrame(fExitButton, fBfly3);

   AddFrame(fHframe0, fBly);
   AddFrame(fHframe2, fBly);

   // Loop over POI & NP, create slider
   // need maps of NP->slider? or just slider->NP
   RooArgSet parameters;
   parameters.add(*fMC->GetParametersOfInterest());
   parameters.add(*fMC->GetNuisanceParameters());
   TIter it = parameters.createIterator();
   RooRealVar *param = NULL;

   // BB: This is the part needed in order to have scrollbars
   fCan = new TGCanvas(this, 100, 100, kFixedSize);
   AddFrame(fCan, new TGLayoutHints(kLHintsExpandY | kLHintsExpandX));
   fVFrame = new TGVerticalFrame(fCan->GetViewPort(), 10, 10);
   fCan->SetContainer(fVFrame);
   // And that't it!
   // Obviously, the parent of other subframes is now fVFrame instead of "this"...

   while ((param = (RooRealVar *)it.Next())) {
      cout << "Adding Slider for " << param->GetName() << endl;
      TGHorizontalFrame *hframek = new TGHorizontalFrame(fVFrame, 0, 0, 0);

      TGLabel *hlabel =
         new TGLabel(hframek, Form("%s = %.3f +%.3f", param->GetName(), param->getVal(), param->getError()));
      TGTripleHSlider *hsliderk = new TGTripleHSlider(hframek, 190, kDoubleScaleBoth, HSId1, kHorizontalFrame,
                                                      GetDefaultFrameBackground(), kFALSE, kFALSE, kFALSE, kFALSE);
      hsliderk->Connect("PointerPositionChanged()", "ModelInspectorGUI", this, "DoSlider()");
      hsliderk->Connect("PositionChanged()", "ModelInspectorGUI", this, "DoSlider()");
      hsliderk->SetRange(param->getMin(), param->getMax());

      hframek->Resize(200, 25);
      fSliderList.Add(hsliderk);
      fFrameList.Add(hframek);

      hsliderk->SetPosition(param->getVal() - param->getError(), param->getVal() + param->getError());
      hsliderk->SetPointerPosition(param->getVal());

      hframek->AddFrame(hlabel, fBly);
      hframek->AddFrame(hsliderk, fBly);
      fVFrame->AddFrame(hframek, fBly);
      fSliderMap[hsliderk] = param->GetName();
      fLabelMap[hsliderk] = hlabel;
   }

   // Set main frame name, map sub windows (buttons), initialize layout
   // algorithm via Resize() and map main frame
   SetWindowName("RooFit/RooStats Model Inspector");
   MapSubwindows();
   Resize(GetDefaultSize());
   MapWindow();

   DoSlider();
}

//______________________________________________________________________________
ModelInspectorGUI::~ModelInspectorGUI()
{
   // Clean up

   Cleanup();
}

//______________________________________________________________________________
void ModelInspectorGUI::CloseWindow()
{
   // Called when window is closed via the window manager.

   delete this;
}

//______________________________________________________________________________
void ModelInspectorGUI::DoText(const char * /*text*/)
{
   // Handle text entry widgets.

   TGTextEntry *te = (TGTextEntry *)gTQSender;
   Int_t id = te->WidgetId();

   switch (id) {
   case HId1: fHslider1->SetPosition(atof(fTbh1->GetString()), static_cast<double>(fHslider1->GetMaxPosition())); break;
   case HId2: fHslider1->SetPointerPosition(atof(fTbh2->GetString())); break;
   case HId3: fHslider1->SetPosition(static_cast<double>(fHslider1->GetMinPosition()), atof(fTbh1->GetString())); break;
   default: break;
   }
   DoSlider();
}

//______________________________________________________________________________
void ModelInspectorGUI::DoFit()
{
   fFitRes = fMC->GetPdf()->fitTo(*fData, Save());
   map<TGTripleHSlider *, const char *>::iterator it;
   ;
   it = fSliderMap.begin();
   for (; it != fSliderMap.end(); ++it) {
      RooRealVar *param = fWS->var(it->second);
      param = (RooRealVar *)fFitRes->floatParsFinal().find(it->second);
      it->first->SetPosition(param->getVal() - param->getError(), param->getVal() + param->getError());
      it->first->SetPointerPosition(param->getVal());
   }
   DoSlider();
}

//______________________________________________________________________________
void ModelInspectorGUI::DoSlider(const char *text)
{
   cout << "." << text << endl;
}

//______________________________________________________________________________
void ModelInspectorGUI::DoSlider()
{
   // Handle slider widgets.

   // char buf[32];

   RooSimultaneous *simPdf = NULL;
   Int_t numCats = 0;
   if (strcmp(fMC->GetPdf()->ClassName(), "RooSimultaneous") == 0) {
      simPdf = (RooSimultaneous *)(fMC->GetPdf());
      RooCategory *channelCat = (RooCategory *)(&simPdf->indexCat());
      numCats = channelCat->numTypes();
   } else {
   }

   /////////////////////////////////////////////
   if (!simPdf) {
      /////////////////////////////////////////////
      // if not SimPdf
      /////////////////////////////////////////////

      // pre loop
      map<TGTripleHSlider *, const char *>::iterator it;
      ;
      delete fPlot;
      fPlot = ((RooRealVar *)fMC->GetObservables()->first())->frame();
      fData->plotOn(fPlot);
      double normCount;

      // high loop
      it = fSliderMap.begin();
      for (; it != fSliderMap.end(); ++it) {
         const char *name = it->second;
         fWS->var(name)->setVal(it->first->GetMaxPosition());
         RooRealVar *param = fWS->var(name);
         fLabelMap[it->first]->SetText(Form("%s = %.3f [%.3f,%.3f]", param->GetName(), it->first->GetPointerPosition(),
                                            it->first->GetMinPosition(), it->first->GetMaxPosition()));
      }
      normCount = fMC->GetPdf()->expectedEvents(*fMC->GetObservables());
      fMC->GetPdf()->plotOn(fPlot, LineColor(kRed), Normalization(normCount, RooAbsReal::NumEvent));

      // low loop
      it = fSliderMap.begin();
      for (; it != fSliderMap.end(); ++it) {
         const char *name = it->second;
         fWS->var(name)->setVal(it->first->GetMinPosition());
      }
      normCount = fMC->GetPdf()->expectedEvents(*fMC->GetObservables());
      fMC->GetPdf()->plotOn(fPlot, LineColor(kGreen), Normalization(normCount, RooAbsReal::NumEvent));

      // central oop
      it = fSliderMap.begin();
      for (; it != fSliderMap.end(); ++it) {
         const char *name = it->second;
         fWS->var(name)->setVal(it->first->GetPointerPosition());
      }
      normCount = fMC->GetPdf()->expectedEvents(*fMC->GetObservables());
      fMC->GetPdf()->plotOn(fPlot, LineColor(kBlue), Normalization(normCount, RooAbsReal::NumEvent));
      fPlot->Draw();

      fCanvas->GetCanvas()->Modified();
      fCanvas->GetCanvas()->Update();
      ////////////////////////////////////////////////////////////////////////////
   } else {
      ////////////////////////////////////////////////////////////////////////////
      // else (is simpdf)
      ////////////////////////////////////////////////////////////////////////////
      RooCategory *channelCat = (RooCategory *)(&simPdf->indexCat());
      //    TIterator* iter = simPdf->indexCat().typeIterator() ;
      Int_t frameIndex = 0;
      for (auto const& tt : *channelCat) {
         auto const& catName = tt.first;

         ++frameIndex;
         fCanvas->GetCanvas()->cd(frameIndex);

         // pre loop
         RooAbsPdf *pdftmp = simPdf->getPdf(catName.c_str());
         RooArgSet *obstmp = pdftmp->getObservables(*fMC->GetObservables());
         RooRealVar *obs = ((RooRealVar *)obstmp->first());

         fPlot = fPlotList.at(frameIndex - 1);
         if (fPlot)
            delete fPlot;
         fPlot = obs->frame();
         fPlotList.at(frameIndex - 1) = fPlot;

         RooFit::MsgLevel msglevel = RooMsgService::instance().globalKillBelow();
         RooMsgService::instance().setGlobalKillBelow(RooFit::WARNING);
         fData->plotOn(fPlot, MarkerSize(1),
                       Cut(Form("%s==%s::%s", channelCat->GetName(), channelCat->GetName(), catName.c_str())),
                       DataError(RooAbsData::None));
         RooMsgService::instance().setGlobalKillBelow(msglevel);

         map<TGTripleHSlider *, const char *>::iterator it;
         ;
         double normCount;

         // high loop
         it = fSliderMap.begin();
         for (; it != fSliderMap.end(); ++it) {
            const char *name = it->second;
            fWS->var(name)->setVal(it->first->GetMaxPosition());
            RooRealVar *param = fWS->var(name);
            fLabelMap[it->first]->SetText(Form("%s = %.3f [%.3f,%.3f]", param->GetName(),
                                               it->first->GetPointerPosition(), it->first->GetMinPosition(),
                                               it->first->GetMaxPosition()));
         }
         normCount = pdftmp->expectedEvents(*obs);
         pdftmp->plotOn(fPlot, LineColor(kRed), LineWidth(2.), Normalization(normCount, RooAbsReal::NumEvent));

         // low loop
         it = fSliderMap.begin();
         for (; it != fSliderMap.end(); ++it) {
            const char *name = it->second;
            fWS->var(name)->setVal(it->first->GetMinPosition());
            RooRealVar *param = fWS->var(name);
            fLabelMap[it->first]->SetText(Form("%s = %.3f [%.3f,%.3f]", param->GetName(),
                                               it->first->GetPointerPosition(), it->first->GetMinPosition(),
                                               it->first->GetMaxPosition()));
         }
         normCount = pdftmp->expectedEvents(*obs);
         pdftmp->plotOn(fPlot, LineColor(kGreen), LineWidth(2.), Normalization(normCount, RooAbsReal::NumEvent));

         // central loop
         it = fSliderMap.begin();
         for (; it != fSliderMap.end(); ++it) {
            const char *name = it->second;
            fWS->var(name)->setVal(it->first->GetPointerPosition());
            RooRealVar *param = fWS->var(name);
            fLabelMap[it->first]->SetText(Form("%s = %.3f [%.3f,%.3f]", param->GetName(),
                                               it->first->GetPointerPosition(), it->first->GetMinPosition(),
                                               it->first->GetMaxPosition()));
         }
         normCount = pdftmp->expectedEvents(*obs);
         if (!fFitRes)
            pdftmp->plotOn(fPlot, LineColor(kBlue), LineWidth(2.), Normalization(normCount, RooAbsReal::NumEvent));
         else {
            pdftmp->plotOn(fPlot, Normalization(normCount, RooAbsReal::NumEvent),
                           VisualizeError(*fFitRes, *fMC->GetNuisanceParameters()), FillColor(kYellow));
            pdftmp->plotOn(fPlot, LineColor(kBlue), LineWidth(2.), Normalization(normCount, RooAbsReal::NumEvent));
            msglevel = RooMsgService::instance().globalKillBelow();
            RooMsgService::instance().setGlobalKillBelow(RooFit::WARNING);
            fData->plotOn(fPlot, MarkerSize(1),
                          Cut(Form("%s==%s::%s", channelCat->GetName(), channelCat->GetName(), catName.c_str())),
                          DataError(RooAbsData::None));
            RooMsgService::instance().setGlobalKillBelow(msglevel);
         }
         fPlot->Draw();
      }
      fCanvas->GetCanvas()->Modified();
      fCanvas->GetCanvas()->Update();
      ///////////////////////////////////////////
      // end if(simPdf)
   }
}

//______________________________________________________________________________
void ModelInspectorGUI::HandleButtons()
{
   // Handle different buttons.

   TGButton *btn = (TGButton *)gTQSender;
   Int_t id = btn->WidgetId();

   switch (id) {
   case HCId1: fHslider1->SetConstrained(fCheck1->GetState()); break;
   case HCId2: fHslider1->SetRelative(fCheck2->GetState()); break;
   default: break;
   }
}
void ModelInspectorGUI::DoExit()
{
   printf("Exit application...");
   gApplication->Terminate(0);
}

void ModelInspector(const char *infile = "", const char *workspaceName = "combined",
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

   new ModelInspectorGUI(w, mc, data);
}
