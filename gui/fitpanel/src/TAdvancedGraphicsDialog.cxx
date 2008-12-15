#include "TAdvancedGraphicsDialog.h"
#include "TGraph.h"
#include "TAxis.h"
#include "TPad.h"

TAdvancedGraphicsDialog::TAdvancedGraphicsDialog(const TGWindow *p, const TGWindow *main):
   TGTransientFrame(p, main, 10, 10, kVerticalFrame), 
   fFitter((TBackCompFitter *) TVirtualFitter::GetFitter())
{
   // Creates the Advanced Graphics Dialog.


   if (!p && !main) {
      MakeZombie();
      return;
   }
   SetCleanup(kDeepCleanup);

   fMainFrame = new TGVerticalFrame(this);

   fTab = new TGTab(fMainFrame, 10, 10);
   fMainFrame->AddFrame(fTab, new TGLayoutHints(kLHintsExpandY | kLHintsExpandX, 5,5,5,0));
   fTab->SetCleanup(kDeepCleanup);
   fTab->Associate(this);

   // Add the first method to the dialog (Contour)
   CreateContourFrame();
   fTab->AddTab("Contour", fContourFrame);

   // Add the second method to the dialog (Scan)
   CreateScanFrame();
   fTab->AddTab("Scan", fScanFrame);

   TGCompositeFrame * frame = new TGHorizontalFrame(fMainFrame);

   fDraw = new TGTextButton(frame, "&Draw", kAGD_BDRAW);
   fDraw->Associate(this);
   frame->AddFrame(fDraw, new TGLayoutHints(kLHintsCenterY | kLHintsExpandX, 5, 5, 0, 0));

   fClose = new TGTextButton(frame, "&Close", kAGD_BCLOSE);
   fClose->Associate(this);
   frame->AddFrame(fClose, new TGLayoutHints(kLHintsCenterY | kLHintsExpandX, 5, 5, 0, 0));

   UInt_t width = 0, height = 0;
   height = fClose->GetDefaultHeight();
   width  = TMath::Max(width, fClose->GetDefaultWidth());
   frame->Resize((width + 20) * 2, height);

   fMainFrame->AddFrame(frame, new TGLayoutHints(kLHintsBottom | kLHintsRight, 0, 0, 0, 0));

   this->AddFrame(fMainFrame, new TGLayoutHints(kLHintsNormal | kLHintsExpandX,0,0,5,5));

   ConnectSlots();

   SetWindowName("Advanced Drawing Tools");

   // map all widgets and calculate size of dialog
   MapSubwindows();

   width  = GetDefaultWidth();
   height = GetDefaultHeight();

   Resize(width, height);

   // position relative to the parent's window
   CenterOnParent();

   // make the message box non-resizable
   SetWMSize(width, height);
   SetWMSizeHints(width, height, width, height, 0, 0);

   SetMWMHints(kMWMDecorAll | kMWMDecorResizeH  | kMWMDecorMaximize |
               kMWMDecorMinimize | kMWMDecorMenu,
               kMWMFuncAll  | kMWMFuncResize    | kMWMFuncMaximize |
               kMWMFuncMinimize,
               kMWMInputModeless);

   // popup dialog and wait till user replies
   MapWindow();
   gClient->WaitFor(this);
}

//______________________________________________________________________________
void TAdvancedGraphicsDialog::CreateContourFrame()
{
   // Create the frame that contains all the necessary information for
   // the Contour method.

   fContourFrame = new TGVerticalFrame(fTab);
   TGHorizontalFrame* frame = new TGHorizontalFrame(fContourFrame);

   TGLabel* label = new TGLabel(frame, "Number of Points: ");
   frame->AddFrame(label, new TGLayoutHints(kLHintsTop | kLHintsLeft, 5, 5, 5, 0));

   fContourPoints = new TGNumberEntry(frame, 40, 
                                5, kAGD_SCANMIN,
                                TGNumberFormat::kNESInteger,
                                TGNumberFormat::kNEAPositive,
                                TGNumberFormat::kNELNoLimits);
   fContourPoints->Resize(130, 20);
   frame->AddFrame(fContourPoints, new TGLayoutHints(kLHintsNormal, 8, 0, 5, 0));
   fContourFrame->AddFrame(frame, new TGLayoutHints(kLHintsExpandX, 5, 5, 0, 0));

   frame = new TGHorizontalFrame(fContourFrame);
   label = new TGLabel(frame, "Parameter 1: ");
   frame->AddFrame(label, new TGLayoutHints(kLHintsTop | kLHintsLeft, 5, 5, 5, 0));

   fContourPar1 = new TGComboBox(frame, kAGD_CONTPAR1);
   AddParameters(fContourPar1);
   fContourPar1->Resize(130, 20);
   fContourPar1->Associate(this);
   TGListBox *lb = fContourPar1->GetListBox();
   lb->Resize(lb->GetWidth(), 200);
   frame->AddFrame(fContourPar1, new TGLayoutHints(kLHintsNormal, 37, 0, 5, 0));
   fContourFrame->AddFrame(frame, new TGLayoutHints(kLHintsExpandX, 5, 5, 0, 0));

   frame = new TGHorizontalFrame(fContourFrame);

   label = new TGLabel(frame, "Parameter 2: ");
   frame->AddFrame(label, new TGLayoutHints(kLHintsTop | kLHintsLeft, 5, 5, 5, 0));

   fContourPar2 = new TGComboBox(frame, kAGD_CONTPAR2);
   AddParameters(fContourPar2);
   fContourPar2->Select(kAGD_PARCOUNTER+1, kFALSE);
   fContourPar2->Resize(130, 20);
   fContourPar2->Associate(this);
   lb = fContourPar2->GetListBox();
   lb->Resize(lb->GetWidth(), 200);
   frame->AddFrame(fContourPar2, new TGLayoutHints(kLHintsNormal, 37, 0, 5, 0));

   fContourFrame->AddFrame(frame, new TGLayoutHints(kLHintsExpandX, 5, 5, 0, 0));

   frame = new TGHorizontalFrame(fContourFrame);

   label = new TGLabel(frame, "Confidence Level: ");
   frame->AddFrame(label, new TGLayoutHints(kLHintsTop | kLHintsLeft, 5, 5, 5, 0));

   fContourError = new TGNumberEntry(frame, 0.683, 5, kAGD_CONTERR,
                                     TGNumberFormat::kNESRealThree,
                                     TGNumberFormat::kNEANonNegative,
                                     TGNumberFormat::kNELNoLimits);
   fContourError->Resize(130, 20);
   frame->AddFrame(fContourError, new TGLayoutHints(kLHintsNormal, 5, 0, 5, 0));

   fContourFrame->AddFrame(frame, new TGLayoutHints(kLHintsExpandX, 5, 5, 0, 5));
}

//______________________________________________________________________________
void TAdvancedGraphicsDialog::CreateScanFrame()
{
   // Create the frame that contains all the necessary information for
   // the Scan method.

   fScanFrame = new TGVerticalFrame(fTab);
   TGHorizontalFrame* frame = new TGHorizontalFrame(fScanFrame);

   TGLabel* label = new TGLabel(frame, "Number of Points: ");
   frame->AddFrame(label, new TGLayoutHints(kLHintsTop | kLHintsLeft, 5, 5, 5, 0));

   fScanPoints = new TGNumberEntry(frame, 40, 
                                5, kAGD_SCANMIN,
                                TGNumberFormat::kNESInteger,
                                TGNumberFormat::kNEAPositive,
                                TGNumberFormat::kNELNoLimits);
   fScanPoints->Resize(140, 20);
   frame->AddFrame(fScanPoints, new TGLayoutHints(kLHintsNormal, 0, 0, 5, 0));
   fScanFrame->AddFrame(frame, new TGLayoutHints(kLHintsExpandX, 5, 5, 0, 0));

   frame = new TGHorizontalFrame(fScanFrame);

   label = new TGLabel(frame, "Parameter: ");
   frame->AddFrame(label, new TGLayoutHints(kLHintsTop | kLHintsLeft, 5, 5, 5, 0));

   fScanPar = new TGComboBox(frame, kAGD_SCANPAR);
   AddParameters(fScanPar);
   fScanPar->Resize(140, 20);
   fScanPar->Associate(this);
   TGListBox *lb = fScanPar->GetListBox();
   lb->Resize(lb->GetWidth(), 200);
   frame->AddFrame(fScanPar, new TGLayoutHints(kLHintsNormal, 39, 0, 5, 0));
   fScanFrame->AddFrame(frame, new TGLayoutHints(kLHintsExpandX, 5, 5, 0, 0));

   frame = new TGHorizontalFrame(fScanFrame);
   label = new TGLabel(frame, "Min: ");
   frame->AddFrame(label, new TGLayoutHints(kLHintsTop | kLHintsLeft, 5, 5, 5, 0));

   double val = fFitter->GetParameter( fScanPar->GetSelected() - kAGD_PARCOUNTER );
   double err = fFitter->GetParError( fScanPar->GetSelected() - kAGD_PARCOUNTER ); 
   fScanMin = new TGNumberEntry(frame, val - 2.*err , 
                                5, kAGD_SCANMIN,
                                TGNumberFormat::kNESRealFour,
                                TGNumberFormat::kNEAAnyNumber,
                                TGNumberFormat::kNELNoLimits);
   fScanMin->Resize(70, 20);
   frame->AddFrame(fScanMin, new TGLayoutHints(kLHintsNormal, 2, 0, 5, 0));

   label = new TGLabel(frame, "Max: ");
   frame->AddFrame(label, new TGLayoutHints(kLHintsTop | kLHintsLeft, 35, 5, 5, 0));
   fScanMax = new TGNumberEntry(frame, val + 2.*err,
                                5, kAGD_SCANMAX,
                                TGNumberFormat::kNESRealFour,
                                TGNumberFormat::kNEAAnyNumber,
                                TGNumberFormat::kNELNoLimits);
   fScanMax->Resize(70, 20);
   frame->AddFrame(fScanMax, new TGLayoutHints(kLHintsNormal, 2, 0, 5, 0));
   fScanFrame->AddFrame(frame, new TGLayoutHints(kLHintsExpandX, 5, 5, 0, 0));
   
}

void TAdvancedGraphicsDialog::AddParameters(TGComboBox* comboBox) 
{
   // Add all the parameters of the VirtualFitter into a comboBox
   // (helper method)

   for ( Int_t i = 0; i < fFitter->GetNumberTotalParameters(); ++i ) {
      comboBox->AddEntry(fFitter->GetParName(i), kAGD_PARCOUNTER + i);
   }
   comboBox->Select(kAGD_PARCOUNTER, kFALSE);
}

//______________________________________________________________________________
void TAdvancedGraphicsDialog::ConnectSlots()
{
   // Connect the slots (buttons mainly + specific methods)

   // Buttons
   fClose->Connect("Clicked()", "TAdvancedGraphicsDialog", this, "CloseWindow()");
   fDraw->Connect("Clicked()", "TAdvancedGraphicsDialog", this, "DoDraw()");

   // Slots for the Scan method
   fScanPar->Connect("Selected(Int_t)", "TAdvancedGraphicsDialog", this, "DoChangedScanPar(Int_t)");
}

//______________________________________________________________________________
void TAdvancedGraphicsDialog::DoChangedScanPar(Int_t selected)
{
   // Changes the Min and Max default values of the scan method,
   // depending on the selected parameter.

   double val = fFitter->GetParameter( selected - kAGD_PARCOUNTER );
   double err = fFitter->GetParError(  selected - kAGD_PARCOUNTER ); 
   fScanMin->SetNumber( val -2 * err );
   fScanMax->SetNumber( val +2 * err );
}

//______________________________________________________________________________
void TAdvancedGraphicsDialog::DoDraw()
{
   // Calls the correspoding method, depending on the selected tab.

   if ( fTab->GetCurrent() == 0 ) {
      DrawContour();
   } else if ( fTab->GetCurrent() == 1 ) {
      DrawScan();
   }
}

//______________________________________________________________________________
void TAdvancedGraphicsDialog::DrawContour()
{
   // Generates all necessary data for the Contour method from its
   // tab. Then it call Virtual Fitter to perform it.

   static TGraph * graph = 0;
   if ( graph )
      delete graph;
   graph = new TGraph( static_cast<int>(fContourPoints->GetNumber()) ); 
   Int_t par1 = fContourPar1->GetSelected() - kAGD_PARCOUNTER;
   Int_t par2 = fContourPar2->GetSelected() - kAGD_PARCOUNTER;
   if ( par1 == par2 ) {
      Error("TAdvancedGraphicsDialog::DrawContour", "Parameters cannot be the same");
      return;
   }
   // contour error is actually the desired confidence level
   Double_t cl = fContourError->GetNumber(); 
   fFitter->Contour( par1, par2, graph, cl);
   graph->SetFillColor(kYellow-10);
   graph->GetXaxis()->SetTitle( fFitter->GetParName(par1) );
   graph->GetYaxis()->SetTitle( fFitter->GetParName(par2) );
   graph->Draw("ALF"); 
   gPad->Update();
}

//______________________________________________________________________________
void TAdvancedGraphicsDialog::DrawScan()
{
   // Generates all necessary data for the Scan method from its
   // tab. Then it call Virtual Fitter to perform it.

   static TGraph * graph = 0;
   if ( graph )
      delete graph;
   graph = new TGraph( static_cast<int>(fScanPoints->GetNumber()) );
   Int_t par = fScanPar->GetSelected() - kAGD_PARCOUNTER;
   fFitter->Scan( par, graph, 
                  fScanMin->GetNumber(),
                  fScanMax->GetNumber() ); 
   graph->SetLineColor(kBlue);
   graph->SetLineWidth(2);
   graph->GetXaxis()->SetTitle(fFitter->GetParName(par) );
   graph->GetYaxis()->SetTitle("FCN" );
   graph->Draw("APL");
   gPad->Update();
}

//______________________________________________________________________________
TAdvancedGraphicsDialog::~TAdvancedGraphicsDialog()
{
   // Cleanup dialog.

   Cleanup();
}
