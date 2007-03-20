
// Simple macro showing capabilities of TGSpeedo widget.
//Author: Bertrand Bellenot
   
#include "TSystem.h"
#include "TGFrame.h"
#include "TGWindow.h"
#include "TGSpeedo.h"

class TGShapedMain : public TGMainFrame {

protected:
   TGSpeedo          *fSpeedo;          // analog meter
   Int_t              fActInfo;         // actual information value
   Bool_t             fRunning;         // kTRUE while updating infos

public:
   TGShapedMain(const TGWindow *p, int w, int h);
   virtual ~TGShapedMain();

   void              CloseWindow();
   TGSpeedo         *GetSpeedo() const { return fSpeedo; }
   Int_t             GetActInfo() const { return fActInfo; }
   Bool_t            IsRunning() const { return fRunning; }
   void              ToggleInfos();
};


//______________________________________________________________________________
TGShapedMain::TGShapedMain(const TGWindow *p, int w, int h) :
   TGMainFrame(p, w, h)
{
   // Constructor.

   fActInfo = 1;
   fRunning = kTRUE;

   fSpeedo = new TGSpeedo(this, 0.0, 100.0, "CPU", "[%]");
   fSpeedo->Connect("OdoClicked()", "TGShapedMain", this, "ToggleInfos()");
   fSpeedo->Connect("LedClicked()", "TGShapedMain", this, "CloseWindow()");
   TGMainFrame::Connect("CloseWindow()", "TGShapedMain", this, "CloseWindow()");
   AddFrame(fSpeedo, new TGLayoutHints(kLHintsCenterX | kLHintsCenterX, 0, 0, 0, 0));
   fSpeedo->SetDisplayText("Used RAM", "[MB]");

   MapSubwindows();
   MapWindow();
   // To avoid closing the window while TGSpeedo is drawing
   DontCallClose();
   Resize(GetDefaultSize());
   // Set fixed size
   SetWMSizeHints(GetDefaultWidth(), GetDefaultHeight(), GetDefaultWidth(), GetDefaultHeight(), 1, 1);
   SetWindowName("ROOT CPU Load Meter");
}

//______________________________________________________________________________
void TGShapedMain::ToggleInfos()
{
   // Toggle information displayed in Analog Meter

   if (fActInfo < 2)
      fActInfo++;
   else
      fActInfo = 0;
   if (fActInfo == 0)
      fSpeedo->SetDisplayText("Total RAM", "[MB]");
   else if (fActInfo == 1)
      fSpeedo->SetDisplayText("Used RAM", "[MB]");
   else if (fActInfo == 2)
      fSpeedo->SetDisplayText("Free RAM", "[MB]");
}

//______________________________________________________________________________
TGShapedMain::~TGShapedMain()
{
   // Destructor.

}

//______________________________________________________________________________
void TGShapedMain::CloseWindow()
{
   // Close Window.

   // stop updating
   fRunning = kFALSE;
   // reset kDontCallClose bit to be able to really close the window
   ResetBit(kDontCallClose);
}

//______________________________________________________________________________
void CPUMeter()
{
   // Main application.

   MemInfo_t memInfo;
   CpuInfo_t cpuInfo;
   Float_t act_load, prev_load = 0.0;
   Int_t i, memUsage, old_memUsage = 0;

   TGShapedMain *mainWindow = new TGShapedMain(0, 500, 200);
   TGSpeedo *speedo = mainWindow->GetSpeedo();

   // set threshold values
   speedo->SetThresholds(12.5, 50.0, 87.5);
   // set threshold colors
   speedo->SetThresholdColors(TGSpeedo::kGreen, TGSpeedo::kOrange, TGSpeedo::kRed);
   // enable threshold
   speedo->EnableThreshold();
   speedo->SetScaleValue(0.0, 5);
   // enable peak marker
   speedo->EnablePeakMark();
   // update the TGSpeedo widget
   gSystem->GetCpuInfo(&cpuInfo);
   gSystem->GetMemInfo(&memInfo);
   gSystem->ProcessEvents();
   while (mainWindow->IsRunning() && mainWindow->IsMapped()) {
      // Get CPU informations
      gSystem->GetCpuInfo(&cpuInfo, 100);
      // actual CPU load
      act_load = cpuInfo.fTotal;
      // Get Memory informations
      gSystem->GetMemInfo(&memInfo);
      // choose which value to display
      if (mainWindow->GetActInfo() == 0)
         memUsage = memInfo.fMemTotal;
      else if (mainWindow->GetActInfo() == 1)
         memUsage = memInfo.fMemUsed;
      else if (mainWindow->GetActInfo() == 2)
         memUsage = memInfo.fMemFree;
      // small threshold to avoid "trembling" needle
      if (fabs(act_load-prev_load) > 0.9) {
         speedo->SetScaleValue(act_load, 10);
         prev_load = act_load;
      }
      // update only if value has changed
      if (memUsage != old_memUsage) {
         speedo->SetOdoValue(memUsage);
         old_memUsage = memUsage;
      }
      // sleep a bit
      gSystem->ProcessEvents();
      gSystem->Sleep(250);
   }
   mainWindow->CloseWindow();
}

