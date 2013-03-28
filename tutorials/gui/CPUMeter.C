
// Simple macro showing capabilities of TGSpeedo widget.
//Author: Bertrand Bellenot
   
#include "TSystem.h"
#include "TGFrame.h"
#include "TGWindow.h"
#include "TGSpeedo.h"

class TGShapedMain : public TGMainFrame {

protected:
   const TGPicture   *fBgnd;           // picture used as mask
   TGSpeedo          *fSpeedo;          // analog meter
   TTimer            *fTimer;           // update timer
   Int_t              fActInfo;         // actual information value

public:
   TGShapedMain(const TGWindow *p, int w, int h);
   virtual ~TGShapedMain();

   void              CloseWindow();
   TGSpeedo         *GetSpeedo() const { return fSpeedo; }
   Int_t             GetActInfo() const { return fActInfo; }
   void              ToggleInfos();

   ClassDef(TGShapedMain, 0)
};


// globals
TGShapedMain *gMainWindow;
TGSpeedo *gSpeedo;
Float_t prev_load;
Int_t old_memUsage;

//______________________________________________________________________________
TGShapedMain::TGShapedMain(const TGWindow *p, int w, int h) :
   TGMainFrame(p, w, h)
{
   // Constructor.

   fActInfo = 1;

   fSpeedo = new TGSpeedo(this, 0.0, 100.0, "CPU", "[%]");
   fSpeedo->Connect("OdoClicked()", "TGShapedMain", this, "ToggleInfos()");
   fSpeedo->Connect("LedClicked()", "TGShapedMain", this, "CloseWindow()");
   Connect("CloseWindow()", "TGShapedMain", this, "CloseWindow()");
   AddFrame(fSpeedo, new TGLayoutHints(kLHintsCenterX | kLHintsCenterX));
   fSpeedo->SetDisplayText("Used RAM", "[MB]");
   fTimer = new TTimer(100);
   fTimer->SetCommand("Update()");

   fBgnd = fSpeedo->GetPicture();
   gVirtualX->ShapeCombineMask(GetId(), 0, 0, fBgnd->GetMask());
   SetBackgroundPixmap(fBgnd->GetPicture());
   SetWMSizeHints(fBgnd->GetWidth(), fBgnd->GetHeight(), fBgnd->GetWidth(),
                  fBgnd->GetHeight(), 1, 1);

   MapSubwindows();
   MapWindow();
   // To avoid closing the window while TGSpeedo is drawing
   DontCallClose();
   // To avoid closing the window while TGSpeedo is drawing
   Resize(GetDefaultSize());
   // Set fixed size
   SetWMSizeHints(GetDefaultWidth(), GetDefaultHeight(), GetDefaultWidth(), 
                  GetDefaultHeight(), 1, 1);
   SetWindowName("ROOT CPU Load Meter");
   fTimer->TurnOn();   
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

   delete fTimer;
   delete fSpeedo;
}

//______________________________________________________________________________
void TGShapedMain::CloseWindow()
{
   // Close Window.

   if (fTimer)
      fTimer->TurnOff();
   DestroyWindow();
}

void Update()
{
   MemInfo_t memInfo;
   CpuInfo_t cpuInfo;
   Float_t act_load = 0.0;
   Int_t memUsage = 0;
   prev_load = act_load;
   old_memUsage = memUsage;

   // Get CPU information
   gSystem->GetCpuInfo(&cpuInfo, 100);
   // actual CPU load
   act_load = cpuInfo.fTotal;
   // Get Memory information
   gSystem->GetMemInfo(&memInfo);
   // choose which value to display
   if (gMainWindow->GetActInfo() == 0)
      memUsage = memInfo.fMemTotal;
   else if (gMainWindow->GetActInfo() == 1)
      memUsage = memInfo.fMemUsed;
   else if (gMainWindow->GetActInfo() == 2)
      memUsage = memInfo.fMemFree;
   // small threshold to avoid "trembling" needle
   if (fabs(act_load-prev_load) > 0.9) {
      gSpeedo->SetScaleValue(act_load, 10);
      prev_load = act_load;
   }
   // update only if value has changed
   if (memUsage != old_memUsage) {
      gSpeedo->SetOdoValue(memUsage);
      old_memUsage = memUsage;
   }
}

//______________________________________________________________________________
void CPUMeter()
{
   // Main application.

   gMainWindow = new TGShapedMain(gClient->GetRoot(), 500, 200);
   gSpeedo = gMainWindow->GetSpeedo();

   // set threshold values
   gSpeedo->SetThresholds(12.5, 50.0, 87.5);
   // set threshold colors
   gSpeedo->SetThresholdColors(TGSpeedo::kGreen, TGSpeedo::kOrange, 
                               TGSpeedo::kRed);
   // enable threshold
   gSpeedo->EnableThreshold();
   gSpeedo->SetScaleValue(0.0, 5);
   // enable peak marker
   gSpeedo->EnablePeakMark();

}

