//  This macro shows how to use a TGImageMap class.
//
// A TGImageMap provides the functionality like a clickable image in    
// with sensitive regions (similar to MAP HTML tag).                 
//Author: Valeriy Onuchin

#include <TGPicture.h>
#include <TGMenu.h>
#include <TGImageMap.h>
#include <TGMsgBox.h>
#include <TGClient.h>

////////////////////////////////////////////////////////////////////////////////
class WorldMap
{
protected:
   TGMainFrame*      fMain;      // main frame
   TGImageMap*       fImageMap;  // image map

   virtual void InitMap();
   virtual void InitRU();
   virtual void InitUS();
//   virtual void InitCN();
   virtual void InitAU();
   virtual void InitFR();
   virtual void InitUK();

public:
   // the name corresponds to TLD code 
   // (http://www.iana.org/cctld/cctld-whois.htm)
   // the value to "country phone code" 
   // (http://www.att.com/traveler/tools/codes.html)
   enum ECountryCode {
      kRU = 7, kUS = 1, kFR = 33, kDE = 49, kCH = 41, kCN = 86, kAU = 61,
      kUK = 44, kUA = 380, kBR = 55
    };

   WorldMap(const char *picName = "worldmap.jpg");
   virtual ~WorldMap() {}

   virtual void Show() { fMain->MapRaised(); }
   TGImageMap* GetImageMap() const { return fImageMap; }
   virtual TString GetTitle() const;

   // slots
   void PrintCode(Int_t code);
};


//__________________________________________________________________________
WorldMap::WorldMap(const char* picName)
{
   //

   fMain = new TGMainFrame(gClient->GetRoot(), 750, 420);

   fImageMap = new TGImageMap(fMain, picName);
   fMain->AddFrame(fImageMap);
   fMain->SetWindowName(GetTitle().Data());
   fMain->SetIconName("World Map");

   TGDimension size = fMain->GetDefaultSize();
   fMain->Resize(size);
   fMain->MapSubwindows();
   InitMap();

   fImageMap->Connect("RegionClicked(Int_t)", "WorldMap", this,
                      "PrintCode(Int_t)");
}

//__________________________________________________________________________
TString WorldMap::GetTitle() const
{
   // title

   return "Country Code (left button). City/Area Codes (right button)";
}

//__________________________________________________________________________
void WorldMap::InitRU()
{
   //

   int x[12] = { 403, 406, 427, 444, 438, 470, 508, 568, 599, 632, 645, 493 };
   int y[12] = { 68, 90, 120, 125, 109, 94, 109, 101, 122, 107, 74, 46 };
   TGRegion reg(12, x, y);
   fImageMap->AddRegion(reg, kRU);
   fImageMap->SetToolTipText(kRU, "Russia");
   TGPopupMenu* pm = fImageMap->CreatePopup(kRU);
   pm->AddLabel("City Codes");
   pm->AddSeparator();
   pm->AddEntry("Moscow = 095", 95);
   pm->AddEntry("Protvino = 0967", 967);
   pm->AddEntry("St.Petersburg = 812", 812);
}

//__________________________________________________________________________
void WorldMap::InitUS()
{
   //

   int x[5] = { 136, 122, 165, 194, 232 };
   int y[5] = { 110, 141, 158, 160, 118 };
   TGRegion reg(5, x, y);
   fImageMap->AddRegion(reg, kUS);

   int alaskaX[4] = { 86, 131, 154, 117 };
   int alaskaY[4] = { 90, 82, 64, 63 };
   TGRegion alaska(4, alaskaX, alaskaY);
   fImageMap->AddRegion(alaska, kUS);
   fImageMap->SetToolTipText(kUS, "USA");

   TGPopupMenu* pm = fImageMap->CreatePopup(kUS);
   pm->AddLabel("Area Codes");
   pm->AddSeparator();
   pm->AddEntry("Illinois = 217", 217);
   pm->AddEntry("New York = 212", 212);
}

//__________________________________________________________________________
void WorldMap::InitFR()
{
   //

   int x[5] = { 349, 353, 368, 368, 358 };
   int y[5] = { 112, 123, 119, 108, 107 };
   TGRegion reg(5, x, y);
   fImageMap->AddRegion(reg, kFR);
   fImageMap->SetToolTipText(kFR, "France");
}

//__________________________________________________________________________
void WorldMap::InitUK()
{
   //

   int x[4] = { 346, 348, 359, 352 };
   int y[4] = { 93, 104, 103, 87 };
   TGRegion reg(4, x, y);
   fImageMap->AddRegion(reg, kUK);
   fImageMap->SetToolTipText(kUK, "United Kingdom");
}

//__________________________________________________________________________
void WorldMap::InitAU()
{
   //

   int x[6] = { 582, 576, 634, 658, 641, 607 };
   int y[6] = { 271, 300, 310, 283, 251, 253 };
   TGRegion reg(6, x, y);
   fImageMap->AddRegion(reg, kAU);
   fImageMap->SetToolTipText(kAU, "Australia");
}

//__________________________________________________________________________
void WorldMap::InitMap()
{
   //

   InitRU();
   InitUS();
   InitFR();
   InitAU();
   InitUK();
   fImageMap->SetToolTipText(GetTitle().Data(), 300);
}

//__________________________________________________________________________
void WorldMap::PrintCode(Int_t code)
{
   //

   EMsgBoxIcon icontype = kMBIconAsterisk;
   Int_t buttons = 0;
   Int_t retval;

   TGMsgBox* box = new TGMsgBox(gClient->GetRoot(), fMain,
                              "Country Code", Form("Country Code=%d",code),
                              icontype, buttons, &retval);
}

void WorldMap()
{
   WorldMap *map = new WorldMap;
   map->Show();
}
