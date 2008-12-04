// A simple example that shows the usage of a TGSplitButton. 
// The checkbutton is used to change the split state of the button.
//
// author, Roel Aaij 13/07/2007

#include <iostream>
#include <TApplication.h>
#include <TGClient.h>
#include <TGButton.h>
#include <TGFrame.h>
#include <TGLayout.h>
#include <TGWindow.h>
#include <TGLabel.h>
#include <TString.h>
#include <TGMenu.h>

// A little class to automatically handle the generation of unique
// widget ids.
enum EMenuIds {
   ID_1,
   ID_2,
   ID_3,
   ID_4,
   ID_5
};

class IDList {
private:
   Int_t nID ;               // Generates unique widget IDs.
public:
   IDList() : nID(0) {}
   ~IDList() {}
   Int_t GetUnID(void) { return ++nID ; }
} ;


class SplitButtonTest : public TGMainFrame {

private:
   TGSplitButton *fMButton;  // Split Button
   TGPopupMenu   *fPopMenu;  // TGpopupMenu that will be attached to
                             // the button.
   IDList         IDs ;      // Generator for unique widget IDs.
   
public:
   SplitButtonTest(const TGWindow *p, UInt_t w, UInt_t h) ;
   virtual ~SplitButtonTest() ;

   void DoExit() ;
   void DoSplit(Bool_t split) ;
   void DoEnable(Bool_t on) ;
   void HandleMenu(Int_t id) ;

   ClassDef(SplitButtonTest, 0)
};
                          
SplitButtonTest::SplitButtonTest(const TGWindow *p, UInt_t w, UInt_t h) 
   : TGMainFrame(p, w, h)   
{
   SetCleanup(kDeepCleanup) ;
   
   Connect("CloseWindow()", "SplitButtonTest", this, "DoExit()") ;
   DontCallClose() ;

   TGVerticalFrame *fVL = new TGVerticalFrame(this, 100, 100) ;
   TGHorizontalFrame *fHL = new TGHorizontalFrame(fVL, 100, 40) ;

   // Create a popup menu.
   fPopMenu = new TGPopupMenu(gClient->GetRoot());
   fPopMenu->AddEntry("Button &1", ID_1);
   fPopMenu->AddEntry("Button &2", ID_2);
   fPopMenu->DisableEntry(ID_2);
   fPopMenu->AddEntry("Button &3", ID_3);
   fPopMenu->AddSeparator();
   
   // Create a split button, the menu is adopted.
   fMButton = new TGSplitButton(fHL, new TGHotString("Button &Options"), 
                                fPopMenu, IDs.GetUnID());

   // It is possible to add entries later
   fPopMenu->AddEntry("En&try with really really long name", ID_4);
   fPopMenu->AddEntry("&Exit", ID_5);   
   
   // Connect the special signal for the activation of items in a menu
   // that belongs to a split button to the slot.
   fMButton->Connect("ItemClicked(Int_t)", "SplitButtonTest", this, 
                     "HandleMenu(Int_t)");

   TGCheckButton *fCButton = new TGCheckButton(fHL, new TGHotString("Split"), 
                                               IDs.GetUnID());
   fCButton->SetState(kButtonDown);
   fCButton->Connect("Toggled(Bool_t)", "SplitButtonTest", this, "DoSplit(Bool_t)");

   // Add frames to their parent for layout.
   fHL->AddFrame(fCButton, new TGLayoutHints(kLHintsCenterX | kLHintsCenterY, 
                                             0, 10, 0, 0)) ;        
   TGCheckButton *fEButton = new TGCheckButton(fHL, new TGHotString("Enable"), 
                                               IDs.GetUnID());
   fEButton->SetState(kButtonDown);
   fEButton->Connect("Toggled(Bool_t)", "SplitButtonTest", this, "DoEnable(Bool_t)");

   // Add frames to their parent for layout.
   fHL->AddFrame(fEButton, new TGLayoutHints(kLHintsCenterX | kLHintsCenterY, 
                                             0, 10, 0, 0)) ;        
   fHL->AddFrame(fMButton, new TGLayoutHints(kLHintsCenterX | kLHintsCenterY));
   fVL->AddFrame(fHL, new TGLayoutHints(kLHintsCenterX | kLHintsCenterY)) ;
   AddFrame(fVL, new TGLayoutHints(kLHintsCenterX | kLHintsCenterY)) ;

   SetWindowName("SplitButton Test") ;
   MapSubwindows() ;
   Resize(GetDefaultSize()) ;
   MapWindow() ;

} ;

SplitButtonTest::~SplitButtonTest()
{
   // Destructor
   Cleanup() ;
}

void SplitButtonTest::DoExit()
{
   // Exit this application via the Exit button or Window Manager.
   // Use one of the both lines according to your needs.
   // Please note to re-run this macro in the same ROOT session,
   // you have to compile it to get signals/slots 'on place'.
   
   //DeleteWindow();            // to stay in the ROOT session
   gApplication->Terminate();   // to exit and close the ROOT session   
}

void SplitButtonTest::DoSplit(Bool_t split)
{
   fMButton->SetSplit(split);
}

void SplitButtonTest::DoEnable(Bool_t on)
{
   if (on)
      fMButton->SetState(kButtonUp);
   else
      fMButton->SetState(kButtonDisabled);
}

void SplitButtonTest::HandleMenu(Int_t id) 
{
   // Activation of menu items in the popup menu are handled in a user
   // defined slot to which the ItemClicked(Int_t) signal is
   // connected.

   switch (id) {
   case ID_1:
      std::cout << "Button 1 was activated" << std::endl;
      break;
   case ID_2:
      std::cout << "Button 2 was activated" << std::endl;
      break;
   case ID_3:
      std::cout << "Button 3 was activated" << std::endl;
      break;
   case ID_4:
      std::cout << "Button with a really really long name was activated" 
                << std::endl;
      break;
   case ID_5:
      DoExit();
      break;
   }
}
void splitbuttonTest() 
{
   new SplitButtonTest(gClient->GetRoot(),100,100);
}

