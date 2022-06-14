/// \file
/// \ingroup tutorial_gui
/// A simple example of creating icon image from XPM data, included into the code.
///
/// \macro_code
///
/// \author Ilka Antcheva   27/09/2007

#include <TGClient.h>
#include <TGButton.h>
#include <TGFrame.h>
#include <TGIcon.h>
#include <TGResourcePool.h>
#include <TGPicture.h>
#include <TString.h>
#include <TApplication.h>

const char * const icon1[] =
{
"16 16 8 1",
"    c None s None",
".   c #808080",
"X   c #FFFF00",
"o   c #c0c0c0",
"O   c black",
"+   c #00FFFF",
"@   c #00FF00",
"#   c white",
"     .....      ",
"   ..XXoooOO    ",
"  .+XXXoooooO   ",
" .@++XXoooo#oO  ",
" .@@+XXooo#ooO  ",
".oo@@+Xoo#ooooO ",
".ooo@+.O.oooooO ",
".oooo@O#OoooooO ",
".oooo#.O.+ooooO ",
".ooo#oo#@X+oooO ",
" .o#oooo@X++oO  ",
" .#ooooo@XX++O  ",
"  .ooooo@@XXO   ",
"   ..ooo@@OO    ",
"     ..OOO      ",
"                "
};

class MyMainFrame : public TGMainFrame {

public:
   MyMainFrame(const TGWindow *p, UInt_t w, UInt_t h);
   virtual ~MyMainFrame();

   void DoExit();

   ClassDef(MyMainFrame, 0)
};

void MyMainFrame::DoExit()
{
   Cleanup();
   gApplication->Terminate(0);
}

MyMainFrame::MyMainFrame(const TGWindow *p, UInt_t w, UInt_t h) :
   TGMainFrame(p, w, h)
{
   // Create a main frame

   TString name = "myicon";
   ULong_t yellow;
   gClient->GetColorByName("yellow", yellow);

   // Create a picture from the XPM data
   TGPicturePool *picpool = gClient->GetResourcePool()->GetPicturePool();
   const TGPicture *iconpic = picpool->GetPicture(name.Data(),(char **)icon1);
   TGIcon *icon = new TGIcon(this, iconpic, 40, 40, kChildFrame, yellow);
   AddFrame(icon, new TGLayoutHints(kLHintsLeft, 1,15,1,1));

   TGTextButton *exit = new TGTextButton(this, "&Exit","gApplication->Terminate(0)");
   AddFrame(exit, new TGLayoutHints(kLHintsExpandX,2,0,2,2));

   SetWindowName("Icon test");
   MapSubwindows();
   Resize(GetDefaultSize());
   MapWindow();
}

MyMainFrame::~MyMainFrame()
{
   // Clean up all widgets, frames and layouthints.
   Cleanup();
}

void iconAsXPMData()
{
   // Popup the GUI...
   new MyMainFrame(gClient->GetRoot(), 350, 80);
}
