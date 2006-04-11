#ifndef GUITEST_H
#define GUITEST_H

#include <stdlib.h>

#include <TROOT.h>
#include <TApplication.h>
#include <TVirtualX.h>

#include <TGListBox.h>
#include <TGClient.h>
#include <TGFrame.h>
#include <TGIcon.h>
#include <TGLabel.h>
#include <TGButton.h>
#include <TGTextEntry.h>
#include <TGMsgBox.h>
#include <TGMenu.h>
#include <TGCanvas.h>
#include <TGComboBox.h>
#include <TGTab.h>
#include <TGSlider.h>
#include <TGDoubleSlider.h>
#include <TGFileDialog.h>
#include <TGTextEdit.h>
#include <TGShutter.h>
#include <TGProgressBar.h>
#include <TRootEmbeddedCanvas.h>
#include <TCanvas.h>
#include <TH1.h>
#include <TH2.h>
#include <TRandom.h>
#include <TSystem.h>
#include <TEnv.h>


enum ETestCommandIdentifiers {
   M_FILE_OPEN,
   M_FILE_SAVE,
   M_FILE_SAVEAS,
   M_FILE_EXIT,

   M_TEST_DLG,
   M_TEST_MSGBOX,
   M_TEST_SLIDER,
   M_TEST_SHUTTER,
   M_TEST_PROGRESS,

   M_HELP_CONTENTS,
   M_HELP_SEARCH,
   M_HELP_ABOUT,

   M_CASCADE_1,
   M_CASCADE_2,
   M_CASCADE_3,

   VId1,
   HId1,
   VId2,
   HId2,

   VSId1,
   HSId1,
   VSId2,
   HSId2
};


struct shutterData_t {
   const char *pixmap_name;
   const char *tip_text;
   Int_t       id;
   TGButton   *button;
};






class TileFrame : public TGCompositeFrame {

private:
   TGCanvas *fCanvas;

public:
   TileFrame(const TGWindow *p);
   virtual ~TileFrame() { }

   void SetCanvas(TGCanvas *canvas) { fCanvas = canvas; }
   Bool_t HandleButton(Event_t *event);
};



class TestMainFrame : public TGMainFrame {

private:
   TGCompositeFrame   *fStatusFrame;
   TGCanvas           *fCanvasWindow;
   TileFrame          *fContainer;
   TGTextEntry        *fTestText;
   TGButton           *fTestButton;

   TGMenuBar          *fMenuBar;
   TGPopupMenu        *fMenuFile, *fMenuTest, *fMenuHelp;
   TGPopupMenu        *fCascadeMenu, *fCascade1Menu, *fCascade2Menu;
   TGLayoutHints      *fMenuBarLayout, *fMenuBarItemLayout, *fMenuBarHelpLayout;

public:
   TestMainFrame(const TGWindow *p, UInt_t w, UInt_t h);
   virtual ~TestMainFrame();

   virtual void CloseWindow();
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t);
};


class TestDialog : public TGTransientFrame {

private:
   TGCompositeFrame    *fFrame1, *fF1, *fF2, *fF3, *fF4, *fF5, *fF6, *fF7;
   TGButton            *fOkButton, *fCancelButton, *fStartB, *fStopB;
   TGButton            *fBtn1, *fBtn2, *fChk1, *fChk2, *fRad1, *fRad2;
   TGPictureButton     *fPicBut1;
   TGRadioButton       *fRadio1, *fRadio2;
   TGCheckButton       *fCheck1;
   TGCheckButton       *fCheckMulti;
   TGListBox           *fListBox;
   TGComboBox          *fCombo;
   TGTab               *fTab;
   TGTextEntry         *fTxt1, *fTxt2;
   TGLayoutHints       *fL1, *fL2, *fL3, *fL4;
   TRootEmbeddedCanvas *fEc1, *fEc2;
   Int_t                fFirstEntry;
   Int_t                fLastEntry;
   Bool_t               fFillHistos;
   TH1F                *fHpx;
   TH2F                *fHpxpy;
   TList               *fCleanup;

   void FillHistos();

public:
   TestDialog(const TGWindow *p, const TGWindow *main, UInt_t w, UInt_t h,
               UInt_t options = kVerticalFrame);
   virtual ~TestDialog();

   virtual void CloseWindow();
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
};


class TestMsgBox : public TGTransientFrame {

private:
   TGCompositeFrame     *f1, *f2, *f3, *f4, *f5;
   TGButton             *fTestButton, *fCloseButton;
   TGPictureButton      *fPictButton;
   TGRadioButton        *fR[4];
   TGCheckButton        *fC[9];
   TGGroupFrame         *fG1, *fG2;
   TGLayoutHints        *fL1, *fL2, *fL3, *fL4, *fL5, *fL6, *fL21;
   TGTextEntry          *fTitle, *fMsg;
   TGTextBuffer         *fTbtitle, *fTbmsg;
   TGLabel              *fLtitle, *fLmsg;
   TGGC                  fRedTextGC;

public:
   TestMsgBox(const TGWindow *p, const TGWindow *main, UInt_t w, UInt_t h,
              UInt_t options = kVerticalFrame);
   virtual ~TestMsgBox();

   virtual void CloseWindow();
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
};


class TestSliders : public TGTransientFrame {

private:
   TGVerticalFrame   *fVframe1, *fVframe2;
   TGLayoutHints     *fBly, *fBfly1;
   TGHSlider         *fHslider1, *fHslider2;
   TGVSlider         *fVslider1;
   TGDoubleVSlider   *fVslider2;
   TGTextEntry       *fTeh1, *fTev1, *fTeh2, *fTev2;
   TGTextBuffer      *fTbh1, *fTbv1, *fTbh2, *fTbv2;

public:
   TestSliders(const TGWindow *p, const TGWindow *main, UInt_t w, UInt_t h);
   virtual ~TestSliders();

   virtual void CloseWindow();
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
};


class TestShutter : public TGTransientFrame {

private:
   TGShutter       *fShutter;
   TGLayoutHints   *fLayout;
   const TGPicture *fDefaultPic;
   TList           *fTrash;

public:
   TestShutter(const TGWindow *p, const TGWindow *main, UInt_t w, UInt_t h);
   ~TestShutter();

   void AddShutterItem(const char *name, shutterData_t data[]);
   virtual void CloseWindow();
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
};


class TestProgress : public TGTransientFrame {

private:
   TGHorizontalFrame *fHframe1;
   TGVerticalFrame   *fVframe1;
   TGLayoutHints     *fHint1, *fHint2, *fHint3, *fHint4, *fHint5;
   TGHProgressBar    *fHProg1, *fHProg2, *fHProg3;
   TGVProgressBar    *fVProg1, *fVProg2;
   TGTextButton      *fGO;

public:
   TestProgress(const TGWindow *p, const TGWindow *main, UInt_t w, UInt_t h);
   virtual ~TestProgress();

   virtual void CloseWindow();
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
};


class Editor : public TGTransientFrame {

private:
   TGTextEdit       *fEdit;   // text edit widget
   TGTextButton     *fOK;     // OK button
   TGLayoutHints    *fL1;     // layout of TGTextEdit
   TGLayoutHints    *fL2;     // layout of OK button

public:
   Editor(const TGWindow *main, UInt_t w, UInt_t h);
   virtual ~Editor();

   void   LoadBuffer(const char *buffer);
   void   LoadFile(const char *file);

   TGTextEdit *GetEditor() const { return fEdit; }

   void   SetTitle();
   void   Popup();
   void   CloseWindow();
   Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
};


#endif
