#ifndef HISTACTION_H
#define HISTACTION_H

//--------------------------------------------------------------
//
//  An example of a control panel with ROOT GUI classes
//  for different actions with 1d histogrammes
//
//  Author: Dmitry Vasiliev (LNS, Catania)
//
//--------------------------------------------------------------

#include <TObjArray.h>
#include <TGFrame.h>
#include <TGButton.h>
#include <TGListBox.h>
#include <TGTextEntry.h>
#include <TGTextBuffer.h>
#include <TGLabel.h>
#include <TRootEmbeddedCanvas.h>
#include <TGTab.h>
#include <TGFSContainer.h>
#include <TGComboBox.h>

enum CommandsId {

  M_CLOSE = 100,

  M_DRAW,
  M_SELECT,
  M_CLEAR_A,
  M_SAVE,
  M_EDIT,
  M_NEXT_A,
  M_PREV_A,

  M_IMPORT,
  M_CLEAR_B,
  M_NEXT_B,
  M_PREV_B,

  M_MULTI,

  M_LIST_A,
  M_LIST_B,

  M_CDUP,
  M_LIST_MODE,
  M_DETAIL_MODE,

  M_FILTER
};


class TPad;

class HistAction : public TGMainFrame {

private:
   enum { kMaxHist = 1000 };
   TGCompositeFrame     *fF0, *fFA, *fFB;
   TRootEmbeddedCanvas  *fCanvasA, *fCanvasB;
   TGListBox            *fListBoxA, *fListBoxB;
   TGCompositeFrame     *fA1, *fA2, *fA3, *fA4, *fA5;
   TGCompositeFrame     *fB1, *fB2, *fB3, *fB4, *fB5;
   TGButton             *fCloseButton;
   TGButton             *fSaveButton, *fEditButton;
   TGButton             *fDrawButton, *fSelectButton, *fClearButtonA;
   TGButton             *fPrevButtonA, *fNextButtonA;
   TGButton             *fGetButton,*fClearButtonB,*fPrevButtonB,*fNextButtonB;
   TGButton             *fLayoutButton[16];
   TGCheckButton        *fMultiButton;
   TGTextEntry          *fName, *fTitle, *fChan, *fRange, *fEntries;
   TGTextBuffer         *fNameBuf, *fTitleBuf,*fChanBuf,*fRangeBuf,*fEntriesBuf;
   TGTextEntry          *fBinCont, *fBinRange;
   TGTextBuffer         *fBinContBuf, *fBinRangeBuf;
   TGTab                *fTab;
   TGCompositeFrame     *fC1, *fC2;
   const TGPicture      *fPcdup;
   const TGPicture      *fPlist;
   const TGPicture      *fPdetail;
   TGPictureButton      *fCdup, *fListMode, *fDetailMode;
   TGListBox            *fDir;
   TGListView           *fFileView;
   TGFileContainer      *fFileCont;
   TGComboBox           *fFilterBox;

   TObjArray            *fHisto; //histo container
   Int_t                 position; //current position in array "fHisto"
   Bool_t                flags[kMaxHist]; //true for highlighted histos (ListBoxA)
   TPad                 *pads[16];//addresses of pads in 4x4 matrix
                                 //(display layout for CanvasA)
   Int_t                 histInd[16];//indices of histos drawn in CanvasA
   Int_t                 horLay[4];//horizontal display layout
   Int_t                 verLay[4];//vertical display layout
   Int_t                 cursorIter;//current true position in array "flags"
   Int_t                 xDiv, yDiv;//parameters for CanvasA division in case
                                   //of automatic display layout

   Bool_t toGreen(Window_t id);
   Bool_t toDefault(Window_t id);
   Bool_t isOverlap();
   Bool_t isLayout();

   void toScan();

public:
   HistAction(const TGWindow *p, UInt_t w, UInt_t h);
   virtual ~HistAction();
   virtual void CloseWindow();
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
   Int_t getNextTrueIndex(); //returns -1 in case of no index found
   void resetIter() {cursorIter = -1;}
   void resetFlags() { for (int i = 0; i < kMaxHist; i++) flags[i] = kFALSE; }
   void setCanvasDivision(Int_t number);
   void drawHist();//draws a histo in case of automatic display layout
   void processBoxB(Int_t par);
   void doubleclickedBoxA(const char *text);
   Bool_t importHist(const char *name);
   Bool_t importFromFile(const char *filename);
   void clearScan();
   void paintHist();//draws a histo in case of user defined display layout

   ClassDef(HistAction,0)
};

#endif
