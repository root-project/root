// @(#)root/treeviewer:$Name$:$Id$
// Author: Rene Brun   08/12/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTreeViewer
#define ROOT_TTreeViewer


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTreeViewer                                                          //
//                                                                      //
// This class is a specialized canvas to browse a Root TTree,           //
// define cuts, make 1-d,2-d,3-d histograms, selection lists.           //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TCanvas
#include "TCanvas.h"
#endif
#ifndef ROOT_TButton
#include "TButton.h"
#endif
#ifndef ROOT_TSlider
#include "TSlider.h"
#endif
#ifndef ROOT_TTimer
#include "TTimer.h"
#endif
#ifndef ROOT_TTree
#include "TTree.h"
#endif
#ifndef ROOT_TAttText
#include "TAttText.h"
#endif

class TButton;
class TPaveVar;

class TTreeViewer : public TCanvas, public TAttText {

protected:
   enum { kDrawExecuting = BIT(17) };

   TString     fTreeName;       //Name of the TTree to be viewed
   TTree       *fTree;          //!pointer to the TTree
   TButton     *fDraw;          //Pointer to the Draw button
   TButton     *fScan;          //Pointer to the Scan button
   TButton     *fBreak;         //Pointer to the Break button
   TButton     *fGopt;          //Pointer to the Graphics option button
   TButton     *fIList;         //Pointer to the EventList In button
   TButton     *fOList;         //Pointer to the EventList Out button
   TButton     *fX;             //Pointer to the X button
   TButton     *fY;             //Pointer to the Y button
   TButton     *fZ;             //Pointer to the Z button
   TButton     *fW;             //Pointer to the W button
   TButton     *fHist;          //Pointer to the Histogram button
   TButton     *fRecord;        //Pointer to the Record button
   TSlider     *fSlider;        //Pointer to the Event slider
   TTimer      *fTimer;         //!Pointer to timer
   Int_t        fTimerInterval; //Timer interval in milliseconds
   Bool_t       fRecordFlag;    //Indication to record the Draw command

public:
   TTreeViewer();
   TTreeViewer(const char *treename, const char *title="TreeViewer", UInt_t ww=520, UInt_t wh=400);
   virtual        ~TTreeViewer();
   virtual void   BuildInterface();
   TPaveVar      *CreateNewVar(const char *varname="");  // *MENU*
   virtual void   ExecuteDraw(Option_t *option="");
   Bool_t         HandleTimer(TTimer *timer);
   TPaveVar      *IsUnder(TButton *button);
   TPaveVar      *IsUnderW(TButton *button);
   TTree         *GetTree()    {return fTree;}
   TButton       *GetDraw()    {return fDraw;}
   TButton       *GetScan()    {return fScan;}
   TButton       *GetBreak()   {return fBreak;}
   TButton       *GetGopt()    {return fGopt;}
   TButton       *GetIList()   {return fIList;}
   TButton       *GetOList()   {return fOList;}
   TButton       *GetX()       {return fX;}
   TButton       *GetY()       {return fY;}
   TButton       *GetZ()       {return fZ;}
   TButton       *GetW()       {return fW;}
   TButton       *GetHist()    {return fHist;}
   TButton       *GetRecord()  {return fRecord;}
   TSlider       *GetSlider()  {return fSlider;}
   TTimer        *GetTimer()   {return fTimer;}
   virtual const char  *GetTreeName() const {return fTreeName.Data();}
   virtual void   MakeClass(const char *classname);  // *MENU*
   virtual void   Reorganize();  // *MENU*
   virtual void   SetTimerInterval(Int_t msec=333) {fTimerInterval=msec;} // *MENU*
   virtual void   SetTreeName(const char *treename); // *MENU*
   virtual void   ToggleRecordCommand();

   //dummies
   virtual void   Divide(Int_t nx=1, Int_t ny=1, Float_t xmargin=0.01, Float_t ymargin=0.01, Int_t color=0);
   virtual void   SetGrid(Int_t valuex = 1, Int_t valuey = 1);
   virtual void   SetGridx(Int_t value = 1);
   virtual void   SetGridy(Int_t value = 1);
   virtual void   SetLogx(Int_t value = 1);
   virtual void   SetLogy(Int_t value = 1);
   virtual void   SetLogz(Int_t value = 1);
   virtual void   SetTickx(Int_t value = 1);
   virtual void   SetTicky(Int_t value = 1);
   virtual void   x3d(Option_t *option="");

   ClassDef(TTreeViewer,1)  //The Tree viewer and Browser
};

inline void TTreeViewer::Divide(Int_t, Int_t, Float_t, Float_t, Int_t) { }
inline void TTreeViewer::SetGrid(Int_t, Int_t) { }
inline void TTreeViewer::SetGridx(Int_t) { }
inline void TTreeViewer::SetGridy(Int_t) { }
inline void TTreeViewer::SetLogx(Int_t) { }
inline void TTreeViewer::SetLogy(Int_t) { }
inline void TTreeViewer::SetLogz(Int_t) { }
inline void TTreeViewer::SetTickx(Int_t) { }
inline void TTreeViewer::SetTicky(Int_t) { }
inline void TTreeViewer::x3d(Option_t *) { }

#endif

