// @(#)root/memstat:$Name$:$Id$
// Author: M.Ivanov -- Anar Manafov (A.Manafov@gsi.de) 28/04/2008

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ALITPCCALIBVIEWERGUI
#define ALITPCCALIBVIEWERGUI

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id: TMemViewerGUI.h,v */

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//  GUI for the AliTPCCalibViewer                                            //
//  used for the calibration monitor                                         //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

// STD
#include <string>
#include <memory>
// ROOT
#include "TGFrame.h"

class TGTextView;
class TGNumberEntry;
class TGComboBox;

class TMemViewerGUI : public TGCompositeFrame
{
protected:
    TMemStat *fViewer;                           // CalibViewer object used for drawing
    TGTextView *fText;                           // text widget
    TGNumberEntry *fNmbStackDeep;                // number entry box for specifying the stack deepness
    TGNumberEntry *fNmbSortDeep;                 // number entry box for specifying the number of stamps
    std::string fCurLib;
    std::string fCurFunc;


protected:
    void MakeContSortStat(TGCompositeFrame *frame);
    void MakeContSortStamp(TGCompositeFrame *frame);
    void MakeContDeep(TGCompositeFrame *frame);
    void MakeContSelect(TGCompositeFrame *frame);
    void MakeStampList(TGCompositeFrame *frame);
    void MakeDrawButton(TGCompositeFrame *frame);
    void MakeSelection(TGCompositeFrame *frame);

    const Option_t *GetOption() const;                  // get option
    void Initialize(Option_t* option);                  // initializes the GUI with default settings and opens tree for drawing
    void MakePrint();                                   // get print

    template< class T >
    void HandleRButtons(Int_t id, Int_t FirstBtnId, T *ViewerSortType);

public:
    TMemViewerGUI(const TGWindow *p, UInt_t w, UInt_t h, Option_t* option = "read");
    virtual ~TMemViewerGUI();
    
    static void ShowGUI();                               // initialize and show GUI for presentation

    void HandleButtonsSortStat(Int_t id = -1);           // handles mutual radio buuton for sort stat
    void HandleButtonsSortStamp(Int_t id = -1);          // handles mutual radio buuton for sort stamp
    void HandleDeep(Long_t id);                          // handles stack deep and nrows
    void HandleStampSelect(const char*);
    void HandleDrawMemStat();
    void HandleFuncSelect(const char*);
    void HandleLibSelect(const char*);

    ClassDef(TMemViewerGUI, 0)
};

//______________________________________________________________________________
template< class T >
void TMemViewerGUI::HandleRButtons(Int_t id, Int_t FirstBtnId, T *ViewerSortType)
{
    // handles mutual radio button exclusions
    *ViewerSortType = static_cast<T>(id - FirstBtnId);
    MakePrint();
}


#endif
