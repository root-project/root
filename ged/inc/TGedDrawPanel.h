// @(#)root/gui:$Name:  $:$Id: TGedDrawPanel.h,v 1.0 2003/06/24 13:41:59 rdm Exp $
// Author: Marek Biskup, Ilka Antcheva   15/07/2003  

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGedDrawPanel
#define ROOT_TGedDrawPanel

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGedDrawPanel                                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

class TPad;
class TGDoubleSlider;
class TGRadioButton;

class TGedDrawPanel : public TGMainFrame {

protected:
   static TGedDrawPanel *fDrawPanel;
   
   TGRadioButton  *fType[12];
   TGRadioButton  *fCoords[4];
   TGRadioButton  *fErrors[5];
   TGDoubleSlider *fSlider;
   
   TObject        *fHistogram;
   TPad           *fRefPad;
   TString         fOption;

   TGedDrawPanel();
   void    Build(void);

   virtual void DrawHistogram();
   virtual void Reset();
   virtual void SetRadio(TGRadioButton **group, Int_t count, Int_t index);
   virtual void ParseRadio(const char * const option, TGRadioButton **group, 
                           const char * const * const data, Int_t index);
   virtual void ReadOption();     // reads options to fOptions
   virtual void AddRadioOption(TGRadioButton **group, const char * const * const data, Int_t count);
   virtual void ProcessSlider(Long_t submsg);
   virtual void SetHistogram(TObject* histogram, TPad* pad);
   virtual void ParseOption(const char * const options);
   virtual void Clean();

public:
   virtual ~TGedDrawPanel();
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t);
   virtual void   CloseWindow();
   static void    ShowPanel(TObject* histogram, TPad* pad);
 
   ClassDef(TGedDrawPanel,0)
};

#endif
