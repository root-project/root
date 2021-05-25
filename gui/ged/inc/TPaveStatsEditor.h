// @(#)root/ged:$Id$
// Author: Ilka  Antcheva 21/06/04

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPaveStatsEditor
#define ROOT_TPaveStatsEditor


#include "TGedFrame.h"

class TGCheckButton;
class TPaveStats;

class TPaveStatsEditor : public TGedFrame {

protected:
   TPaveStats         *fPaveStats;        ///< TPaveStats object
   // widgets for stat options
   TGCheckButton      *fHistoName;        ///< histo name check box
   TGCheckButton      *fEntries;          ///< entries' number check box
   TGCheckButton      *fMean;             ///< mean value check box
   TGCheckButton      *fRMS;              ///< RMS check box
   TGCheckButton      *fUnderflow;        ///< underflow number check box
   TGCheckButton      *fOverflow;         ///< overflow number check box
   TGCheckButton      *fIntegral;         ///< integral of bins check box
   TGCheckButton      *fSkewness;         ///< skewness check box
   TGCheckButton      *fKurtosis;         ///< kurtosis check box
   TGCheckButton      *fStatsErrors;      ///< statistics error check box
   // widgets for fit options
   TGCheckButton      *fNameValues;       ///< parameters' name/values check box
   TGCheckButton      *fErrors;           ///< error check box
   TGCheckButton      *fChisquare;        ///< Chisquare check box
   TGCheckButton      *fProbability;      ///< probability check box

   virtual void ConnectSignals2Slots();

public:
   TPaveStatsEditor(const TGWindow *p=0,
                    Int_t width = 140, Int_t height = 30,
                    UInt_t options = kChildFrame,
                    Pixel_t back = GetDefaultFrameBackground());
   virtual ~TPaveStatsEditor();

   virtual void   SetModel(TObject* obj);
   virtual void   DoStatOptions();
   virtual void   DoFitOptions();
   virtual void   SetValuesON(Bool_t on);

   ClassDef(TPaveStatsEditor,0)  // GUI for editing TPaveStats
};

#endif
