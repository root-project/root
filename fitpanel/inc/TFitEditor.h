// @(#)root/fitpanel:$Name:  $:$Id: TFitEditor.h,v 1.2 2006/10/05 21:33:21 rdm Exp $
// Author: Ilka Antcheva, Lorenzo Moneta 10/08/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFitEditor
#define ROOT_TFitEditor

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFitEditor                                                           //
//                                                                      //
// Allows to explore and compare various fits.                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif
#ifndef ROOT_TGButton
#include "TGButton.h"
#endif


//--- Object types

enum EObjectType {
   kObjectHisto,
   kObjectGraph,
   kObjectGraph2D,
   kObjectHStack,
   kObjectTree
};


class TGTab;
class TPad;
class TGLabel;
class TGComboBox;
class TGTextEntry;
class TGNumberEntry;
class TGDoubleHSlider;
class TGNumberEntry;
class TAxis;
class TF1;


class TFitEditor : public TGMainFrame {

protected:
   TGTab               *fTab;              // tab widget holding the editor
   TGCompositeFrame    *fTabContainer;     // main tab container
   TGCompositeFrame    *fGeneral;          // general tab
   TGCompositeFrame    *fMinimization;     // minimization tab
   TGTextButton        *fFitButton;        // performs fitting
   TGTextButton        *fResetButton;      // resets fit parameters
   TGTextButton        *fCloseButton;      // close the fit panel
   TGLabel             *fObjLabel;         // contains fitted object name
   TGLabel             *fSelLabel;         // contains selected fit function
   TGComboBox          *fFuncList;         // contains function list
   TGTextEntry         *fEnteredFunc;      // contains user function file name
   TGTextButton        *fUserButton;       // opens a dialog for user-defined fit method
   TGRadioButton       *fNone;             // set no operation mode
   TGRadioButton       *fAdd;              // set addition mode
   TGRadioButton       *fConv;             // set convolution mode
   TGTextButton        *fSetParam;         // open set parameters dialog
   TGCheckButton       *fIntegral;         // switch on/off option 'integral'
   TGCheckButton       *fBestErrors;       // switch on/off option 'improve errors'
   TGCheckButton       *fUseRange;         // switch on/off option 'use function range'
   TGCheckButton       *fAdd2FuncList;     // switch on/off option 'add to list'
   TGCheckButton       *fAllWeights1;      // switch on/off option 'all weights=1'
   TGCheckButton       *fImproveResults;   // switch on/off option 'improve fit results'
   TGComboBox          *fMethodList;       // contains method list
   TGCheckButton       *fLinearFit;        // switch on/off linear fit option
   TGCheckButton       *fNoChi2;           // switch on/off option 'No Chi-square'
   TGCheckButton       *fNoStoreDrawing;   // switch on/off 'no store/drwing' option
   TGCheckButton       *fNoDrawing;        // switch on/off 'no drawing' option
   TGCheckButton       *fDrawSame;         // switch on/off fit function drawing
   TGTextButton        *fDrawAdvanced;     // opens a dialog for advanced draw options
   TGDoubleHSlider     *fSliderX;          // slider to set fit range along x-axis
   TGDoubleHSlider     *fSliderY;          // slider to set fit range along y-axis
   TGDoubleHSlider     *fSliderZ;          // slider to set fit range along z-axis
   TGNumberEntry       *fRobustValue;      // contains robust value for linear fit
   TGRadioButton       *fOptDefault;       // set default printing mode
   TGRadioButton       *fOptVerbose;       // set printing mode to 'Verbose'
   TGRadioButton       *fOptQuiet;         // set printing mode to 'Quiet'
   TVirtualPad         *fParentPad;        // pad containing the object
   TObject             *fFitObject;        // selected object to fit
   EObjectType          fType;             // object type info
   Int_t                fDim;              // object dimension
   TAxis               *fXaxis;            // x-axis
   TAxis               *fYaxis;            // y-axis
   TAxis               *fZaxis;            // z-axis
   Double_t             fXmin;             // x-min
   Double_t             fXmax;             // x-max
   TString              fPlus;             // string for addition ('+' or "++")
   TString              fFunction;         // selected function to fit
   TString              fFitOption;        // fitting options
   TString              fDrawOption;       // graphics option for drawing
   TF1                 *fFitFunc;          // function used for fitting

   Int_t     fPx1old,
             fPy1old,
             fPx2old,
             fPy2old;

   static  TGComboBox *BuildFunctionList(TGFrame *parent, Int_t id);
   static  TGComboBox *BuildMethodList(TGFrame *parent, Int_t id);
   void    Init();
   Int_t    CheckFunctionString(const char* str);

private:
   TFitEditor(const TFitEditor&);              // not implemented
   TFitEditor& operator=(const TFitEditor&);   // not implemented

public:
   TFitEditor(const TVirtualPad* pad, const TObject *obj);
   virtual ~TFitEditor();

   virtual void   CloseWindow();
   virtual void   DoAddtoList();
   virtual void   DoAdvancedOptions();
   virtual void   DoAllWeights1();
   virtual void   DoClose();
   virtual void   DoDrawSame();
   virtual void   DoFit();
   virtual void   DoEnteredFunction();
   virtual void   DoFunction(Int_t sel);
   virtual void   DoImproveResults();
   virtual void   DoBestErrors();
   virtual void   DoIntegral();
   virtual void   DoAddition(Bool_t on);
   virtual void   DoNoOperation(Bool_t);
   virtual void   DoLinearFit();
   virtual void   DoMethod(Int_t);
   virtual void   DoNoChi2();
   virtual void   DoNoDrawing();
   virtual void   DoNoStoreDrawing();
   virtual void   DoPrintOpt(Bool_t);
   virtual void   DoReset();
   virtual void   DoRobust();
   virtual void   DoSetParameters();
   virtual void   DoSliderXMoved();
   virtual void   DoSliderXPressed();
   virtual void   DoSliderXReleased();
   virtual void   DoSliderYMoved();
   virtual void   DoSliderYPressed();
   virtual void   DoSliderYReleased();
   virtual void   DoSliderZMoved();
   virtual void   DoSliderZPressed();
   virtual void   DoSliderZReleased();
   virtual void   DoUserDialog();
   virtual void   DoUseRange();

   virtual void   SetFunction(const char *function);
   virtual void   RecursiveRemove(TObject* obj);

   ClassDef(TFitEditor,0)  //new fit panel interface
};

#endif
