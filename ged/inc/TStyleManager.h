// @(#)root/ged:$Id: TStyleManager.h,v 1.0 2005/09/08
// Author: Denis Favre-Miville   08/09/05

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TStyleManager
#define ROOT_TStyleManager

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TStyleManager                                                       //
//                                                                      //
//  This class provides a Graphical User Interface to manage styles     //
//       in ROOT. It allows the user to edit styles, import / export    //
//       them to macros, apply a style on the selected object or on     //
//       all canvases, change gStyle.                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif
#ifndef ROOT_TGNumberEntry
#include "TGNumberEntry.h"
#endif

class TGButtonGroup;
class TGCheckButton;
class TGColorSelect;
class TGComboBox;
class TGCompositeFrame;
class TGedMarkerSelect;
class TGedPatternSelect;
class TGFileInfo;
class TGFontTypeComboBox;
class TGHButtonGroup;
class TGHorizontal3DLine;
class TGHorizontalFrame;
class TGLabel;
class TGLayoutHints;
class TGLineStyleComboBox;
class TGLineWidthComboBox;
class TGMainFrame;
class TGMatrixLayout;
class TGMenuBar;
class TGPicture;
class TGPictureButton;
class TGPopupMenu;
class TGRadioButton;
class TGStatusBar;
class TGTab;
class TGTextButton;
class TGTextEntry;
class TGToolBar;
class TGVerticalFrame;
class TList;
class TObject;
class TStyle;
class TStylePreview;
class TVirtualPad;

class TStyleManager : public TGMainFrame {

private:
   static TStyleManager *fgStyleManager;     // singleton style manager

   TStyle              *fCurSelStyle;        // current selected style
   Bool_t               fLastChoice;         //=kTRUE if the user choose OK in the last TStyleDialog
   Bool_t               fRealTimePreview;    //=kTRUE if auto refreshed preview
   Int_t                fCurTabNum;          // current opened tab number
   Int_t                fCurTabAxisNum;      // current opened axis tab number
   UInt_t               fSMWidth;            // style manager's width
   UInt_t               fSMHeight;           // style manager's height
   Bool_t               fStyleChanged;       //=kTRUE if the style has been modified

   Bool_t               fMoreAndNotLess;     //=kTRUE when editor is open
   Bool_t               fSigSlotConnected;   //=kTRUE when signal/slots connected
   Bool_t               fAllAndNotCurrent;   //=kTRUE when apply on 'All canvases'
   TList               *fTrashListFrame;     // to avoid memory leak
   TList               *fTrashListLayout;    // to avoid memory leak

   TGMenuBar           *fMenuBar;            // the main window menu bar
   TGPopupMenu         *fMenuStyle;          // the 'Style' popup menu
   TGPopupMenu         *fImportCascade;      // Cascaded menu 'Import'
   TGPopupMenu         *fMenuHelp;           // the 'Help' popup menu

   TGToolBar           *fToolBar;            // the tool bar
   TGPictureButton     *fToolBarNew;         // tool bar 'New' button
   TGPictureButton     *fToolBarDelete;      // tool bar 'Delete' button
   TGPictureButton     *fToolBarImportCanvas;// tool bar 'Import from canvas' button
   TGPictureButton     *fToolBarImportMacro; // tool bar 'Import from macro' button
   TGPictureButton     *fToolBarExport;      // tool bar 'Export' button
   TGPictureButton     *fToolBarHelp;        // tool bar 'Help' button
   const TGPicture     *fToolBarNewPic;      // tool bar 'New' picture
   const TGPicture     *fToolBarDeletePic;   // tool bar 'Delete' picture
   const TGPicture     *fToolBarImportCanvasPic;// tool bar 'Import from canvas' picture
   const TGPicture     *fToolBarImportMacroPic; // tool bar 'Import from macro' picture
   const TGPicture     *fToolBarExportPic;   // tool bar 'Export' picture
   const TGPicture     *fToolBarHelpPic;     // tool bar 'Help' picture
   TGHorizontal3DLine  *fHorizontal3DLine;   // a line under the tool bar

   TGLabel             *fListLabel;          // label 'Available Styles:'
   TGComboBox          *fListComboBox;       // list of available styles
   TGFileInfo          *fCurMacro;           // current macro
   TGLabel             *fCurStylabel;        // label 'gStyle is set to:'
   TGTextEntry         *fCurStyle;           // label showing gStyle's name
   TGLabel             *fCurPadLabel;        // label 'Canvas:'
   TGTextEntry         *fCurPadTextEntry;    // label showing current pad's name
   TVirtualPad         *fCurPad;             // current pad
   TGLabel             *fCurObjLabel;        // label 'Object:'
   TGTextEntry         *fCurObjTextEntry;    // label showing current object's name
   TObject             *fCurObj;             // current object
   TGCheckButton       *fPreviewButton;      // if checked, preview is visible
   TGCheckButton       *fPreviewRealTime;    // if checked, real time preview
   TStylePreview       *fPreviewWindow;      // preview
   TGPictureButton     *fMakeDefault;        // selected style becom gStyle
   const TGPicture     *fMakeDefaultPic;     // button picture

   TGHButtonGroup      *fApplyOnGroup;       // 'Apply on' button group
   TGRadioButton       *fApplyOnAll;         // 'Apply on' button group
   TGRadioButton       *fApplyOnSel;         // 'Apply on' button group
   TGTextButton        *fApplyOnButton;      // apply style on object(s)

   TGTextButton        *fMoreLess;           // open/close the editor
   TGStatusBar         *fStatusBar;          // status bar

   TGVerticalFrame     *fEditionFrame;       // editor
   TGTab               *fEditionTab;         // editor's tabs
   TGTab               *fHistosTab;          // histos' tabs
   TGTab               *fAxisTab;            // axis' tabs
   TGHorizontalFrame   *fEditionButtonFrame; // editor's buttons

   TGTextButton        *fEditionHelp;           // help button
   TGTextButton        *fEditionUpdatePreview;  // update preview button
   TGTextButton        *fEditionReset;          // reset button

   TGColorSelect       *fFillColor;          // general fill color selection widget
   TGedPatternSelect   *fFillStyle;          // general fill pattern selection widget
   TGLineWidthComboBox *fHatchesLineWidth;   // general hatches width combo box
   TGNumberEntry       *fHatchesSpacing;     // general hatches spacing number entry
   TGColorSelect       *fTextColor;          // general text color selection widget
   TGNumberEntry       *fTextSize;           // general text size number entry
   TGCheckButton       *fTextSizeInPixels;   // general text size check box
   TGFontTypeComboBox  *fTextFont;           // general text font combo box
   TGComboBox          *fTextAlign;          // general text align combo box
   TGNumberEntry       *fTextAngle;          // general text angle number entry
   TGColorSelect       *fLineColor;          // general line color selection widget
   TGLineWidthComboBox *fLineWidth;          // general line width combo box
   TGLineStyleComboBox *fLineStyle;          // general line style combo box
   TGTextButton        *fLineStyleEdit;      // general line style editor open button
   TGColorSelect       *fMarkerColor;        // general marker color selection widget
   TGedMarkerSelect    *fMarkerStyle;        // general marker style entry
   TGComboBox          *fMarkerSize;         // general marker size combo box
   TGNumberEntry       *fScreenFactor;       // general screen factor number entry
   TGColorSelect       *fCanvasColor;        // canvas fill color selection widget
   TGNumberEntry       *fCanvasDefX;         // canvas abscissa number entry
   TGNumberEntry       *fCanvasDefY;         // canvas ordinate number entry
   TGNumberEntry       *fCanvasDefW;         // canvas width number entry
   TGNumberEntry       *fCanvasDefH;         // canvas height number entry
   TGButtonGroup       *fCanvasBorderMode;   // canvas border mode button group
   TGLineWidthComboBox *fCanvasBorderSize;   // canvas border size combo box
   TGCheckButton       *fOptDateBool;        // canvas date show/hide check box
   TGColorSelect       *fAttDateTextColor;   // canvas date color selection widget
   TGNumberEntry       *fAttDateTextSize;    // canvas date size number entry
   TGCheckButton       *fAttDateTextSizeInPixels;  // canvas date size check box
   TGComboBox          *fOptDateFormat;      // canvas date format text entry
   TGFontTypeComboBox  *fAttDateTextFont;    // canvas date font combo box
   TGNumberEntry       *fAttDateTextAngle;   // canvas date angle number entry
   TGComboBox          *fAttDateTextAlign;   // canvas date align combo box
   TGNumberEntry       *fDateX;              // canvas date abscissa number entry
   TGNumberEntry       *fDateY;              // canvas date ordinate number entry
   TGNumberEntry       *fPadLeftMargin;      // pad left margin number entry
   TGNumberEntry       *fPadRightMargin;     // pad right margin number entry
   TGNumberEntry       *fPadTopMargin;       // pad top margin number entry
   TGNumberEntry       *fPadBottomMargin;    // pad bottom margin number entry
   TGButtonGroup       *fPadBorderMode;      // pad border mode button group
   TGLineWidthComboBox *fPadBorderSize;      // pad border size combo box
   TGColorSelect       *fPadColor;           // pad fill color selection widget
   TGCheckButton       *fPadTickX;           // pad ticks along X show/hide check box
   TGCheckButton       *fPadTickY;           // pad ticks along Y show/hide check box
   TGCheckButton       *fPadGridX;           // pad grid along X show/hide check box
   TGCheckButton       *fPadGridY;           // pad grid along Y show/hide check box
   TGColorSelect       *fGridColor;          // pad grid line color selection widget
   TGLineWidthComboBox *fGridWidth;          // pad grid line width combo box
   TGLineStyleComboBox *fGridStyle;          // pad grid line style combo box
   TGColorSelect       *fHistFillColor;      // histograms fill color selection widget
   TGedPatternSelect   *fHistFillStyle;      // histograms fill pattern selection widget
   TGColorSelect       *fHistLineColor;      // histograms fill color selection widget
   TGLineWidthComboBox *fHistLineWidth;      // histograms line width combo box
   TGLineStyleComboBox *fHistLineStyle;      // histograms line style combo box
   TGNumberEntry       *fBarWidth;           // histograms bar width number entry
   TGNumberEntry       *fBarOffset;          // histograms bar offset number entry
   TGCheckButton       *fHistMinimumZero;    // histograms minimum zero check box
   TGTextEntry         *fPaintTextFormat;    // histograms format text entry
   TGNumberEntry       *fNumberContours;     // histograms number of contours number entry
   TGNumberEntry       *fLegoInnerR;         // histograms lego inner radius number entry
   TGColorSelect       *fFrameFillColor;     // frame fill color selection widget
   TGedPatternSelect   *fFrameFillStyle;     // frame fill pattern selection widget
   TGColorSelect       *fFrameLineColor;     // frame line color selection widget
   TGLineWidthComboBox *fFrameLineWidth;     // frame line width combo box
   TGLineStyleComboBox *fFrameLineStyle;     // frame line style combo box
   TGTextButton        *fPaletteEdit;        // palette editor open button
   TGButtonGroup       *fFrameBorderMode;    // frame border mode button group
   TGLineWidthComboBox *fFrameBorderSize;    // frame border size combo box
   TGColorSelect       *fFuncColor;          // function color selection widget
   TGLineWidthComboBox *fFuncWidth;          // function width number entry
   TGLineStyleComboBox *fFuncStyle;          // function line style combo box
   TGCheckButton       *fDrawBorder;         // function border show/hide check box
   TGNumberEntry       *fEndErrorSize;       // end error size number entry
   TGNumberEntry       *fErrorX;             // error along abscissa number entry
   TGNumberEntry       *fTimeOffsetDate;     // axis time offset (mm/dd/yyyy) number entry
   TGNumberEntry       *fTimeOffsetTime;     // axis time offset (hh:mm:ss) number entry
   TGCheckButton       *fStripDecimals;      // axis label's decimal part show/hide check box
   TGTextButton        *fApplyOnXYZ;         // axis apply on XYZ text button
   TGNumberEntry       *fXTitleSize;         // X axis title size number entry
   TGCheckButton       *fXTitleSizeInPixels; // X axis title size check box
   TGColorSelect       *fXTitleColor;        // X axis title color selection widget
   TGNumberEntry       *fXTitleOffset;       // X axis title offset number entry
   TGFontTypeComboBox  *fXTitleFont;         // X axis title font combo box
   TGNumberEntry       *fXLabelSize;         // X axis label size number entry
   TGCheckButton       *fXLabelSizeInPixels; // X axis label size check box
   TGColorSelect       *fXLabelColor;        // X axis label color selection widget
   TGNumberEntry       *fXLabelOffset;       // X axis label offset number entry
   TGFontTypeComboBox  *fXLabelFont;         // X axis label font combo box
   TGColorSelect       *fXAxisColor;         // X axis color selection widget
   TGNumberEntry       *fXTickLength;        // X axis tick length number entry
   TGCheckButton       *fOptLogx;            // X axis logarithmic scale check box
   TGNumberEntry       *fXNdivMain;          // X axis primary division number entry
   TGNumberEntry       *fXNdivSub;           // X axis secondary division number entry
   TGNumberEntry       *fXNdivSubSub;        // X axis tertiary division number entry
   TGCheckButton       *fXNdivisionsOptimize;// X axis division optimization check box
   TGNumberEntry       *fYTitleSize;         // Y axis title size number entry
   TGCheckButton       *fYTitleSizeInPixels; // Y axis title size check box
   TGColorSelect       *fYTitleColor;        // Y axis title color selection widget
   TGNumberEntry       *fYTitleOffset;       // Y axis title offset number entry
   TGFontTypeComboBox  *fYTitleFont;         // Y axis title font combo box
   TGNumberEntry       *fYLabelSize;         // Y axis label size number entry
   TGCheckButton       *fYLabelSizeInPixels; // Y axis label size check box
   TGColorSelect       *fYLabelColor;        // Y axis label color selection widget
   TGNumberEntry       *fYLabelOffset;       // Y axis label offset number entry
   TGFontTypeComboBox  *fYLabelFont;         // Y axis label font combo box
   TGColorSelect       *fYAxisColor;         // Y axis color selection widget
   TGNumberEntry       *fYTickLength;        // Y axis tick length number entry
   TGCheckButton       *fOptLogy;            // Y axis logarithmic scale check box
   TGNumberEntry       *fYNdivMain;          // Y axis primary division number entry
   TGNumberEntry       *fYNdivSub;           // Y axis secondary division number entry
   TGNumberEntry       *fYNdivSubSub;        // Y axis tertiary division number entry
   TGCheckButton       *fYNdivisionsOptimize;// Y axis division optimization check box
   TGNumberEntry       *fZTitleSize;         // Z axis title size number entry
   TGCheckButton       *fZTitleSizeInPixels; // Z axis title size check box
   TGColorSelect       *fZTitleColor;        // Z axis title color selection widget
   TGNumberEntry       *fZTitleOffset;       // Z axis title offset number entry
   TGFontTypeComboBox  *fZTitleFont;         // Z axis title font combo box
   TGNumberEntry       *fZLabelSize;         // Z axis label size number entry
   TGCheckButton       *fZLabelSizeInPixels; // Z axis label size check box
   TGColorSelect       *fZLabelColor;        // Z axis label color selection widget
   TGNumberEntry       *fZLabelOffset;       // Z axis label offset number entry
   TGFontTypeComboBox  *fZLabelFont;         // Z axis label font combo box
   TGColorSelect       *fZAxisColor;         // Z axis color selection widget
   TGNumberEntry       *fZTickLength;        // Z axis tick length number entry
   TGCheckButton       *fOptLogz;            // Z axis logarithmic scale check box
   TGNumberEntry       *fZNdivMain;          // Z axis primary division number entry
   TGNumberEntry       *fZNdivSub;           // Z axis secondary division number entry
   TGNumberEntry       *fZNdivSubSub;        // Z axis tertiary division number entry
   TGCheckButton       *fZNdivisionsOptimize;// Z axis division optimization check box
   TGCheckButton       *fOptTitle;           // title show/hide check box
   TGColorSelect       *fTitleColor;         // title fill color selection widget
   TGedPatternSelect   *fTitleStyle;         // title fill pattern selection widget
   TGColorSelect       *fTitleTextColor;     // title text color selection widget
   TGNumberEntry       *fTitleFontSize;      // title font size number entry
   TGCheckButton       *fTitleFontSizeInPixels; // title font size check box
   TGFontTypeComboBox  *fTitleFont;          // title font combo box
   TGComboBox          *fTitleAlign;         // title align combo box
   TGLabel             *fTitleBorderSizeLabel;  // label 'Title's'
   TGLineWidthComboBox *fTitleBorderSize;    // title border size combo box
   TGNumberEntry       *fTitleX;             // title abscissa number entry
   TGNumberEntry       *fTitleY;             // title ordinate number entry
   TGNumberEntry       *fTitleW;             // title width number entry
   TGNumberEntry       *fTitleH;             // title height number entry
   TGLabel             *fLegendBorderSizeLabel; // label 'Legend's'
   TGLineWidthComboBox *fLegendBorderSize;   // legend border size combo box
   TGColorSelect       *fStatColor;          // stats fill color selection widget
   TGedPatternSelect   *fStatStyle;          // stats fill pattern selection widget
   TGColorSelect       *fStatTextColor;      // stats text color selection widget
   TGNumberEntry       *fStatFontSize;       // stats font size number entry
   TGCheckButton       *fStatFontSizeInPixels;  // stats font size check box
   TGFontTypeComboBox  *fStatFont;           // stats font type combo box
   TGNumberEntry       *fStatX;              // stats abscissa number entry
   TGNumberEntry       *fStatY;              // stats ordinate number entry
   TGNumberEntry       *fStatW;              // stats width number entry
   TGNumberEntry       *fStatH;              // stats height number entry
   TGLabel             *fStatBorderSizeLabel;   // label 'stats' shadow
   TGLineWidthComboBox *fStatBorderSize;     // stats border size combo box
   TGCheckButton       *fOptStatName;        // stats name show/hide check box
   TGCheckButton       *fOptStatEntries;     // stats entries show/hide check box
   TGCheckButton       *fOptStatOverflow;    // stats overflow show/hide check box
   TGCheckButton       *fOptStatMean;        // stats mean show/hide check box
   TGCheckButton       *fOptStatUnderflow;   // stats underflow show/hide check box
   TGCheckButton       *fOptStatRMS;         // stats RMS show/hide check box
   TGCheckButton       *fOptStatSkewness;    // stats Skewness show/hide check box
   TGCheckButton       *fOptStatIntegral;    // stats integral show/hide check box
   TGCheckButton       *fOptStatKurtosis;    // stats kurtosis show/hide check box
   TGCheckButton       *fOptStatErrors;      // stats errors check box
   TGLabel             *fStatFormatLabel;    // label 'stats format'
   TGTextEntry         *fStatFormat;         // stats format text entry
   TGCheckButton       *fOptFitValues;       // fit values show/hide check box
   TGCheckButton       *fOptFitErrors;       // fit errors check box
   TGCheckButton       *fOptFitProbability;  // fit probability show/hide check box
   TGCheckButton       *fOptFitChi;          // fit Chi show/hide check box
   TGLabel             *fFitFormatLabel;     // label 'fit format'
   TGTextEntry         *fFitFormat;          // fit format text entry
   TGTextEntry         *fHeaderPS;           // ps/pdf header text entry
   TGTextEntry         *fTitlePS;            // ps/pdf title text entry
   TGButtonGroup       *fColorModelPS;       // ps/pdf color model button group
   TGRadioButton       *fColorModelPSRGB;    // RGB radio button
   TGRadioButton       *fColorModelPSCMYK;   // CMYB radio button
   TGNumberEntry       *fLineScalePS;        // ps/pdf line scale number entry
   TGComboBox          *fPaperSizePredef;    // ps/pdf paper size combo box
   Bool_t               fPaperSizeEnCm;      //=kTRUE if the paper size is in cm
   TGNumberEntry       *fPaperSizeX;         // ps/pdf paper size width number entry
   TGNumberEntry       *fPaperSizeY;         // ps/pdf paper size height number entry

   TGLayoutHints       *fLayoutExpandX;               // often used layout
   TGLayoutHints       *fLayoutExpandXMargin;         // often used layout
   TGLayoutHints       *fLayoutExpandXY;              // often used layout
   TGLayoutHints       *fLayoutExpandXYMargin;        // often used layout
   TGLayoutHints       *fLayoutExpandXCenterYMargin;  // often used layout

   void AddMenus(TGCompositeFrame *p);
   void DoNew();
   void DoDelete();
   void DoRename();
   void DoExport();
   void DoExit();
   void DoHelp(Int_t i);
   void DoImportCanvas();
   void CreateMacro();

   void AddToolbar(TGCompositeFrame *p);

   void AddTopLevelInterface(TGCompositeFrame *cf);
   void UpdateStatusBar();
   void UpdateEditor(Int_t tabNum);
   void ConnectAll();
   void DisconnectAll();
   void ConnectEditor(Int_t tabNum);
   void DisconnectEditor(Int_t tabNum);
   void DoEditor();

   void AddEdition(TGCompositeFrame *p);

   void CreateTabGeneral(TGCompositeFrame *tab);
   void AddGeneralLine(TGCompositeFrame *f);
   void AddGeneralFill(TGCompositeFrame *f);
   void AddGeneralText(TGCompositeFrame *f);
   void AddGeneralMarker(TGCompositeFrame *f);

   void CreateTabCanvas(TGCompositeFrame *tab);
   void AddCanvasFill(TGCompositeFrame *f);
   void AddCanvasGeometry(TGCompositeFrame *f);
   void AddCanvasBorder(TGCompositeFrame *f);
   void AddCanvasDate(TGCompositeFrame *f);

   void CreateTabPad(TGCompositeFrame *tab);
   void AddPadMargin(TGCompositeFrame *f);
   void AddPadBorder(TGCompositeFrame *f);
   void AddPadFill(TGCompositeFrame *f);
   void AddPadTicks(TGCompositeFrame *f);
   void AddPadGrid(TGCompositeFrame *f);

   void CreateTabHistos(TGCompositeFrame *tab);
   void CreateTabHistosHistos(TGCompositeFrame *tab);
   void AddHistosHistosFill(TGCompositeFrame *f);
   void AddHistosHistosLine(TGCompositeFrame *f);
   void AddHistosHistosBar(TGCompositeFrame *f);
   void AddHistosHistosContours(TGCompositeFrame *f);
   void AddHistosHistosAxis(TGCompositeFrame *f);
   void AddHistosHistosLegoInnerR(TGCompositeFrame *f);

   void CreateTabHistosFrames(TGCompositeFrame *tab);
   void AddHistosFramesFill(TGCompositeFrame *f);
   void AddHistosFramesLine(TGCompositeFrame *f);
   void AddHistosFramesBorder(TGCompositeFrame *f);

   void CreateTabHistosGraphs(TGCompositeFrame *tab);
   void AddHistosGraphsLine(TGCompositeFrame *f);
   void AddHistosGraphsBorder(TGCompositeFrame *f);
   void AddHistosGraphsErrors(TGCompositeFrame *f);

   void CreateTabAxis(TGCompositeFrame *tab);
   void CreateTabAxisX(TGCompositeFrame *tab);
   void AddAxisXTitle(TGCompositeFrame *f);
   void AddAxisXLine(TGCompositeFrame *f);
   void AddAxisXLabels(TGCompositeFrame *f);
   void AddAxisXDivisions(TGCompositeFrame *f);

   void CreateTabAxisY(TGCompositeFrame *tab);
   void AddAxisYTitle(TGCompositeFrame *f);
   void AddAxisYLine(TGCompositeFrame *f);
   void AddAxisYLabels(TGCompositeFrame *f);
   void AddAxisYDivisions(TGCompositeFrame *f);

   void CreateTabAxisZ(TGCompositeFrame *tab);
   void AddAxisZTitle(TGCompositeFrame *f);
   void AddAxisZLine(TGCompositeFrame *f);
   void AddAxisZLabels(TGCompositeFrame *f);
   void AddAxisZDivisions(TGCompositeFrame *f);

   void CreateTabTitle(TGCompositeFrame *tab);
   void AddTitleFill(TGCompositeFrame *f);
   void AddTitleBorderSize(TGCompositeFrame *f);
   void AddTitleText(TGCompositeFrame *f);
   void AddTitleGeometry(TGCompositeFrame *f);

   void CreateTabStats(TGCompositeFrame *tab);
   void AddStatsFill(TGCompositeFrame *f);
   void AddStatsText(TGCompositeFrame *f);
   void AddStatsGeometry(TGCompositeFrame *f);
   void AddStatsStats(TGCompositeFrame *f);
   void AddStatsFit(TGCompositeFrame *f);

   void CreateTabPsPdf(TGCompositeFrame *tab);
   void AddPsPdfHeader(TGCompositeFrame *f);
   void AddPsPdfTitle(TGCompositeFrame *f);
   void AddPsPdfColorModel(TGCompositeFrame *f);
   void AddPsPdfPaperSize(TGCompositeFrame *f);
   void AddPsPdfLineScale(TGCompositeFrame *f);

   void                 AddTitle(TGCompositeFrame *f, const char *s);
   TGColorSelect       *AddColorEntry(TGCompositeFrame *f, Int_t id);
   TGedPatternSelect   *AddFillStyleEntry(TGCompositeFrame *f, Int_t id);
   TGedMarkerSelect    *AddMarkerStyleEntry(TGCompositeFrame *f, Int_t id);
   TGComboBox          *AddMarkerSizeEntry(TGCompositeFrame *f, Int_t id);
   TGNumberEntry       *AddNumberEntry(TGCompositeFrame *f, Int_t e1, Int_t e2,
                           Int_t e3, Int_t id, const char *s, Double_t init, Int_t digits,
                           TGNumberFormat::EStyle nfS, TGNumberFormat::EAttribute nfA,
                           TGNumberFormat::ELimit nfL, Double_t min, Double_t max);
   TGLineWidthComboBox *AddLineWidthEntry(TGCompositeFrame *f, Int_t id);
   TGLineStyleComboBox *AddLineStyleEntry(TGCompositeFrame *f, Int_t id);
   TGTextButton        *AddTextButton(TGCompositeFrame *f, const char *s, Int_t id);
   TGFontTypeComboBox  *AddFontTypeEntry(TGCompositeFrame *f, Int_t id);
   TGComboBox          *AddTextAlignEntry(TGCompositeFrame *f, Int_t id);
   TGButtonGroup       *AddBorderModeEntry(TGCompositeFrame *f, Int_t id1, Int_t id2, Int_t id3);
   TGComboBox          *AddDateFormatEntry(TGCompositeFrame *f, Int_t id);
   TGCheckButton       *AddCheckButton(TGCompositeFrame *f, const char *s, Int_t id, Int_t e1 = 0, Int_t e2 = 2);
   TGTextEntry         *AddTextEntry(TGCompositeFrame *f, const char *s, Int_t id);
   TGComboBox          *AddPaperSizeEntry(TGCompositeFrame *f, Int_t id);

public:
   TStyleManager(const TGWindow *);
   virtual ~TStyleManager();

   static void Show();
   static void Terminate();
   static TStyleManager *&GetSM();

   void Init();
   void Hide();

   void SetCurSelStyle(TStyle *style) { fCurSelStyle = style; }
   void SetLastChoice(Bool_t choice)  { fLastChoice = choice; }

   void DoMenu(Int_t menuID);                // SLOT
   void DoImportMacro(Bool_t create);        // SLOT
   void DoListSelect();                      // SLOT
   void DoRealTime(Bool_t b);                // SLOT
   void DoPreview(Bool_t b);                 // SLOT
   void DoPreviewClosed();                   // SLOT
   void DoMakeDefault();                     // SLOT
   void DoApplyOnSelect(Int_t i);            // SLOT
   void DoApplyOn();                         // SLOT
   void DoMoreLess();                        // SLOT
   void DoEditionUpdatePreview();            // SLOT
   void DoChangeTab(Int_t i);                // SLOT
   void DoChangeAxisTab(Int_t i);            // SLOT
   void BuildList(TStyle *style = 0);
   void DoSelectNoCanvas();                  // SLOT
   void DoSelectCanvas(TVirtualPad *pad,
         TObject *obj, Int_t mouseButton);   // SLOT
   void CloseWindow();                       // SLOT

// GENERAL
   void ModFillColor();                      // SLOT
   void ModFillStyle();                      // SLOT
   void ModHatchesLineWidth();               // SLOT
   void ModHatchesSpacing();                 // SLOT
   void ModMarkerColor();                    // SLOT
   void ModMarkerStyle();                    // SLOT
   void ModMarkerSize();                     // SLOT
   void ModScreenFactor();                   // SLOT
   void ModLineColor();                      // SLOT
   void ModLineWidth();                      // SLOT
   void ModLineStyle();                      // SLOT
   void ModLineStyleEdit();                  // SLOT
   void ModTextColor();                      // SLOT
   void ModTextSize();                       // SLOT
   void ModTextSizeInPixels(Bool_t b);       // SLOT
   void ModTextFont();                       // SLOT
   void ModTextAlign();                      // SLOT
   void ModTextAngle();                      // SLOT

// CANVAS
   void ModCanvasColor();                    // SLOT
   void ModCanvasDefX();                     // SLOT
   void ModCanvasDefY();                     // SLOT
   void ModCanvasDefW();                     // SLOT
   void ModCanvasDefH();                     // SLOT
   void ModCanvasBorderMode();               // SLOT
   void ModCanvasBorderSize();               // SLOT
   void ModOptDateBool();                    // SLOT
   void ModAttDateTextColor();               // SLOT
   void ModAttDateTextSize();                // SLOT
   void ModAttDateTextSizeInPixels(Bool_t b);// SLOT
   void ModOptDateFormat();                  // SLOT
   void ModAttDateTextFont();                // SLOT
   void ModAttDateTextAngle();               // SLOT
   void ModAttDateTextAlign();               // SLOT
   void ModDateX();                          // SLOT
   void ModDateY();                          // SLOT

// PAD
   void ModPadTopMargin();                   // SLOT
   void ModPadBottomMargin();                // SLOT
   void ModPadLeftMargin();                  // SLOT
   void ModPadRightMargin();                 // SLOT
   void ModPadBorderMode();                  // SLOT
   void ModPadBorderSize();                  // SLOT
   void ModPadColor();                       // SLOT
   void ModPadTickX();                       // SLOT
   void ModPadTickY();                       // SLOT
   void ModPadGridX();                       // SLOT
   void ModPadGridY();                       // SLOT
   void ModGridColor();                      // SLOT
   void ModGridWidth();                      // SLOT
   void ModGridStyle();                      // SLOT

 // HISTOS HISTOS
   void ModHistFillColor();                  // SLOT
   void ModHistFillStyle();                  // SLOT
   void ModHistLineColor();                  // SLOT
   void ModHistLineWidth();                  // SLOT
   void ModHistLineStyle();                  // SLOT
   void ModBarWidth();                       // SLOT
   void ModBarOffset();                      // SLOT
   void ModHistMinimumZero();                // SLOT
   void ModPaintTextFormat();                // SLOT
   void ModNumberContours();                 // SLOT
   void ModLegoInnerR();                     // SLOT

// HISTOS FRAMES
   void ModFrameFillColor();                 // SLOT
   void ModFrameFillStyle();                 // SLOT
   void ModFrameLineColor();                 // SLOT
   void ModFrameLineWidth();                 // SLOT
   void ModFrameLineStyle();                 // SLOT
   void ModPaletteEdit();                    // SLOT
   void ModFrameBorderMode();                // SLOT
   void ModFrameBorderSize();                // SLOT

// HISTOS GRAPHS
   void ModFuncColor();                      // SLOT
   void ModFuncWidth();                      // SLOT
   void ModFuncStyle();                      // SLOT
   void ModDrawBorder();                     // SLOT
   void ModEndErrorSize();                   // SLOT
   void ModErrorX();                         // SLOT

// AXIS
   void ModTimeOffset();                     // SLOT
   void ModStripDecimals();                  // SLOT
   void ModApplyOnXYZ();                     // SLOT

// AXIS X AXIS
   void ModXTitleSize();                     // SLOT
   void ModXTitleSizeInPixels(Bool_t b);     // SLOT
   void ModXTitleColor();                    // SLOT
   void ModXTitleOffset();                   // SLOT
   void ModXTitleFont();                     // SLOT
   void ModXLabelSize();                     // SLOT
   void ModXLabelSizeInPixels(Bool_t b);     // SLOT
   void ModXLabelColor();                    // SLOT
   void ModXLabelOffset();                   // SLOT
   void ModXLabelFont();                     // SLOT
   void ModXAxisColor();                     // SLOT
   void ModXTickLength();                    // SLOT
   void ModOptLogx();                        // SLOT
   void ModXNdivisions();                    // SLOT

// AXIS Y AXIS
   void ModYTitleSize();                     // SLOT
   void ModYTitleSizeInPixels(Bool_t b);     // SLOT
   void ModYTitleColor();                    // SLOT
   void ModYTitleOffset();                   // SLOT
   void ModYTitleFont();                     // SLOT
   void ModYLabelSize();                     // SLOT
   void ModYLabelSizeInPixels(Bool_t b);     // SLOT
   void ModYLabelColor();                    // SLOT
   void ModYLabelOffset();                   // SLOT
   void ModYLabelFont();                     // SLOT
   void ModYAxisColor();                     // SLOT
   void ModYTickLength();                    // SLOT
   void ModOptLogy();                        // SLOT
   void ModYNdivisions();                    // SLOT

// AXIS Z AXIS
   void ModZTitleSize();                     // SLOT
   void ModZTitleSizeInPixels(Bool_t b);     // SLOT
   void ModZTitleColor();                    // SLOT
   void ModZTitleOffset();                   // SLOT
   void ModZTitleFont();                     // SLOT
   void ModZLabelSize();                     // SLOT
   void ModZLabelSizeInPixels(Bool_t b);     // SLOT
   void ModZLabelColor();                    // SLOT
   void ModZLabelOffset();                   // SLOT
   void ModZLabelFont();                     // SLOT
   void ModZAxisColor();                     // SLOT
   void ModZTickLength();                    // SLOT
   void ModOptLogz();                        // SLOT
   void ModZNdivisions();                    // SLOT

// TITLES
   void ModOptTitle();                       // SLOT
   void ModTitleFillColor();                 // SLOT
   void ModTitleStyle();                     // SLOT
   void ModTitleTextColor();                 // SLOT
   void ModTitleFontSize();                  // SLOT
   void ModTitleFontSizeInPixels(Bool_t b);  // SLOT
   void ModTitleFont();                      // SLOT
   void ModTitleAlign();                     // SLOT
   void ModTitleBorderSize();                // SLOT
   void ModTitleX();                         // SLOT
   void ModTitleY();                         // SLOT
   void ModTitleW();                         // SLOT
   void ModTitleH();                         // SLOT
   void ModLegendBorderSize();               // SLOT

// STATS
   void ModStatColor(Pixel_t color);         // SLOT
   void ModStatStyle(Style_t pattern);       // SLOT
   void ModStatTextColor(Pixel_t color);     // SLOT
   void ModStatFontSize();                   // SLOT
   void ModStatFontSizeInPixels(Bool_t b);   // SLOT
   void ModStatFont();                       // SLOT
   void ModStatX();                          // SLOT
   void ModStatY();                          // SLOT
   void ModStatW();                          // SLOT
   void ModStatH();                          // SLOT
   void ModStatBorderSize();                 // SLOT
   void ModOptStat();                        // SLOT
   void ModStatFormat(const char *sformat);  // SLOT
   void ModOptFit();                         // SLOT
   void ModFitFormat(const char *fitformat); // SLOT

// PS / PDF
   void ModHeaderPS();                       // SLOT
   void ModTitlePS();                        // SLOT
   void ModColorModelPS();                   // SLOT
   void ModLineScalePS();                    // SLOT
   void ModPaperSizePredef();                // SLOT
   void ModPaperSizeXY();                    // SLOT

   ClassDef(TStyleManager, 0) // Graphical User Interface for managing styles
};

#endif
