/* @(#)root/gui:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ enum EButtonState;
#pragma link C++ enum EFrameState;
#pragma link C++ enum EFrameType;
#pragma link C++ enum EFrameCleanup;
#pragma link C++ enum EMWMHints;
#pragma link C++ enum ELayoutHints;
#pragma link C++ enum ETableLayoutHints;
#pragma link C++ enum EMenuEntryState;
#pragma link C++ enum EMenuEntryType;
#pragma link C++ enum EMsgBoxIcon;
#pragma link C++ enum EMsgBoxButton;
#pragma link C++ enum ETextJustification;
#pragma link C++ enum EWidgetStatus;
#pragma link C++ enum EWidgetMessageTypes;
#pragma link C++ enum TGNumberFormat::EStyle;
#pragma link C++ enum TGNumberFormat::EAttribute;
#pragma link C++ enum TGNumberFormat::ELimit;
#pragma link C++ enum TGNumberFormat::EStepSize;
#pragma link C++ enum TGGroupFrame::ETitlePos;



#pragma link C++ function MK_MSG;
#pragma link C++ function GET_MSG;
#pragma link C++ function GET_SUBMSG;
// This replaces the old
// #pragma link C++ global gClient;
// after the fix for ROOT-6106
// #pragma link C++ function TGClient::Instance;

#pragma link C++ class TGObject;
#pragma link C++ class TGClient;
#pragma link C++ class TGWindow;
#pragma link C++ class TGUnknownWindowHandler;
#pragma link C++ class TGIdleHandler;
#pragma link C++ class TGPicture;
#pragma link C++ class TGSelectedPicture;
#pragma link C++ class TGPicturePool;
#pragma link C++ class TGDimension;
#pragma link C++ class TGPosition;
#pragma link C++ class TGLongPosition;
#pragma link C++ class TGInsets;
#pragma link C++ class TGRectangle;
#pragma link C++ class TGFrame;
#pragma link C++ class TGCompositeFrame;
#pragma link C++ class TGVerticalFrame;
#pragma link C++ class TGHorizontalFrame;
#pragma link C++ class TGHeaderFrame;
#pragma link C++ class TGMainFrame;
#pragma link C++ class TGTransientFrame;
#pragma link C++ class TGGroupFrame;
#pragma link C++ class TGFrameElement;
#pragma link C++ class TGLayoutHints;
#pragma link C++ class TGTableLayoutHints;
#pragma link C++ class TGLayoutManager;
#pragma link C++ class TGVerticalLayout;
#pragma link C++ class TGHorizontalLayout;
#pragma link C++ class TGRowLayout;
#pragma link C++ class TGColumnLayout;
#pragma link C++ class TGMatrixLayout;
#pragma link C++ class TGTableLayout;
#pragma link C++ class TGTileLayout;
#pragma link C++ class TGListLayout;
#pragma link C++ class TGListDetailsLayout;
#pragma link C++ class TGString;
#pragma link C++ class TGHotString;
#pragma link C++ class TGWidget;
#pragma link C++ class TGIcon;
#pragma link C++ class TGLabel;
#pragma link C++ class TGButton;
#pragma link C++ class TGTextButton;
#pragma link C++ class TGPictureButton;
#pragma link C++ class TGCheckButton;
#pragma link C++ class TGRadioButton;
#pragma link C++ class TGSplitButton;
#pragma link C++ class TGButtonGroup;
#pragma link C++ class TGVButtonGroup;
#pragma link C++ class TGHButtonGroup;
#pragma link C++ class TGTextBuffer;
#pragma link C++ class TGTextEntry;
#pragma link C++ class TGMsgBox;
#pragma link C++ class TGInputDialog;
#pragma link C++ class TGMenuBar;
#pragma link C++ class TGPopupMenu;
#pragma link C++ class TGMenuTitle;
#pragma link C++ class TGMenuEntry;
#pragma link C++ class TGShutter;
#pragma link C++ class TGShutterItem;
#pragma link C++ class TGHorizontal3DLine;
#pragma link C++ class TGVertical3DLine;
#pragma link C++ class TGGC;
#pragma link C++ class TGGCPool;
#pragma link C++ class TGProgressBar;
#pragma link C++ class TGHProgressBar;
#pragma link C++ class TGVProgressBar;
#pragma link C++ class TGNumberFormat;
#pragma link C++ class TGNumberEntryField;
#pragma link C++ class TGNumberEntry;
#pragma link C++ class TGNumberEntryLayout;
#pragma link C++ class TGPack;
#pragma link C++ class TGFrameElementPack;

#pragma link C++ enum EFileDialogMode;
#pragma link C++ enum EFSSortMode;
#pragma link C++ enum EListViewMode;
#pragma link C++ enum EScrollBarMode;
#pragma link C++ enum ESliderType;
#pragma link C++ enum EDoubleSliderScale;
#pragma link C++ enum ETextLayoutFlags;
#pragma link C++ enum EFontWeight;
#pragma link C++ enum EFontSlant;

#pragma link C++ global gCurrentRegion;

#pragma link C++ struct ToolBarData_t;
#pragma link C++ struct FontMetrics_t;
#pragma link C++ struct FontAttributes_t;

#pragma link C++ class TGScrollBarElement;
#pragma link C++ class TGScrollBar;
#pragma link C++ class TGHScrollBar;
#pragma link C++ class TGVScrollBar;
#pragma link C++ class TGViewPort;
#pragma link C++ class TGCanvas;
#pragma link C++ class TGLBEntry;
#pragma link C++ class TGTextLBEntry;
#pragma link C++ class TGLineLBEntry;
#pragma link C++ class TGIconLBEntry;
#pragma link C++ class TGLBContainer;
#pragma link C++ class TGListBox;
#pragma link C++ class TGComboBoxPopup;
#pragma link C++ class TGComboBox;
#pragma link C++ class TGLineStyleComboBox;
#pragma link C++ class TGLineWidthComboBox;
#pragma link C++ class TGFontTypeComboBox;
#pragma link C++ class TGFSComboBox;
#pragma link C++ class TGTreeLBEntry;
#pragma link C++ class TGTabElement;
#pragma link C++ class TGTabLayout;
#pragma link C++ class TGTab;
#pragma link C++ class TGSlider;
#pragma link C++ class TGVSlider;
#pragma link C++ class TGHSlider;
#pragma link C++ class TGDoubleSlider;
#pragma link C++ class TGDoubleVSlider;
#pragma link C++ class TGDoubleHSlider;
#pragma link C++ class TGTripleVSlider;
#pragma link C++ class TGTripleHSlider;
#pragma link C++ class TGLVEntry;
#pragma link C++ class TGContainer;
#pragma link C++ class TGLVContainer;
#pragma link C++ class TGListView;
#pragma link C++ class TGMimeTypes;
#pragma link C++ class TGFileItem;
#pragma link C++ class TGFileContainer;
#pragma link C++ class TGFileDialog;
#pragma link C++ class TGFileInfo;
#pragma link C++ class TGStatusBar;
#pragma link C++ class TGToolTip;
#pragma link C++ class TGToolBar;
#pragma link C++ class TGListTreeItem;
#pragma link C++ class TGListTreeItemStd;
#pragma link C++ class TGListTree;
#pragma link C++ class TGSplitter;
#pragma link C++ class TGVSplitter;
#pragma link C++ class TGHSplitter;
#pragma link C++ class TGVFileSplitter;
#pragma link C++ class TGText;
#pragma link C++ class TGTextLine;
#pragma link C++ class TGView;
#pragma link C++ class TGViewFrame;
#pragma link C++ class TGTextView;
#pragma link C++ class TGTextEdit;
#pragma link C++ class TGSearchDialog;
#pragma link C++ class TGPrintDialog;
#pragma link C++ class TGGotoDialog;
#pragma link C++ class TGSearchType;
#pragma link C++ class TGRegion;
#pragma link C++ class TGRegionWithId;
#pragma link C++ class TGImageMap;
#pragma link C++ class TGApplication;
#pragma link C++ class TGXYLayout;
#pragma link C++ class TGXYLayoutHints;
#pragma link C++ class TGResourcePool;
#pragma link C++ class TGFont;
#pragma link C++ class TGFontPool;
#pragma link C++ class TGTextLayout;

#pragma link C++ enum EDragType;
#pragma link C++ enum EGuiBldAction;
#pragma link C++ enum EDNDFlags;

#pragma link C++ global gDragManager;
#pragma link C++ global gGuiBuilder;
#pragma link C++ global gDNDManager;

#pragma link C++ nestedclass;

#pragma link C++ enum EMdiResizerPlacement;
#pragma link C++ enum EMdiResizingModes;
#pragma link C++ enum EMdiHints;
#pragma link C++ enum EMdiArrangementModes;
#pragma link C++ enum EMdiGeometryMask;

#pragma link C++ class TRootGuiFactory;
#pragma link C++ class TRootApplication;
#pragma link C++ class TRootCanvas;
#pragma link C++ class TRootEmbeddedCanvas;
#pragma link C++ class TRootBrowserLite;
#pragma link C++ class TRootContextMenu;
#pragma link C++ class TRootDialog;
#pragma link C++ class TRootControlBar;
#pragma link C++ class TRootHelpDialog;

#pragma link C++ class TGColorFrame;
#pragma link C++ class TG16ColorSelector;
#pragma link C++ class TGColorPopup;
#pragma link C++ class TGColorSelect;
#pragma link C++ class TGColorPalette;
#pragma link C++ class TGColorPick;
#pragma link C++ class TGColorDialog;
#pragma link C++ class TGFontDialog;
#pragma link C++ class TGFontDialog::FontProp_t;
#pragma link C++ class TGDockableFrame;
#pragma link C++ class TGUndockedFrame;
#pragma link C++ class TGDockButton;
#pragma link C++ class TGDockHideButton;
#pragma link C++ class TGMdiMenuBar;
#pragma link C++ class TGMdiFrameList;
#pragma link C++ class TGMdiGeometry;
#pragma link C++ class TGMdiMainFrame;
#pragma link C++ class TGMdiContainer;
#pragma link C++ class TGMdiFrame;
#pragma link C++ class TGMdiWinResizer;
#pragma link C++ class TGMdiVerticalWinResizer;
#pragma link C++ class TGMdiHorizontalWinResizer;
#pragma link C++ class TGMdiCornerWinResizer;
#pragma link C++ class TGMdiButtons;
#pragma link C++ class TGMdiTitleIcon;
#pragma link C++ class TGMdiTitleBar;
#pragma link C++ class TGMdiDecorFrame;
#pragma link C++ class TVirtualDragManager;
#pragma link C++ class TGuiBuilder;
#pragma link C++ class TGuiBldAction;
#pragma link C++ class TGRedirectOutputGuard;
#pragma link C++ class TGPasswdDialog;
#pragma link C++ class TGTextEditor;
#pragma link C++ class TGSpeedo;
#pragma link C++ class TDNDData;
#pragma link C++ class TGDNDManager;
#pragma link C++ class TGDragWindow;
#pragma link C++ class TGTableCell;
#pragma link C++ class TGTableHeader;
#pragma link C++ class TGTable;
#pragma link C++ class TTableRange;
#pragma link C++ class TGTableFrame;
#pragma link C++ class TGSimpleTable;
#pragma link C++ class TGTableHeaderFrame;
#pragma link C++ class TGSimpleTableInterface;
#pragma link C++ class TGCommandPlugin;
#pragma link C++ class TGFileBrowser;
#pragma link C++ class TBrowserPlugin;
#pragma link C++ class TRootBrowser;

#pragma link C++ class TGSplitFrame;
#pragma link C++ class TGSplitTool;
#pragma link C++ class TGRectMap;
#pragma link C++ class TGShapedFrame;
#pragma link C++ class TGEventHandler;

#pragma link C++ class TGTextViewStreamBuf;
#pragma link C++ class TGTextViewostream;

#endif
