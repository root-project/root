/* @(#)root/gui:$Name:  $:$Id: LinkDef3.h,v 1.11 2005/10/14 10:56:07 rdm Exp $ */

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

#pragma link C++ enum EDragType;
#pragma link C++ enum EGuiBldAction;

#pragma link C++ global gDragManager;
#pragma link C++ global gGuiBuilder;

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
#pragma link C++ class TRootBrowser;
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

#endif
