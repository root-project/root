/* @(#)root/gpad:$Name:  $:$Id: LinkDef.h,v 1.1.1.1 2000/05/16 17:00:41 rdm Exp $ */

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

// enum EColor
#pragma link C++ global kWhite;
#pragma link C++ global kBlack;
#pragma link C++ global kRed;
#pragma link C++ global kGreen;
#pragma link C++ global kBlue;
#pragma link C++ global kYellow;
#pragma link C++ global kMagenta;
#pragma link C++ global kCyan;

// enum ELineStyle
#pragma link C++ global kSolid;
#pragma link C++ global kDashed;
#pragma link C++ global kDotted;
#pragma link C++ global kDashDotted;

#pragma link C++ class TAttCanvas+;
#pragma link C++ class TButton+;
#pragma link C++ class TCanvas-;
#pragma link C++ class TClassTree-;
#pragma link C++ class TControlBar+;
#pragma link C++ class TControlBarButton+;
#pragma link C++ class TDialogCanvas+;
#pragma link C++ class TAttLineCanvas+;
#pragma link C++ class TAttFillCanvas+;
#pragma link C++ class TAttTextCanvas+;
#pragma link C++ class TAttMarkerCanvas+;
#pragma link C++ class TDrawPanelHist+;
#pragma link C++ class TFitPanel+;
#pragma link C++ class TFitPanelGraph+;
#pragma link C++ class TGroupButton+;
#pragma link C++ class TInspectCanvas+;
#pragma link C++ class TPad-;
#pragma link C++ class TPaveClass+;
#pragma link C++ class TSlider+;
#pragma link C++ class TSliderBox+;

#endif
