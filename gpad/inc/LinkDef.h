/* @(#)root/gpad:$Name:  $:$Id: LinkDef.h,v 1.2 2000/11/21 20:19:18 brun Exp $ */

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

// enum EMarkerStyle
#pragma link C++ global kDot;
#pragma link C++ global kPlus;
#pragma link C++ global kStar;
#pragma link C++ global kCircle;
#pragma link C++ global kMultiply;
#pragma link C++ global kFullCircle;
#pragma link C++ global kFullSquare;
#pragma link C++ global kFullTriangleUp;
#pragma link C++ global kFullTriangleDown;
#pragma link C++ global kFullStar;
#pragma link C++ global kFullDotSmall;
#pragma link C++ global kFullDotMedium;
#pragma link C++ global kFullDotLarge;
#pragma link C++ global kOpenCircle;
#pragma link C++ global kOpenSquare;
#pragma link C++ global kOpenTriangleUp;
#pragma link C++ global kOpenDiamond;
#pragma link C++ global kOpenCross;
#pragma link C++ global kOpenStar;

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
