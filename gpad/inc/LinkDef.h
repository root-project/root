/* @(#)root/gpad:$Name:  $:$Id: LinkDef.h,v 1.3 2001/02/13 08:30:33 brun Exp $ */

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

// Gtypes.h enums
#pragma link C++ enum EColor;
#pragma link C++ enum ELineStyle;
#pragma link C++ enum EMarkerStyle;

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
