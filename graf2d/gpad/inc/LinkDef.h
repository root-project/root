/* @(#)root/gpad:$Id$ */

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

#pragma link C++ class TAttCanvas+;
#pragma link C++ class TButton+;
#pragma link C++ class TCanvas-;
#pragma link C++ class TClassTree-;
#pragma link C++ class TColorWheel+;
#pragma link C++ class TControlBar+;
#pragma link C++ class TControlBarButton+;
#pragma link C++ class TDialogCanvas+;
#pragma link C++ class TGroupButton+;
#pragma link C++ class TInspectCanvas+;
#pragma link C++ class TPad-;
#pragma link C++ class TPaveClass+;
#pragma link C++ class TSlider+;
#pragma link C++ class TSliderBox+;
#pragma link C++ class TView+;
#pragma link C++ class TViewer3DPad;
#pragma link C++ class TPadPainter;
#pragma link C++ class TRatioPlot+;

#ifdef ROOT7_TDisplayItem
#pragma link C++ class ROOT::Experimental::TDisplayItem+;
#pragma link C++ class std::vector<ROOT::Experimental::TDisplayItem*>+;
#pragma link C++ class ROOT::Experimental::TPadDisplayItem+;
#pragma link C++ class ROOT::Experimental::TUniqueDisplayItem<TPad>+;
#pragma link C++ class ROOT::Experimental::TOrdinaryDisplayItem<TH1>+;
#pragma link C++ class ROOT::Experimental::Detail::TMenuItem+;
#pragma link C++ class std::vector<ROOT::Experimental::Detail::TMenuItem*>+;
#pragma link C++ class ROOT::Experimental::Detail::TCheckedMenuItem+;
#pragma link C++ class ROOT::Experimental::Detail::TMenuArgument+;
#pragma link C++ class std::vector<ROOT::Experimental::Detail::TMenuArgument>+;
#pragma link C++ class ROOT::Experimental::Detail::TArgsMenuItem+;
#pragma link C++ class ROOT::Experimental::TMenuItems+;
#pragma link C++ class ROOT::Experimental::TObjectDrawable+;
#pragma link C++ class ROOT::Experimental::Detail::TPadUserCoordBase+;
#pragma link C++ class ROOT::Experimental::Detail::TPadLinearUserCoord+;
#pragma link C++ struct ROOT::Experimental::Internal::TPadHorizVert+;
#pragma link C++ struct ROOT::Experimental::TPadExtent+;
#pragma link C++ struct ROOT::Experimental::TPadPos+;
#pragma link C++ class std::vector<std::unique_ptr<ROOT::Experimental::TDrawable>>+;
#pragma link C++ class ROOT::Experimental::Internal::TPadBase+;
#pragma link C++ class ROOT::Experimental::TPad+;
#pragma link C++ class ROOT::Experimental::TPadDrawable+;
#pragma link C++ class ROOT::Experimental::TCanvas+;
#pragma link C++ class ROOT::Experimental::TPadCoord::Pixel+;
#pragma link C++ class ROOT::Experimental::TPadCoord::Normal+;
#pragma link C++ class ROOT::Experimental::TPadCoord::User+;
//#pragma link C++ class ROOT::Experimental::TPadCoord::CoordSysBase<ROOT::Experimental::TPadCoord::Pixel>+;
//#pragma link C++ class ROOT::Experimental::TPadCoord::CoordSysBase<ROOT::Experimental::TPadCoord::Normal>+;
//#pragma link C++ class ROOT::Experimental::TPadCoord::CoordSysBase<ROOT::Experimental::TPadCoord::User>+;
#endif

#endif
