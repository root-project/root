/* @(#)root/hist:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

// need to correctly generate dictionary for display items, used in v7 histpainter,
// but currently histpainter does not creates dictionary at all
#pragma extra_include "ROOT/TDisplayItem.hxx";

#pragma link C++ class ROOT::Experimental::THistDrawingOpts<1>+;
#pragma link C++ class ROOT::Experimental::THistDrawingOpts<2>+;
#pragma link C++ class ROOT::Experimental::THistDrawingOpts<3>+;
#pragma link C++ class ROOT::Experimental::THistDrawable<1>+;
#pragma link C++ class ROOT::Experimental::THistDrawable<2>+;
#pragma link C++ class ROOT::Experimental::THistDrawable<3>+;
#pragma link C++ class ROOT::Experimental::TDrawableBase<ROOT::Experimental::THistDrawable<1>>+;
#pragma link C++ class ROOT::Experimental::TDrawableBase<ROOT::Experimental::THistDrawable<2>>+;
#pragma link C++ class ROOT::Experimental::TDrawableBase<ROOT::Experimental::THistDrawable<3>>+;
#pragma link C++ class ROOT::Experimental::THistDrawableBase<ROOT::Experimental::THistDrawable<1>>+;
#pragma link C++ class ROOT::Experimental::THistDrawableBase<ROOT::Experimental::THistDrawable<2>>+;
#pragma link C++ class ROOT::Experimental::THistDrawableBase<ROOT::Experimental::THistDrawable<3>>+;
#pragma link C++ class ROOT::Experimental::TOrdinaryDisplayItem<ROOT::Experimental::THistDrawable<1>>+;
#pragma link C++ class ROOT::Experimental::TOrdinaryDisplayItem<ROOT::Experimental::THistDrawable<2>>+;
#pragma link C++ class ROOT::Experimental::TOrdinaryDisplayItem<ROOT::Experimental::THistDrawable<3>>+;
#pragma link C++ class ROOT::Experimental::Internal::TUniWeakPtr<ROOT::Experimental::Detail::THistImplPrecisionAgnosticBase<1> >+;
#pragma link C++ class ROOT::Experimental::Internal::TUniWeakPtr<ROOT::Experimental::Detail::THistImplPrecisionAgnosticBase<2>>+;
#pragma link C++ class ROOT::Experimental::Internal::TUniWeakPtr<ROOT::Experimental::Detail::THistImplPrecisionAgnosticBase<3>>+;
#pragma link C++ class ROOT::Experimental::TStringEnumAttr<ROOT::Experimental::THistDrawingOpts<1>::EStyle>+;
#pragma link C++ class ROOT::Experimental::TStringEnumAttr<ROOT::Experimental::THistDrawingOpts<2>::EStyle>+;
#pragma link C++ class ROOT::Experimental::TStringEnumAttr<ROOT::Experimental::THistDrawingOpts<3>::EStyle>+;
#pragma link C++ class ROOT::Experimental::TDrawingAttr<ROOT::Experimental::TStringEnumAttr<ROOT::Experimental::THistDrawingOpts<1>::EStyle>>+;
#pragma link C++ class ROOT::Experimental::TDrawingAttr<ROOT::Experimental::TStringEnumAttr<ROOT::Experimental::THistDrawingOpts<2>::EStyle>>+;
#pragma link C++ class ROOT::Experimental::TDrawingAttr<ROOT::Experimental::TStringEnumAttr<ROOT::Experimental::THistDrawingOpts<3>::EStyle>>+;


#endif
