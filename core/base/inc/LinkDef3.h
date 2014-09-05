/* @(#)root/base:$Id$ */

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

// kDefaultScrollBarWidth is not a global but an enum constant.
// Removing it from the list of selected globals until enum matching
// is not implemented
//#pragma link C++ global kDefaultScrollBarWidth;
//------
#pragma link C++ global kNone;
#pragma link C++ global kCopyFromParent;
#pragma link C++ global kParentRelative;
#pragma link C++ global kWABackPixmap;
#pragma link C++ global kWABackPixel;
#pragma link C++ global kWABorderPixmap;
#pragma link C++ global kWABorderPixel;
#pragma link C++ global kWABorderWidth;
#pragma link C++ global kWABitGravity;
#pragma link C++ global kWAWinGravity;
#pragma link C++ global kWABackingStore;
#pragma link C++ global kWABackingPlanes;
#pragma link C++ global kWABackingPixel;
#pragma link C++ global kWAOverrideRedirect;
#pragma link C++ global kWASaveUnder;
#pragma link C++ global kWAEventMask;
#pragma link C++ global kWADontPropagate;
#pragma link C++ global kWAColormap;
#pragma link C++ global kWACursor;

#pragma link C++ global kNoEventMask;
#pragma link C++ global kKeyPressMask;
#pragma link C++ global kKeyReleaseMask;
#pragma link C++ global kButtonPressMask;
#pragma link C++ global kButtonReleaseMask;
#pragma link C++ global kPointerMotionMask;
#pragma link C++ global kButtonMotionMask;
#pragma link C++ global kExposureMask;
#pragma link C++ global kStructureNotifyMask;
#pragma link C++ global kEnterWindowMask;
#pragma link C++ global kLeaveWindowMask;
#pragma link C++ global kFocusChangeMask;
#pragma link C++ global kOwnerGrabButtonMask;
#pragma link C++ global kColormapChangeMask;

#pragma link C++ global kKeyShiftMask;
#pragma link C++ global kKeyLockMask;
#pragma link C++ global kKeyControlMask;
#pragma link C++ global kKeyMod1Mask;
#pragma link C++ global kButton1Mask;
#pragma link C++ global kButton2Mask;
#pragma link C++ global kButton3Mask;
#pragma link C++ global kButton4Mask;
#pragma link C++ global kButton5Mask;
#pragma link C++ global kAnyModifier;

#pragma link C++ global kGCFunction;
#pragma link C++ global kGCPlaneMask;
#pragma link C++ global kGCForeground;
#pragma link C++ global kGCBackground;
#pragma link C++ global kGCLineWidth;
#pragma link C++ global kGCLineStyle;
#pragma link C++ global kGCCapStyle;
#pragma link C++ global kGCJoinStyle;
#pragma link C++ global kGCFillStyle;
#pragma link C++ global kGCFillRule;
#pragma link C++ global kGCTile;
#pragma link C++ global kGCStipple;
#pragma link C++ global kGCTileStipXOrigin;
#pragma link C++ global kGCTileStipYOrigin;
#pragma link C++ global kGCFont;
#pragma link C++ global kGCSubwindowMode;
#pragma link C++ global kGCGraphicsExposures;
#pragma link C++ global kGCClipXOrigin;
#pragma link C++ global kGCClipYOrigin;
#pragma link C++ global kGCClipMask;
#pragma link C++ global kGCDashOffset;
#pragma link C++ global kGCDashList;
#pragma link C++ global kGCArcMode;

#pragma link C++ global kDoRed;
#pragma link C++ global kDoGreen;
#pragma link C++ global kDoBlue;

#pragma link C++ global kPAColormap;
#pragma link C++ global kPADepth;
#pragma link C++ global kPASize;
#pragma link C++ global kPAHotspot;
#pragma link C++ global kPAReturnPixels;
#pragma link C++ global kPACloseness;

#pragma link C++ global kPrimarySelection;
#pragma link C++ global kCutBuffer;
#pragma link C++ global kMaxPixel;

// #pragma link C++ global gPerfStats;
#pragma link C++ global gMonitoringWriter;
#pragma link C++ global gMonitoringReader;

#pragma link C++ enum EGuiConstants;
#pragma link C++ enum EGEventType;
#pragma link C++ enum EGraphicsFunction;
#pragma link C++ enum EGraphicsFunction;
#pragma link C++ enum EMouseButton;
#pragma link C++ enum EXMagic;
#pragma link C++ enum EInitialState;
#pragma link C++ enum EKeySym;
#pragma link C++ enum EEventType;
#pragma link C++ enum ECursor;
#pragma link C++ global kNumCursors;

#pragma link C++ typedef timespec_t;
// #pragma link C++ typedef Handle_t;
// #pragma link C++ typedef Display_t;
// #pragma link C++ typedef Visual_t;
// #pragma link C++ typedef Window_t;
// #pragma link C++ typedef Pixmap_t;
// #pragma link C++ typedef Drawable_t;
// #pragma link C++ typedef Region_t;
// #pragma link C++ typedef Colormap_t;
// #pragma link C++ typedef Cursor_t;
// #pragma link C++ typedef FontH_t;
// #pragma link C++ typedef KeySym_t;
// #pragma link C++ typedef Atom_t;
// #pragma link C++ typedef GContext_t;
// #pragma link C++ typedef FontStruct_t;
// #pragma link C++ typedef Mask_t;
// #pragma link C++ typedef Time_t;

#pragma link C++ struct Event_t;
#pragma link C++ struct SetWindowAttributes_t;
#pragma link C++ struct WindowAttributes_t;
#pragma link C++ struct GCValues_t;
#pragma link C++ struct ColorStruct_t;
#pragma link C++ struct PictureAttributes_t;
#pragma link C++ struct Segment_t;
#pragma link C++ struct Point_t;
#pragma link C++ struct Rectangle_t;
#pragma link C++ struct timespec;

#pragma link C++ function operator<<(std::ostream&, const TTimeStamp&);
#pragma link C++ function operator<<(TBuffer&, const TTimeStamp&);
#pragma link C++ function operator>>(TBuffer&, TTimeStamp&);
#pragma link C++ function operator==(const TTimeStamp&, const TTimeStamp&);
#pragma link C++ function operator!=(const TTimeStamp&, const TTimeStamp&);
#pragma link C++ function operator< (const TTimeStamp&, const TTimeStamp&);
#pragma link C++ function operator<=(const TTimeStamp&, const TTimeStamp&);
#pragma link C++ function operator> (const TTimeStamp&, const TTimeStamp&);
#pragma link C++ function operator>=(const TTimeStamp&, const TTimeStamp&);

#pragma link C++ class TTimeStamp+;
#pragma link C++ class TFileInfo+;
#pragma link C++ class TFileInfoMeta+;
#pragma link C++ class TFileCollection+;
#pragma link C++ class TVirtualAuth;
#pragma link C++ class TVirtualMutex;
#pragma link C++ class TLockGuard;
#pragma link C++ class TRedirectOutputGuard;
#pragma link C++ class TVirtualPerfStats;
#pragma link C++ enum TVirtualPerfStats::EEventType;
#pragma link C++ class TVirtualMonitoringWriter;
#pragma link C++ class TVirtualMonitoringReader;
#pragma link C++ class TObjectSpy;
#pragma link C++ class TObjectRefSpy;
#pragma link C++ class TUri;
#pragma link C++ function operator==(const TUri&, const TUri&);
#pragma link C++ class TUrl;
#pragma link C++ class TInetAddress-;
#pragma link C++ class TVirtualTableInterface+;
#pragma link C++ class TBase64;

// Insure the creation of the TClass object for pairs that might be
// inside the cintdlls.
#pragma extra_include "string";
// insure using namespace std and declaration of std::pair
#pragma extra_include "Rpair.h";
#include <utility>

#pragma link C++ class std::pair<char*,int>+;
#pragma link C++ class std::pair<char*,long>+;
#pragma link C++ class std::pair<char*,float>+;
#pragma link C++ class std::pair<char*,double>+;
#pragma link C++ class std::pair<char*,void*>+;
#pragma link C++ class std::pair<char*,char*>+;
#pragma link C++ class std::pair<std::string,int>+;
#pragma link C++ class std::pair<string,long>+;
#pragma link C++ class std::pair<string,float>+;
#pragma link C++ class std::pair<string,double>+;
#pragma link C++ class std::pair<string,void*>+;
#pragma link C++ class std::pair<int,int>+;
#pragma link C++ class std::pair<int,long>+;
#pragma link C++ class std::pair<int,float>+;
#pragma link C++ class std::pair<int,double>+;
#pragma link C++ class std::pair<int,void*>+;
#pragma link C++ class std::pair<int,char*>+;
#pragma link C++ class std::pair<long,int>+;
#pragma link C++ class std::pair<long,long>+;
#pragma link C++ class std::pair<long,float>+;
#pragma link C++ class std::pair<long,double>+;
#pragma link C++ class std::pair<long,void*>+;
#pragma link C++ class std::pair<long,char*>+;
#pragma link C++ class std::pair<float,int>+;
#pragma link C++ class std::pair<float,long>+;
#pragma link C++ class std::pair<float,float>+;
#pragma link C++ class std::pair<float,double>+;
#pragma link C++ class std::pair<float,void*>+;
#pragma link C++ class std::pair<float,char*>+;
#pragma link C++ class std::pair<double,int>+;
#pragma link C++ class std::pair<double,long>+;
#pragma link C++ class std::pair<double,float>+;
#pragma link C++ class std::pair<double,double>+;
#pragma link C++ class std::pair<double,void*>+;
#pragma link C++ class std::pair<double,char*>+;

#pragma link C++ class std::pair<const char*,int>+;
#pragma link C++ class std::pair<const char*,long>+;
#pragma link C++ class std::pair<const char*,float>+;
#pragma link C++ class std::pair<const char*,double>+;
#pragma link C++ class std::pair<const char*,void*>+;
#pragma link C++ class std::pair<const char*,char*>+;
#pragma link C++ class std::pair<const std::string,int>+;
#pragma link C++ class std::pair<const std::string,long>+;
#pragma link C++ class std::pair<const std::string,float>+;
#pragma link C++ class std::pair<const std::string,double>+;
#pragma link C++ class std::pair<const std::string,void*>+;
#pragma link C++ class std::pair<const int,int>+;
#pragma link C++ class std::pair<const int,long>+;
#pragma link C++ class std::pair<const int,float>+;
#pragma link C++ class std::pair<const int,double>+;
#pragma link C++ class std::pair<const int,void*>+;
#pragma link C++ class std::pair<const int,char*>+;
#pragma link C++ class std::pair<const long,int>+;
#pragma link C++ class std::pair<const long,long>+;
#pragma link C++ class std::pair<const long,float>+;
#pragma link C++ class std::pair<const long,double>+;
#pragma link C++ class std::pair<const long,void*>+;
#pragma link C++ class std::pair<const long,char*>+;
#pragma link C++ class std::pair<const float,int>+;
#pragma link C++ class std::pair<const float,long>+;
#pragma link C++ class std::pair<const float,float>+;
#pragma link C++ class std::pair<const float,double>+;
#pragma link C++ class std::pair<const float,void*>+;
#pragma link C++ class std::pair<const float,char*>+;
#pragma link C++ class std::pair<const double,int>+;
#pragma link C++ class std::pair<const double,long>+;
#pragma link C++ class std::pair<const double,float>+;
#pragma link C++ class std::pair<const double,double>+;
#pragma link C++ class std::pair<const double,void*>+;
#pragma link C++ class std::pair<const double,char*>+;

#pragma extra_include "Rtypes.h";
#pragma link C++ class TParameter<Bool_t>+;
#pragma link C++ class TParameter<Float_t>+;
#pragma link C++ class TParameter<Double_t>+;
#pragma link C++ class TParameter<Int_t>+;
#pragma link C++ class TParameter<Long_t>+;
#pragma link C++ class TParameter<Long64_t>+;

#endif
