/* @(#)root/gui:$Name:  $:$Id: WidgetMessageTypes.h,v 1.1.1.1 2000/05/16 17:00:42 rdm Exp $ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_WidgetMessageTypes
#define ROOT_WidgetMessageTypes


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// WidgetMessageTypes                                                   //
//                                                                      //
// System predefined widget message types. Message types are constants  //
// that indicate which widget sent the message and by which widget      //
// function (sub-message). Make sure your own message types don't clash //
// whith the ones defined in this file. ROOT reserves all message ids   //
// between 0 - 1000. User defined messages should be in the range       //
// 1001 - 10000. Sub-messages must always be in the range 1-255.        //
// To use MK_MSG() just cast your message id's to an EWidgetMessageType.//
//                                                                      //
//////////////////////////////////////////////////////////////////////////

enum EWidgetMessageTypes {
   kC_COMMAND          = 1,
      kCM_MENU            = 1,
      kCM_MENUSELECT      = 2,
      kCM_BUTTON          = 3,
      kCM_CHECKBUTTON     = 4,
      kCM_RADIOBUTTON     = 5,
      kCM_LISTBOX         = 6,
      kCM_COMBOBOX        = 7,
      kCM_TAB             = 8,
   kC_HSCROLL          = 2,
   kC_VSCROLL          = 3,
      kSB_LINEUP          = 1,
      kSB_LINEDOWN        = 2,
      kSB_PAGEUP          = 3,
      kSB_PAGEDOWN        = 4,
      kSB_SLIDERTRACK     = 5,
      kSB_SLIDERPOS       = 6,
   kC_TEXTENTRY        = 4,
      kTE_TEXTCHANGED     = 1,
      kTE_ENTER           = 2,
   kC_CONTAINER        = 5,
      kCT_ITEMCLICK       = 1,
      kCT_ITEMDBLCLICK    = 2,
      kCT_SELCHANGED      = 3,
   kC_HSLIDER          = 6,
   kC_VSLIDER          = 7,
      kSL_POS             = 1,
      kSL_TRACK           = 2,
   kC_LISTTREE         = 8,
   kC_TEXTVIEW         = 9,
      kTXT_ISMARKED       = 1,
      kTXT_DATACHANGE     = 2,
      kTXT_CLICK2         = 3,
      kTXT_CLICK3         = 4,
      kTXT_F3             = 5,
   kC_MSGMAX           = 10000
};


// Message cracking routines
inline Int_t MK_MSG(EWidgetMessageTypes msg, EWidgetMessageTypes submsg)
                                   { return (msg << 8) + submsg; }
inline Int_t GET_MSG(Long_t val)    { return Int_t((val >> 8) & 255); }
inline Int_t GET_SUBMSG(Long_t val) { return Int_t(val & 255); }

#endif
