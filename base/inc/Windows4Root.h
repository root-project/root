/* @(#)root/base:$Name$:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_Windows4Root
#define ROOT_Windows4Root


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// This include file is necessary to solve a problem with the original  //
//  Windows.h file from Microsoft                                       //
// The native Windows.h redefines the following names:                  //
//     RemoveDirectory                                                  //
//     GetClassName                                                     //
//     GetTextAlign                                                     //
//     GetTextColor                                                     //
//                                                                      //
//   This include file references the original Windows.h file           //
//   and undefined these symbols in exit.                               //
//////////////////////////////////////////////////////////////////////////

#ifndef __CINT__
#include <windows.h>



#undef OpenSemaphore

#undef RemoveDirectory
#undef GetClassName
#undef GetTextAlign
#undef GetTextColor

#undef SetTextAlign
#undef SetTextColor
#undef UpdateWindow
#undef SetClipRegion


    #undef      ClearWindow
    #undef      ClosePixmap
    #undef      CloseWindow
    #undef      CopyPixmap
    #undef      DrawBox
    #undef      DrawCellArray
    #undef      DrawFillArea
    #undef      DrawLine
    #undef      DrawPolyLine
    #undef      DrawPolyMarker
    #undef      DrawText
    #undef      GetCharacterUp

    #undef      GetDoubleBuffer
    #undef      GetPixel
    #undef      GetPlanes
    #undef      GetRGB
    #undef      GetTextExtent
    #undef      InitWindow
    #undef      MoveWindow
    #undef      OpenPixmap
    #undef      PutByte
    #undef      QueryPointer
    #undef      RescaleWindow
    #undef      ResizePixmap
    #undef      ResizeWindow
    #undef      SelectWindow
    #undef      SetCharacterUp
    #undef      SetClipOFF
    #undef      SetClipRegion
    #undef      SetCursor
    #undef      SetDrawMode
    #undef      SetFillColor
    #undef      SetFillStyle
    #undef      SetLineColor
    #undef      SetLineType
    #undef      SetLineStyle
    #undef      SetLineWidth
    #undef      SetMarkerColor
    #undef      SetMarkerSize
    #undef      SetMarkerStyle
    #undef      SetRGB
    #undef      SetTextAlign
    #undef      SetTextColor
    #undef      SetTextFont
    #undef      SetTextFont
    #undef      SetTextSize
    #undef      UpdateWindow
    #undef      Warp
    #undef      WritePixmap

    #undef      CreateWindow

#   ifndef ROOT_TGWin32Object
#      undef GetObject
#      undef GetClassInfo
#   endif

#else
   typedef HANDLE void *
#endif


#endif
