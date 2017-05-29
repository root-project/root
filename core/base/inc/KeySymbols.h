/* @(#)root/base:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_KeySymbols
#define ROOT_KeySymbols


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// KeySymbols                                                           //
//                                                                      //
// Key symbols used by the GUI classes.                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "GuiTypes.h"

enum EKeySym {
   kKey_Escape              = 0x1000,          // misc keys
   kKey_Tab                 = 0x1001,
   kKey_Backtab             = 0x1002,
   kKey_Backspace           = 0x1003,
   kKey_Return              = 0x1004,
   kKey_Enter               = 0x1005,
   kKey_Insert              = 0x1006,
   kKey_Delete              = 0x1007,
   kKey_Pause               = 0x1008,
   kKey_Print               = 0x1009,
   kKey_SysReq              = 0x100a,

   kKey_Home                = 0x1010,          // cursor movement
   kKey_End                 = 0x1011,
   kKey_Left                = 0x1012,
   kKey_Up                  = 0x1013,
   kKey_Right               = 0x1014,
   kKey_Down                = 0x1015,
   kKey_Prior               = 0x1016,
   kKey_PageUp              = kKey_Prior,
   kKey_Next                = 0x1017,
   kKey_PageDown            = kKey_Next,

   kKey_Shift               = 0x1020,          // modifiers
   kKey_Control             = 0x1021,
   kKey_Meta                = 0x1022,
   kKey_Alt                 = 0x1023,
   kKey_CapsLock            = 0x1024,
   kKey_NumLock             = 0x1025,
   kKey_ScrollLock          = 0x1026,

   kKey_F1                  = 0x1030,          // function keys
   kKey_F2                  = 0x1031,
   kKey_F3                  = 0x1032,
   kKey_F4                  = 0x1033,
   kKey_F5                  = 0x1034,
   kKey_F6                  = 0x1035,
   kKey_F7                  = 0x1036,
   kKey_F8                  = 0x1037,
   kKey_F9                  = 0x1038,
   kKey_F10                 = 0x1039,
   kKey_F11                 = 0x103a,
   kKey_F12                 = 0x103b,
   kKey_F13                 = 0x103c,
   kKey_F14                 = 0x103d,
   kKey_F15                 = 0x103e,
   kKey_F16                 = 0x103f,
   kKey_F17                 = 0x1040,
   kKey_F18                 = 0x1041,
   kKey_F19                 = 0x1042,
   kKey_F20                 = 0x1043,
   kKey_F21                 = 0x1044,
   kKey_F22                 = 0x1045,
   kKey_F23                 = 0x1046,
   kKey_F24                 = 0x1047,
   kKey_F25                 = 0x1048,          // F25 .. F35 only on X11
   kKey_F26                 = 0x1049,
   kKey_F27                 = 0x104a,
   kKey_F28                 = 0x104b,
   kKey_F29                 = 0x104c,
   kKey_F30                 = 0x104d,
   kKey_F31                 = 0x104e,
   kKey_F32                 = 0x104f,
   kKey_F33                 = 0x1050,
   kKey_F34                 = 0x1051,
   kKey_F35                 = 0x1052,

   kKey_Space               = 0x20,   // 7 bit printable ASCII, for convenience
   kKey_Exclam              = 0x21,   // they map straight to ASCII
   kKey_QuoteDbl            = 0x22,
   kKey_NumberSign          = 0x23,
   kKey_Dollar              = 0x24,
   kKey_Percent             = 0x25,
   kKey_Ampersand           = 0x26,
   kKey_Apostrophe          = 0x27,
   kKey_ParenLeft           = 0x28,
   kKey_ParenRight          = 0x29,
   kKey_Asterisk            = 0x2a,
   kKey_Plus                = 0x2b,
   kKey_Comma               = 0x2c,
   kKey_Minus               = 0x2d,
   kKey_Period              = 0x2e,
   kKey_Slash               = 0x2f,
   kKey_0                   = 0x30,
   kKey_1                   = 0x31,
   kKey_2                   = 0x32,
   kKey_3                   = 0x33,
   kKey_4                   = 0x34,
   kKey_5                   = 0x35,
   kKey_6                   = 0x36,
   kKey_7                   = 0x37,
   kKey_8                   = 0x38,
   kKey_9                   = 0x39,
   kKey_Colon               = 0x3a,
   kKey_Semicolon           = 0x3b,
   kKey_Less                = 0x3c,
   kKey_Equal               = 0x3d,
   kKey_Greater             = 0x3e,
   kKey_Question            = 0x3f,
   kKey_At                  = 0x40,
   kKey_A                   = 0x41,
   kKey_B                   = 0x42,
   kKey_C                   = 0x43,
   kKey_D                   = 0x44,
   kKey_E                   = 0x45,
   kKey_F                   = 0x46,
   kKey_G                   = 0x47,
   kKey_H                   = 0x48,
   kKey_I                   = 0x49,
   kKey_J                   = 0x4a,
   kKey_K                   = 0x4b,
   kKey_L                   = 0x4c,
   kKey_M                   = 0x4d,
   kKey_N                   = 0x4e,
   kKey_O                   = 0x4f,
   kKey_P                   = 0x50,
   kKey_Q                   = 0x51,
   kKey_R                   = 0x52,
   kKey_S                   = 0x53,
   kKey_T                   = 0x54,
   kKey_U                   = 0x55,
   kKey_V                   = 0x56,
   kKey_W                   = 0x57,
   kKey_X                   = 0x58,
   kKey_Y                   = 0x59,
   kKey_Z                   = 0x5a,
   kKey_BracketLeft         = 0x5b,
   kKey_Backslash           = 0x5c,
   kKey_BracketRight        = 0x5d,
   kKey_AsciiCircum         = 0x5e,
   kKey_Underscore          = 0x5f,
   kKey_QuoteLeft           = 0x60,
   kKey_a                   = 0x61,
   kKey_b                   = 0x62,
   kKey_c                   = 0x63,
   kKey_d                   = 0x64,
   kKey_e                   = 0x65,
   kKey_f                   = 0x66,
   kKey_g                   = 0x67,
   kKey_h                   = 0x68,
   kKey_i                   = 0x69,
   kKey_j                   = 0x6a,
   kKey_k                   = 0x6b,
   kKey_l                   = 0x6c,
   kKey_m                   = 0x6d,
   kKey_n                   = 0x6e,
   kKey_o                   = 0x6f,
   kKey_p                   = 0x70,
   kKey_q                   = 0x71,
   kKey_r                   = 0x72,
   kKey_s                   = 0x73,
   kKey_t                   = 0x74,
   kKey_u                   = 0x75,
   kKey_v                   = 0x76,
   kKey_w                   = 0x77,
   kKey_x                   = 0x78,
   kKey_y                   = 0x79,
   kKey_z                   = 0x7a,
   kKey_BraceLeft           = 0x7b,
   kKey_Bar                 = 0x7c,
   kKey_BraceRight          = 0x7d,
   kKey_AsciiTilde          = 0x7e,

   kKey_Unknown             = 0xffff,

   kAnyKey                  = 0           // maps to any key, used by GrabKey()
};

#endif
