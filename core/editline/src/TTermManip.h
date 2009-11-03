// @(#)root/editline:$Id$
// Author: Axel Naumann, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef INCLUDE_TTERMMANIP_H
#define INCLUDE_TTERMMANIP_H

#include <map>
#include <cstring>
#include <stdio.h>

extern "C" typedef int (*PutcFunc_t)(int);

// setupterm must be called before TTermManip can be created!
class TTermManip {
public:
   TTermManip();
   ~TTermManip() { ResetTerm(); }

   bool SetColor(unsigned char r, unsigned char g, unsigned char b);
   bool SetColor(int idx);

   int GetColorIndex(unsigned char r, unsigned char g, unsigned char b);

   void
   StartUnderline() {
      if (!fCurrentlyUnderlined) {
         WriteTerm(fStartUnderline);
         fCurrentlyUnderlined = true;
      }
   }


   void
   StopUnderline() {
      if (fCurrentlyUnderlined) {
         WriteTerm(fStopUnderline);
         fCurrentlyUnderlined = false;
      }
   }

   void StartBold();
   void StopBold();


   bool ResetTerm();
   void SetDefaultColor();

private:
   class Color {
   public:
      Color(unsigned char r = 0, unsigned char g = 0, unsigned char b = 0):
         fR((r* 1001) / 256),
         fG((g* 1001) / 256),
         fB((b* 1001) / 256) {
         // Re-normalize RGB components from 0 to 255 to 0 to 1000
      }


      bool
      operator <(const Color& c) const {
         return fR < c.fR
                || (fR == c.fR && (fG < c.fG
                                   || (fG == c.fG && fB < c.fB)));
      }


      int fR, fG, fB;
   };

   char* GetTermStr(const char* cap);
   int GetTermNum(const char* cap);

   bool WriteTerm(char* termstr);

   bool WriteTerm(char* termstr, int i);

   static int
   DefaultPutchar(int c) {
      // tputs takes int(*)(char) on solaris, so wrap putchar
      return putchar(c);
   }


   int fNumColors; // number of available colors
   bool fAnsiColors; // whether fSetFg, Bg use ANSI
   char* fSetFg; // set foreground color
   char* fSetBold; // set bold color
   char* fSetDefault; // set normal color
   char* fStartUnderline; // start underline;
   char* fStopUnderline; // stop underline;
   PutcFunc_t fPutc;
   int fCurrentColorIdx;   // index if the currently active color
   bool fCurrentlyBold;  // whether bold is active
   bool fCurrentlyUnderlined;  // whether underlining is active
};

#endif // INCLUDE_TTERMMANIP_H
