// @(#)root/editline:$Id$
// Author: Axel Naumann, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TTermManip.h"
#include <strings.h>

#ifndef _MSC_VER
# include "rlcurses.h"
#else
# include "win32vt100.h"
#endif

TTermManip::TTermManip():
   fNumColors(-1),
   fAnsiColors(true),
   fSetFg(0),
   fSetBold(0),
   fSetDefault(0),
   fStartUnderline(0),
   fStopUnderline(0),
   fPutc((PutcFunc_t)DefaultPutchar),
   fCurrentColorIdx(-1),
   fCurrentlyBold(false),
   fCurrentlyUnderlined(false)
{
   // Initialize color management
   ResetTerm();
   // Use colors where possible
   fNumColors = GetTermNum("colors");
   if (fNumColors > 1) {
      fSetFg = GetTermStr("setaf");
      fAnsiColors = true;

      if (!fSetFg) {
         fSetFg = GetTermStr("setf");
         fAnsiColors = false;
      }
   }

   fSetBold = GetTermStr("bold");
   // "sgr0" doesn't reset window size
   fSetDefault = GetTermStr("sgr0");

   if (!fSetDefault) {
      fSetDefault = GetTermStr("rs2");
   }
   fStartUnderline = GetTermStr("smul");
   fStopUnderline = GetTermStr("rmul");
}


void
TTermManip::SetDefaultColor() {
   // Set terminal to the default color.
   if (fCurrentlyBold || fCurrentColorIdx != -1) {
      WriteTerm(fSetDefault);
      fCurrentlyBold = false;
      fCurrentColorIdx = -1;
   }
   if (fCurrentlyUnderlined) {
      WriteTerm(fStopUnderline);
      fCurrentlyUnderlined = false;      
   }
}

void
TTermManip::StartBold() {
   // want bold
   if (!fCurrentlyBold) {
      if (fSetBold) {
         WriteTerm(fSetBold);
      }
      fCurrentlyBold = true;
   }
}


void
TTermManip::StopBold() {
   // turn bold off
   if (fCurrentlyBold) {
      if (fSetDefault && fCurrentlyBold) {
         WriteTerm(fSetDefault);
      }
      fCurrentlyBold = false;
      if (fCurrentColorIdx != -1) {
         int ci = fCurrentColorIdx;
         fCurrentColorIdx = -1;
         SetColor(ci);
      }
   }
}


int
TTermManip::GetColorIndex(unsigned char r, unsigned char g, unsigned char b) {
   // Determine the color index givenan RGB triplet, each within [0..255].
   int idx = -1;

   if (fNumColors > 255) {
      static unsigned char rgb256[256][3] = {{0}};
      if (rgb256[0][0] == 0) {
         // initialize the array with the expected standard colors:
         // (from http://frexx.de/xterm-256-notes)
         unsigned char rgbidx = 0;
         // this is not what I see, though it's supposedly the default:
         //   rgb[0][0] =   0; rgb[0][1] =   0; rgb[0][1] =   0;
         // use this instead, just to be on the safe side:
         rgb256[0][0] =  46; rgb256[0][1] =  52; rgb256[0][1] =  64;
         rgb256[1][0] = 205; rgb256[1][1] =   0; rgb256[1][1] =   0;
         rgb256[2][0] =   0; rgb256[2][1] = 205; rgb256[2][1] =   0;
         rgb256[3][0] = 205; rgb256[3][1] = 205; rgb256[3][1] =   0;
         rgb256[4][0] =   0; rgb256[4][1] =   0; rgb256[4][1] = 238;
         rgb256[5][0] = 205; rgb256[5][1] =   0; rgb256[5][1] = 205;
         rgb256[6][0] =   0; rgb256[6][1] = 205; rgb256[6][1] = 205;
         rgb256[7][0] = 229; rgb256[7][1] = 229; rgb256[7][1] = 229;

         // this is not what I see, though it's supposedly the default:
         //   rgb256[ 8][0] = 127; rgb256[ 8][1] = 127; rgb256[ 8][1] = 127;
         // use this instead, just to be on the safe side:
         rgb256[ 8][0] =   0; rgb256[ 8][1] =   0; rgb256[ 8][1] =   0;
         rgb256[ 9][0] = 255; rgb256[ 9][1] =   0; rgb256[ 9][1] =   0;
         rgb256[10][0] =   0; rgb256[10][1] = 255; rgb256[10][1] =   0;
         rgb256[11][0] = 255; rgb256[11][1] = 255; rgb256[11][1] =   0;
         rgb256[12][0] =  92; rgb256[12][1] =  92; rgb256[12][1] = 255;
         rgb256[13][0] = 255; rgb256[13][1] =   0; rgb256[13][1] = 255;
         rgb256[14][0] =   0; rgb256[14][1] = 255; rgb256[14][1] = 255;
         rgb256[15][0] = 255; rgb256[15][1] = 255; rgb256[15][1] = 255;

         for (unsigned char red = 0; red < 6; ++red) {
            for (unsigned char green = 0; green < 6; ++green) {
               for (unsigned char blue = 0; blue < 6; ++blue) {
                  rgbidx = 16 + (red * 36) + (green * 6) + blue;
                  rgb256[rgbidx][0] = red ? (red * 40 + 55) : 0;
                  rgb256[rgbidx][1] = green ? (green * 40 + 55) : 0;
		  rgb256[rgbidx][2] = blue ? (blue * 40 + 55) : 0;
               }
            }
         }
         // colors 232-255 are a grayscale ramp, intentionally leaving out
         // black and white
         for (unsigned char gray = 0; gray < 24; ++gray) {
            unsigned char level = (gray * 10) + 8;
            rgb256[232 + gray][0] = level;
            rgb256[232 + gray][1] = level;
            rgb256[232 + gray][2] = level;
         }
      }

      // Find the closest index.
      // A: the closest color match (square of geometric distance in RGB)
      // B: the closest brightness match
      // Treat them equally, which suppresses differences
      // in color due to squared distance.

      // start with black:
      idx = 0;
      int graylvl = (r + g + b)/3;
      long mindelta = (r*r + g*g + b*b) + graylvl;
      if (mindelta) {
         for (unsigned int i = 1; i < 256; ++i) {
            long delta = (rgb256[i][0] + rgb256[i][1] + rgb256[i][2])/3 - graylvl;
            if (delta < 0) delta = -delta;
            delta += (r-rgb256[i][0])*(r-rgb256[i][0]) +
                     (g-rgb256[i][1])*(g-rgb256[i][1]) +
                     (b-rgb256[i][2])*(b-rgb256[i][2]);
            
            if (delta < mindelta) {
               mindelta = delta;
               idx = i;
               if (mindelta == 0) break;
            }
         }
      }
   } else if (fNumColors > 1) {
      int sum = r + g + b;
      r = r > sum / 4;
      g = g > sum / 4;
      b = b > sum / 4;

      if (fAnsiColors) {
         idx = r + (g * 2) + (b * 4);
      } else {
         idx = (r * 4) + (g * 2) + b;
      }
   }
   return idx;
}

bool
TTermManip::SetColor(unsigned char r, unsigned char g, unsigned char b) {
   // RGB colors range from 0 to 255
   return SetColor(GetColorIndex(r, g, b));
}

bool
TTermManip::SetColor(int idx) {
   // Set color to a certain index as returned by GetColorIdx.
   if (fSetFg && idx != fCurrentColorIdx) {
      WriteTerm(fSetFg, idx);
      fCurrentColorIdx = idx;
   }
   return true;
} // SetColor


char*
TTermManip::GetTermStr(const char* cap) {
   char capid[10];
   strncpy(capid, cap, sizeof(capid) - 1);
   capid[sizeof(capid) - 1] = 0; // force 0 termination
   char* termstr = tigetstr(capid);

   if (termstr == (char*) -1) {
      //printf("ERROR unknown capability %s\n", cap);
      return NULL;
   } else if (termstr == 0) {
      // printf("ERROR capability %s not supported\n", cap);
      return NULL;
   }
   return termstr;
}

int
TTermManip::GetTermNum(const char* cap) {
   char capid[10];
   strncpy(capid, cap, sizeof(capid) - 1);
   capid[sizeof(capid) - 1] = 0; // force 0 termination
   return tigetnum(capid);
}


bool
TTermManip::ResetTerm() {
   WriteTerm(fSetDefault);
   WriteTerm(fStopUnderline);

   fCurrentColorIdx = -1;
   fCurrentlyBold = false;
   fCurrentlyUnderlined = false;
   return true;
} // ResetTerm


bool
TTermManip::WriteTerm(char* termstr) {
   if (!termstr) {
      return false;
   }
   tputs(tparm(termstr, 0, 0, 0, 0, 0, 0, 0, 0, 0), 1, fPutc);

   /*if (tputs(tparm(termstr, 0, 0, 0, 0, 0, 0, 0, 0, 0), 1, fPutc) == ERR) {
      printf("ERROR writing %s\n", termstr);
      return false;
      }*/
   fflush(stdout);
   return true;
}


bool
TTermManip::WriteTerm(char* termstr, int i) {
   if (!termstr) {
      return false;
   }
   tputs(tparm(termstr, i, 0, 0, 0, 0, 0, 0, 0, 0), 1, fPutc);

   /*if (tputs(tparm(termstr, i, 0, 0, 0, 0, 0, 0, 0, 0), 1, fPutc) == ERR) {
      printf("ERROR writing %s %d\n", termstr, i);
      return false;
      }*/
   fflush(stdout);
   return true;
}


#ifdef TEST_TERMMANIP
void
testcolor(TTermManip& tm, int r, int g, int b) {
   tm.SetColor(r, g, b);

   if (r % 2) {
      tm.StartUnderline();
   }
   printf("HELLO %d %d %d\n", r, g, b);

   if (r % 2) {
      tm.StopUnderline();
   }
}


void
testall(TTermManip& tm, int h) {
   testcolor(tm, h, 0, 0);
   testcolor(tm, 0, h, 0);
   testcolor(tm, 0, 0, h);
   testcolor(tm, h, h, 0);
   testcolor(tm, h, 0, h);
   testcolor(tm, 0, h, h);
   testcolor(tm, h, h, h);
}


int
main(int, char*[]) {
   int errcode;

   if (ERR == setupterm(0, 1, &errcode)) {
      printf("ERROR in setupterm: %d\n", errcode);
      return 1;
   }
   TTermManip tm;
   testall(tm, 31);
   testall(tm, 127);
   testall(tm, 128);
   testall(tm, 255);

   testcolor(tm, 0, 0, 0);

   return 0;
} // main


#endif
