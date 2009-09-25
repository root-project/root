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
   fColorCapable(false),
   fUsePairs(false),
   fAnsiColors(true),
   fCanChangeColors(false),
   fOrigColors(0),
   fInitColor(0),
   fInitPair(0),
   fSetPair(0),
   fSetFg(0),
   fSetBold(0),
   fSetDefault(0),
   fStartUnderline(0),
   fStopUnderline(0),
   fPutc((PutcFunc_t)DefaultPutchar),
   fCurrentColorIdx(-1),
   fCurrentlyBold(false),
   fCurrentlyUnderlined(false) {
   // Initialize color management
   fOrigColors = GetTermStr("oc");
   ResetTerm();
   // Use pairs where possible
   fInitPair = GetTermStr("initp");
   fUsePairs = (fInitPair != 0);

   if (fUsePairs) {
      fSetPair = GetTermStr("scp");
      fCanChangeColors = true;
   } else {
      fInitColor = GetTermStr("initc");
      fCanChangeColors = (fInitColor != 0);
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

   fColorCapable = fUsePairs || fSetFg;
}


int
TTermManip::AllocColor(const Color& col) {
   ColorMap_t::iterator iCol2 = fColors.find(col);
   int idx = -1;

   if (iCol2 != fColors.end()) {
      idx = iCol2->second;
   } else {
      // inserted; set pair idx starting at fgStartColIdx
      idx = fColors.size() - 1 + fgStartColIdx;
      fColors[col] = idx;
      char* initcolcmd = 0;

      if (fUsePairs) {
         initcolcmd = tparm(fInitPair, idx, 0, 0, 0, col.fR, col.fG, col.fB, 0, 0);
      } else if (fInitColor) {
         initcolcmd = tparm(fInitColor, idx, col.fR, col.fG, col.fB, 0, 0, 0, 0, 0);
      }

      if (initcolcmd) {
         tputs(initcolcmd, 1, fPutc);
      }
   }
   return idx;
} // AllocColor


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
   // RGB colors range from 0 to 255
   if (fCanChangeColors) {
      return AllocColor(Color(r, g, b));
   } else {
      int sum = r + g + b;
      r = r > sum / 4;
      g = g > sum / 4;
      b = b > sum / 4;
      int idx = 0;

      if (fAnsiColors) {
         idx = r + (g * 2) + (b * 4);
      } else {
         idx = (r * 4) + (g * 2) + b;
      }
      return idx;
   }
   return -1;
}

bool
TTermManip::SetColor(unsigned char r, unsigned char g, unsigned char b) {
   // RGB colors range from 0 to 255
   return SetColor(GetColorIndex(r, g, b));
}

bool
TTermManip::SetColor(int idx) {
   // Set color to a certain index as returned by GetColorIdx.
   if (fCanChangeColors) {
      if (idx != fCurrentColorIdx) {
         if (fSetPair) {
            WriteTerm(fSetPair, idx);
         } else if (fSetFg) {
            WriteTerm(fSetFg, idx);
         }
         fCurrentColorIdx = idx;
      }
   } else {
      if (fSetFg && idx != fCurrentColorIdx) {
         WriteTerm(fSetFg, idx);
         fCurrentColorIdx = idx;
      }
   }
   return true;
} // SetColor


char*
TTermManip::GetTermStr(const char* cap) {
   char capid[8];
   strcpy(capid, cap);
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
   char capid[8];
   strcpy(capid, cap);
   return tigetnum(capid);
}


bool
TTermManip::ResetTerm() {
   WriteTerm(fSetDefault);
   WriteTerm(fStopUnderline);

   if (fOrigColors) {
      WriteTerm(fOrigColors);
   }
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
