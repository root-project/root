// @(#)root/gl:$Name:  $:$Id: TGLStopwatch.h,v 1.1.1.1 2007/04/04 16:01:43 mtadel Exp $
// Author:  Richard Maunder  25/05/2005

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLStopwatch
#define ROOT_TGLStopwatch

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLStopwatch                                                         //
//                                                                      //
// Stopwatch object for timing GL work. We do not use the TStopwatch as //
// we need to perform GL flushing to get accurate times + we record     //
// timing overheads here.                                               //
//
// MT: Bypassed all of the overhead stuff. It does not seem reasonable
// anyway. Besides it was being initialized outside of a valid GL
// context and coused random crashes (especially on 64-bit machines with
// nvidia cards).
//
//////////////////////////////////////////////////////////////////////////

class TGLStopwatch
{
private:
   // Fields
   Double_t        fStart;           //! start time (millisec)

   static Bool_t   fgInitOverhead;   //! overhead calculated?
   static Double_t fgOverhead;       //! timing overhead (millisecs)

   // Methods
   void     InitOverhead(void)  const;
   Double_t GetClock(void)      const;
   Double_t WaitForTick(void)   const;
   void     FinishDrawing(void) const;


public:
   TGLStopwatch();
   virtual ~TGLStopwatch(); // ClassDef introduces virtual fns

   void     Start();
   Double_t Lap() const;
   Double_t End();

   ClassDef(TGLStopwatch,0) // a GL stopwatch utility class
};

#endif // ROOT_TGLStopwatch
