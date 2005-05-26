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

/*************************************************************************
 * TGLStopwatch - TODO
 *
 *
 *
 *************************************************************************/
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
