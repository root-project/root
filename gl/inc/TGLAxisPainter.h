// @(#)root/gl:$Name:  $:$Id: TGLAxisPainter.h,v 1.1 2006/06/14 10:00:00 couet Exp $
// Author:  Timur Pocheptsov  14/06/2006
                                                                                
/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLAxisPainter
#define ROOT_TGLAxisPainter

#include <utility>
#include <vector>

#include "Rtypes.h"

/*
   TGLAxisPainter defines interface for axis painters.
   Now, we have only one concrete axis painter, which
   uses TGAxis class and TVirtualX to do all work.
   In future, TGAxis must be replaced by real
   3d axis painter.
*/

class TGLPlotFrame;

class TGLAxisPainter {
public:
   typedef std::pair<Double_t, Double_t> Range_t;

   virtual ~TGLAxisPainter();

   virtual void SetRanges(const Range_t &xRange, const Range_t &yRange, const Range_t &zRange) = 0;
   virtual void SetZLevels(std::vector<Double_t> &zLevels) = 0;

   virtual void Paint(Int_t context) = 0;

   ClassDef(TGLAxisPainter, 0) //Base for axis painters
};

class TAxis;
class TH1;

/*
   This painter obtains 3d coordinates converted into 2d coordinates in a window system,
   draws them via TGAxis.
*/

class TGL2DAxisPainter : public TGLAxisPainter {
private:
   Range_t         fRangeX;
   Range_t         fRangeY;
   Range_t         fRangeZ;
   TGLPlotFrame   *fPlotFrame;
   TAxis          *fAxisX;
   TAxis          *fAxisY;
   TAxis          *fAxisZ;

public:
   TGL2DAxisPainter(TH1 *hist);

   void SetPlotFrame(TGLPlotFrame *frame);
   void SetRanges(const Range_t &xRange, const Range_t &yRange, const Range_t &zRange);
   void SetZLevels(std::vector<Double_t> &zLevels);
   void Paint(Int_t context);

   ClassDef(TGL2DAxisPainter, 0) //Default painter, uses TGAxis to make its work
};

#endif
