 ////////////////////////////////////////////////////////////////////////////
// $Id$
//
// PlexCalib
//
// PlexCalib is an *interface* 
//   Use this class as a common mix-in that any "Calibrator" can
//   derive from.  This allows Plexus/PlexHandle to generate a 
//   fully filled PlexSEIdAltL knowning *only* about this interface.
//
// Author:  R. Hatcher 2001.11.15
//
////////////////////////////////////////////////////////////////////////////

#ifndef PLEXCALIB_H
#define PLEXCALIB_H

// not inheriting from TObject so we need an explicit Rtypes
#include "Rtypes.h"
#include "SEIdAltLItem.h"

class PlexCalib {

public:

   virtual SEIdAltLItem CalibStripEnd(const Int_t& seid,
                                          Int_t adc, Double_t time) const = 0;

   virtual ~PlexCalib() {;}

private:
   
   ClassDef(PlexCalib,1)
};

#endif  // PLEXCALIB_H
