// @(#)root/minuit2:$Name:  $:$Id: DavidonErrorUpdator.h,v 1.3.6.3 2005/11/29 11:08:34 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_DavidonErrorUpdator
#define ROOT_Minuit2_DavidonErrorUpdator

#include "Minuit2/MinimumErrorUpdator.h"

namespace ROOT {

   namespace Minuit2 {


class DavidonErrorUpdator : public MinimumErrorUpdator {

public:

  DavidonErrorUpdator() {}
  
  virtual ~DavidonErrorUpdator() {}

  virtual MinimumError Update(const MinimumState&, const MinimumParameters&,
			      const FunctionGradient&) const;

private:

};

  }  // namespace Minuit2

}  // namespace ROOT

#endif  // ROOT_Minuit2_DavidonErrorUpdator
