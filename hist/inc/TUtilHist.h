// @(#)root/hist:$Name:  $:$Id: TUtilHist.h,v 1.3 2004/01/25 20:33:32 brun Exp $
// Author: Rene Brun   14/09/2002

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TUtilHist
#define ROOT_TUtilHist


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TUtilHist                                                            //
//                                                                      //
// misc histogram  utilities                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TVirtualUtilHist
#include "TVirtualUtilHist.h"
#endif
#ifndef ROOT_TMatrixFBasefwd
#include "TMatrixFBasefwd.h"
#endif
#ifndef ROOT_TMatrixDBasefwd
#include "TMatrixDBasefwd.h"
#endif
#ifndef ROOT_TVectorFfwd
#include "TVectorFfwd.h"
#endif
#ifndef ROOT_TVectorDfwd
#include "TVectorDfwd.h"
#endif

class TUtilHist : public TVirtualUtilHist {


public:
   TUtilHist();
   virtual     ~TUtilHist();
   virtual void  InitStandardFunctions();
   virtual void  PaintMatrix(const TMatrixFBase &m, Option_t *option);
   virtual void  PaintMatrix(const TMatrixDBase &m, Option_t *option);
   virtual void  PaintVector(const TVectorF     &v, Option_t *option);
   virtual void  PaintVector(const TVectorD     &v, Option_t *option);

   ClassDef(TUtilHist,0)  //misc. histogram  utilities
};

#endif
