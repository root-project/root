// @(#)root/hist:$Name:  $:$Id: TUtilHist.h,v 1.1 2002/09/14 16:19:13 brun Exp $
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


#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TMatrix;
class TMatrixD;
class TVector;
class TVectorD;

class TUtilHist : public TNamed {


public:
   TUtilHist();
   virtual     ~TUtilHist();
   virtual void  InitStandardFunctions();
   virtual void  PaintMatrix(TMatrix &m, Option_t *option);
   virtual void  PaintMatrix(TMatrixD &m, Option_t *option);
   virtual void  PaintVector(TVector &v, Option_t *option);
   virtual void  PaintVector(TVectorD &v, Option_t *option);

   ClassDef(TUtilHist,0)  //misc histogram  utilities
};

#endif
