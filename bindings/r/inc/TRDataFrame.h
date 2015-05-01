// @(#)root/r:$Id$
// Author: Omar Zapata   30/05/2015


/*************************************************************************
 * Copyright (C) 2013-2015, Omar Andres Zapata Mesa                      *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_R_TRDataFrame
#define ROOT_R_TRDataFrame

#ifndef ROOT_R_RExports
#include<RExports.h>
#endif

//________________________________________________________________________________________________________
/**
   This is a base class to create DataFrames from ROOT to R


   @ingroup R
*/

namespace ROOT {
   namespace R {

     static Rcpp::internal::NamedPlaceHolder Label;
     
      class TRDataFrame: public TObject {
         friend class TRInterface;
         friend SEXP Rcpp::wrap<TRDataFrame>(const TRDataFrame &f);
      protected:
         Rcpp::DataFrame df;
      public:
         TRDataFrame();
         TRDataFrame(const TRDataFrame &_df);
         #include <TRDataFrame__ctors.h>
        

         ClassDef(TRDataFrame, 0) //
      };
   }
}



#endif
