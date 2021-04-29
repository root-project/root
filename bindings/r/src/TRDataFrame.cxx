// Author: Omar Zapata  Omar.Zapata@cern.ch   2015

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include<TRDataFrame.h>


using namespace ROOT::R;

//______________________________________________________________________________
TRDataFrame::TRDataFrame(): TObject()
{
   df = Rcpp::DataFrame::create();
}

//______________________________________________________________________________
TRDataFrame::TRDataFrame(const TRDataFrame &_df): TObject(_df)
{
   df = _df.df;
}

//______________________________________________________________________________
TRDataFrame::Binding TRDataFrame::operator[](const TString &name)
{
   return Binding(df, name);
}
