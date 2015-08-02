/*************************************************************************
 * Copyright (C) 2013-2015, Omar Andres Zapata Mesa                      *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include<TRDataFrame.h>
////////////////////////////////////////////////////////////////////////////////


using namespace ROOT::R;
ClassImp(TRDataFrame)


////////////////////////////////////////////////////////////////////////////////

TRDataFrame::TRDataFrame(): TObject()
{
    df = Rcpp::DataFrame::create();
}

////////////////////////////////////////////////////////////////////////////////

TRDataFrame::TRDataFrame(const TRDataFrame &_df): TObject(_df)
{
    df = _df.df;
}

////////////////////////////////////////////////////////////////////////////////

TRDataFrame::Binding TRDataFrame::operator[](const TString &name)
{
    return Binding(df,name);
}
