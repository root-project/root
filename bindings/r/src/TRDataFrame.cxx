/*************************************************************************
 * Copyright (C) 2013-2015, Omar Andres Zapata Mesa                      *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include<TRDataFrame.h>
//______________________________________________________________________________
/* Begin_Html
<center><h2>TRDataFrame class</h2></center>

<p>
The TRDataFrame class lets you pass DataFrames from/to  R's environment<br>
</p>
End_Html
*/


using namespace ROOT::R;
ClassImp(TRDataFrame)


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
    return Binding(df,name);
}
