// @(#)root/hist:$Id$
// Author: Benjamin Bannier, August 2016

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_THnChain
#define ROOT_THnChain

#include <TObject.h>

#include <string>
#include <vector>

class TAxis;
class TH1;
class TH2;
class TH3;
class THnBase;

/** \class THnChain
A class to chain together multiple histograms.

This class allows to chain together any `THnBase`-derived (`THn` or `THnSparse`)
histograms from multiple files. Operations on the axes and projections are
supported. The intent is to allow convenient merging merging of projections
of high-dimensional histograms.

\code{.cpp}
// `file1.root` and `file2.root` contain a `THnSparse` named `hsparse`.
THnChain hs("hsparse");
hs.AddFile("file1.root");
hs.AddFile("file2.root");

// Project out axis 0, integrate over other axes.
TH1* h0 = hs.Projection(0);

// Project out axis 0, integrate over other axes in their active ranges.
hs.GetAxis(1)->SetRangeUser(0, 0.1); // select a subrange
TH1* h0 = hs.Projection(0);
\endcode
*/

class THnChain : public TObject
{
 public:
   /// Default constructor.
   ///
   /// \param name name of the histogram to work on
   explicit THnChain(const char* name) : fName(name) {}

   void AddFile(const char* fileName);

   TAxis* GetAxis(Int_t i) const;

   TH1* Projection(Int_t xDim, Option_t* option = "") const;

   TH2* Projection(Int_t yDim, Int_t xDim, Option_t* option = "") const;

   TH3* Projection(Int_t xDim, Int_t yDim, Int_t zDim, Option_t* option = "") const;

   THnBase* ProjectionND(Int_t ndim, const Int_t* dim, Option_t* option = "") const;

 private:
   std::string fName;               ///< name of the histogram

   std::vector<std::string> fFiles; ///< a list of files to extract the histogram from
   std::vector<TAxis*> fAxes;       ///< the list of histogram axes

   TObject* ProjectionAny(Int_t ndim, const Int_t* dim, Option_t* option = "") const;

   THnBase* ReadHistogram(const char* fileName) const;

   void SetupAxes(THnBase& hs) const;

   static bool CheckConsistency(const THnBase& h, const std::vector<TAxis*>& axes);

   ClassDefOverride(THnChain, 0);
};

#endif
