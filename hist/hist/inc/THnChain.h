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

   /// Add a new file to this chain.
   ///
   /// \param fileName path of the file to add
   void AddFile(const char* fileName);

   /// Get an axis from the histogram.
   ///
   /// \param i index of the axis to retrieve
   ///
   /// This function requires that a file containing the histogram was
   /// already added with `AddFile`.
   ///
   /// Properties set on the axis returned by `GetAxis` are propagated to all
   /// histograms in the chain, so it can e.g. be used to set ranges for
   /// projections.
   TAxis* GetAxis(Int_t i) const;

   /// See `THnBase::Projection` for the intended behavior.
   TH1* Projection(Int_t xDim, Option_t* option = "") const;

   /// See `THnBase::Projection` for the intended behavior.
   TH2* Projection(Int_t yDim, Int_t xDim, Option_t* option = "") const;

   /// See `THnBase::Projection` for the intended behavior.
   TH3* Projection(Int_t xDim, Int_t yDim, Int_t zDim, Option_t* option = "") const;

   /// See `THnBase::Projection` for the intended behavior.
   THnBase* ProjectionND(Int_t ndim, const Int_t* dim, Option_t* option = "") const;

 private:
   std::string fName; ///< name of the histogram

   std::vector<std::string> fFiles; ///< a list of files to extract the histogram from
   std::vector<TAxis*> fAxes;       ///< the list of histogram axes

   /// Projects all histograms in the chain.
   ///
   /// See `THnBase::Projection` for parameters and their semantics.
   TObject* ProjectionAny(Int_t ndim, const Int_t* dim, Option_t* option = "") const;

   /// Retrieve a histogram from a file.
   ///
   /// \param fileName path of the file to read.
   THnBase* ReadHistogram(const char* fileName) const;

   /// Copy the properties of all axes to a histogram.
   ///
   /// \param hs histogram whose axes should be updated
   void SetupAxes(THnBase& hs) const;

   /// Ensure a histogram has axes similar to the ones we expect.
   ///
   /// \param h histogram to verify
   /// \param axes expected set of axes
   static bool CheckConsistency(const THnBase& h, const std::vector<TAxis*>& axes);

   ClassDef(THnChain, 0);
};

#endif
