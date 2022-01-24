// @(#)root/hist:$Id$
// Author: Benjamin Bannier, August 2016

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "THnChain.h"

#include "TArray.h"
#include "TAxis.h"
#include "TDirectory.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "THnBase.h"
#include "TMath.h"

/// Add a new file to this chain.
///
/// \param fileName path of the file to add
void THnChain::AddFile(const char* fileName)
{
   fFiles.emplace_back(fileName);

   // Initialize axes from first seen instance.
   if (fAxes.empty()) {
      THnBase* hs = ReadHistogram(fileName);

      if (hs) {
         const Int_t naxes = hs->GetNdimensions();
         for (Int_t i = 0; i < naxes; ++i) {
            fAxes.push_back(hs->GetAxis(i));
         }
      } else {
          Warning("AddFile",
                  "Could not find histogram %s in file %s",
                  fName.c_str(),
                  fileName);
      }
   }
}

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
TAxis* THnChain::GetAxis(Int_t i) const
{
   if (i < 0 || i >= static_cast<Int_t>(fAxes.size())) {
      return nullptr;
   }

   return fAxes[i];
}

/// Projects all histograms in the chain.
///
/// See `THnBase::Projection` for parameters and their semantics.
TObject* THnChain::ProjectionAny(Int_t ndim, const Int_t* dim, Option_t* option) const
{
   if (ndim <= 0) {
      return nullptr;
   }

   TObject* h_merged = nullptr;
   for (const auto& file : fFiles) {
      THnBase* hs = ReadHistogram(file.c_str());

      if (!hs) {
         Warning("ProjectionAny",
                 "Could not find histogram %s in file %s",
                 fName.c_str(),
                 file.c_str());

         continue;
      }

      if (!CheckConsistency(*hs, fAxes)) {
         Warning("ProjectionAny",
                 "Histogram %s from file %s is inconsistent with the histogram from file %s",
                 fName.c_str(),
                 file.c_str(),
                 fFiles[0].c_str());

         continue;
      }

      SetupAxes(*hs);

      // Perform projection.
      TObject* h = nullptr;

      if (ndim == 1) {
         h = hs->Projection(dim[0], option);
      } else if (ndim == 2) {
         h = hs->Projection(dim[0], dim[1], option);
      } else if (ndim == 3) {
         h = hs->Projection(dim[0], dim[1], dim[2], option);
      } else {
         h = hs->ProjectionND(ndim, dim, option);
      }

      delete hs;

      // Add this histogram.
      if (h_merged) {
         if (ndim < 3) {
            static_cast<TH1*>(h_merged)->Add(static_cast<TH1*>(h));
         } else {
            static_cast<THnBase*>(h_merged)->Add(static_cast<THnBase*>(h));
         }

         delete h;
      } else {
         h_merged = h;
      }
   }

   return h_merged;
}

/// Retrieve a histogram from a file.
///
/// \param fileName path of the file to read.
THnBase* THnChain::ReadHistogram(const char* fileName) const
{
   TDirectory::TContext ctxt(gDirectory);

   TFile* f = TFile::Open(fileName);

   if (!f) {
      return nullptr;
   }

   THnBase* hs = nullptr;
   f->GetObject(fName.c_str(), hs);
   delete f;

   return hs;
}

/// Copy the properties of all axes to a histogram.
///
/// \param hs histogram whose axes should be updated
void THnChain::SetupAxes(THnBase& hs) const
{
   const Int_t naxes = fAxes.size();
   for (Int_t i = 0; i < naxes; ++i) {
      const TAxis* ax_ref = fAxes[i];
      TAxis* ax = hs.GetAxis(i);
      ax_ref->Copy(*ax);
   }
}

/// Ensure a histogram has axes similar to the ones we expect.
///
/// \param h histogram to verify
/// \param axes expected set of axes
bool THnChain::CheckConsistency(const THnBase& h, const std::vector<TAxis*>& axes)
{
   // We would prefer to directly use `TH1::CheckEqualAxes` here;
   // however it is protected so we inherit the parts we care about.
   // FIXME(bbannier): It appears that functionality like `TH1::CheckEqualAxes` could
   // just as well live in `TAxis` so that anyone using axes could make use of it.
   const Int_t naxes = h.GetNdimensions();
   const Int_t naxes2 = axes.size();

   if (naxes != naxes2) {
      return false;
   }

   for (Int_t i = 0; i < naxes; ++i) {
      const TAxis* ax1 = h.GetAxis(i);
      const TAxis* ax2 = axes[i];

      if (ax1->GetNbins() != ax2->GetNbins()) {
         return false;
      }

      // Copied from `TH1::CheckAxisLimits.`
      if (!TMath::AreEqualRel(ax1->GetXmin(), ax2->GetXmin(), 1.E-12) ||
          !TMath::AreEqualRel(ax1->GetXmax(), ax2->GetXmax(), 1.E-12)) {
         return false;
      }

      // Copied from `TH1::CheckBinLimits`.
      const TArrayD* h1Array = ax1->GetXbins();
      const TArrayD* h2Array = ax2->GetXbins();
      Int_t fN = h1Array->fN;
      if (fN != 0) {
         if (h2Array->fN != fN) {
            return false;
         } else {
            for (int ibin = 0; ibin < fN; ++ibin) {
               if (!TMath::AreEqualRel(h1Array->GetAt(ibin), h2Array->GetAt(ibin), 1E-10)) {
                  return false;
               }
            }
         }
      }

      // We ignore checking for equal bin labels here. A check
      // for that is implemented in `TH1::CheckBinLabels`.
   }

   return true;
}

/// See `THnBase::Projection` for the intended behavior.
TH1* THnChain::Projection(Int_t xDim, Option_t* option) const
{
   // Forwards to `THnBase::Projection()`.
   Int_t dim[1] = {xDim};
   return static_cast<TH1*>(ProjectionAny(1, dim, option));
}

/// See `THnBase::Projection` for the intended behavior.
TH2* THnChain::Projection(Int_t yDim, Int_t xDim, Option_t* option) const
{
   // Forwards to `THnBase::Projection()`.
   Int_t dim[2] = {xDim, yDim};
   return static_cast<TH2*>(ProjectionAny(2, dim, option));
}

/// See `THnBase::Projection` for the intended behavior.
TH3* THnChain::Projection(Int_t xDim, Int_t yDim, Int_t zDim, Option_t* option) const
{
   // Forwards to `THnBase::Projection()`.
   Int_t dim[3] = {xDim, yDim, zDim};
   return static_cast<TH3*>(ProjectionAny(3, dim, option));
}

/// See `THnBase::Projection` for the intended behavior.
THnBase* THnChain::ProjectionND(Int_t ndim, const Int_t* dim, Option_t* option) const
{
   // Forwards to `THnBase::ProjectionND()`.
   return static_cast<THnBase*>(ProjectionAny(ndim, dim, option));
}
