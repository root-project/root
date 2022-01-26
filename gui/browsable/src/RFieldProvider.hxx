/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_Browsable_RFieldProvider
#define ROOT_Browsable_RFieldProvider

#include "TH1.h"
#include "TMath.h"
#include <map>
#include <memory>
#include <string>
#include <utility>

#include <ROOT/Browsable/RProvider.hxx>

#include <ROOT/RFieldVisitor.hxx>
#include <ROOT/RPageStorage.hxx>
#include <ROOT/RNTupleView.hxx>

#include "RFieldHolder.hxx"

using namespace ROOT::Experimental::Browsable;

using namespace std::string_literals;

template<typename T>
using RField = ROOT::Experimental::RField<T>;

// ==============================================================================================

/** \class RFieldProvider
\ingroup rbrowser
\brief Base class for provider of RNTuple drawing
\author Sergey Linev <S.Linev@gsi.de>
\date 2021-03-09
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RFieldProvider : public RProvider {
   class RDrawVisitor : public ROOT::Experimental::Detail::RFieldVisitor {
   private:
      std::shared_ptr<ROOT::Experimental::Detail::RPageSource> fNtplSource;
      std::unique_ptr<TH1> fHist;

      /** Test collected entries if it looks like integer values and one can use better binning */
      void TestHistBuffer()
      {
         auto len = fHist->GetBufferLength();
         auto buf = fHist->GetBuffer();

         if (!buf || (len < 5))
            return;

         Double_t min = buf[1], max = buf[1];
         Bool_t is_integer = kTRUE;

         for (Int_t n = 0; n < len; ++n) {
            Double_t v = buf[2 + 2*n];
            if (v > max) max = v;
            if (v < min) min = v;
            if (TMath::Abs(v - TMath::Nint(v)) > 1e-5) { is_integer = kFALSE; break; }
         }

         // special case when only integer values in short range - better binning
         if (is_integer && (max-min < 100)) {
            max += 2;
            if (min > 1) min -= 2;
            int npoints = TMath::Nint(max - min);
            std::unique_ptr<TH1> h1 = std::make_unique<TH1F>(fHist->GetName(), fHist->GetTitle(), npoints, min, max);
            h1->SetDirectory(nullptr);
            for (Int_t n = 0; n < len; ++n)
               h1->Fill(buf[2 + 2*n], buf[1 + 2*n]);
            std::swap(fHist, h1);
         }
      }

      template<typename T>
      void FillHistogram(const RField<T> &field)
      {
         std::string title = "Drawing of RField "s + field.GetName();

         fHist = std::make_unique<TH1F>("hdraw", title.c_str(), 100, 0, 0);
         fHist->SetDirectory(nullptr);

         auto bufsize = (fHist->GetBufferSize() - 1) / 2;
         int cnt = 0;
         if (bufsize > 10) bufsize-=3; else bufsize = -1;

         auto view = ROOT::Experimental::RNTupleView<T>(field.GetOnDiskId(), fNtplSource.get());
         for (auto i : view.GetFieldRange()) {
            fHist->Fill(view(i));
            if (++cnt == bufsize) {
               TestHistBuffer();
               ++cnt;
            }
         }
         if (cnt < bufsize)
            TestHistBuffer();

         fHist->BufferEmpty();
      }

      void FillStringHistogram(const RField<std::string> &field)
      {
         std::map<std::string, int> values;

         int nentries = 0;

         auto view = ROOT::Experimental::RNTupleView<std::string>(field.GetOnDiskId(), fNtplSource.get());
         for (auto i : view.GetFieldRange()) {
             std::string v = view(i);
             nentries++;
             auto iter = values.find(v);
             if (iter != values.end())
                iter->second++;
             else if (values.size() >= 50)
                return;
             else
                values[v] = 0;
         }

         // now create histogram with labels

         std::string title = "Drawing of RField "s + field.GetName();
         fHist = std::make_unique<TH1F>("h",title.c_str(),3,0,3);
         fHist->SetDirectory(nullptr);
         fHist->SetStats(0);
         fHist->SetEntries(nentries);
         fHist->SetCanExtend(TH1::kAllAxes);
         for (auto &entry : values)
            fHist->Fill(entry.first.c_str(), entry.second);
         fHist->LabelsDeflate();
         fHist->Sumw2(kFALSE);
      }

   public:
      explicit RDrawVisitor(std::shared_ptr<ROOT::Experimental::Detail::RPageSource> ntplSource)
         : fNtplSource(ntplSource)
      {
      }

      TH1 *MoveHist() {
         return fHist.release();
      }

      void VisitField(const ROOT::Experimental::Detail::RFieldBase & /* field */) final {}
      void VisitBoolField(const RField<bool> &field) final { FillHistogram(field); }
      void VisitFloatField(const RField<float> &field) final { FillHistogram(field); }
      void VisitDoubleField(const RField<double> &field) final { FillHistogram(field); }
      void VisitCharField(const RField<char> &field) final { FillHistogram(field); }
      void VisitInt8Field(const RField<std::int8_t> &field) final { FillHistogram(field); }
      void VisitInt16Field(const RField<std::int16_t> &field) final { FillHistogram(field); }
      void VisitIntField(const RField<int> &field) final { FillHistogram(field); }
      void VisitInt64Field(const RField<std::int64_t> &field) final { FillHistogram(field); }
      void VisitStringField(const RField<std::string> &field) final { FillStringHistogram(field); }
      void VisitUInt16Field(const RField<std::uint16_t> &field) final { FillHistogram(field); }
      void VisitUInt32Field(const RField<std::uint32_t> &field) final { FillHistogram(field); }
      void VisitUInt64Field(const RField<std::uint64_t> &field) final { FillHistogram(field); }
      void VisitUInt8Field(const RField<std::uint8_t> &field) final { FillHistogram(field); }
   }; // class RDrawVisitor

public:
   // virtual ~RFieldProvider() = default;

   TH1 *DrawField(RFieldHolder *holder)
   {
      if (!holder) return nullptr;

      auto ntplSource = holder->GetNtplSource();
      std::string name = holder->GetParentName();

      std::unique_ptr<ROOT::Experimental::Detail::RFieldBase> field;
      {
         auto descriptorGuard = ntplSource->GetSharedDescriptorGuard();
         field = descriptorGuard->GetFieldDescriptor(holder->GetId()).CreateField(descriptorGuard.GetRef());
      }
      name.append(field->GetName());

      RDrawVisitor drawVisitor(ntplSource);
      field->AcceptVisitor(drawVisitor);
      return drawVisitor.MoveHist();
   }
};

#endif
