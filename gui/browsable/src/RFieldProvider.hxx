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
#include <string>

#include <ROOT/Browsable/RProvider.hxx>

#include <ROOT/RNTuple.hxx>
// #include <ROOT/RNTupleDescriptor.hxx>
// #include <ROOT/RMiniFile.hxx>

#include "RFieldHolder.hxx"

using namespace ROOT::Experimental::Browsable;

using namespace std::string_literals;


// ==============================================================================================

/** \class RFieldProvider
\ingroup rbrowser
\brief Base class for provider of RNTuple drawing
\author Sergey Linev <S.Linev@gsi.de>
\date 2021-03-09
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RFieldProvider : public RProvider {
public:

   // virtual ~RFieldProvider() = default;

   /** Test collected entries if it looks like integer values and one can use better binning */
   TH1F *TestHistBuffer(TH1F *hist)
   {
      auto len = hist->GetBufferLength();
      auto buf = hist->GetBuffer();

      if (!buf || (len < 5)) return hist;

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
         TH1F *h1 = new TH1F(hist->GetName(), hist->GetTitle(), npoints, min, max);
         h1->SetDirectory(nullptr);
         for (Int_t n = 0; n < len; ++n)
            h1->Fill(buf[2 + 2*n], buf[1 + 2*n]);
         delete hist;
         return h1;
      }

      return hist;
   }


   template<typename T>
   TH1 *FillHistogram(std::shared_ptr<ROOT::Experimental::RNTupleReader> &tuple, const std::string &field_name)
   {
      std::string title = "Drawing of RField "s + field_name.c_str();

      auto h1 = new TH1F("hdraw", title.c_str(), 100, 0, 0);
      h1->SetDirectory(nullptr);

      auto bufsize = (h1->GetBufferSize() - 1) / 2;
      int cnt = 0;
      if (bufsize > 10) bufsize-=3; else bufsize = -1;

      auto view = tuple->GetView<T>(field_name);
      for (auto i : tuple->GetEntryRange()) {
         h1->Fill(view(i));
         if (++cnt == bufsize) {
            h1 = TestHistBuffer(h1);
            ++cnt;
         }
      }
      if (cnt < bufsize)
         h1 = TestHistBuffer(h1);

      h1->BufferEmpty();
      return h1;
   }

   TH1 *FillStringHistogram(std::shared_ptr<ROOT::Experimental::RNTupleReader> &tuple, const std::string &field_name)
   {
      std::map<std::string, int> values;

      int nentries = 0;

      auto view = tuple->GetView<std::string>(field_name);
      for (auto i : tuple->GetEntryRange()) {
          std::string v = view(i);
          nentries++;
          auto iter = values.find(v);
          if (iter != values.end())
             iter->second++;
          else if (values.size() >= 50)
             return nullptr;
          else
             values[v] = 0;
      }

      // now create histogram with labels

      std::string title = "Drawing of RField "s + field_name.c_str();
      TH1F *h = new TH1F("h",title.c_str(),3,0,3);
      h->SetDirectory(nullptr);
      h->SetStats(0);
      h->SetEntries(nentries);
      h->SetCanExtend(TH1::kAllAxes);
      for (auto &entry : values)
         h->Fill(entry.first.c_str(), entry.second);
      h->LabelsDeflate();
      h->Sumw2(kFALSE);
      return h;
   }

   TH1 *DrawField(RFieldHolder *holder)
   {
      if (!holder) return nullptr;

      auto tuple = holder->GetNTuple();
      std::string name = holder->GetParentName();

      auto &field = tuple->GetDescriptor().GetFieldDescriptor(holder->GetId());
      name.append(field.GetFieldName());

      if (field.GetTypeName() == "double"s)
         return FillHistogram<double>(tuple, name);
      else if (field.GetTypeName() == "float"s)
         return FillHistogram<float>(tuple, name);
      else if (field.GetTypeName() == "int"s)
         return FillHistogram<int>(tuple, name);
      else if (field.GetTypeName() == "std::int32_t"s)
         return FillHistogram<int32_t>(tuple, name);
      else if (field.GetTypeName() == "std::uint32_t"s)
         return FillHistogram<uint32_t>(tuple, name);
      else if (field.GetTypeName() == "std::string"s)
         return FillStringHistogram(tuple, name);

      return nullptr;
   }

};

#endif
