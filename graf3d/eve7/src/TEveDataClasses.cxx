#include "ROOT/TEveDataClasses.hxx"

using namespace ROOT::Experimental;
namespace REX = ROOT::Experimental;

#include "TParticle.h"
#include "TRandom.h"

#include "TROOT.h"

TEveDataCollection::TEveDataCollection(const char* n, const char* t) :
   TEveElementList(n, t)
{
   fChildClass = TEveDataItem::Class();

   // HACK TO POPULATE WITH RANDOM PARTICLES

   fItemClass  = TParticle::Class();
   fFilterExpr = "i.Pt() > 1 && std::abs(i.Eta()) < 1";

   auto vp = new std::vector<TParticle>;
   auto &v = *vp;
   v.reserve(20);

   TRandom &r = * gRandom;
   r.SetSeed(0);

   for (int i = 1; i <= 20; ++i)
   {
      double pt  = r.Uniform(0.5, 20);
      double eta = r.Uniform(-2.55, 2.55);
      double phi = r.Uniform(0, TMath::TwoPi());

      double px = pt * std::cos(phi);
      double py = pt * std::sin(phi);
      double pz = pt * (1. / (std::tan(2*std::atan(std::exp(-eta)))));

      printf("%2d: pt=%.2f, eta=%.2f, phi=%.2f\n", i, pt, eta, phi);

      v.push_back(TParticle(0, 0, 0, 0, 0, 0,
                            px, py, pz, std::sqrt(px*px + py*py + pz*pz + 80*80),
                            0, 0, 0, 0));

      TString pname; pname.Form("Particle %2d", i);
      AddItem(&v.back(), pname.Data(), "");
   }

   // END HACK
}

void TEveDataCollection::AddItem(void *data_ptr, const char* n, const char* t)
{
   auto el = new TEveDataItem(n, t);
   AddElement(el);
   fItems.push_back({data_ptr, el});
}

void TEveDataCollection::ApplyFilter()
{
   std::function<bool(void*)> ffoo;

   TString s;
   s.Form("*((std::function<bool(%s*)>*)%p) = [](%s* p){%s &i=*p; printf(\"%%f %%f\\n\", i.Pt(), i.Eta()); return (%s); }",
          fItemClass->GetName(), &ffoo, fItemClass->GetName(), fItemClass->GetName(),
          fFilterExpr.Data());

   printf("%s\n", s.Data());

   gROOT->ProcessLine(s.Data());

   for (auto &ii : fItems)
   {
      bool res = ffoo(ii.fDataPtr);

      printf("Item:%s -- filter result = %d\n", ii.fItemPtr->GetElementName(), res);

      ii.fItemPtr->SetFiltered( ! res );
   }
}

//==============================================================================

TEveDataItem::TEveDataItem(const char* n, const char* t) :
   TEveElementList(n, t)
{
}

//==============================================================================

TEveDataTable::TEveDataTable(const char* n, const char* t) :
   TEveElementList(n, t)
{
   fChildClass = TEveDataColumn::Class();
}

//==============================================================================

TEveDataColumn::TEveDataColumn(const char* n, const char* t) :
   TEveElementList(n, t)
{
}
