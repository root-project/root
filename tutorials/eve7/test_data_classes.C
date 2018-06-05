#include "ROOT/TEveManager.hxx"
#include "ROOT/TEveDataClasses.hxx"

#include "TParticle.h"
#include "TRandom.h"
#include "TSystem.h"

std::vector<TParticle> ext_col;

void fill_ext_col(int N)
{
   ext_col.clear();
   ext_col.reserve(N);

   TRandom &r = * gRandom;
   r.SetSeed(0);

   for (int i = 1; i <= N; ++i)
   {
      double pt  = r.Uniform(0.5, 20);
      double eta = r.Uniform(-2.55, 2.55);
      double phi = r.Uniform(0, TMath::TwoPi());

      double px = pt * std::cos(phi);
      double py = pt * std::sin(phi);
      double pz = pt * (1. / (std::tan(2*std::atan(std::exp(-eta)))));

      printf("%2d: pt=%.2f, eta=%.2f, phi=%.2f\n", i, pt, eta, phi);

      ext_col.push_back
         (TParticle(0, 0, 0, 0, 0, 0,
                    px, py, pz, std::sqrt(px*px + py*py + pz*pz + 80*80),
                    0, 0, 0, 0 ));
   }
}


void test_data_classes()
{
   namespace REX = ROOT::Experimental;

   gSystem->Load("libROOTEve.so");

   fill_ext_col(30);

   // --------------------------------

   auto col = new REX::TEveDataCollection();

   col->SetItemClass(TParticle::Class());

   {
      int i = 1;
      for (auto &p : ext_col)
      {
         TString pname; pname.Form("Particle %2d", i++);

         col->AddItem(&p, pname.Data(), "");
      }
   }
   col->SetFilterExpr("i.Pt() > 1 && std::abs(i.Eta()) < 1");
   col->ApplyFilter();

   // --------------------------------

   auto tbl = new REX::TEveDataTable();

   tbl->SetCollection(col);

   auto c1 = new REX::TEveDataColumn("phi");
   c1->SetExpressionAndType("i.Phi()", REX::TEveDataColumn::FT_Double);
   c1->SetPrecision(1);
   tbl->AddElement(c1);

   auto c2 = new REX::TEveDataColumn("is_central");
   c2->SetExpressionAndType("std::abs(i.Eta()) < 1.0", REX::TEveDataColumn::FT_Bool);
   tbl->AddElement(c2);
}
