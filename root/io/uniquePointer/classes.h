#ifndef __CLASSES_H__
#define __CLASSES_H__

#include <memory>
#include <vector>
#include <list>
#include <string>
#include "TH1F.h"
#include "TRandom.h"
#include "ROOT/TSeq.h"

class A{
public:
   A(): TH1FFixedUPtr(new TH1F("","b",64,0,4)),
        TH1FPtr(new TH1F("","b",64,0,4))
   {
   };

   A(const char* meta): TH1FBaseUPtr(new TH1F((std::string(meta)+"_base").c_str(),"b",64,0,4)),
                        TH1FUPtr(new TH1F((std::string(meta)+"_var").c_str(),"b",64,0,4)),
                        TH1FFixedUPtr(new TH1F((std::string(meta)+"_fixed").c_str(),"b",64,0,4)),
                        TH1FPtr(new TH1F(meta,"b",64,0,4))
   {
      TH1FUPtrs.emplace_back(new TH1F((std::string(meta)+"_unique1").c_str(),"b",64,0,4));
      TH1FUPtrs.emplace_back(new TH1F((std::string(meta)+"_unique2").c_str(),"b",64,0,4));
      TH1FUPtrs.emplace_back(new TH1F((std::string(meta)+"_unique3").c_str(),"b",64,0,4));
      
      gRandom->SetSeed(1);
      TH1FBaseUPtr->FillRandom("gaus");
      gRandom->SetSeed(1);
      TH1FUPtr->FillRandom("gaus");
      gRandom->SetSeed(1);
      TH1FFixedUPtr->FillRandom("gaus");
      gRandom->SetSeed(1);
      TH1FPtr->FillRandom("gaus");

      for (auto&& h : TH1FUPtrs) {
         gRandom->SetSeed(1);
         h->FillRandom("gaus");
      }
   }

   ~A(){
      delete TH1FPtr;
   }

   void Randomize() {
      auto rndMizeBins = [](TH1F* h){
         gRandom->SetSeed(1);
         for (auto i : ROOT::TSeqI(h->GetNbinsX())) {
            auto bContent = h->GetBinContent(i);
            auto bVar = bContent*.6;
            h->SetBinContent(i, fabs(gRandom->Gaus(bContent,bVar)));
         }
      };
      rndMizeBins(TH1FPtr);
      rndMizeBins(TH1FUPtr.get());
      for (auto&& h : TH1FUPtrs) rndMizeBins(h.get());
   }

   TH1F* GetHPtr() {return TH1FPtr;}
   TH1*  GetHBaseUPtr() {return TH1FBaseUPtr.get();}
   TH1F* GetHUPtr() {return TH1FUPtr.get();}
   TH1F* GetHFixedUPtr() {return TH1FFixedUPtr.get();}
   TH1F* GetHUPtrAt(unsigned int i) {
      //return TH1FUPtrs.at(i).get()
      auto it = TH1FUPtrs.begin();
      for (unsigned int j=0;j<i;++j) it++;
      return it->get();
      
   }

private:
   std::unique_ptr<TH1>  TH1FBaseUPtr;
   std::unique_ptr<TH1F> TH1FUPtr;
   std::unique_ptr<TH1F> TH1FFixedUPtr; //->
   std::list<std::unique_ptr<TH1F>> TH1FUPtrs;
   double d = .5;
   TH1F* TH1FPtr; // ->
};

#endif
