#ifndef __CLASSES_H__
#define __CLASSES_H__

#include <memory>
#include <vector>
#include <list>
#include <string>
#include <cmath>

#include "TH1F.h"
#include "TRandom.h"
#include "ROOT/TSeq.hxx"

class A{
public:
   A(): fTH1FFixedUPtr(new TH1F("","b",64,0,4)),
        fTH1FPtr(new TH1F("","b",64,0,4))
   {
   };

   A(const char* meta): fTH1FBaseUPtr(new TH1F((std::string(meta)+"_base").c_str(),"b",64,0,4)),
                        fTH1FUPtr(new TH1F((std::string(meta)+"_var").c_str(),"b",64,0,4)),
                        fTH1FFixedUPtr(new TH1F((std::string(meta)+"_fixed").c_str(),"b",64,0,4)),
                        fTH1FPtr(new TH1F(meta,"b",64,0,4))
   {
      fTH1FUPtrs.emplace_back(new TH1F((std::string(meta)+"_unique1").c_str(),"b",64,0,4));
      fTH1FUPtrs.emplace_back(new TH1F((std::string(meta)+"_unique2").c_str(),"b",64,0,4));
      fTH1FUPtrs.emplace_back(new TH1F((std::string(meta)+"_unique3").c_str(),"b",64,0,4));
      
      gRandom->SetSeed(1);
      fTH1FBaseUPtr->FillRandom("gaus");
      gRandom->SetSeed(1);
      fTH1FUPtr->FillRandom("gaus");
      gRandom->SetSeed(1);
      fTH1FFixedUPtr->FillRandom("gaus");
      gRandom->SetSeed(1);
      fTH1FPtr->FillRandom("gaus");

      for (auto&& h : fTH1FUPtrs) {
         gRandom->SetSeed(1);
         h->FillRandom("gaus");
      }
   }

   ~A(){
      delete fTH1FPtr;
   }

   void Randomize() {
      auto rndMizeBins = [](TH1F* h){
         gRandom->SetSeed(1);
         for (auto i : ROOT::TSeqI(h->GetNbinsX())) {
            auto bContent = h->GetBinContent(i);
            auto bVar = bContent*.6;
            h->SetBinContent(i, std::fabs(gRandom->Gaus(bContent,bVar)));
         }
      };
      rndMizeBins(fTH1FPtr);
      rndMizeBins(fTH1FUPtr.get());
      for (auto&& h : fTH1FUPtrs) rndMizeBins(h.get());
   }

   TH1F* GetHPtr() {return fTH1FPtr;}
   TH1*  GetHBaseUPtr() {return fTH1FBaseUPtr.get();}
   TH1F* GetHUPtr() {return fTH1FUPtr.get();}
   TH1F* GetHFixedUPtr() {return fTH1FFixedUPtr.get();}
   TH1F* GetHUPtrAt(unsigned int i) {
      //return fTH1FUPtrs.at(i).get()
      auto it = fTH1FUPtrs.begin();
      for (unsigned int j=0;j<i;++j) it++;
      return it->get();
      
   }

private:
   std::unique_ptr<TH1>  fTH1FBaseUPtr;
   std::unique_ptr<TH1F> fTH1FUPtr;
   std::unique_ptr<TH1F> fTH1FFixedUPtr; //->
   std::list<std::unique_ptr<TH1F>> fTH1FUPtrs;
   double d = .5;
   TH1F* fTH1FPtr; // ->
};

class Aconst
{
   std::unique_ptr<const TH1> fPtr;

 public:
   Aconst(const char *c) : fPtr(new TH1F(c, c, 64, -1, 1)) {}
   Aconst() {}
};

#endif
