#ifndef __CLASSES_H__
#define __CLASSES_H__

#include <memory>
#include <string>
#include "TH1F.h"
#include "TRandom.h"
#include "ROOT/TSeq.h"

class A{
public:
   A():TH1FPtr(new TH1F("","b",64,0,4)),
       TH1FUPtr(new TH1F("","b",64,0,4)){};

   A(const char* meta): TH1FPtr(new TH1F(meta,"b",64,0,4)),
                        TH1FUPtr(new TH1F((std::string(meta)+"_unique").c_str(),"b",64,0,4)){
      gRandom->SetSeed(1);
      TH1FPtr->FillRandom("gaus");
      gRandom->SetSeed(1);
      TH1FUPtr->FillRandom("gaus");
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
   }

   TH1F* GetHPtr() {return TH1FPtr;}
   TH1F* GetHUPtr() {return TH1FUPtr.get();}

private:
   std::unique_ptr<TH1F> TH1FUPtr;
   double d;
   TH1F* TH1FPtr; // ->
};

#endif
