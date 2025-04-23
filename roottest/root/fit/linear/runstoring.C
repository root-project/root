//#include "TLinearFitter.h"
#include "TFormula.h"
#include "TFile.h"
#include "TRandom.h"
#include "TStopwatch.h"

void writefitter()
{
#ifdef __CINT__
   G__SetCatchException(0);  
#endif
   Int_t npoints = 100;
   Int_t ndim = 3;
   TFormula *f = new TFormula("f", "x[0]++x[1]++x[2]");
   //TLinearFitter *lf = new TLinearFitter(3, "x[0]++x[1]++x[2]", "");
   TLinearFitter *lf = new TLinearFitter(f);

   Double_t *x = new Double_t[ndim];
   Double_t par[3];
   par[0]=1; par[1]=2; par[2]=3;
   Double_t y;
   for (Int_t ipoint = 0; ipoint<npoints; ipoint++){
      for (Int_t idim=0; idim<ndim; idim++){
         x[idim]=gRandom->Uniform(-10, 10);
      }
      y = f->EvalPar(x, par)+gRandom->Gaus(0, 1);
      lf->AddPoint(x, y);
   }
   // lf->AddTempMatrices();
   TFile *file = TFile::Open("linfitter.root", "RECREATE");
   lf->Write("linfitter");
   lf->Write("linfitter2");
   lf->Eval();
   TVectorD param;
   lf->GetParameters(param);
   param.Print();
   lf->EvalRobust(0.9);

   file->Write();
   file->Close();

}

void readfitter()
{
#ifdef __CINT__
   // G__SetCatchException(0);
#endif
   TVectorD vect;  
   TFile *f = TFile::Open("linfitter.root");
   if (!f) return;
   TLinearFitter *lf = (TLinearFitter*)f->Get("linfitter");
   TLinearFitter *lf2 = (TLinearFitter*)f->Get("linfitter2");
  
   if (!lf) {printf("no fitter\n"); return;}
   if (!lf2) {printf("no fitter2\n"); return;}
   printf("fitter retrieved\n");

   printf("current parameter values for fitter1:\n");
   lf->Eval();
   lf->GetParameters(vect);
   vect.Print();
   printf("current parameter values for fitter2:\n");
   lf2->Eval();
   lf2->GetParameters(vect);
   vect.Print();

   lf->Add(lf2);

//    TFormula *form = new TFormula("f", "x[0]++x[1]++x[2]");
//    Int_t npoints = 1000;
//    Int_t ndim = 3;
//    Double_t *x = new Double_t[ndim];
//    Double_t par[3];
//    par[0]=1; par[1]=2; par[2]=3;
//    Double_t y;
//    for (Int_t ipoint = 0; ipoint<npoints; ipoint++){
//       for (Int_t idim=0; idim<ndim; idim++){
//          x[idim]=gRandom->Uniform(-10, 10);
//       }
//       y = form->EvalPar(x, par)+gRandom->Gaus(0, 1);
//       lf->AddPoint(x, y);
//    }

  lf->Eval();

  lf->GetParameters(vect);
  printf("paramter values for the sum of fitters:\n");
  vect.Print();

  printf("try the robust fitting\n");
  lf->EvalRobust(0.9);
  lf->GetParameters(vect);
  vect.Print();

   f->Close();

}




void writeformula()
{
#ifdef __CINT__
   G__SetCatchException(0);  
#endif
   TFormula *f = new TFormula("ffffff", "x[0]++x[1]++x[2]");

    TFile *file = TFile::Open("formula.root", "RECREATE");
    f->Write("formula");
    file->Write();
    file->Close();
}

void readformula()
{
   TFile *file = TFile::Open("formula.root");
   TFormula *f = (TFormula*)file->Get("formula");
   if (!f){
      printf("formula not found\n");
      return;
   }
   Double_t x[3];
   x[0]=1; x[1]=1; x[2]=1;
   Double_t par[3];
   par[0]=1; par[1]=1; par[2]=1;
   Double_t res = f->EvalPar(x, par);
   printf("res=%f\n", res);

   TFormula *part1 = (TFormula*)f->GetLinearPart(0);
   part1->Print();
   TFormula *part2 = (TFormula*)f->GetLinearPart(1);
   part2->Print();

   file->Close();
}

int runstoring() {
   writeformula();
   readformula();
   writefitter();
   readfitter();
   return 0;
}
