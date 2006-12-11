
#include "TVirtualFFT.h"
#include "TRandom.h"

//testing by doing forward and backward transforms
//Author: Anna Kreshuk

void testc2c()
{
//test c2c
   Int_t N = 100;
   Double_t *in = new Double_t[2*N];
   Double_t *out = new Double_t[2*N];
   for (Int_t i=0; i<2*N; i++)
      in[i] = gRandom->Uniform(-100, 100);
   TVirtualFFT *fftc2cf = TVirtualFFT::FFT(1, &N, "C2CF", "M", "O");
   fftc2cf->SetPoints(in);
   fftc2cf->Transform();
   fftc2cf->GetPoints(out);
   TVirtualFFT *fftc2cb = TVirtualFFT::FFT(1, &N, "C2CB", "M", "O");
   fftc2cb->SetPoints(out);
   fftc2cb->Transform();
   fftc2cb->GetPoints(out);
   Int_t ndiff = 0;
   for (Int_t i=0; i<N; i++){
      if ((in[i]-out[i]/N)>0.01){
         printf("i=%d, different, in[i] = %f, out[i]=%f\n", i, in[i], out[i]);
         ndiff++;
      }
   }
   printf("ndiff=%d\n", ndiff);
   delete [] in;
   delete [] out;
   delete fftc2cf;
   delete fftc2cb;
}

void testc2r()
{
//test c2r and r2c

   Int_t N = 100;
   Double_t *in = new Double_t[N];
   Double_t *out = new Double_t[2*(N/2+1)];
   TVirtualFFT *fftr2c = TVirtualFFT::FFT(1, &N, "R2C", "M", "O");
   for (Int_t i=0; i<N; i++)
      in[i] = gRandom->Uniform(-10, 10);
   fftr2c->SetPoints(in);
   fftr2c->Transform();
   fftr2c->GetPoints(out);
   TVirtualFFT *fftc2r = TVirtualFFT::FFT(1, &N, "C2R", "M", "O");
   fftc2r->SetPoints(out);
   fftc2r->Transform();
   fftc2r->GetPoints(out);
   Int_t ndiff = 0;
   for (Int_t i=0; i<N; i++){
      if (in[i]-out[i]/N > 0.01){
         printf("diff: in[%d]=%f, out[%d]=%f\n", i, in[i], i, out[i]);
         ndiff++;
      }
   }
   printf("ndiff=%d\n", ndiff);
} 

void testdht()
{
//test discrete hartley transforms

   Int_t N = 100;
   Double_t *in = new Double_t[N];
   Double_t *out = new Double_t[N];
   TVirtualFFT *dht = TVirtualFFT::FFT(1, &N, "DHT", "M", "O");
   for (Int_t i=0; i<N; i++)
      in[i] = gRandom->Uniform(-10, 10);
   dht->SetPoints(in);
   dht->Transform();
   dht->GetPoints(out);
   dht->SetPoints(out);
   dht->Transform();
   dht->GetPoints(out);
   Int_t ndiff = 0;
   for (Int_t i=0; i<N; i++){
      if (TMath::Abs(in[i]-out[i]/N )> 0.01){
         printf("diff: in[%d]=%f, out[%d]=%f\n", i, in[i], i, out[i]);
         ndiff++;
      }
   }
   printf("ndiff=%d\n", ndiff);
} 

void testhc()
{
//test the halfcomplex format
   Int_t N = 1000;
   Double_t *in = new Double_t[N];
   Double_t *out = new Double_t[N];
   TVirtualFFT *ffthc = TVirtualFFT::FFT(1, &N, "R2HC", "M", "O");
   for (Int_t i=0; i<N; i++)
      in[i] = gRandom->Uniform(-10, 10);
   ffthc->SetPoints(in);
   ffthc->Transform();
   ffthc->GetPoints(out);
   TVirtualFFT *ffthcr = TVirtualFFT::FFT(1, &N, "HC2R", "M", "O");
   ffthcr->SetPoints(out);
   ffthcr->Transform();
   ffthcr->GetPoints(out);
   Int_t ndiff = 0;
   for (Int_t i=0; i<N; i++){
      if (TMath::Abs(in[i]-out[i]/N )> 0.01){
         printf("diff: in[%d]=%f, out[%d]=%f\n", i, in[i], i, out[i]);
         ndiff++;
      }
   }
   printf("ndiff=%d\n", ndiff);
} 

void testr2r()
{
//test sine and cosine transforms
   Int_t N = 1000;
   Double_t *in = new Double_t[N];
   Double_t *out = new Double_t[N];
   Int_t kind = 0;
   TVirtualFFT *cos = TVirtualFFT::SineCosine(1, &N, &kind, "M", "O");
   for (Int_t i=0; i<N; i++)
      in[i] = gRandom->Uniform(-10, 10);
   cos->SetPoints(in);
   cos->Transform();
   cos->GetPoints(out);
   cos->SetPoints(out);
   cos->Transform();
   cos->GetPoints(out);
   Int_t ndiff = 0;
   for (Int_t i=0; i<N; i++){
      if (TMath::Abs(in[i]-out[i]/(2*(N-1)) )> 0.01){
         printf("diff: in[%d]=%f, out[%d]=%f\n", i, in[i], i, out[i]);
         ndiff++;
      }
   }
   printf("ndiff_cos1=%d\n", ndiff);

   kind = 1;
   cos->Init("M", 0, &kind);
   cos->SetPoints(in);
   cos->Transform();
   cos->GetPoints(out);
   kind = 2;
   cos->Init("M", 0, &kind);
   cos->SetPoints(out);
   cos->Transform();
   cos->GetPoints(out);
   ndiff = 0;
   for (Int_t i=0; i<N; i++){
      if (TMath::Abs(in[i]-out[i]/(2*N))> 0.01){
         printf("diff: in[%d]=%f, out[%d]=%f\n", i, in[i], i, out[i]);
         ndiff++;
      }
   }
   printf("ndiff_cos2=%d\n", ndiff);

   kind = 3;
   cos->Init("M", 0, &kind);
   cos->SetPoints(in);
   cos->Transform();
   cos->GetPoints(out);
   cos->SetPoints(out);
   cos->Transform();
   cos->GetPoints(out);
   ndiff = 0;
   for (Int_t i=0; i<N; i++){
      if (TMath::Abs(in[i]-out[i]/(2*N))> 0.01){
         printf("diff: in[%d]=%f, out[%d]=%f\n", i, in[i], i, out[i]);
         ndiff++;
      }
   }
   printf("ndiff_cos3=%d\n", ndiff);
   delete cos;

   kind = 4;
   TVirtualFFT *sin = TVirtualFFT::SineCosine(1, &N, &kind, "M", "O");
   for (Int_t i=0; i<N; i++)
      in[i] = gRandom->Uniform(-10, 10);
   sin->SetPoints(in);
   sin->Transform();
   sin->GetPoints(out);
   sin->SetPoints(out);
   sin->Transform();
   sin->GetPoints(out);
   ndiff = 0;
   for (Int_t i=0; i<N; i++){
      if (TMath::Abs(in[i]-out[i]/(2*(N+1)) )> 0.01){
         printf("diff: in[%d]=%f, out[%d]=%f\n", i, in[i], i, out[i]);
         ndiff++;
      }
   }
   printf("ndiff_sin1=%d\n", ndiff);

   kind = 5;
   sin->Init("M", 0, &kind);
   sin->SetPoints(in);
   sin->Transform();
   sin->GetPoints(out);
   kind = 6;
   sin->Init("M", 0, &kind);
   sin->SetPoints(out);
   sin->Transform();
   sin->GetPoints(out);
   ndiff = 0;
   for (Int_t i=0; i<N; i++){
      if (TMath::Abs(in[i]-out[i]/(2*N))> 0.01){
         printf("diff: in[%d]=%f, out[%d]=%f\n", i, in[i], i, out[i]);
         ndiff++;
      }
   }
   printf("ndiff_sin2=%d\n", ndiff);

   kind = 7;
   sin->Init("M", 0, &kind);
   sin->SetPoints(in);
   sin->Transform();
   sin->GetPoints(out);
   sin->SetPoints(out);
   sin->Transform();
   sin->GetPoints(out);
   ndiff = 0;
   for (Int_t i=0; i<N; i++){
      if (TMath::Abs(in[i]-out[i]/(2*N))> 0.01){
         printf("diff: in[%d]=%f, out[%d]=%f\n", i, in[i], i, out[i]);
         ndiff++;
      }
   }
   printf("ndiff_sin3=%d\n", ndiff);
   delete sin;



   delete [] in;
   delete [] out;

   //delete cos2;
} 
