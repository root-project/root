
#include "TMath.h"

Double_t ChisquareQuantile(Double_t p, Double_t ndf)
{
//Evaluate the quantiles of the chi-squared probability distribution function.
//Algorithm AS 91   Appl. Statist. (1975) Vol.24, P.35
//Incorporates the suggested changes in AS R85 (vol.40(1), pp.233-5, 1991) 
//Parameters:
//1st - the probability value, at which the quantile is computed
//2nd - number of degrees of freedom
      
   Double_t c[]={0, 0.01, 0.222222, 0.32, 0.4, 1.24, 2.2,
                 4.67, 6.66, 6.73, 13.32, 60.0, 70.0,
                 84.0, 105.0, 120.0, 127.0, 140.0, 175.0,
                 210.0, 252.0, 264.0, 294.0, 346.0, 420.0,
                 462.0, 606.0, 672.0, 707.0, 735.0, 889.0,
                 932.0, 966.0, 1141.0, 1182.0, 1278.0, 1740.0,
                 2520.0, 5040.0};
   Double_t e = 5e-7;
   Double_t aa = 0.6931471806;
   Int_t maxit = 20;
   Double_t ch, p1, p2, q, t, a, b, x;
   Double_t s1, s2, s3, s4, s5, s6;

   if (ndf <= 0) return 0;

   Double_t g = TMath::LnGamma(0.5*ndf);

   Double_t xx = 0.5 * ndf;
   Double_t cp = xx - 1;
   if (ndf >= TMath::Log(p)*(-c[5])){
   //starting approximation for ndf less than or equal to 0.32
      if (ndf > c[3]) {
         x = TMath::NormQuantile(p);
         //starting approximation using Wilson and Hilferty estimate
         p1=c[2]/ndf;
         ch = ndf*TMath::Power((x*TMath::Sqrt(p1) + 1 - p1), 3);
         if (ch > c[6]*ndf + 6)
            ch = -2 * (TMath::Log(1-p) - cp * TMath::Log(0.5 * ch) + g);
      } else {
         ch = c[4];
         a = TMath::Log(1-p);
         do{
            q = ch;
            p1 = 1 + ch * (c[7]+ch);
            p2 = ch * (c[9] + ch * (c[8] + ch));
            t = -0.5 + (c[7] + 2 * ch) / p1 - (c[9] + ch * (c[10] + 3 * ch)) / p2;
            ch = ch - (1 - TMath::Exp(a + g + 0.5 * ch + cp * aa) *p2 / p1) / t;
         }while (TMath::Abs(q/ch - 1) > c[1]);
      }
   } else {
      ch = TMath::Power((p * xx * TMath::Exp(g + xx * aa)),(1./xx));
      if (ch < e) return ch;
   }
//call to algorithm AS 239 and calculation of seven term  Taylor series
   for (Int_t i=0; i<maxit; i++){
      q = ch;
      p1 = 0.5 * ch;
      p2 = p - TMath::Gamma(xx, p1);

      t = p2 * TMath::Exp(xx * aa + g + p1 - cp * TMath::Log(ch));
      b = t / ch;
      a = 0.5 * t - b * cp;
      s1 = (c[19] + a * (c[17] + a * (c[14] + a * (c[13] + a * (c[12] +c[11] * a))))) / c[24];
      s2 = (c[24] + a * (c[29] + a * (c[32] + a * (c[33] + c[35] * a)))) / c[37];
      s3 = (c[19] + a * (c[25] + a * (c[28] + c[31] * a))) / c[37];
      s4 = (c[20] + a * (c[27] + c[34] * a) + cp * (c[22] + a * (c[30] + c[36] * a))) / c[38];
      s5 = (c[13] + c[21] * a + cp * (c[18] + c[26] * a)) / c[37];
      s6 = (c[15] + cp * (c[23] + c[16] * cp)) / c[38];
      ch = ch + t * (1 + 0.5 * t * s1 - b * cp * (s1 - b * (s2 - b * (s3 - b * (s4 - b * (s5 - b * s6))))));
      if (TMath::Abs(q / ch - 1) > e) break;
   }
   return ch;
}
