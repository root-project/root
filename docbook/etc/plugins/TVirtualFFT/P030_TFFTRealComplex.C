void P030_TFFTRealComplex()
{
   gPluginMgr->AddHandler("TVirtualFFT", "fftwr2c", "TFFTRealComplex",
      "FFTW", "TFFTRealComplex(Int_t,Int_t *, Bool_t)");
}
