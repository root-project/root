void P010_TFFTComplex()
{
   gPluginMgr->AddHandler("TVirtualFFT", "fftwc2c", "TFFTComplex",
      "FFTW", "TFFTComplex(Int_t, Int_t *,Bool_t)");
}
