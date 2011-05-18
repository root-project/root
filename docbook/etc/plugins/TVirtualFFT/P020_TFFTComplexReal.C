void P020_TFFTComplexReal()
{
   gPluginMgr->AddHandler("TVirtualFFT", "fftwc2r", "TFFTComplexReal",
      "FFTW", "TFFTComplexReal(Int_t,Int_t *, Bool_t)");
}
