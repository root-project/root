void P040_TFFTReal()
{
   gPluginMgr->AddHandler("TVirtualFFT", "fftwr2r", "TFFTReal",
      "FFTW", "TFFTReal(Int_t, Int_t *,Bool_t)");
}
