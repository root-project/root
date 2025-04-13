#include "PEvent.hh"

ClassImp(PEvent);
ClassImp(QRawPulse);
ClassImp(QRawTriggerPulse);


//______________________________________________________________________________
QRawPulse::QRawPulse()  : TObject(){
    //
    // empty ctor: set invalid channel number
    //
    fNsamples = 0;
    fSample = NULL;
}

//______________________________________________________________________________
QRawPulse::QRawPulse(const QRawPulse& P) : TObject(), fSample(0) {
    //
    //  copy ctor
    //
    Resize(P.fNsamples);

    if ( fNsamples <= 0 ) return;

    memcpy(fSample,P.fSample, fNsamples*sizeof(UInt_t));
    
}

//_____________________________________________________________________
QRawPulse::QRawPulse(Int_t nsamples, UInt_t* s) : TObject(), fSample(0) {
    //
    // ctor
    //
    Resize(nsamples);

    if ( fNsamples <= 0 ) return;

    memcpy(fSample,s, fNsamples*sizeof(UInt_t));

}

//_____________________________________________________________________
QRawPulse& QRawPulse::operator=(const QRawPulse& P){
    //
    // assignement operator
    //

    Resize(P.fNsamples);
    if ( fNsamples <= 0 ) return *this;

    memcpy(fSample,P.fSample, fNsamples*sizeof(UInt_t));
  
    return *this;
}

//_____________________________________________________________________
QRawPulse::~QRawPulse() {
    //
    // dtor
    //
    Reset();
}

//_____________________________________________________________________
void QRawPulse::Reset() {
    if (fSample) delete [] fSample;
    fNsamples = 0;
    
}


//_____________________________________________________________________
void QRawPulse::Resize( Int_t n ) {
    
    if ( fSample && n != fNsamples ) {
        delete [] fSample;
        fNsamples = 0;
    }

    if ( n <= 0 ) return;

    fNsamples = n;
    fSample = new UInt_t[fNsamples];
    
}

//______________________________________________________________________________
QRawTriggerPulse::QRawTriggerPulse() : fChannel(-9999) {
}

//______________________________________________________________________________
QRawTriggerPulse::QRawTriggerPulse( const QRawTriggerPulse& P ) : QRawPulse(P), fChannel(-9999) {
}

//_____________________________________________________________________
QRawTriggerPulse::QRawTriggerPulse(Int_t nsamples, UInt_t* s) : QRawPulse(nsamples,s), fChannel(-9999){
    //
    // ctor
    //
}


//_____________________________________________________________________
PEvent::PEvent(){
  //
  //default  ctor
  //
}

//_____________________________________________________________________
PEvent::PEvent(const QRawTriggerPulse& rawPulse) : fRawTriggerPulse(rawPulse) {
  //
  // ctor
  //
}


