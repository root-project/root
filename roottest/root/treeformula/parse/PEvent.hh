#ifndef _QEVENT_HH
#define _QEVENT_HH
#include <TObject.h>
#include <TClass.h>

using namespace std;


class QRawPulse : public TObject {
    public:
        QRawPulse();

        virtual ~QRawPulse();

        QRawPulse(Int_t, UInt_t*);

        QRawPulse(const QRawPulse& P);

        QRawPulse& operator=(const QRawPulse& P);

        const UInt_t* GetSample() const { return fSample; };

    protected:

        void Resize(Int_t n); 
        void Reset();
        Int_t fNsamples;           // number of ADC samples 
        UInt_t* fSample;           //[fNsamples]

        ClassDefOverride(QRawPulse,1);
};

class QRawTriggerPulse : public QRawPulse {
    public:
        QRawTriggerPulse();
        QRawTriggerPulse( const QRawTriggerPulse& );
        QRawTriggerPulse( Int_t, UInt_t* );
        virtual ~QRawTriggerPulse() {}

    protected:
        Int_t fChannel;

        ClassDefOverride(QRawTriggerPulse,1);
};

 
class PEvent : public TObject {
    public:

        PEvent();	
        PEvent(const QRawTriggerPulse&);
        virtual ~PEvent () {};

        QRawTriggerPulse fRawTriggerPulse;      // triggering pulse

        ClassDef (PEvent, 1) 
};

#endif

