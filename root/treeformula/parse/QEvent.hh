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

        ClassDef(QRawPulse,1);
};

class QRawTriggerPulse : public QRawPulse {
    public:
        QRawTriggerPulse();
        QRawTriggerPulse( const QRawTriggerPulse& );
        QRawTriggerPulse( Int_t, UInt_t* );
        virtual ~QRawTriggerPulse() {}

    protected:
        Int_t fChannel;

        ClassDef(QRawTriggerPulse,1);
};

 
class QEvent : public TObject {
    public:

        QEvent();	
        QEvent(const QRawTriggerPulse&);
        virtual ~QEvent () {};

        QRawTriggerPulse fRawTriggerPulse;      // triggering pulse

        ClassDef (QEvent, 1) 
};

#endif

