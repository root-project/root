#include "TH1D.h"
#include "TObject.h"

//______________________________________________________________________________
// QRawPulseR class
//______________________________________________________________________________
class QRawPulseR : public TObject 
{
	public:
		QRawPulseR();
		~QRawPulseR() override;
		const QRawPulseR& operator=(const QRawPulseR& P);
		void SetDataHist(const int, double*);
		void SetChannel(const Int_t chan) {fChannel = chan;}

	protected:
		TH1D fDataHist;
		Int_t fChannel;

	friend class QRawTriggerPulseR;
	ClassDefOverride(QRawPulseR, 1)
};

ClassImp(QRawPulseR);

QRawPulseR::QRawPulseR()
{
	fChannel = -9999;
	fDataHist.SetDirectory(0);
}

QRawPulseR::~QRawPulseR() 
{
}

const QRawPulseR& QRawPulseR::operator=(const QRawPulseR& P)
{
	fChannel = P.fChannel;
	fDataHist = P.fDataHist;

	return *this;
}

void QRawPulseR::SetDataHist(const int n, double* samples)
{
	fDataHist.Reset();
	fDataHist.SetBins(n, 0, 1);
	for (int i = 1; i <= n; ++i) {
		fDataHist.SetBinContent(i, samples[i-1]);
	}
}


//______________________________________________________________________________
// QRawTriggerPulseR class
//______________________________________________________________________________
class QRawTriggerPulseR : public QRawPulseR 
{
	public:
		QRawTriggerPulseR();
		~QRawTriggerPulseR() override;
		const QRawTriggerPulseR& operator=(const QRawTriggerPulseR& P);
		Int_t GetTriggerPosition() const { return ftrigger_position; }

	private:
		Int_t ftrigger_position; 

	ClassDefOverride(QRawTriggerPulseR, 1);
};

ClassImp(QRawTriggerPulseR);

QRawTriggerPulseR::QRawTriggerPulseR() : QRawPulseR() 
{
	ftrigger_position = -9999;
}

QRawTriggerPulseR::~QRawTriggerPulseR() 
{
}

const QRawTriggerPulseR& QRawTriggerPulseR::operator=(const QRawTriggerPulseR& P)
{
	fDataHist = P.fDataHist;
	fChannel = P.fChannel;
	ftrigger_position = P.ftrigger_position;

	return *this;
}


//______________________________________________________________________________
// QRawEventR class
//______________________________________________________________________________
class QRawEventR : public TObject 
{ 
	public:
		QRawEventR();	
		~QRawEventR() override;
		void SetRawPulse(const QRawTriggerPulseR);

	private:
		QRawPulseR fraw_pulse;
		QRawTriggerPulseR fraw_trigger_pulse;

	ClassDef (QRawEventR, 1)
};

ClassImp(QRawEventR);

QRawEventR::QRawEventR() 
{
}

QRawEventR::~QRawEventR()
{
}

void QRawEventR::SetRawPulse(const QRawTriggerPulseR pulse)
{
	fraw_trigger_pulse = pulse;
	fraw_pulse = pulse;
}
