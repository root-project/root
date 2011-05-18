#ifndef MATERIAL_H
#define MATERIAL_H

#include <TObject.h>
#include <TString.h>

//////////////// NdbMaterial /////////////////
class NdbMaterial : public TObject
{
protected:
	TString	eMnemonic;		// Mnemonic of the element XX-AAAii
	TString eName;			// Name of isotope "Gold"
	Long_t	eId;			// AZ code = Z*10000 + A*10 + ISO
	Int_t	eZ;			// Z = number of protons
	Int_t	eA;			// A = mass number
	Int_t	eIS;			// Isomeric state of element
	Float_t	eMass;			// Mass of the element
	Float_t	eMassExcess;		// M-A
	Float_t eHalfLife;		// Half life
	Float_t	eAbundance;		// Percentage of natural isotope
	Float_t	eDensity;		// Density of element
	Float_t	eMeltingPoint;		// Melting point in K
	Float_t eBoilingPoint;		// Boiling point in K

//	TDecayMode	eDecayMode;	// Decay mode of element

	TObjArray	*mf;		// Array of File types

public:
	NdbMaterial() {}

	NdbMaterial(
		const char *aName,
		const char *aMnemonic,
		      Int_t aZ,
		      Int_t anA,
		      Int_t anIso=0,
		      Float_t aMass=0.0,
		      Float_t aMassExcess=0.0,
		      Float_t aHalfLife=0.0,
		      Float_t aAbundance=0.0,
		      Float_t aDensity=0.0,
		      Float_t aMeltingPoint=0.0,
		      Float_t aBoilingPoint=0.0)
	: TObject(), eMnemonic(aMnemonic), eName(aName) {
		eZ	= aZ;
		eA	= anA;
		eIS	= anIso;
		eId	= eZ*10000 + eA*10 + eIS;
		if (aMass!=0.0)
			eMass = aMass;
		else
			eMass = (Float_t)anA;
		eMassExcess	= aMassExcess;
		eAbundance	= aAbundance;
		eHalfLife	= aHalfLife;
		eDensity	= aDensity;
		eMeltingPoint	= aMeltingPoint;
		eBoilingPoint	= aBoilingPoint;
	}

	NdbMaterial( NdbMaterial& elem )
		: TObject(), eName(elem.eName)
	{
		eZ    = elem.eZ;
		eA    = elem.eA;
		eIS   = elem.eIS;
		eId   = elem.eId;
		eMass = elem.eMass;
		eMassExcess	= elem.eMassExcess;
		eDensity	= elem.eDensity;
		eMeltingPoint	= elem.eMeltingPoint;
		eBoilingPoint	= elem.eBoilingPoint;
		eAbundance	= elem.eAbundance;
		eHalfLife	= elem.eHalfLife;
		eMeltingPoint	= elem.eMeltingPoint;
		eBoilingPoint	= elem.eBoilingPoint;
	}

	~NdbMaterial() {}

	// --- Virtual functions ---
	Int_t Compare(const TObject *o) const
		{ return ((eId==((NdbMaterial*)o)->eId)? 0 :
				(eId > ((NdbMaterial*)o)->eId)? 1 : -1 ); }

	// --- Access functions ---
	inline TString Name()		const { return eName; }
	inline TString Mnemonic()	const { return eMnemonic; }
	inline Long_t  Id()		const { return eId; }
	inline Int_t   A()		const { return eA; }
	inline Int_t   Z()		const { return eZ; }
	inline Int_t   ISO()		const { return eIS; }
	inline Float_t Mass()		const { return eMass; }
	inline Float_t MassExcess()	const { return eMassExcess; }
	inline Float_t HalfLife()	const { return eHalfLife; }
	inline Float_t Abundance()	const { return eAbundance; }
	inline Float_t Density()	const { return eDensity; }
	inline Float_t MeltingPoint()	const { return eMeltingPoint; }
	inline Float_t BoilingPoint()	const { return eBoilingPoint; }

	ClassDef(NdbMaterial,1)

}; // NdbMaterial

#endif
