/*
 * $Header$
 * $Log$
 *
 * Array of Elements with names and mnemonics
 */

#ifndef __XSELEMENT_LIST_H
#define __XSELEMENT_LIST_H

#include <TGListBox.h>

#define XSEL_SORTBY_NAME	0
#define XSEL_SORTBY_MNEMONIC	1
#define XSEL_SORTBY_Z		2

/* =================== XSElementList ===================== */
class XSElementList : public TGListBox
{
protected:
	Int_t			sortBy;

public:
	XSElementList(TGWindow *p, Int_t sortby = XSEL_SORTBY_NAME);
	~XSElementList();

	void		SelectZ(UInt_t Z);
	UInt_t		CurrentZ();

protected:
	Int_t		Compare(int i, int j) const;
	Int_t           Compare(const TObject *o) const { return TObject::Compare(o); }
	UInt_t		GetZ( TGTextLBEntry *entry );
	void		Sort(int *index);
	Int_t		ElementString(Int_t idx, char *buf);

	//ClassDef(XSElementList,1)
}; // XSElementList

#endif
