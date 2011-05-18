/*
 * $Header$
 * $Log$
 *
 * Fills a listbox with the elements sorted, by name, mnemonic, or Z
 */

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#include <TGFrame.h>
#include <TGLayout.h>
#include <TGWidget.h>
#include <TGWindow.h>
#include <TGPicture.h>

#include "XSVarious.h"
#include "XSElementList.h"

//ClassImp(XSElementList)

/* ----- XSElementList ----- */
XSElementList::XSElementList(TGWindow* p, Int_t sortby)
	: TGListBox(p,50,50)
{
	UInt_t	i;
	int	*index = new int[XSelements->GetSize()];

	sortBy = sortby;

	for (i=0; i<XSelements->GetSize(); i++)
		index[i] = i;

	Sort(index);

	for (i=0; i<XSelements->GetSize(); i++) {
		char	str[100];
		Int_t	Z = ElementString(index[i]+1, str);
		TGString	*tgstr = new TGString(str);
		TGTextLBEntry	*lbe = new TGTextLBEntry(fLbc,tgstr,Z,
		                                fixedGC,fixedFontStruct);

		TGLayoutHints	*lhints = new TGLayoutHints(
					kLHintsExpandX | kLHintsTop);

		fItemVsize = TMath::Max(fItemVsize, lbe->GetDefaultHeight());

		AddEntry((TGLBEntry*)lbe, lhints);
	}

	delete [] index;
} // XSElementList

/* ----- ~XSElementList ----- */
XSElementList::~XSElementList()
{
} // ~XSElementList

/* ----- Compare ----- */
Int_t
XSElementList::Compare(int i, int j) const
{
	switch (sortBy) {
		case XSEL_SORTBY_NAME:
			return strcmp(XSelements->Name(i+1),
					XSelements->Name(j+1));

		case XSEL_SORTBY_MNEMONIC:
			return strcmp(XSelements->Mnemonic(i+1),
					XSelements->Mnemonic(j+1));

		case XSEL_SORTBY_Z:
			return	(i<j? -1 : i>j? 1 : 0);

		default:
			return 0;
	}
} // Compare

/* ----- Sort ----- */
void
XSElementList::Sort(int *index)
{
	// --- Perform a bubble sort ---
	for (UInt_t i=0; i<XSelements->GetSize(); i++) {
		int change = 0;
		for (UInt_t j=XSelements->GetSize()-1; j>i; j--) {
			if (Compare(index[j],index[j-1]) < 0) {
				int t      = index[j];
				index[j]   = index[j-1];
				index[j-1] = t;
				change     = 1;
			}
		}
		if (!change)
			return;
	}
} // Sort

/* ----- ElementString ----- */
Int_t
XSElementList::ElementString(Int_t Z, char *buf)
{
	switch (sortBy) {
		case XSEL_SORTBY_NAME:
			sprintf(buf,"%-20s %-4s %d",
				XSelements->Name(Z),
				XSelements->Mnemonic(Z),
				Z);
			break;

		case XSEL_SORTBY_MNEMONIC:
			sprintf(buf,"%-4s %-20s %d",
				XSelements->Mnemonic(Z),
				XSelements->Name(Z),
				Z);
			break;

		case XSEL_SORTBY_Z:
			sprintf(buf,"%3d   %-4s %-20s",
				Z,
				XSelements->Mnemonic(Z),
				XSelements->Name(Z));
			break;
	}
	return Z;
} // ElementString

/* ----- GetZ ------- */
UInt_t
XSElementList::GetZ( TGTextLBEntry *entry )
{
	if (entry == NULL) return 0;

	const TGString	*txt = entry->GetText();
	if (txt == NULL) return 0;

	const char	*str = txt->GetString();
	if (str == NULL) return 0;

	// Search the string for the Z
	while (*str && !isdigit(*str))
		str++;
	return atol(str);
} // GetZ

/* ----- SelectZ ----- */
void
XSElementList::SelectZ( UInt_t Z )
{
	TGTextLBEntry	*f;
	TGFrameElement	*el;

	if (Z<1 || Z>XSelements->GetSize()) return;

	TGLBContainer *ct = (TGLBContainer *) fVport->GetContainer();

	TIter	next(ct->GetList());
	while ((el = (TGFrameElement *) next())) {
		f = (TGTextLBEntry *) el->fFrame;
		if (GetZ(f) == Z) {
			f->Activate(kTRUE);
			return;
		}
	}
	return;
} // SelectZ

/* ----- CurrentZ ----- */
UInt_t
XSElementList::CurrentZ( )
{
        TGTextLBEntry	*entry = (TGTextLBEntry*)(GetSelectedEntry());
	return	GetZ(entry);
} // CurrentZ
