/*
 * $Header$
 * $Log$
 */

#ifndef __XSELEMENT_DLG_H
#define __XSELEMENT_DLG_H

#include <TGTab.h>
#include <TGFrame.h>
#include <TGButton.h>
#include <TGLayout.h>

#include "XSElementList.h"
#include "XSPeriodicTable.h"

/* =========== XSElementDlg ============== */
class XSElementDlg : public TGTransientFrame
{
private:
	UInt_t			*selectedZ;

	XSPeriodicTable		*pTable;
	TGTab			*tabMenu;
	TGButton		*okButton,
				*closeButton;
	TGCompositeFrame	*buttonFrame,
				*nameFrame,
				*mnemonicFrame,
				*zFrame;
	XSElementList		*nameListBox,
				*mnemonicListBox,
				*zListBox;
	TGLayoutHints		*buttonLayoutHints,
				*frameLayoutHints,
				*lHints,
				*lHints2;

public:
	XSElementDlg(const TGWindow *p, const TGWindow *main,
			UInt_t *retZ, UInt_t w=600, UInt_t h=350);
	~XSElementDlg();

	virtual void	CloseWindow();
	virtual Bool_t	ProcessButton(Long_t param);
	virtual Bool_t	ProcessMessage(Long_t msg,
				Long_t param1, Long_t param2);

	//ClassDef(XSElementDlg,1)
}; // XSElementDlg

#endif
