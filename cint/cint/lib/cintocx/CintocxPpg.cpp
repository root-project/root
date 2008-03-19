// CintocxPpg.cpp : Implementation of the CCintocxPropPage property page class.

#include "stdafx.h"
#include "cintocx.h"
#include "CintocxPpg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif


IMPLEMENT_DYNCREATE(CCintocxPropPage, COlePropertyPage)


/////////////////////////////////////////////////////////////////////////////
// Message map

BEGIN_MESSAGE_MAP(CCintocxPropPage, COlePropertyPage)
	//{{AFX_MSG_MAP(CCintocxPropPage)
	// NOTE - ClassWizard will add and remove message map entries
	//    DO NOT EDIT what you see in these blocks of generated code !
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()


/////////////////////////////////////////////////////////////////////////////
// Initialize class factory and guid

IMPLEMENT_OLECREATE_EX(CCintocxPropPage, "CINTOCX.CintocxPropPage.1",
	0x3a1e344, 0xc39c, 0x11d0, 0xba, 0xdd, 0xf4, 0x35, 0xb0, 0x60, 0x51, 0xd6)


/////////////////////////////////////////////////////////////////////////////
// CCintocxPropPage::CCintocxPropPageFactory::UpdateRegistry -
// Adds or removes system registry entries for CCintocxPropPage

BOOL CCintocxPropPage::CCintocxPropPageFactory::UpdateRegistry(BOOL bRegister)
{
	if (bRegister)
		return AfxOleRegisterPropertyPageClass(AfxGetInstanceHandle(),
			m_clsid, IDS_CINTOCX_PPG);
	else
		return AfxOleUnregisterClass(m_clsid, NULL);
}


/////////////////////////////////////////////////////////////////////////////
// CCintocxPropPage::CCintocxPropPage - Constructor

CCintocxPropPage::CCintocxPropPage() :
	COlePropertyPage(IDD, IDS_CINTOCX_PPG_CAPTION)
{
	//{{AFX_DATA_INIT(CCintocxPropPage)
	// NOTE: ClassWizard will add member initialization here
	//    DO NOT EDIT what you see in these blocks of generated code !
	//}}AFX_DATA_INIT
}


/////////////////////////////////////////////////////////////////////////////
// CCintocxPropPage::DoDataExchange - Moves data between page and properties

void CCintocxPropPage::DoDataExchange(CDataExchange* pDX)
{
	//{{AFX_DATA_MAP(CCintocxPropPage)
	// NOTE: ClassWizard will add DDP, DDX, and DDV calls here
	//    DO NOT EDIT what you see in these blocks of generated code !
	//}}AFX_DATA_MAP
	DDP_PostProcessing(pDX);
}


/////////////////////////////////////////////////////////////////////////////
// CCintocxPropPage message handlers
