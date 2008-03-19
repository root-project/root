// CintocxPpg.h : Declaration of the CCintocxPropPage property page class.

////////////////////////////////////////////////////////////////////////////
// CCintocxPropPage : See CintocxPpg.cpp.cpp for implementation.

class CCintocxPropPage : public COlePropertyPage
{
	DECLARE_DYNCREATE(CCintocxPropPage)
	DECLARE_OLECREATE_EX(CCintocxPropPage)

// Constructor
public:
	CCintocxPropPage();

// Dialog Data
	//{{AFX_DATA(CCintocxPropPage)
	enum { IDD = IDD_PROPPAGE_CINTOCX };
		// NOTE - ClassWizard will add data members here.
		//    DO NOT EDIT what you see in these blocks of generated code !
	//}}AFX_DATA

// Implementation
protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

// Message maps
protected:
	//{{AFX_MSG(CCintocxPropPage)
		// NOTE - ClassWizard will add and remove member functions here.
		//    DO NOT EDIT what you see in these blocks of generated code !
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()

};
