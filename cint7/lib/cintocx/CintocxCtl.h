// CintocxCtl.h : Declaration of the CCintocxCtrl OLE control class.

#include "G__ci.h"

/////////////////////////////////////////////////////////////////////////////
#include <afxmt.h>
#define QUEDEPTH 3
enum EVENTID { INITEVENT, EVALEVENT, TERMINATEEVENT };
class CCintocxCtrl;
class CintEventQue {
 private:
  int pushp;
  int popp;
  char Arg[QUEDEPTH][G__LONGLINE];
  CCintocxCtrl *eventorigin[QUEDEPTH];
  EVENTID Eid[QUEDEPTH];
  CCriticalSection CriticalSection;
  int IsEmpty() { return(pushp==popp); }
  int IsFull() { return(popp==(pushp+1)%QUEDEPTH); }
 public:
  CintEventQue() ;
  void Reset() ;
  int Push(char* buf,CCintocxCtrl* origin,EVENTID id);
  int Pop(char* buf,CCintocxCtrl** porigin,EVENTID* pid);
};


/////////////////////////////////////////////////////////////////////////////
// CCintocxCtrl : See CintocxCtl.cpp for implementation.

class CCintocxCtrl : public COleControl
{
	DECLARE_DYNCREATE(CCintocxCtrl)

// Constructor
public:
	CCintocxCtrl();

// Overrides

	// Drawing function
	virtual void OnDraw(
				CDC* pdc, const CRect& rcBounds, const CRect& rcInvalid);

	// Persistence
	virtual void DoPropExchange(CPropExchange* pPX);

	// Reset control state
	virtual void OnResetState();

// Implementation
protected:
	~CCintocxCtrl();

	DECLARE_OLECREATE_EX(CCintocxCtrl)    // Class factory and guid
	DECLARE_OLETYPELIB(CCintocxCtrl)      // GetTypeInfo
	DECLARE_PROPPAGEIDS(CCintocxCtrl)     // Property page IDs
	DECLARE_OLECTLTYPE(CCintocxCtrl)		// Type name and misc status

// Message maps
	//{{AFX_MSG(CCintocxCtrl)
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg void OnDestroy();
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()

// Dispatch maps
	//{{AFX_DISPATCH(CCintocxCtrl)
	CString m_result;
	afx_msg void OnResultChanged();
	afx_msg long Eval(LPCTSTR expr);
	afx_msg long Interrupt();
	afx_msg long Init(LPCTSTR com);
	afx_msg BOOL IsIdle();
	afx_msg long Stepmode();
	afx_msg long Terminate();
	afx_msg void Reset();
	//}}AFX_DISPATCH
	DECLARE_DISPATCH_MAP()

	afx_msg void AboutBox();

// Event maps
	//{{AFX_EVENT(CCintocxCtrl)
	void FireEvalDone()
		{FireEvent(eventidEvalDone,EVENT_PARAM(VTS_NONE));}
	//}}AFX_EVENT
	DECLARE_EVENT_MAP()

// Dispatch and event IDs
public:
	enum {
	//{{AFX_DISP_ID(CCintocxCtrl)
	dispidResult = 1L,
	dispidEval = 2L,
	dispidInterrupt = 3L,
	dispidInit = 4L,
	dispidIsIdle = 5L,
	dispidStepmode = 6L,
	dispidTerminate = 7L,
	dispidReset = 8L,
	eventidEvalDone = 1L,
	//}}AFX_DISP_ID
	};

// added by hand
private:
  static int IsBusy;
  static int IsCintThreadActive;

#ifndef WIN32THREAD
#else
  static DWORD CintThreadID;
  static HANDLE CintThreadHandle;
#endif

#ifndef WIN32EVENT
  static CEvent CintEvent;
#else
  static HANDLE CintEvent;
#endif

  static CintEventQue EventQue;
  static CCriticalSection CriticalSection;

  // start cint thread
  static void StartCintThread();
  // thread body
  static DWORD WINAPI CintThread(LPVOID lpvThreadParm);
};
