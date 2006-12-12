// CintocxCtl.cpp : Implementation of the CCintocxCtrl OLE control class.

#include "stdafx.h"
#include "cintocx.h"
#include "CintocxCtl.h"
#include "CintocxPpg.h"


#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
CintEventQue::CintEventQue() { Reset(); }
void CintEventQue::Reset() { 
  CriticalSection.Lock(10); // timeout 10sec
  pushp=popp=0; 
  CriticalSection.Unlock();
}

int CintEventQue::Push(char* buf,CCintocxCtrl* origin,EVENTID id) {
  int flag;
  CriticalSection.Lock(10); // timeout 10sec
  if(!IsFull()) {
    strcpy(Arg[pushp],buf);
    eventorigin[pushp]=origin;
    Eid[pushp]=id;
    pushp = (pushp+1)%QUEDEPTH;
    flag=0;
  }
  else {
    flag=1;
  }
  CriticalSection.Unlock();
  return(flag);
}

int CintEventQue::Pop(char* buf,CCintocxCtrl** porigin,EVENTID* pid) {
  int flag;
  CriticalSection.Lock(10); // timeout 10sec
  if(!IsEmpty()) {
    strcpy(Arg[pushp],buf);
    strcpy(buf,Arg[popp]);
    *porigin=eventorigin[popp];
    *pid=Eid[popp];
    popp = (popp+1)%QUEDEPTH;
    flag=0;
  }
  else {
    flag=1;
  }
  CriticalSection.Unlock();
  return(flag);
}


/////////////////////////////////////////////////////////////////////////////

IMPLEMENT_DYNCREATE(CCintocxCtrl, COleControl)


/////////////////////////////////////////////////////////////////////////////
// Message map

BEGIN_MESSAGE_MAP(CCintocxCtrl, COleControl)
	//{{AFX_MSG_MAP(CCintocxCtrl)
	ON_WM_CREATE()
	ON_WM_DESTROY()
	//}}AFX_MSG_MAP
	ON_OLEVERB(AFX_IDS_VERB_EDIT, OnEdit)
	ON_OLEVERB(AFX_IDS_VERB_PROPERTIES, OnProperties)
END_MESSAGE_MAP()


/////////////////////////////////////////////////////////////////////////////
// Dispatch map

BEGIN_DISPATCH_MAP(CCintocxCtrl, COleControl)
	//{{AFX_DISPATCH_MAP(CCintocxCtrl)
	DISP_PROPERTY_NOTIFY(CCintocxCtrl, "Result", m_result, OnResultChanged, VT_BSTR)
	DISP_FUNCTION(CCintocxCtrl, "Eval", Eval, VT_I4, VTS_BSTR)
	DISP_FUNCTION(CCintocxCtrl, "Interrupt", Interrupt, VT_I4, VTS_NONE)
	DISP_FUNCTION(CCintocxCtrl, "Init", Init, VT_I4, VTS_BSTR)
	DISP_FUNCTION(CCintocxCtrl, "IsIdle", IsIdle, VT_BOOL, VTS_NONE)
	DISP_FUNCTION(CCintocxCtrl, "Stepmode", Stepmode, VT_I4, VTS_NONE)
	DISP_FUNCTION(CCintocxCtrl, "Terminate", Terminate, VT_I4, VTS_NONE)
	DISP_FUNCTION(CCintocxCtrl, "Reset", Reset, VT_EMPTY, VTS_NONE)
	//}}AFX_DISPATCH_MAP
	DISP_FUNCTION_ID(CCintocxCtrl, "AboutBox", DISPID_ABOUTBOX, AboutBox, VT_EMPTY, VTS_NONE)
END_DISPATCH_MAP()


/////////////////////////////////////////////////////////////////////////////
// Event map

BEGIN_EVENT_MAP(CCintocxCtrl, COleControl)
	//{{AFX_EVENT_MAP(CCintocxCtrl)
	EVENT_CUSTOM("EvalDone", FireEvalDone, VTS_NONE)
	//}}AFX_EVENT_MAP
END_EVENT_MAP()


/////////////////////////////////////////////////////////////////////////////
// Property pages

// TODO: Add more property pages as needed.  Remember to increase the count!
BEGIN_PROPPAGEIDS(CCintocxCtrl, 1)
	PROPPAGEID(CCintocxPropPage::guid)
END_PROPPAGEIDS(CCintocxCtrl)


/////////////////////////////////////////////////////////////////////////////
// Initialize class factory and guid

IMPLEMENT_OLECREATE_EX(CCintocxCtrl, "CINTOCX.CintocxCtrl.1",
	0x3a1e343, 0xc39c, 0x11d0, 0xba, 0xdd, 0xf4, 0x35, 0xb0, 0x60, 0x51, 0xd6)


/////////////////////////////////////////////////////////////////////////////
// Type library ID and version

IMPLEMENT_OLETYPELIB(CCintocxCtrl, _tlid, _wVerMajor, _wVerMinor)


/////////////////////////////////////////////////////////////////////////////
// Interface IDs

const IID BASED_CODE IID_DCintocx =
		{ 0x3a1e341, 0xc39c, 0x11d0, { 0xba, 0xdd, 0xf4, 0x35, 0xb0, 0x60, 0x51, 0xd6 } };
const IID BASED_CODE IID_DCintocxEvents =
		{ 0x3a1e342, 0xc39c, 0x11d0, { 0xba, 0xdd, 0xf4, 0x35, 0xb0, 0x60, 0x51, 0xd6 } };


/////////////////////////////////////////////////////////////////////////////
// Control type information

static const DWORD BASED_CODE _dwCintocxOleMisc =
	OLEMISC_INVISIBLEATRUNTIME |
	OLEMISC_ACTIVATEWHENVISIBLE |
	OLEMISC_SETCLIENTSITEFIRST |
	OLEMISC_INSIDEOUT |
	OLEMISC_CANTLINKINSIDE |
	OLEMISC_RECOMPOSEONRESIZE;

IMPLEMENT_OLECTLTYPE(CCintocxCtrl, IDS_CINTOCX, _dwCintocxOleMisc)


/////////////////////////////////////////////////////////////////////////////
// CCintocxCtrl::CCintocxCtrlFactory::UpdateRegistry -
// Adds or removes system registry entries for CCintocxCtrl

BOOL CCintocxCtrl::CCintocxCtrlFactory::UpdateRegistry(BOOL bRegister)
{
	if (bRegister)
		return AfxOleRegisterControlClass(
			AfxGetInstanceHandle(),
			m_clsid,
			m_lpszProgID,
			IDS_CINTOCX,
			IDB_CINTOCX,
			TRUE,                       //  Insertable
			_dwCintocxOleMisc,
			_tlid,
			_wVerMajor,
			_wVerMinor);
	else
		return AfxOleUnregisterClass(m_clsid, m_lpszProgID);
}


/////////////////////////////////////////////////////////////////////////////
// CCintocxCtrl::CCintocxCtrl - Constructor

CCintocxCtrl::CCintocxCtrl()
{
	InitializeIIDs(&IID_DCintocx, &IID_DCintocxEvents);

	// Start cint thread only once
	if(0==IsCintThreadActive) StartCintThread();
}


/////////////////////////////////////////////////////////////////////////////
// CCintocxCtrl::~CCintocxCtrl - Destructor

CCintocxCtrl::~CCintocxCtrl()
{
	// TODO: Cleanup your control's instance data here.
}


/////////////////////////////////////////////////////////////////////////////
// CCintocxCtrl::OnDraw - Drawing function

void CCintocxCtrl::OnDraw(
			CDC* pdc, const CRect& rcBounds, const CRect& rcInvalid)
{
	// TODO: Replace the following code with your own drawing code.
	//pdc->FillRect(rcBounds, CBrush::FromHandle((HBRUSH)GetStockObject(WHITE_BRUSH)));
	//pdc->Ellipse(rcBounds);
#if 0
        // this part has problem with VC++6.0
	pdc->ExtTextOut(rcBounds.left,
				    rcBounds.top,
					ETO_CLIPPED,
					rcBounds,
					"CINT OCX(hidden)",
					16,
					NULL);
#endif
}


/////////////////////////////////////////////////////////////////////////////
// CCintocxCtrl::DoPropExchange - Persistence support

void CCintocxCtrl::DoPropExchange(CPropExchange* pPX)
{
	ExchangeVersion(pPX, MAKELONG(_wVerMinor, _wVerMajor));
	COleControl::DoPropExchange(pPX);

	// TODO: Call PX_ functions for each persistent custom property.
}


/////////////////////////////////////////////////////////////////////////////
// CCintocxCtrl::OnResetState - Reset control to default state

void CCintocxCtrl::OnResetState()
{
	COleControl::OnResetState();  // Resets defaults found in DoPropExchange

	// TODO: Reset any other control state here.
	Reset();
}


/////////////////////////////////////////////////////////////////////////////
// CCintocxCtrl::AboutBox - Display an "About" box to the user

void CCintocxCtrl::AboutBox()
{
	CDialog dlgAbout(IDD_ABOUTBOX_CINTOCX);
	dlgAbout.DoModal();
}


/////////////////////////////////////////////////////////////////////////////
// CCintocxCtrl message handlers


/////////////////////////////////////////////////////////////////////////////
long CCintocxCtrl::Init(LPCTSTR com) 
{
	// schedule initialize cint event
	if(0==EventQue.Push((char*)com,this,INITEVENT)) {
	  if(0==IsCintThreadActive) StartCintThread();
#ifndef WIN32EVENT
	  CintEvent.PulseEvent();
#else
	  PulseEvent(CintEvent);
#endif
	  return(0);
	}
	else {
#ifndef WIN32EVENT
	  CintEvent.PulseEvent();
#else
	  PulseEvent(CintEvent);
#endif
	  MessageBeep((WORD)(-1));
	  return(1);
	}
}

/////////////////////////////////////////////////////////////////////////////
long CCintocxCtrl::Eval(LPCTSTR expr) 
{
	// Push event que and notify it to CintThread
	if(0==EventQue.Push((char*)expr,this,EVALEVENT)) {
	  if(0==IsCintThreadActive) StartCintThread();
#ifndef WIN32EVENT
	  CintEvent.PulseEvent();
#else
	  PulseEvent(CintEvent);
#endif
	  return(0);
	}
	else {
#ifndef WIN32EVENT
	  CintEvent.PulseEvent();
#else
	  PulseEvent(CintEvent);
#endif
	  MessageBeep((WORD)(-1));
	  return(1);
	}
}

/////////////////////////////////////////////////////////////////////////////
#define TERMEVENT
long CCintocxCtrl::Terminate() 
{
	// TODO: Add your dispatch handler code here

	// terminates the application
#ifdef TERMEVENT
	// following scheme never worked
	if(IsCintThreadActive) {
	  if(0==EventQue.Push("",NULL,TERMINATEEVENT)) {
#ifndef WIN32EVENT
	  CintEvent.PulseEvent();
#else
	  PulseEvent(CintEvent);
#endif
	    return(0);
	  }
	  else {
	    MessageBeep((WORD)(-1));
	    return(1);
	  }
	}
#else
	exit(0);
#endif

	return(0);
}

/////////////////////////////////////////////////////////////////////////////
long CCintocxCtrl::Stepmode() 
{
	CriticalSection.Lock(10); // timeout 10sec
	G__AllocConsole();
	G__stepmode(0); // set step mode and wait until one interpreted 
	G__breakkey(0); // statement finishes
	CriticalSection.Unlock();
	return 0;
}

/////////////////////////////////////////////////////////////////////////////
long CCintocxCtrl::Interrupt() 
{
	if(IsBusy) {
	  CriticalSection.Lock(INFINITE); // timeout infinite
	  G__AllocConsole();
	  G__stepmode(1); // 
	  G__breakkey(0); // All threads stops here
	  G__stepmode(0); // reset step mode 
	  CriticalSection.Unlock();
	  return 0;
	}
	else {
	  MessageBeep((WORD)(-1));
	  return 1;
	}
}

/////////////////////////////////////////////////////////////////////////////
BOOL CCintocxCtrl::IsIdle() 
{
	return(!IsBusy);
}


/////////////////////////////////////////////////////////////////////////////
int CCintocxCtrl::OnCreate(LPCREATESTRUCT lpCreateStruct) 
{
	if (COleControl::OnCreate(lpCreateStruct) == -1)
		return -1;
	
	return 0;
}

/////////////////////////////////////////////////////////////////////////////
void CCintocxCtrl::OnDestroy() 
{
	COleControl::OnDestroy();

	// schedule destroy event
	EventQue.Push(NULL,this,TERMINATEEVENT);
#ifndef WIN32EVENT
	CintEvent.PulseEvent();
#else
	PulseEvent(CintEvent);
#endif
}

/////////////////////////////////////////////////////////////////////////////
void CCintocxCtrl::OnResultChanged() 
{
	// TODO: Add notification handler code

	SetModifiedFlag();
}



/*********************************************************************
* value2string()
*********************************************************************/
void value2string(char *arg,G__value *result)
{
  switch(result->type) {
  case 'C':
    strcpy(arg,(char*)result->obj.i);
    break;
  case 'd':
  case 'f':
    sprintf(arg,"%g",result->obj.d);
    break;
  default:
    if(isupper(result->type)) sprintf(arg,"0x%x",result->obj.i);
    else                      sprintf(arg,"%d",result->obj.i);
  }
}

/*********************************************************************
* static member function for CINT Thread
*********************************************************************/
/////////////////////////////////////////////////////////////////////////////
void CCintocxCtrl::StartCintThread()
{
	if(0==IsCintThreadActive) {
#ifndef WIN32THREAD
	  AfxBeginThread((AFX_THREADPROC)CintThread,(LPVOID)NULL);
#else
    	  CintThreadHandle = CreateThread(NULL,0
			    	   ,(LPTHREAD_START_ROUTINE)CintThread
			    	   ,(LPVOID)NULL,0
			    	   ,&CintThreadID);
#endif
	  G__setautoconsole(0);
	  IsCintThreadActive=1;
	}
}


/////////////////////////////////////////////////////////////////////////////
DWORD WINAPI CCintocxCtrl::CintThread(LPVOID lpvThreadParm)
{
   DWORD dwResult=0;
   int flag=1;
   G__value result;
   int iresult;
   char arg[G__LONGLINE];
   CCintocxCtrl *origin;
   EVENTID eid;

#ifndef WIN32EVENT
   CSingleLock WaitForCintEvent(&CintEvent);
#else
   CintEvent = CreateEvent(NULL,FALSE,FALSE,"CintEvent");
#endif

   while(flag) {
      // Wait for event from GUI thread
      IsBusy=0;
#ifndef WIN32EVENT
      WaitForCintEvent.Lock(INFINITE);
#else
      WaitForSingleObject(CintEvent,INFINITE);
#endif
      IsBusy=1;

      while(0==EventQue.Pop(arg,&origin,&eid)) {
        switch(eid) {
        case INITEVENT:
	  // start cint with specified argument
	  iresult=G__init_cint(arg);  
	  sprintf(arg,"%d",iresult); 
	  origin->m_result = arg;
	  break;
        case EVALEVENT:
	  // start cint with default if not started already
	  if(!G__getcintready()) iresult=G__init_cint("cint");  
	  // evaluate scheduled expression
	  result=G__calc(arg);
	  // translate return value to string
	  value2string(arg,&result);
	  origin->m_result = arg;
	  break;
        case TERMINATEEVENT:
	  // terminate cint and kill thread
	  G__scratch_all();
	  goto terminate_thread;
        }
	// notify evaldone to GUI thread
        origin->FireEvalDone();
      }
   } // while(flag)

terminate_thread:
   IsBusy=0;
   IsCintThreadActive = 0 ;
#ifndef WIN32EVENT
   WaitForCintEvent.Unlock(3);
#else
   CloseHandle(CintEvent);
#endif

   G__FreeConsole();
   
   return(dwResult);   
}

int CCintocxCtrl::IsBusy = 0;
int CCintocxCtrl::IsCintThreadActive = 0;

#ifndef WIN32THREAD
#else
DWORD CCintocxCtrl::CintThreadID;
HANDLE CCintocxCtrl::CintThreadHandle;
#endif

#ifndef WIN32EVENT
CEvent CCintocxCtrl::CintEvent;
#else
HANDLE CCintocxCtrl::CintEvent;
#endif

CintEventQue CCintocxCtrl::EventQue;
CCriticalSection CCintocxCtrl::CriticalSection;



void CCintocxCtrl::Reset() 
{
	// Reset EventQue asynchrnously, can be dangerous
	EventQue.Reset();
}

