/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/* =======================================================
	Simple Win32API Application for Drawing Graphics.

	WinMain.c: Defines the entry point for the application.
	Last modified Time-stamp: <02/01/13 09:48:12 hata>            
	Proposed by K.Hata <kazuhiko.hata@nifty.ne.jp>
========================================================== */

#define	STRICT

#include "StdAfx.h"
#include "resource.h"
#include "WndProc.h"

#include "G__ci.h"          /* Cint header file */

#define MAX_LOADSTRING 100

TCHAR szTitle[MAX_LOADSTRING]="graph01";   	// The title bar text
TCHAR szWindowClass[MAX_LOADSTRING]="MainWindowClass";		// Window class name

// Global Variables:
HINSTANCE hInst;			// current instance

// Foward declarations of functions included in this code module:
ATOM RegMainWindowClass( HINSTANCE );
BOOL InitInstance( HINSTANCE, int );

// Entry point of application:
int APIENTRY WinMain(HINSTANCE hInstance,
                     HINSTANCE hPrevInstance,
                     LPSTR     lpCmdLine,
                     int       nCmdShow )
{
 	// TODO: Place code here.
	MSG				msg;
	HACCEL hAccelTable;
	extern void G__c_setup(); /* defined in G__clink.c */

#ifdef TEST
	G__init_cint("cint -T Script.c"); /* initialize Cint */
#else
	G__init_cint("cint Script.c"); /* initialize Cint */
#endif
	G__c_setup(); /* Initialize dictionary that includes DrawRect4 */

	// Initialize global strings
	LoadString(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
	LoadString(hInstance, IDC_MAIN_WINDOW_CLASS,   szWindowClass, MAX_LOADSTRING);

	if(RegMainWindowClass(hInstance)==0)	// Register Window class
		return	0;

#if TEST
	/* Allocate Console window for debugging purpose */
	G__FreeConsole();
	G__AllocConsole();
#endif

	// Perform application initialization:
	if( !InitInstance( hInstance, nCmdShow ) ) 
		return FALSE;

	hAccelTable = LoadAccelerators(hInstance, (LPCTSTR)IDC_ACCEL1);

	// Main message loop:
	while(GetMessage(&msg,NULL,0,0))	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}

	G__scratch_all(); /* Clean up Cint */

	return	msg.wParam;
}

//
//  FUNCTION: MyRegisterClass()
//
//  PURPOSE: Registers the window class.
//
//  COMMENTS:
//
//    This function and its usage is only necessary if you want this code
//    to be compatible with Win32 systems prior to the 'RegisterClassEx'
//    function that was added to Windows 95. It is important to call this function
//    so that the application will get 'well formed' small icons associated
//    with it.
//
ATOM RegMainWindowClass( HINSTANCE hInstance )
{
	WNDCLASSEX wcex;

	wcex.cbSize = sizeof(WNDCLASSEX); 
	wcex.style			= CS_HREDRAW | CS_VREDRAW ; 
	wcex.lpfnWndProc	= (WNDPROC)WndProc;
	wcex.cbClsExtra		= 0;
	wcex.cbWndExtra		= 0;
	wcex.hInstance		= hInstance;
	wcex.hIcon			=  LoadIcon(hInstance, (LPCTSTR)IDI_APP);
	wcex.hCursor		= LoadCursor(NULL, IDC_ARROW);
	wcex.hbrBackground	= (HBRUSH)(COLOR_WINDOW+1);
	wcex.lpszMenuName	= (LPCSTR)IDC_MAIN_MENU;
	wcex.lpszClassName	= szWindowClass;
	wcex.hIconSm		= LoadIcon(wcex.hInstance, (LPCTSTR)IDI_SMALL);

	return RegisterClassEx( &wcex );
}

//
//   FUNCTION: InitInstance(HANDLE, int)
//
//   PURPOSE: Saves instance handle and creates main window
//
//   COMMENTS:
//
//        In this function, we save the instance handle in a global variable and
//        create and display the main program window.
//
BOOL InitInstance( HINSTANCE hInstance, int nCmdShow )
{
   HWND hWnd;

   hInst = hInstance; // Store instance handle in our global variable
 
   hWnd = CreateWindow(szWindowClass,			// pointer to registered class name
						szTitle,				// pointer to window name
						WS_OVERLAPPEDWINDOW,    // | WS_VISIBLE | WS_VSCROLL | ES_MULTILINE,
						                        // window style
						CW_USEDEFAULT,          // horizontal position of window
						0,                      // vertical position of window
						CW_USEDEFAULT,,         // window width
						0,                      // window height
						NULL,                   // handle to parent or owner window
						NULL,                   //handle to menu or child-window identifier
						hInstance,				// handle to application instance
						NULL);                  // pointer to window-creation data

   if( !hWnd ) 
   {
      return FALSE;
   }

	ShowWindow( hWnd,nCmdShow );
	UpdateWindow( hWnd );

   return TRUE;
}
