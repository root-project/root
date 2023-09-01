// asview.cpp : Defines the entry point for the application.
//

#include "stdafx.h"

void usage()
{
	printf( "Usage: asview [-h]|[image]\n");
	printf( "Where: image - filename of the image to display.\n");
}

LRESULT CALLBACK MyWindowProc( HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam );
void show_system_error();


HINSTANCE hinst; 
HWND hWnd = NULL ;
void *bmbits = NULL ;
BITMAPINFO *bmi = NULL ;


int APIENTRY 
WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR lpCmdLine, int nCmdShow)
{
	char *image_file = "../apps/rose512.jpg" ;
	
//	char *image_file = "../apps/fore.xpm" ;
	ASImage *im ;
	WNDCLASSEX wnd_class ; 
	ASVisual *asv ;
	MSG msg; 

	if( lpCmdLine != NULL && strncmp( lpCmdLine, "-h", 2 ) == 0 )
	{
		usage();
		return 0;
	}else if( lpCmdLine  != NULL && strlen(lpCmdLine) > 0 ) 
		image_file = lpCmdLine ;
	else
		usage();

	if( image_file[0] == '\"' ) 
	{
		int i = 0;
		while( image_file[i+1] != '\0' && image_file[i+1] !=  '\"')
		{
			image_file[i] = image_file[i+1] ; 
			++i ;
		}
		image_file[i] = '\0' ;
	}

	asv = create_asvisual( NULL, 0, 0, NULL );

	im = file2ASImage( image_file, 0xFFFFFFFF, SCREEN_GAMMA, 0, NULL );
	if( im == NULL ) 
	{
		MessageBox( NULL, "Unable to load image from file.", image_file, MB_OK | MB_ICONINFORMATION );
		return 0 ;
	}
	
	/* converting result into BMP file ( as an example ) */
	/* ASImage2file( im, NULL, "asview.bmp", ASIT_Bmp, NULL ); */
	
	/* The following could be used to dump JPEG version of the image into
	 * stdout : */
	/* ASImage2file( im, NULL, NULL, ASIT_Jpeg, NULL ); */


	bmbits = NULL ;
	// Convert ASImage into DIB: 
	bmi = ASImage2DBI( asv, im, 0, 0, im->width, im->height, &bmbits );

	if( bmi == NULL ) 
	{
		MessageBox( NULL, "Failed to convert image into Windows bitmap.", image_file, MB_OK | MB_ICONINFORMATION );
		return 0 ;
	}

	memset( &wnd_class, 0x00, sizeof(wnd_class));
	wnd_class.cbSize = sizeof(wnd_class);
	wnd_class.hInstance = hInstance ;
	wnd_class.lpszClassName = "ASView" ;
	wnd_class.lpfnWndProc = MyWindowProc ;
    wnd_class.hIcon = LoadIcon((HINSTANCE) NULL, IDI_APPLICATION); 
    wnd_class.hCursor = LoadCursor((HINSTANCE) NULL, IDC_ARROW); 
	wnd_class.hbrBackground = (struct HBRUSH__ *)GetStockObject(WHITE_BRUSH); 
        
	if( !RegisterClassEx( &wnd_class ) ) 
	{
		show_system_error();
		return 0 ;
	}

	hinst = hInstance ;
	/* Now let us create a window and display image in that window : */
	hWnd = CreateWindow( "ASView",  image_file,
						 WS_OVERLAPPEDWINDOW,  
						 CW_USEDEFAULT, CW_USEDEFAULT, 
						 bmi->bmiHeader.biWidth, bmi->bmiHeader.biHeight,          
						 (HWND) NULL, (HMENU)NULL, hinst, NULL );
	if (!hWnd) 
	{
		show_system_error();
        return FALSE; 
	}

    // Show the window and paint its contents. 
    ShowWindow(hWnd, nCmdShow); 
    UpdateWindow(hWnd); 

    // Start the message loop. 
    while (GetMessage(&msg, hWnd, 0, 0) > 0 ) 
    { 
        TranslateMessage(&msg); 
        DispatchMessage(&msg); 
	} 
	return 0;
}

LRESULT CALLBACK 
MyWindowProc( HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam )
{
	if( uMsg == WM_PAINT )
	{
		// Paint image in responce to WM_PAINT event : 
		if( bmi != NULL && bmbits != NULL ) 
		{
			PAINTSTRUCT ps ;
			HDC dc  = BeginPaint(hWnd, &ps );
			StretchDIBits(	dc,                // handle to device context
							0, 0, bmi->bmiHeader.biWidth, bmi->bmiHeader.biHeight, 
							0, 0, bmi->bmiHeader.biWidth, bmi->bmiHeader.biHeight,  
							bmbits, bmi, DIB_RGB_COLORS, SRCCOPY );	
			EndPaint(hWnd, &ps );
  
		}
		return 0;
	}
	
	return DefWindowProc( hwnd, uMsg, wParam, lParam ) ;
}

void
show_system_error()
{
	LPVOID lpMsgBuf;
	FormatMessage( 
		FORMAT_MESSAGE_ALLOCATE_BUFFER | 
		FORMAT_MESSAGE_FROM_SYSTEM | 
		FORMAT_MESSAGE_IGNORE_INSERTS,
		NULL, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR) &lpMsgBuf,
		0, NULL );
	MessageBox( NULL, (LPCTSTR)lpMsgBuf, "ASView System Error", MB_OK | MB_ICONINFORMATION );
	// Free the buffer.
	LocalFree( lpMsgBuf );
}


