/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file lib/win32api/winfunc.h
 ************************************************************************
 * Description:
 *  Create WIN32 API function interface
 ************************************************************************
 * Copyright(c) 1995~2001  Masaharu Goto (MXJ02154@niftyserve.or.jp)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

/***********************************************************************
* Author's Personal note
*  Check out for WinInet.h  Ftp* functions
***********************************************************************/


#ifndef G__WINFUNC_H
#define G__WINFUNC_H

#ifdef __MAKECINT__

#pragma link off all functions;
#pragma link off all classes;
#pragma link off all globals;
#pragma link off all typedefs;

#pragma link C global G__WINFUNC_H;

/***********************************************************************
* Types
***********************************************************************/
#pragma link C class tagFILETIME;
#pragma link C class _FILETIME;
#pragma link C typedef FILETIME;

#pragma link C class _SYSTEMTIME;
#pragma link C typedef SYSTEMTIME;
#pragma link C typedef PSYSTEMTIME;
#pragma link C typedef LPSYSTEMTIME;

#pragma link C class _TIME_ZONE_INFORMATION;
#pragma link C typedef TIME_ZONE_INFORMATION;
#pragma link C typedef PTIME_ZONE_INFORMATION;
#pragma link C typedef LPTIME_ZONE_INFORMATION;

#pragma link C class _COMMTIMEOUTS;
#pragma link C typedef COMMTIMEOUTS;
#pragma link C typedef LPCOMMTIMEOUTS;

#pragma link C class _DCB;
#pragma link C typedef DCB;
#pragma link C typedef LPDCB;

#pragma link C class _OVERLAPPED;
#pragma link C typedef OVERLAPPED;
#pragma link C typedef LPOVERLAPPED;

#pragma link C class _WIN32_FIND_DATA;
#pragma link C class _WIN32_FIND_DATAA;
#pragma link C class _WIN32_FIND_DATAW;
#pragma link C typedef WIN32_FIND_DATA;
#pragma link C typedef LPWIN32_FIND_DATA;

#pragma link C class _SECURITY_ATTRIBUTES;
#pragma link C typedef SECURITY_ATTRIBUTES;
#pragma link C typedef LPSECURITY_ATTRIBUTES;

#pragma link C typedef PTHREAD_START_ROUTINE;
#pragma link C typedef LPTHREAD_START_ROUTINE;

#pragma link C class RTL_CRITICAL_SECTION;
#pragma link C typedef CRITICAL_SECTION;
#pragma link C typedef LPCRITICAL_SECTION;

#pragma link C typedef LPCHOOSECOLOR;

#pragma link C typedef LPCHOOSEFONT;

#pragma link C typedef LPPAGESETUPDLG;

#pragma link C typedef LPPRINTDLG;

#pragma link C typedef LPFINDREPLACE;

#pragma link C typedef LPOVERLAPPED_COMPLETION_ROUTINE;

#pragma link C typedef BOOL;
#pragma link C typedef BOOLEAN;
#pragma link C typedef BYTE;
#pragma link C typedef DWORD;
#pragma link C typedef WORD;
#pragma link C typedef HANDLE;
#pragma link C typedef HINSTANCE;
#pragma link C typedef LPCVOID;
#pragma link C typedef UCHAR;
#pragma link C typedef USHORT;
#pragma link C typedef UINT;
#pragma link C typedef ULONG;
#pragma link C typedef HMODULE;
#pragma link C typedef LARGE_INTEGER;
#pragma link C typedef PVOID;
#pragma link C typedef WCHAR;
#pragma link C typedef DWORDLONG;
typedef LONG LRESULT;
#pragma link C typedef LRESULT;
#pragma link C typedef WPARAM;
#pragma link C typedef LPARAM;


/***********************************************************************
* Constants
***********************************************************************/
#define GENERIC_READ                (0x80000000) /* from WINNT.H */
#define GENERIC_WRITE               (0x40000000) /* from WINNT.H */
#define FILE_SHARE_READ             (0x00000001) /* from WINNT.H */
#define FILE_SHARE_WRITE            (0x00000002) /* from WINNT.H */
#define FILE_FLAG_SEQUENTIAL_SCAN   0x08000000
#define CREATE_NEW          1
#define CREATE_ALWAYS       2
#define OPEN_EXISTING       3
#define OPEN_ALWAYS         4
#define TRUNCATE_EXISTING   5

#ifndef _MAC
#define INVALID_HANDLE_VALUE        ((HANDLE)(-1))
#define DELETE                      0x00010000L

#define FILE_BEGIN                  0
#define FILE_CURRENT                1
#define FILE_END                    2
#endif

#define FILE_ATTRIBUTE_READONLY         0x00000001
#define FILE_ATTRIBUTE_HIDDEN           0x00000002
#define FILE_ATTRIBUTE_SYSTEM           0x00000004
#define FILE_ATTRIBUTE_DIRECTORY        0x00000010
#define FILE_ATTRIBUTE_ARCHIVE          0x00000020
#define FILE_ATTRIBUTE_NORMAL           0x00000080
#define FILE_ATTRIBUTE_TEMPORARY        0x00000100

#define FILE_FLAG_WRITE_THROUGH     0x80000000
#define FILE_FLAG_RANDOM_ACCESS     0x10000000

#define TIME_ZONE_ID_INVALID        0xFFFFFFFF
#define TIME_ZONE_ID_UNKNOWN        0
#define TIME_ZONE_ID_STANDARD       1
#define TIME_ZONE_ID_DAYLIGHT       2

#define INVALID_HANDLE_VALUE (HANDLE)-1
#define INVALID_FILE_SIZE (DWORD)0xFFFFFFFF

#define FILE_BEGIN           0
#define FILE_CURRENT         1
#define FILE_END             2

#define TIME_ZONE_ID_INVALID (DWORD)0xFFFFFFFF

#define WAIT_FAILED (DWORD)0xFFFFFFFF

#define DRIVE_UNKNOWN     0
#define DRIVE_NO_ROOT_DIR 1
#define DRIVE_REMOVABLE   2
#define DRIVE_FIXED       3
#define DRIVE_REMOTE      4
#define DRIVE_CDROM       5
#define DRIVE_RAMDISK     6


#define GetFreeSpace(w)                 (0x100000L)


#define FILE_TYPE_UNKNOWN   0x0000
#define FILE_TYPE_DISK      0x0001
#define FILE_TYPE_CHAR      0x0002
#define FILE_TYPE_PIPE      0x0003
#define FILE_TYPE_REMOTE    0x8000


#define STD_INPUT_HANDLE    (DWORD)-10
#define STD_OUTPUT_HANDLE   (DWORD)-11
#define STD_ERROR_HANDLE      (DWORD)-12

#define NOPARITY            0
#define ODDPARITY           1
#define EVENPARITY          2
#define MARKPARITY          3
#define SPACEPARITY         4

#define ONESTOPBIT          0
#define ONE5STOPBITS        1
#define TWOSTOPBITS         2

#define IGNORE              0       // Ignore signal
#define INFINITE            0xFFFFFFFF  // Infinite timeout

//
// Basud rates at which the communication device operates
//

#define CBR_110             110
#define CBR_300             300
#define CBR_600             600
#define CBR_1200            1200
#define CBR_2400            2400
#define CBR_4800            4800
#define CBR_9600            9600
#define CBR_14400           14400
#define CBR_19200           19200
#define CBR_38400           38400
#define CBR_56000           56000
#define CBR_57600           57600
#define CBR_115200          115200
#define CBR_128000          128000
#define CBR_256000          256000

//
// Error Flags
//

#define CE_RXOVER           0x0001  // Receive Queue overflow
#define CE_OVERRUN          0x0002  // Receive Overrun Error
#define CE_RXPARITY         0x0004  // Receive Parity Error
#define CE_FRAME            0x0008  // Receive Framing error
#define CE_BREAK            0x0010  // Break Detected
#define CE_TXFULL           0x0100  // TX Queue is full
#define CE_PTO              0x0200  // LPTx Timeout
#define CE_IOE              0x0400  // LPTx I/O Error
#define CE_DNS              0x0800  // LPTx Device not selected
#define CE_OOP              0x1000  // LPTx Out-Of-Paper
#define CE_MODE             0x8000  // Requested mode unsupported

#define IE_BADID            (-1)    // Invalid or unsupported id
#define IE_OPEN             (-2)    // Device Already Open
#define IE_NOPEN            (-3)    // Device Not Open
#define IE_MEMORY           (-4)    // Unable to allocate queues
#define IE_DEFAULT          (-5)    // Error in default parameters
#define IE_HARDWARE         (-10)   // Hardware Not Present
#define IE_BYTESIZE         (-11)   // Illegal Byte Size
#define IE_BAUDRATE         (-12)   // Unsupported BaudRate

//
// Events
//

#define EV_RXCHAR           0x0001  // Any Character received
#define EV_RXFLAG           0x0002  // Received certain character
#define EV_TXEMPTY          0x0004  // Transmitt Queue Empty
#define EV_CTS              0x0008  // CTS changed state
#define EV_DSR              0x0010  // DSR changed state
#define EV_RLSD             0x0020  // RLSD changed state
#define EV_BREAK            0x0040  // BREAK received
#define EV_ERR              0x0080  // Line status error occurred
#define EV_RING             0x0100  // Ring signal detected
#define EV_PERR             0x0200  // Printer error occured
#define EV_RX80FULL         0x0400  // Receive buffer is 80 percent full
#define EV_EVENT1           0x0800  // Provider specific event 1
#define EV_EVENT2           0x1000  // Provider specific event 2

//
// Escape Functions
//

#define SETXOFF             1       // Simulate XOFF received
#define SETXON              2       // Simulate XON received
#define SETRTS              3       // Set RTS high
#define CLRRTS              4       // Set RTS low
#define SETDTR              5       // Set DTR high
#define CLRDTR              6       // Set DTR low
#define RESETDEV            7       // Reset device if possible
#define SETBREAK            8       // Set the device break line.
#define CLRBREAK            9       // Clear the device break line.

//
// PURGE function flags.
//
#define PURGE_TXABORT       0x0001  // Kill the pending/current writes to the comm port.
#define PURGE_RXABORT       0x0002  // Kill the pending/current reads to the comm port.
#define PURGE_TXCLEAR       0x0004  // Kill the transmit queue if there.
#define PURGE_RXCLEAR       0x0008  // Kill the typeahead buffer if there.

#define LPTx                0x80    // Set if ID is for LPT device

//
// Modem Status Flags
//
#define MS_CTS_ON           ((DWORD)0x0010)
#define MS_DSR_ON           ((DWORD)0x0020)
#define MS_RING_ON          ((DWORD)0x0040)
#define MS_RLSD_ON          ((DWORD)0x0080)

//
// WaitSoundState() Constants
//

#define S_QUEUEEMPTY        0
#define S_THRESHOLD         1
#define S_ALLTHRESHOLD      2

//
// Accent Modes
//

#define S_NORMAL      0
#define S_LEGATO      1
#define S_STACCATO    2

//
// SetSoundNoise() Sources
//

#define S_PERIOD512   0     // Freq = N/512 high pitch, less coarse hiss
#define S_PERIOD1024  1     // Freq = N/1024
#define S_PERIOD2048  2     // Freq = N/2048 low pitch, more coarse hiss
#define S_PERIODVOICE 3     // Source is frequency from voice channel (3)
#define S_WHITE512    4     // Freq = N/512 high pitch, less coarse hiss
#define S_WHITE1024   5     // Freq = N/1024
#define S_WHITE2048   6     // Freq = N/2048 low pitch, more coarse hiss
#define S_WHITEVOICE  7     // Source is frequency from voice channel (3)

#define S_SERDVNA     (-1)  // Device not available
#define S_SEROFM      (-2)  // Out of memory
#define S_SERMACT     (-3)  // Music active
#define S_SERQFUL     (-4)  // Queue full
#define S_SERBDNT     (-5)  // Invalid note
#define S_SERDLN      (-6)  // Invalid note length
#define S_SERDCC      (-7)  // Invalid note count
#define S_SERDTP      (-8)  // Invalid tempo
#define S_SERDVL      (-9)  // Invalid volume
#define S_SERDMD      (-10) // Invalid mode
#define S_SERDSH      (-11) // Invalid shape
#define S_SERDPT      (-12) // Invalid pitch
#define S_SERDFQ      (-13) // Invalid frequency
#define S_SERDDR      (-14) // Invalid duration
#define S_SERDSR      (-15) // Invalid source
#define S_SERDST      (-16) // Invalid state

#define NMPWAIT_WAIT_FOREVER            0xffffffff
#define NMPWAIT_NOWAIT                  0x00000001
#define NMPWAIT_USE_DEFAULT_WAIT        0x00000000

/* Flags returned by LocalFlags (in addition to LMEM_DISCARDABLE) */
#define LMEM_DISCARDED      0x4000
#define LMEM_LOCKCOUNT      0x00FF

//
// dwCreationFlag values
//

#define DEBUG_PROCESS               0x00000001
#define DEBUG_ONLY_THIS_PROCESS     0x00000002

#define CREATE_SUSPENDED            0x00000004

#define DETACHED_PROCESS            0x00000008

#define CREATE_NEW_CONSOLE          0x00000010

#define NORMAL_PRIORITY_CLASS       0x00000020
#define IDLE_PRIORITY_CLASS         0x00000040
#define HIGH_PRIORITY_CLASS         0x00000080
#define REALTIME_PRIORITY_CLASS     0x00000100

#define CREATE_NEW_PROCESS_GROUP    0x00000200
#define CREATE_UNICODE_ENVIRONMENT  0x00000400

#define CREATE_SEPARATE_WOW_VDM     0x00000800
#define CREATE_SHARED_WOW_VDM       0x00001000

#define CREATE_DEFAULT_ERROR_MODE   0x04000000
#define CREATE_NO_WINDOW            0x08000000

#define PROFILE_USER                0x10000000
#define PROFILE_KERNEL              0x20000000
#define PROFILE_SERVER              0x40000000

/*
 * Dialog Box Command IDs
 */
#define IDOK                1
#define IDCANCEL            2
#define IDABORT             3
#define IDRETRY             4
#define IDIGNORE            5
#define IDYES               6
#define IDNO                7
// #if(WINVER >= 0x0400)
#define IDCLOSE         8
#define IDHELP          9
//#endif /* WINVER >= 0x0400 */

/*
 * MessageBox() Flags
 */
#define MB_OK                       0x00000000L
#define MB_OKCANCEL                 0x00000001L
#define MB_ABORTRETRYIGNORE         0x00000002L
#define MB_YESNOCANCEL              0x00000003L
#define MB_YESNO                    0x00000004L
#define MB_RETRYCANCEL              0x00000005L

#define MB_ICONHAND                 0x00000010L
#define MB_ICONQUESTION             0x00000020L
#define MB_ICONEXCLAMATION          0x00000030L
#define MB_ICONASTERISK             0x00000040L

// #if(WINVER >= 0x0400)
#define MB_ICONWARNING              MB_ICONEXCLAMATION
#define MB_ICONERROR                MB_ICONHAND
//#endif /* WINVER >= 0x0400 */

#define MB_ICONINFORMATION          MB_ICONASTERISK
#define MB_ICONSTOP                 MB_ICONHAND

#define MB_DEFBUTTON1               0x00000000L
#define MB_DEFBUTTON2               0x00000100L
#define MB_DEFBUTTON3               0x00000200L
//#if(WINVER >= 0x0400)
#define MB_DEFBUTTON4               0x00000300L
//#endif /* WINVER >= 0x0400 */

#define MB_APPLMODAL                0x00000000L
#define MB_SYSTEMMODAL              0x00001000L
#define MB_TASKMODAL                0x00002000L
//#if(WINVER >= 0x0400)
#define MB_HELP                     0x00004000L // Help Button
#define MB_RIGHT                    0x00080000L
#define MB_RTLREADING               0x00100000L
//#endif /* WINVER >= 0x0400 */

#define MB_NOFOCUS                  0x00008000L
#define MB_SETFOREGROUND            0x00010000L
#define MB_DEFAULT_DESKTOP_ONLY     0x00020000L
#define MB_SERVICE_NOTIFICATION     0x00040000L

#define MB_TYPEMASK                 0x0000000FL
//#if(WINVER >= 0x0400)
#define MB_USERICON                 0x00000080L
//#endif /* WINVER >= 0x0400 */
#define MB_ICONMASK                 0x000000F0L
#define MB_DEFMASK                  0x00000F00L
#define MB_MODEMASK                 0x00003000L
#define MB_MISCMASK                 0x0000C000L


/**************** wingdi.h ****************/

#define CLR_INVALID     0xFFFFFFFF

/* Brush Styles */
#define BS_SOLID            0
#define BS_NULL             1
#define BS_HOLLOW           BS_NULL
#define BS_HATCHED          2
#define BS_PATTERN          3
#define BS_INDEXED          4
#define BS_DIBPATTERN       5
#define BS_DIBPATTERNPT     6
#define BS_PATTERN8X8       7
#define BS_DIBPATTERN8X8    8
#define BS_MONOPATTERN      9

/* Hatch Styles */
#define HS_HORIZONTAL       0       /* ----- */
#define HS_VERTICAL         1       /* ||||| */
#define HS_FDIAGONAL        2       /* \\\\\ */
#define HS_BDIAGONAL        3       /* ///// */
#define HS_CROSS            4       /* +++++ */
#define HS_DIAGCROSS        5       /* xxxxx */

/* Pen Styles */
#define PS_SOLID            0
#define PS_DASH             1       /* -------  */
#define PS_DOT              2       /* .......  */
#define PS_DASHDOT          3       /* _._._._  */
#define PS_DASHDOTDOT       4       /* _.._.._  */
#define PS_NULL             5
#define PS_INSIDEFRAME      6
#define PS_USERSTYLE        7
#define PS_ALTERNATE        8
#define PS_STYLE_MASK       0x0000000F

#define PS_ENDCAP_ROUND     0x00000000
#define PS_ENDCAP_SQUARE    0x00000100
#define PS_ENDCAP_FLAT      0x00000200
#define PS_ENDCAP_MASK      0x00000F00

#define PS_JOIN_ROUND       0x00000000
#define PS_JOIN_BEVEL       0x00001000
#define PS_JOIN_MITER       0x00002000
#define PS_JOIN_MASK        0x0000F000

#define PS_COSMETIC         0x00000000
#define PS_GEOMETRIC        0x00010000
#define PS_TYPE_MASK        0x000F0000

#define AD_COUNTERCLOCKWISE 1
#define AD_CLOCKWISE        2

/* Device Parameters for GetDeviceCaps() */
#define DRIVERVERSION 0     /* Device driver version                    */
#define TECHNOLOGY    2     /* Device classification                    */
#define HORZSIZE      4     /* Horizontal size in millimeters           */
#define VERTSIZE      6     /* Vertical size in millimeters             */
#define HORZRES       8     /* Horizontal width in pixels               */
#define VERTRES       10    /* Vertical height in pixels                */
#define BITSPIXEL     12    /* Number of bits per pixel                 */
#define PLANES        14    /* Number of planes                         */
#define NUMBRUSHES    16    /* Number of brushes the device has         */
#define NUMPENS       18    /* Number of pens the device has            */
#define NUMMARKERS    20    /* Number of markers the device has         */
#define NUMFONTS      22    /* Number of fonts the device has           */
#define NUMCOLORS     24    /* Number of colors the device supports     */
#define PDEVICESIZE   26    /* Size required for device descriptor      */
#define CURVECAPS     28    /* Curve capabilities                       */
#define LINECAPS      30    /* Line capabilities                        */
#define POLYGONALCAPS 32    /* Polygonal capabilities                   */
#define TEXTCAPS      34    /* Text capabilities                        */
#define CLIPCAPS      36    /* Clipping capabilities                    */
#define RASTERCAPS    38    /* Bitblt capabilities                      */
#define ASPECTX       40    /* Length of the X leg                      */
#define ASPECTY       42    /* Length of the Y leg                      */
#define ASPECTXY      44    /* Length of the hypotenuse                 */

/* Background Modes */
#define TRANSPARENT         1
#define OPAQUE              2
#define BKMODE_LAST         2

/* Graphics Modes */

#define GM_COMPATIBLE       1
#define GM_ADVANCED         2
#define GM_LAST             2

/* PolyDraw and GetPath point types */
#define PT_CLOSEFIGURE      0x01
#define PT_LINETO           0x02
#define PT_BEZIERTO         0x04
#define PT_MOVETO           0x06

/* Mapping Modes */
#define MM_TEXT             1
#define MM_LOMETRIC         2
#define MM_HIMETRIC         3
#define MM_LOENGLISH        4
#define MM_HIENGLISH        5
#define MM_TWIPS            6
#define MM_ISOTROPIC        7
#define MM_ANISOTROPIC      8

/* Min and Max Mapping Mode values */
#define MM_MIN              MM_TEXT
#define MM_MAX              MM_ANISOTROPIC
#define MM_MAX_FIXEDSCALE   MM_TWIPS

/* Coordinate Modes */
#define ABSOLUTE            1
#define RELATIVE            2

/* Stock Logical Objects */
#define WHITE_BRUSH         0
#define LTGRAY_BRUSH        1
#define GRAY_BRUSH          2
#define DKGRAY_BRUSH        3
#define BLACK_BRUSH         4
#define NULL_BRUSH          5
#define HOLLOW_BRUSH        NULL_BRUSH
#define WHITE_PEN           6
#define BLACK_PEN           7
#define NULL_PEN            8
#define OEM_FIXED_FONT      10
#define ANSI_FIXED_FONT     11
#define ANSI_VAR_FONT       12
#define SYSTEM_FONT         13
#define DEVICE_DEFAULT_FONT 14
#define DEFAULT_PALETTE     15
#define SYSTEM_FIXED_FONT   16



/***********************************************************************
***********************************************************************
* API FUNCTIONS
***********************************************************************
***********************************************************************/

/***********************************************************************
* DLL 
***********************************************************************/
HINSTANCE LoadLibrary(LPCSTR lpszLibFileName);
#pragma link C func LoadLibrary;

void FreeLibrary(HINSTANCE hinst);
#pragma link C func FreeLibrary;

FARPROC GetProcAddress(HINSTANCE hinst,LPCSTR lpszProcName);
#pragma link C func GetProcAddress;

/* int GetModuleUsage(HINSTANCE hinst); */
/* #pragma link C func GetModuleUsage */

/***********************************************************************
* Environment Variable
***********************************************************************/
BOOL SetEnvironmentVariable(LPCSTR lpName,LPCSTR lpValue);
#pragma link C func SetEnvironmentVariable;

/***********************************************************************
* User Info
***********************************************************************/
#ifndef G__SYMANTEC
BOOL GetUserName(LPTSTR lpBuffer,LPDWORD nSize);
#pragma link C func GetUserName;
#endif

BOOL SetComputerName(LPCSTR lpComputerName);
#pragma link C func SetComputerName;

BOOL GetComputerName(LPTSTR lpBuffer,LPDWORD nSize);
#pragma link C func GetComputerName;

#ifndef G__SYMANTEC
BOOL LookupAccountName(LPCTSTR lpSystemName,LPCTSTR lpAccountName,PSID Sid
                      ,LPDWORD cbSid,LPTSTR ReferencedDomainName
                      ,LPDWORD cbReferenceDomainName,PSID_NAME_USE peUse);
#pragma link C func LookupAccountName;
#pragma link C class  _SID_NAME_USE;
#pragma link C typedef SID_NAME_USE;
#endif

#ifndef G__SYMANTEC
BOOL EqualPrefixSid(PSID pSid1,PSID pSid2);
#pragma link C func EqualPrefixSid;
#endif

/***********************************************************************
* Thread
***********************************************************************/
HANDLE CreateThread(LPSECURITY_ATTRIBUTES lpThreadAttributes,
                    DWORD dwStackSize,
                    LPTHREAD_START_ROUTINE lpStartAddress,
                    LPVOID lpParameter,
                    DWORD dwCreationFlags,
                    LPDWORD lpThreadId
                    );
#pragma link C func CreateThread;

// BOOL SwitchToThread();
// #pragma link C func SwitchToThread;

// HANDLE GetCurentThread();
// #pragma link C func GetCurrentThread;

DWORD GetCurrentThreadId();
#pragma link C func GetCurrentThreadId;

VOID InitializeCriticalSection(LPCRITICAL_SECTION lpCriticalSection);
#pragma link C func InitializeCriticalSection;

VOID EnterCriticalSection(LPCRITICAL_SECTION lpCriticalSection);
#pragma link C func EnterCriticalSection;

VOID LeaveCriticalSection(LPCRITICAL_SECTION lpCriticalSection);
#pragma link C func LeaveCriticalSection;

//BOOL TryEnterCriticalSection(LPCRITICAL_SECTION lpCriticalSection);
//#pragma link C func TryEnterCriticalSection;

VOID DeleteCriticalSection(LPCRITICAL_SECTION lpCriticalSection);
#pragma link C func DeleteCriticalSection;

/***********************************************************************
* File
***********************************************************************/
BOOL CopyFile(LPCTSTR lpExistingFileName,LPCTSTR lpNewFileName
	      ,BOOL bFailIfExists);
#pragma link C func CopyFile;

DWORD GetCurrentDirectory(DWORD nBufferLength,LPTSTR lpBuffer);
#pragma link C func GetCurrentDirectory;

BOOL SetCurrentDirectory(LPCSTR lpPathName);
#pragma link C func SetCurrentDirectory;

BOOL GetDiskFreeSpace(LPCTSTR lpRootPathName,LPDWORD lpSectorsPerCluster
		      ,LPDWORD lpByteperSector,LPDWORD lpNumberOfFreeClusters
		      ,LPDWORD lpTotalNumberOfClusters);
#pragma link C func GetDiskFreeSpace;

DWORD GetFileAttributes(LPCTSTR lpFileName);
#pragma link C func GetFileAttributes;

DWORD GetFileSize(HANDLE hFile,LPDWORD lpFleSizeHigh);
#pragma link C func GetFileSize;

DWORD GetFileType(HANDLE hFile);
#pragma link C func GetFileType;

DWORD GetFullPathName(LPCTSTR lpFileName,DWORD nBufferLength,LPTSTR lpBuffer,char **lpFilePart);
#pragma link C func GetFullPathName;

BOOL LockFile(HANDLE hFile,DWORD dwFileOffsetLow,DWORD dwFileOffsetHigh
	      ,DWORD nNumberofBytesToLockLow,DWORD nNumberOfBytesTOLockHigh);
#pragma link C func LockFile;

BOOL MoveFile(LPCTSTR lpExistingFileName,LPCTSTR lpNewFileName);
#pragma link C func MoveFile;

BOOL ReadFile(HANDLE hFile,LPVOID lpBuffer,DWORD nNumberOfBytesToRead
	      ,LPDWORD lpNumberOfBytesRead,LPOVERLAPPED lpOverlapped);
#pragma link C func ReadFile;

BOOL RemoveDirectory(LPCTSTR lpPathName);
#pragma link C func RemoveDirectory;

BOOL CreateDirectory(LPCSTR lpPathName,LPVOID lpSecurityAttributes);
#pragma link C func CreateDirectory;

BOOL WriteFile(HANDLE hFile,LPCVOID lpBuffer,DWORD nNumberOfBytesToWrite,LPDWORD lpNumberOfBytesWritten,LPOVERLAPPED lpOverlapped);
#pragma link C func WriteFile;


HANDLE FindFirstFile(LPCSTR lpFileName, LPWIN32_FIND_DATA lpFindFileData);
#pragma link C func FindFirstFile;

BOOL FindNextFile(HANDLE hFindFile, LPWIN32_FIND_DATA lpFindFileData);
#pragma link C func FindNextFile;

BOOL FindClose(HANDLE hFindFile);
#pragma link C func FindClose;


LONG CompareFileTime(const FILETIME *lpFileTime1, const FILETIME *lpFileTime2);
#pragma link C func CompareFileTime;

BOOL GetFileTime(HANDLE hFile, FILETIME *lpftCreation,
                FILETIME *lpftLastAccess, FILETIME *lpftLastWrite);
#pragma link C func GetFileTime;

BOOL SetFileTime(HANDLE hFile, const FILETIME *lpftCreation,
                const FILETIME *lpftLastAccess,
                const FILETIME *lpftLastWrite);
#pragma link C func SetFileTime;

/* BOOL FIsTask(HTASK hTask); */

DWORD SetFilePointer(HANDLE hFile, LONG lDistanceToMove,
                     LONG *lpDistanceToMoveHigh, DWORD dwMoveMethod);
#pragma link C func SetFilePointer;

BOOL SetEndOfFile(HANDLE hFile);
#pragma link C func SetEndOfFile;

UINT GetTempFileName (LPCSTR lpPathName, LPCSTR lpPrefixString,
                UINT uUnique, LPSTR lpTempFileName);
#pragma link C func GetTempFileName;

BOOL DeleteFile(LPCSTR lpFileName);
#pragma link C func DeleteFile;

BOOL LocalFileTimeToFileTime(const FILETIME *lpLocalFileTime, FILETIME *lpFileTime);
#pragma link C func LocalFileTimeToFileTime;

BOOL FileTimeToLocalFileTime(const FILETIME *lpFileTime, FILETIME *lpLocalFileTime);
#pragma link C func FileTimeToLocalFileTime;

BOOL FileTimeToSystemTime(const FILETIME *lpFileTime, SYSTEMTIME *lpSystemTime);
#pragma link C func FileTimeToSystemTime;

BOOL SystemTimeToFileTime(const SYSTEMTIME *lpSystemTime, FILETIME *lpFileTime);
#pragma link C func SystemTimeToFileTime;

DWORD GetModuleFileName(HMODULE hModule,LPSTR lpFilename,DWORD nSize);
#pragma link C func GetModuleFileName;

#if 0
LPCSTR DirName(LPCSTR lpFilename);
#pragma link C func DirName;
#endif


/***********************************************************************
* Time
***********************************************************************/

void GetSystemTime(SYSTEMTIME *lpSystemTime);
#pragma link C func GetSystemTime;

void GetLocalTime(SYSTEMTIME *);
#pragma link C func GetLocalTime;

DWORD GetTimeZoneInformation(
                LPTIME_ZONE_INFORMATION lpTimeZoneInformation);
#pragma link C func GetTimeZoneInformation;

BOOL SetTimeZoneInformation(
                const TIME_ZONE_INFORMATION *lpTimeZoneInformation);
#pragma link C func SetTimeZoneInformation;

DWORD GetCurrentProcessId(void);
#pragma link C func GetCurrentProcessId;

/***********************************************************************
* IOCTL 
***********************************************************************/
BOOL DeviceIoControl(HANDLE hDevice,DWORD dwIoControlCode,LPVOID lpInBuffer,
		     DWORD nInBufferSize,LPVOID lpOutBuffer,
		     DWORD nOutBufferSize,LPDWORD lpBytesReturned,
		     LPOVERLAPPED lpOverlapped);
#pragma link C func DeviceIoControl;

/***********************************************************************
* COMM port
***********************************************************************/
HANDLE CreateFile(LPCTSTR lpFileName,DWORD dwDesiredAccess,DWORD dwShareMode,LPSECURITY_ATTRIBUTES lpSecurityAttributes,DWORD dwCreationDisposition,DWORD dwFlagsAndAttributes,HANDLE hTemplateFile);
#pragma link C func CreateFile;

BOOL GetCommState(HANDLE hFile,LPDCB lpDCB);
#pragma link C func GetCommState;

BOOL SetCommState(HANDLE hFile,LPDCB lpDCB);
#pragma link C func SetCommState;

BOOL SetCommTimeouts(HANDLE hFile,LPCOMMTIMEOUTS lpCommTimeouts);
#pragma link C func SetCommTimeouts;

BOOL GetCommTimeouts(HANDLE hFile,LPCOMMTIMEOUTS lpCommTimeouts);
#pragma link C func GetCommTimeouts;

BOOL EscapeCommFunction(HANDLE hFile,DWORD dwFunc);
#pragma link C func EscapeCommFunction;

BOOL CloseHandle(HANDLE hObject);
#pragma link C func CloseHandle;

BOOL SetCommMask(HANDLE hFile,DWORD dwEvtMask);
#pragma link C func SetCommMask;

BOOL WaitCommEvent(HANDLE hFile,LPDWORD lpEvMask,LPOVERLAPPED lpOverlapped);
#pragma link C func WaitCommEvent;

/***********************************************************************
* Event
***********************************************************************/
DWORD WaitForSingleObject(HANDLE hHandle,DWORD dwMilliseconds);
#pragma link C func WaitForSingleObject;

DWORD WaitForMultipleObjects(DWORD nCOunt,const HANDLE* lpHandles,BOOL bWaitAll,DWORD dwMilliseconds);
#pragma link C func WaitForMultipleObjects;

BOOL PulseEvent(HANDLE hEvent);
#pragma link C func PulseEvent;

BOOL ResetEvent(HANDLE hEvent);
#pragma link C func ResetEvent;

BOOL CreateEvent(LPSECURITY_ATTRIBUTES lpEventAttributes,BOOL bManualReset,BOOL bIniitialState,LPCTSTR lpName);
#pragma link C func CreateEvent;

/***********************************************************************
* Window
***********************************************************************/
BOOL AdjustWindowRect(LPRECT lprc,DWORD dwStyle,BOOL fMenu);
#pragma link C func AdjustWindowRect;

HWND CreateWindow(LPCTSTR lpszClassName,LPCTSTR lpszWindowName
		  ,DWORD dwStyle,int x,int y,int nWidth,int nHeight
		  ,HWND hwndParent,HMENU hmenu,HANDLE hinst,LPVOID lpvParam);
#pragma link C func CreateWindow;

BOOL DestroyWindow(HWND hwnd);
#pragma link C func DestroyWindow;

BOOL GetWindow(HWND hWnd,UINT uCmd);
#pragma link C func GetWindow;

int GetWindowText(HWND hWnd,LPTSTR lpString,int nMaxCount);
#pragma link C func GetWindowText;

BOOL MoveWindow(HWND hWnd,int X,int Y,int nWidth,int nHeight,BOOL bRepaing);
#pragma link C func MoveWindow;

BOOL SetWindowText(HWND hWnd,LPCTSTR lpString);
#pragma link C func SetWindowText;


/***********************************************************************
* Caret
***********************************************************************/
void CreateCaret(HWND hwnd,HBITMAP hbmp,int nWidth,int nHeight);
#pragma link C func CreateCaret;

BOOL GetCaretPos(LPPOINT lpPoint);
#pragma link C func GetCaretPos;

BOOL HideCaret(HWND hWnd);
#pragma link C func HideCaret;

/***********************************************************************
* Clipboard
***********************************************************************/
BOOL CloseClipboard();
#pragma link C func CloseClipboard;

BOOL EmptyClipboard();
#pragma link C func EmptyClipboard;

HANDLE GetClipboardData(UINT uFormat);
#pragma link C func GetClipboardData;

BOOL OpenClipboard(HWND hWndNewOwner);
#pragma link C func OpenClipboard;

/***********************************************************************
* Message Box
***********************************************************************/
int MessageBox(HWND hWnd,LPCTSTR lpText,LPCTSTR lpCaption,UINT uType);
#pragma link C func MessageBox;

/***********************************************************************
* Common Dialog Box
***********************************************************************/
#ifndef G__SYMANTEC
BOOL ChooseColor(LPCHOOSECOLOR lpcc);
#pragma link C func ChooseColor;

BOOL ChooseFont(LPCHOOSEFONT lpcf);
#pragma link C func ChooseFont;

HWND FindText(LPFINDREPLACE lpfr);
#pragma link C func FindText;

short GetFileTitle(LPCTSTR lpszFile,LPTSTR lpszTitle,WORD cbBuf);
#pragma link C func GetFileTitle;

BOOL GetOpenFileName(LPOPENFILENAME lpofn);
#pragma link C func GetOpenFileName;

BOOL GetSaveFileName(LPOPENFILENAME lpofn);
#pragma link C func GetSaveFileName;

BOOL PageSetupDlg(LPPAGESETUPDLG lppsd);
#pragma link C func PageSetupDlg;

BOOL PrintDlg(LPPRINTDLG lppd);
#pragma link C func PrintDlg;

HWND ReplaceText(LPFINDREPLACE lpfr);
#pragma link C func ReplaceText;
#endif

/***********************************************************************
* Timer
***********************************************************************/
void Sleep(DWORD dwMilliseconds);
#pragma link C func Sleep;

/***********************************************************************
* Console
***********************************************************************/
BOOL AllocConsole();
#pragma link C func AllocConsole();
BOOL FreeConsole();
#pragma link C func FreeConsole();
BOOL SetConsoleTitle(LPCTSTR title);
#pragma link C func SetConsoleTitle;


/***********************************************************************
* Error
***********************************************************************/
BOOL Beep(DWORD dwFreq,DWORD dwDuration);
#pragma link C func Beep;

BOOL ExitWindows(DWORD dwReserved,UINT uReserved);
#pragma link C func ExitWindows;

DWORD GetLastError();
#pragma link C func GetLastError;

void ExitProcess(UINT uExitCode);
#pragma link C func ExitProcess;

/***********************************************************************
* Graphics
***********************************************************************/
#pragma link C class HDC__;
#pragma link C typedef HDC;

BOOL AngleArc(HDC, int, int, DWORD, FLOAT, FLOAT);
#pragma link C func AngleArc;

BOOL PolyPolyline(HDC, /* CONST */ POINT *, /* CONST */ DWORD *, DWORD);
#pragma link C func PolyPolyline;
BOOL GetWorldTransform(HDC, LPXFORM);
#pragma link C func GetWorldTransform;
BOOL SetWorldTransform(HDC, /* CONST */ XFORM *);
#pragma link C func SetWorldTransform;
BOOL ModifyWorldTransform(HDC, /* CONST */ XFORM *, DWORD);
#pragma link C func ModifyWorldTransform;
BOOL CombineTransform(LPXFORM, /* CONST */ XFORM *, /* CONST */ XFORM *);
#pragma link C func CombineTransform;
HBITMAP CreateDIBSection(HDC, /* CONST */ BITMAPINFO *, UINT, VOID **, HANDLE, DWORD);
#pragma link C func CreateDIBSction;
UINT GetDIBColorTable(HDC, UINT, UINT, RGBQUAD *);
#pragma link C func GetDIBColorTable;
UINT SetDIBColorTable(HDC, UINT, UINT, /* CONST */ RGBQUAD *);
#pragma link C func SetDIBColorTable;

HRGN  CreatePolygonRgn(/* CONST */ POINT *, int, int);
#pragma link C func CreatePolygonRgn;
BOOL  DPtoLP(HDC, LPPOINT, int);
#pragma link C func DPtoLP;
BOOL  LPtoDP(HDC, LPPOINT, int);
#pragma link C func LPtoDP;
BOOL  Polygon(HDC, /* CONST */ POINT *, int);
#pragma link C func Polygon;
BOOL  Polyline(HDC, /* CONST */ POINT *, int);
#pragma link C func Polyline;

BOOL  PolyBezier(HDC, /* CONST */ POINT *, DWORD);
#pragma link C func PolyBezier;
BOOL  PolyBezierTo(HDC, /* CONST */ POINT *, DWORD);
#pragma link C func PolyBezierTo;
BOOL  PolylineTo(HDC, /* CONST */ POINT *, DWORD);
#pragma link C func PolylineTo;

BOOL  SetViewportExtEx(HDC, int, int, LPSIZE);
#pragma link C func SetViewportExtEx;
BOOL  SetViewportOrgEx(HDC, int, int, LPPOINT);
#pragma link C func SetViewportOrgEx;
BOOL  SetWindowExtEx(HDC, int, int, LPSIZE);
#pragma link C func SetWindowExtEx;
BOOL  SetWindowOrgEx(HDC, int, int, LPPOINT);
#pragma link C func SetWindowOrgEx;

BOOL  OffsetViewportOrgEx(HDC, int, int, LPPOINT);
#pragma link C func OffsetViewportOrgEx;
BOOL  OffsetWindowOrgEx(HDC, int, int, LPPOINT);
#pragma link C func OffsetWindowOrgEx;
BOOL  ScaleViewportExtEx(HDC, int, int, int, int, LPSIZE);
#pragma link C func ScaleViewportExtEx;
BOOL  ScaleWindowExtEx(HDC, int, int, int, int, LPSIZE);
#pragma link C func ScaleWindowExtEx;
BOOL  SetBitmapDimensionEx(HBITMAP, int, int, LPSIZE);
#pragma link C func SetBitmapDimensionEx;
BOOL  SetBrushOrgEx(HDC, int, int, LPPOINT);
#pragma link C func SetBrushOrgEx;

int   GetTextFace(HDC, int, LPSTR);
#pragma link C func GetTextFace;

DWORD GetKerningPairs(HDC, DWORD, LPKERNINGPAIR);
#pragma link C func GetKerningPairs;

BOOL  GetDCOrgEx(HDC,LPPOINT);
#pragma link C func GetDCOrgEx;
BOOL  FixBrushOrgEx(HDC,int,int,LPPOINT);
#pragma link C func FixBrushOrgEx;
BOOL  UnrealizeObject(HGDIOBJ);
#pragma link C func UnrealizeObject;

BOOL  GdiFlush();
#pragma link C func GdiFlush;
DWORD GdiSetBatchLimit(DWORD);
#pragma link C func GdiSetBatchLimit;
DWORD GdiGetBatchLimit();
#pragma link C func GdiGetBatchLimit;

BOOL DrawState(HDC, HBRUSH, DRAWSTATEPROC, LPARAM, WPARAM, int, int, int, int, UINT);
#pragma link C func DrawState;

/* Kazu Hata's request */
#pragma link C func CreatePen;
#pragma link C func SelectObject;
#pragma link C func CreateHatchBrush; 
#pragma link C func Rectangle;
#pragma link C func DeleteObject;
#pragma link C func GetStockObject;

#if G__MSC_VER>1310
char* _strupr(char*);
#pragma link C func _strupr;
char* _strlwr(char*);
#pragma link C func _strlwr;
#else
char* strupr(char*);
#pragma link C func strupr;
char* strlwr(char*);
#pragma link C func strlwr;
#endif

COLORREF RGB(char a,char b,char c);
#pragma link C MACRO RGB;




/**********************************************************************/
#endif // __MAKECINT__

#endif // G__WINFUNC_H


