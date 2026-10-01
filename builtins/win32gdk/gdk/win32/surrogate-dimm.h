#ifndef __SURROGATE_DIMM_H__
#define __SURROGATE_DIMM_H__

/* The Win32api headers doesn't include <dimm.h>, thus we need
 * this file, which covers just the stuff we need from <dimm.h>.
 */

typedef struct IActiveIMMApp IActiveIMMApp;
typedef struct IActiveIMMMessagePumpOwner IActiveIMMMessagePumpOwner;

/* Dummy vtable structs that contain real names and arg lists for
 * only those methods we need.
 */
typedef struct {
   HRESULT(__stdcall * QueryInterface) (IActiveIMMApp * This,
                                        REFIID riid, void *ppvObject);
   /* Dummy method prototypes for those we don't use */
   ULONG(__stdcall * dummy_AddRef) ();
   ULONG(__stdcall * dummy_Release) ();
   HRESULT(__stdcall * dummy_AssociateContext) ();
   HRESULT(__stdcall * dummy_ConfigureIMEA) ();
   HRESULT(__stdcall * dummy_ConfigureIMEW) ();
   HRESULT(__stdcall * dummy_CreateContext) ();
   HRESULT(__stdcall * dummy_DestroyContext) ();
   HRESULT(__stdcall * dummy_EnumRegisterWordA) ();
   HRESULT(__stdcall * dummy_EnumRegisterWordW) ();
   HRESULT(__stdcall * dummy_EscapeA) ();
   HRESULT(__stdcall * dummy_EscapeW) ();
   HRESULT(__stdcall * dummy_GetCandidateListA) ();
   HRESULT(__stdcall * dummy_GetCandidateListW) ();
   HRESULT(__stdcall * dummy_GetCandidateListCountA) ();
   HRESULT(__stdcall * dummy_GetCandidateListCountW) ();
   HRESULT(__stdcall * dummy_GetCandidateWindow) ();
   HRESULT(__stdcall * dummy_GetCompositionFontA) ();
   HRESULT(__stdcall * dummy_GetCompositionFontW) ();
   HRESULT(__stdcall * dummy_GetCompositionStringA) ();
   HRESULT(__stdcall * dummy_GetCompositionStringW) ();
   HRESULT(__stdcall * dummy_GetCompositionWindow) ();
   HRESULT(__stdcall * dummy_GetContext) ();
   HRESULT(__stdcall * dummy_GetConversionListA) ();
   HRESULT(__stdcall * dummy_GetConversionListW) ();
   HRESULT(__stdcall * dummy_GetConversionStatus) ();

   HRESULT(__stdcall * GetDefaultIMEWnd) (IActiveIMMApp * This,
                                          HWND hWnd, HWND * phDefWnd);

   HRESULT(__stdcall * dummy_GetDescriptionA) ();
   HRESULT(__stdcall * dummy_GetDescriptionW) ();
   HRESULT(__stdcall * dummy_GetGuideLineA) ();
   HRESULT(__stdcall * dummy_GetGuideLineW) ();
   HRESULT(__stdcall * dummy_GetIMEFileNameA) ();
   HRESULT(__stdcall * dummy_GetIMEFileNameW) ();
   HRESULT(__stdcall * dummy_GetOpenStatus) ();
   HRESULT(__stdcall * dummy_GetProperty) ();
   HRESULT(__stdcall * dummy_GetRegisterWordStyleA) ();
   HRESULT(__stdcall * dummy_GetRegisterWordStyleW) ();
   HRESULT(__stdcall * dummy_GetStatusWindowPos) ();
   HRESULT(__stdcall * dummy_GetVirtualKey) ();
   HRESULT(__stdcall * dummy_InstallIMEA) ();
   HRESULT(__stdcall * dummy_InstallIMEW) ();

   HRESULT(__stdcall * IsIME) (IActiveIMMApp * This, HKL hKL);
   HRESULT(__stdcall * IsUIMessageA) (IActiveIMMApp * This,
                                      HWND hWndIME,
                                      UINT msg,
                                      WPARAM wParam, LPARAM lParam);
   HRESULT(__stdcall * dummy_IsUIMessageW) ();
   HRESULT(__stdcall * dummy_NotifyIME) ();
   HRESULT(__stdcall * dummy_RegisterWordA) ();
   HRESULT(__stdcall * dummy_RegisterWordW) ();
   HRESULT(__stdcall * dummy_ReleaseContext) ();
   HRESULT(__stdcall * dummy_SetCandidateWindow) ();
   HRESULT(__stdcall * dummy_SetCompositionFontA) ();
   HRESULT(__stdcall * dummy_SetCompositionFontW) ();
   HRESULT(__stdcall * dummy_SetCompositionStringA) ();
   HRESULT(__stdcall * dummy_SetCompositionStringW) ();
   HRESULT(__stdcall * dummy_SetCompositionWindow) ();
   HRESULT(__stdcall * dummy_SetConversionStatus) ();
   HRESULT(__stdcall * dummy_SetOpenStatus) ();
   HRESULT(__stdcall * dummy_SetStatusWindowPos) ();
   HRESULT(__stdcall * dummy_SimulateHotKey) ();
   HRESULT(__stdcall * dummy_UnregisterWordA) ();
   HRESULT(__stdcall * dummy_UnregisterWordW) ();

   HRESULT(__stdcall * Activate) (IActiveIMMApp * This, BOOL restore);
   HRESULT(__stdcall * Deactivate) (IActiveIMMApp * This);
   HRESULT(__stdcall * OnDefWindowProc) (IActiveIMMApp * This,
                                         HWND hWnd,
                                         UINT Msg,
                                         WPARAM wParam,
                                         LPARAM lParam,
                                         LRESULT * plResult);

   HRESULT(__stdcall * dummy_FilterClientWindows) ();

   HRESULT(__stdcall * GetCodePageA) (IActiveIMMApp * This,
                                      HKL hKL, UINT * uCodePage);
   HRESULT(__stdcall * GetLangId) (IActiveIMMApp * This,
                                   HKL hKL, LANGID * plid);

   HRESULT(__stdcall * dummy_AssociateContextEx) ();
   HRESULT(__stdcall * dummy_DisableIME) ();
   HRESULT(__stdcall * dummy_GetImeMenuItemsA) ();
   HRESULT(__stdcall * dummy_GetImeMenuItemsW) ();
   HRESULT(__stdcall * dummy_EnumInputContext) ();
} IActiveIMMAppVtbl;

struct IActiveIMMApp {
   IActiveIMMAppVtbl *lpVtbl;
};

typedef struct {
   HRESULT(__stdcall * dummy_QueryInterface) ();
   ULONG(__stdcall * dummy_AddRef) ();
   ULONG(__stdcall * dummy_Release) ();

   HRESULT(__stdcall * Start) (IActiveIMMMessagePumpOwner * This);
   HRESULT(__stdcall * End) (IActiveIMMMessagePumpOwner * This);
   HRESULT(__stdcall * OnTranslateMessage) (IActiveIMMMessagePumpOwner *
                                            This, MSG * pMSG);

   HRESULT(__stdcall * dummy_Pause) ();
   HRESULT(__stdcall * dummy_Resume) ();
} IActiveIMMMessagePumpOwnerVtbl;

struct IActiveIMMMessagePumpOwner {
   IActiveIMMMessagePumpOwnerVtbl *lpVtbl;
};

static UUID CLSID_CActiveIMM = {
   0x4955DD33, 0xB159, 0x11d0, {0x8F, 0xCF, 0x00, 0xAA, 0x00, 0x6B, 0xCC,
                                0x59}
};
static IID IID_IActiveIMMApp = {
   0x08C0E040, 0x62D1, 0x11D1, {0x93, 0x26, 0x00, 0x60, 0xB0, 0x67, 0xB8,
                                0x6E}
};
static IID IID_IActiveIMMMessagePumpOwner = {
   0xB5CF2CFA, 0x8AEB, 0x11D1, {0x93, 0x64, 0x00, 0x60, 0xB0, 0x67, 0xB8,
                                0x6E}
};

#endif                          /* __SURROGATE_DIMM_H__ */
