// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine(fine@vxcern.cern.ch)   13/08/96

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//   It contains the dummy implementations of ShowMembers, Streamer     //
//       and operator >>   for all WIN32 and WINNT classes              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TCritSection.h"
#include "TContextMenuItem.h"
#include "TGWin32.h"
#include "TGWin32Object.h"
#include "TGWin32Pen.h"
#include "TGWin32PixmapObject.h"
#include "TGWin32WindowsObject.h"
#include "TWin32MenuItem.h"
#include "TWin32ContextMenuImp.h"
#include "TWin32ControlBarImp.h"
#include "TWin32Canvas.h"
#include "TWin32Dialog.h"
#include "TWin32GuiFactory.h"
#include "TWin32Application.h"
#include "TWin32CallBackList.h"
#include "TWin32TreeViewCtrl.h"
#include "TWin32ListViewCtrl.h"
#include "TWin32CommCtrl.h"
#include "TWin32BrowserImp.h"

#include "TBuffer.h"

#ifdef NEVER
//*-*-*-*-*-*-*-*-*-*-*-*  ShowMembers for Win32  *-*-*-*-*-*-*-*-*-*-*
//______________________________________________________________________________
void TWin32Application::ShowMembers(TMemberInspector &insp, char *parent){;}

//______________________________________________________________________________
void TWin32CallBackList::ShowMembers(TMemberInspector &insp, char *parent){;}
//______________________________________________________________________________
void TCallBackObject::ShowMembers(TMemberInspector &insp, char *parent){;}

//______________________________________________________________________________
void TCritSection::ShowMembers(TMemberInspector &insp, char *parent){;}

//______________________________________________________________________________
void TContextMenuItem::ShowMembers(TMemberInspector &insp, char *parent){;}

//______________________________________________________________________________
void TGWin32Switch::ShowMembers(TMemberInspector &insp, char *parent){;}

//______________________________________________________________________________
void TGWin32Object::ShowMembers(TMemberInspector &insp, char *parent){;}

//______________________________________________________________________________
void TGWin32Pen::ShowMembers(TMemberInspector &insp, char *parent){;}

//______________________________________________________________________________
void TGWin32PixmapObject::ShowMembers(TMemberInspector &insp, char *parent){;}

//______________________________________________________________________________
void TGWin32WindowsObject::ShowMembers(TMemberInspector &insp, char *parent){;}

//______________________________________________________________________________
void TWin32MenuItem::ShowMembers(TMemberInspector &insp, char *parent){;}

//______________________________________________________________________________
void TWin32Canvas::ShowMembers(TMemberInspector &insp, char *parent){;}

//______________________________________________________________________________
void TWin32ContextMenuImp::ShowMembers(TMemberInspector &insp, char *parent){;}

//______________________________________________________________________________
void TWin32ControlBarImp::ShowMembers(TMemberInspector &insp, char *parent){;}

//______________________________________________________________________________
void TWin32Dialog::ShowMembers(TMemberInspector &insp, char *parent){;}
//______________________________________________________________________________
void TWin32Menu::ShowMembers(TMemberInspector &insp, char *parent){;}
//______________________________________________________________________________
void TWin32TreeViewCtrl::ShowMembers(TMemberInspector &insp, char *parent){;}
//______________________________________________________________________________
void TWin32ListViewCtrl::ShowMembers(TMemberInspector &insp, char *parent){;}

//______________________________________________________________________________
void TWin32CommCtrl::ShowMembers(TMemberInspector &insp, char *parent){;}
//______________________________________________________________________________
void TWin32BrowserImp::ShowMembers(TMemberInspector &insp, char *parent){;}



//*-*-*-*-*-*-*-*-*-*-*-*  Streamer for Win32  *-*-*-*-*-*-*-*-*-*-*
//______________________________________________________________________________
void TWin32Application::Streamer(TBuffer &b){;}

//______________________________________________________________________________
void TWin32CallBackList::Streamer(TBuffer &b){;}
//______________________________________________________________________________
void TCallBackObject::Streamer(TBuffer &b){;}

//______________________________________________________________________________
void TCritSection::Streamer(TBuffer &b){;}

//______________________________________________________________________________
void TContextMenuItem::Streamer(TBuffer &b){;}

//______________________________________________________________________________
void TGWin32Switch::Streamer(TBuffer &b){;}

//______________________________________________________________________________
void TGWin32Object::Streamer(TBuffer &b){;}

//______________________________________________________________________________
void TGWin32Pen::Streamer(TBuffer &b){;}

//______________________________________________________________________________
void TGWin32PixmapObject::Streamer(TBuffer &b){;}

//______________________________________________________________________________
void TGWin32WindowsObject::Streamer(TBuffer &b){;}

//______________________________________________________________________________
void TWin32MenuItem::Streamer(TBuffer &b){;}

//______________________________________________________________________________
void TWin32Canvas::Streamer(TBuffer &b){;}

//______________________________________________________________________________
void TWin32ContextMenuImp::Streamer(TBuffer &b){;}

//______________________________________________________________________________
void TWin32ControlBarImp::Streamer(TBuffer &b){;}

//______________________________________________________________________________
void TWin32Dialog::Streamer(TBuffer &b){;}
//______________________________________________________________________________
void TWin32Menu::Streamer(TBuffer &b){;}

//______________________________________________________________________________
void TWin32CommCtrl::Streamer(TBuffer &b){;}
//______________________________________________________________________________
void TWin32TreeViewCtrl::Streamer(TBuffer &b){;}
//______________________________________________________________________________
void TWin32ListViewCtrl::Streamer(TBuffer &b){;}

//______________________________________________________________________________
void TWin32BrowserImp::Streamer(TBuffer &b){;}


//*-*-*-*-*-*-*-*-*-*-*-*  operator >> for Win32  *-*-*-*-*-*-*-*-*-*-*
//______________________________________________________________________________
TBuffer &operator>>(TBuffer &buf, TWin32Application *&obj) {return buf;}
//______________________________________________________________________________
TBuffer &operator>>(TBuffer &buf, TWin32CallBackList *&obj) {return buf;}
//______________________________________________________________________________
TBuffer &operator>>(TBuffer &buf, TCallBackObject *&obj) {return buf;}
//______________________________________________________________________________
TBuffer &operator>>(TBuffer &buf, TCritSection *&obj) {return buf;}
//______________________________________________________________________________
TBuffer &operator>>(TBuffer &buf, TContextMenuItem *&obj) {return buf;}
//______________________________________________________________________________
TBuffer &operator>>(TBuffer &buf, TGWin32Switch *&obj) {return buf;}
//______________________________________________________________________________
TBuffer &operator>>(TBuffer &buf, TGWin32Object *&obj) {return buf;}
//______________________________________________________________________________
TBuffer &operator>>(TBuffer &buf, TGWin32Pen *&obj) {return buf;}
//______________________________________________________________________________
TBuffer &operator>>(TBuffer &buf, TGWin32PixmapObject *&obj) {return buf;}
//______________________________________________________________________________
TBuffer &operator>>(TBuffer &buf, TGWin32WindowsObject *&obj) {return buf;}
//______________________________________________________________________________
TBuffer &operator>>(TBuffer &buf, TWin32MenuItem *&obj) {return buf;}
//______________________________________________________________________________
TBuffer &operator>>(TBuffer &buf, TWin32Canvas *&obj) {return buf;}
//______________________________________________________________________________
TBuffer &operator>>(TBuffer &buf, TWin32ContextMenuImp *&obj) {return buf;}
//______________________________________________________________________________
TBuffer &operator>>(TBuffer &buf, TWin32ControlBarImp *&obj) {return buf;}
//______________________________________________________________________________
TBuffer &operator>>(TBuffer &buf, TWin32Dialog *&obj) {return buf;}
//______________________________________________________________________________
TBuffer &operator>>(TBuffer &buf, TWin32Menu *&obj) {return buf;}

//______________________________________________________________________________
TBuffer &operator>>(TBuffer &buf, TWin32BrowserImp *&obj) {return buf;}
//______________________________________________________________________________
TBuffer &operator>>(TBuffer &buf, TWin32TreeViewCtrl *&obj) {return buf;}
//______________________________________________________________________________
TBuffer &operator>>(TBuffer &buf, TWin32ListViewCtrl *&obj) {return buf;}
//______________________________________________________________________________
TBuffer &operator>>(TBuffer &buf, TWin32CommCtrl *&obj) {return buf;}
#endif
