// @(#)root/win32:$Name$:$Id$
// Author: 
// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine   01/01/96

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TWin32CallBackList                                                   //
//                                                                      //
// A doubly linked list. All classes inheriting from TObject can be     //
// inserted in a TList. Before being inserted into the list the object  //
// pointer is wrapped in a TObjLink object which contains, besides      //
// the object pointer also a previous and next pointer.                 //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TWin32CallBackList
#include "TWin32CallBackList.h"
#endif

// ClassImp(TWin32CallBackList)

//______________________________________________________________________________
TWin32CallBackList::TWin32CallBackList(UINT Idx, CallBack_t DefProc) {
           fDefIdx = Idx;
           AddFirst(new TCallBackObject(fDefIdx,DefProc));
     }


//______________________________________________________________________________
// TCallBackObject& TWin32CallBackList::operator[](UINT message)
// This must be replace with [] operator

//______________________________________________________________________________
void TWin32CallBackList::AddCallBack(UINT message, CallBack_t Wnd_Proc, TGWin32Object *father)
{

//*-*  Create new CallBack entry "WndProc" for the event "message"
//*-*
//*-*  Looking for the existen object carrying a "message"
//*-*  and create a new ones if it needs.

   TObject *obj, *newobj = 0;
   UINT msg;

   TObjLink *lnk = fFirst;

//*-* Take first non-default object if any

   while (lnk && !newobj) {
     obj = lnk->GetObject();
     msg = ((TCallBackObject *)obj)->TakeMessage();

//*-*
//*-* Change CallBack Function entry for this event
//*-*
     if (msg == message) {
               // return  (TCallBackObject&) obj;  // This event was introduced
               ((TCallBackObject*) obj)->SetWindowProc(Wnd_Proc);
               ((TCallBackObject*) obj)->SetFather(father);
     }
     if (msg  > message) { // The new entry must be introduced
       newobj = (TObject *)new TCallBackObject(message, Wnd_Proc);
       obj = Before(obj);
       AddAfter(obj,newobj);
       ((TCallBackObject *)newobj)->SetMessage(message);
//       return (TCallBackObject& )newobj;
       ((TCallBackObject* )newobj)->SetWindowProc(Wnd_Proc);
       ((TCallBackObject* )newobj)->SetFather(father);
     }
     lnk = lnk->Next();
   }
   if (!newobj) {
    AddLast((TObject *)new TCallBackObject(message, Wnd_Proc, father));
   }
return;
}

#ifndef WIN32
//______________________________________________________________________________
TCallBackObject &TWin32CallBackList::operator[](UINT message) const {
//*-*  Looking for the existen object carrying a "message"
//*-*  and return either a found object or the default one

   TCallBackObject *obj;
   TObjLink *lnk = fFirst;

//*-* Take first non-default object if any

   while (lnk) {
     obj = (TCallBackObject *)lnk->GetObject();
     if (obj->TakeMessage() == message) return *obj;  // the event was found
     lnk = lnk->Next();
   }
   return  (TCallBackObject)*First(); // There is no special event- return a default one
}

#endif


//______________________________________________________________________________
LRESULT TWin32CallBackList::operator()(HWND hwnd, UINT message,
                               WPARAM wParam, LPARAM lParam) { // Call Callback

//*-*  Call CallBack Function with parameters
//*-*
//*-*  Looking for the existen object carrying a "message"
//*-*  and return either a found object or the default one

   typedef LRESULT (CALLBACK *CallFatherProc)(void *,HWND,UINT,WPARAM,LPARAM);

   TCallBackObject *obj;
   TObjLink *lnk = fFirst;
   CallBack_t fWndProc;
   UINT msg;
   TGWin32Object *winobj = 0;
   CallFatherProc lpfProc;

   fWndProc = ((TCallBackObject *)First())->TakeWindowProc(); // There is no special event- return a default one

//*-* Take first non-default object if any

   while (lnk) {
     obj = (TCallBackObject *)lnk->GetObject();
     msg = obj->TakeMessage();
     if (msg == message) {
        fWndProc = obj->TakeWindowProc();  // the event was found
                winobj = obj->GetFather();
        lpfProc = (CallFatherProc) fWndProc;
     break;
     }
     lnk = lnk->Next();
   }

   return winobj ? (LRESULT) (lpfProc(winobj,hwnd, message, wParam, lParam))
                 : ((WNDPROC)fWndProc)(hwnd, message, wParam, lParam);

 }


// ClassImp(TCallBackObject)

#ifndef WIN32
//______________________________________________________________________________
TCallBackObject::TCallBackObject(){
   fMessage = 0;
   fWinProc =(WNDPROC*)DefWindowProc;
}
#endif
//______________________________________________________________________________
TCallBackObject::TCallBackObject(UINT Msg, CallBack_t WinProc, TGWin32Object *father){
   fMessage = Msg;
   fWinProc = WinProc;
   fFather  = father;
}
