// @(#)root/base:$Name:  $:$Id: TQObject.cxx,v 1.10 2001/04/22 23:09:18 rdm Exp $
// Author: Valeriy Onuchin & Fons Rademakers   15/10/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// This is the ROOT implementation of the Qt object communication       //
// mechanism (see also http://www.troll.no/qt/metaobjects.html)         //
//                                                                      //
// Signals and slots are used for communication between objects.        //
// When an object has changed in some way that might be interesting     //
// for the outside world, it emits a signal to tell whoever is          //
// listening. All slots that are connected to this signal will be       //
// activated (called). It is even possible to connect a signal          //
// directly to another signal (this will emit the second signal         //
// immediately whenever the first is emitted.) There is no limitation   //
// on the number of slots that can be connected to a signal.            //
// The slots will be activated in the order they were connected         //
// to the signal. This mechanism allows objects to be easily reused,    //
// because the object that emits a signal does not need to know         //
// to which objects the signals are connected.                          //
// Together, signals and slots make up a powerfull component            //
// programming mechanism.                                               //
//                                                                      //
// This implementation is provided by                                   //
// Valeriy Onuchin (onuchin@sirius.ihep.su).                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//***************************** Signals *****************************
//___________________________________________________________________
//
//             Destroyed()
//
// Signal emitted when object is destroyed.
// This signal could be connected to some garbage-collector object.
//
//___________________________________________________________________
//
//             ChangedBy(const char *method_name)
//
// This signal is emitted when some important data members of
// the object were changed. method_name parameter can be used
// as an identifier of the modifier method.
//
//___________________________________________________________________
//
//             Message(const char *msg)
//
// General purpose message signal
//
/////////////////////////////////////////////////////////////////////

#include "TQObject.h"
#include "TQConnection.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TMethod.h"
#include "TBaseClass.h"
#include "TDataType.h"
#include "TInterpreter.h"
#include "TClass.h"
#include "G__ci.h"
#include <iostream.h>

#ifdef HAVE_CONFIG
#include "config.h"
#endif


void *gTQSender; // A pointer to the object that sent the last signal.
                 // Getting access to the sender might be practical
                 // when many signals are connected to a single slot.


ClassImpQ(TQObject)
ClassImpQ(TQObjSender)
ClassImpQ(TQClass)

/////////////////// internal use static functions ///////////////////
//______________________________________________________________________________
static char *ResolveTypes(const char *method)
{
   // Resolve any typedefs in the method signature. For example:
   // func(Float_t,Int_t) becomes func(float,int).
   // The returned string must be deleted by the user.

   if (!method || !*method) return 0;

   char *str = new char[strlen(method)+1];
   if (str) strcpy(str, method);

   TString res;

   char *s = strtok(str, "(");
   res = s;
   res += "(";

   Bool_t first = kTRUE;
   while ((s = strtok(0, ",)"))) {
      char *s1, s2 = 0;
      if ((s1 = strchr(s, '*'))) {
         *s1 = 0;
         s2  = '*';
      }
      if (!s1 && (s1 = strchr(s, '&'))) {
         *s1 = 0;
         s2  = '&';
      }
      TDataType *dt = gROOT->GetType(s);
      if (s1) *s1 = s2;
      if (!first) res += ",";
      if (dt) {
         res += dt->GetFullTypeName();
         if (s1) res += s1;
      } else
         res += s;
      first = kFALSE;
   }

   res += ")";

   delete [] str;
   str = new char[res.Length()+1];
   strcpy(str, res.Data());

   return str;
}

//______________________________________________________________________________
static char *CompressName(const char *method_name)
{
   // Removes "const" words and blanks from full (with prototype)
   // method name.
   //
   //  Example: CompressName(" Draw(const char *, const char *,
   //                               Option_t * , Int_t , Int_t)")
   //
   // Returns the string "Draw(char*,char*,char*,int,int)"
   // The returned string must be deleted by the user.

   if (!method_name || !*method_name) return 0;

   char *str = new char[strlen(method_name)+1];
   if (str) strcpy(str, method_name);

   char *tmp = str;

   // substitute "const" with white spaces
   while ((tmp = strstr(tmp,"const"))) {
      for (int i = 0; i < 5; i++) *(tmp+i) = ' ';
   }

   tmp = str;
   char *s;
   s = str;

   Bool_t quote = kFALSE;
   while (*tmp) {
      if (*tmp == '\"')
         quote = quote ? kFALSE : kTRUE;
      if (*tmp != ' ' || quote)
         *s++ = *tmp;
      tmp++;
   }
   *s = '\0';

   s = ResolveTypes(str);

   delete [] str;

   return s;
}

//______________________________________________________________________________
static TMethod *GetMethodWithPrototype(TClass *cl, const char *method,
                                       const char *proto)
{
   // Almost the same as TClass::GetMethodWithPrototype().

   if (!gInterpreter) return 0;

   Long_t faddr = (Long_t)gInterpreter->GetInterfaceMethodWithPrototype(cl,
                                    (char *)method, (char *)proto);
   if (!faddr) return 0;

   TMethod *m;
   TIter next_method(cl->GetListOfMethods());

   // Look for a method in this class
   while ((m = (TMethod *) next_method())) {
      if (faddr == (Long_t)m->InterfaceMethod()) return m;
   }

   TIter next_base(cl->GetListOfBases());
   TBaseClass *base;

   // loop over all base classes
   while ((base = (TBaseClass *)next_base())) {
      TClass *c;
      if ((c = base->GetClassPointer())) {
         if ((m = GetMethodWithPrototype(c, method, proto))) return m;
      }
   }
   return 0;
}

//______________________________________________________________________________
static TMethod *GetMethod(TClass* cl, const char *method, const char *params)
{
   // Almost the same as TClass::GetMethod().

   if (!gInterpreter) return 0;

   Long_t faddr = (Long_t)gInterpreter->GetInterfaceMethod(cl,
                                    (char *)method, (char *)params);
   if (!faddr) return 0;

   TMethod *m;
   TIter next_method(cl->GetListOfMethods());

   // Look for a method in this class
   while ((m = (TMethod *) next_method())) {
      if (faddr == (Long_t)m->InterfaceMethod()) return m;
   }

   TIter next_base(cl->GetListOfBases());
   TBaseClass *base;

   // loop over all base classes
   while ((base = (TBaseClass *)next_base())) {
      TClass *c;
      if ((c = base->GetClassPointer())) {
         if ((m = GetMethod(c,method,params))) return m;
      }
   }
   return 0;
}

//______________________________________________________________________________
static Bool_t CheckConnectArgs(TClass *sender_class, const char *signal,
                               TClass *receiver_class, const char *slot)
{
   // Checking of consitency of sender/receiver methods/arguments.

   char *signal_method = new char[strlen(signal)+1];
   if (signal_method) strcpy(signal_method, signal);

   char *signal_proto;
   char *tmp;

   if ((signal_proto = strchr(signal_method,'('))) {
      // substitute first '(' symbol with '\0'
      *signal_proto++ = '\0';
      // substitute last ')' symbol with '\0'
      if ((tmp = strrchr(signal_proto,')'))) *tmp = '\0';
   }

   if (!signal_proto) signal_proto = ""; // avoid zero strings

   TMethod *signalMethod = GetMethodWithPrototype(sender_class,
                                                  signal_method,
                                                  signal_proto);
   if (!signalMethod) {
      sender_class->Error("CheckConnectArgs",
                          Form("method %s(%s) does not exist",
                          signal_method, signal_proto));
      return kFALSE;
   }

#if defined(CHECK_COMMENT_STRING)
   const char *comment = signalMethod->GetCommentString();

   if (!comment || !strlen(comment) || strstr(comment,"*SIGNAL")){
      sender_class->Error("CheckConnectArgs",
                          Form("method %s(%s),"
                               "to declare signal use comment //*SIGNAL*",
                          signal_method, signal_proto));
      return kFALSE;
   }
#endif

   // cleaning
   if (signal_method)  {
      delete [] signal_method;
      signal_method =  0;
   }

   // case of slot_method is intrepreted function
   if (!receiver_class) return kTRUE;

   char *slot_method = new char[strlen(slot)+1];
   if (slot_method) strcpy(slot_method, slot);

   char *slot_proto;
   char *slot_params = 0;

   if ((slot_proto =  strchr(slot_method,'('))) {

      // substitute first '(' symbol with '\0'
      *slot_proto++ = '\0';

      // substitute last ')' symbol with '\0'
      if ((tmp = strrchr(slot_proto,')'))) *tmp = '\0';
   }

   if (!slot_proto) slot_proto = "";     // avoid zero strings
   if (slot_proto &&
       (slot_params = strchr(slot_proto,'='))) *slot_params = ' ';

   TMethod *slotMethod  = !slot_params ?
                          GetMethodWithPrototype(receiver_class,
                                                 slot_method,
                                                 slot_proto) :
                          GetMethod(receiver_class,
                                    slot_method, slot_params);
   if (!slotMethod) {
      if (!slot_params) {
         receiver_class->Error("CheckConnectArgs",
                               Form("method %s(%s) does not exist",
                               slot_method, slot_proto));
      } else {
         receiver_class->Error("CheckConnectArgs",
                               Form("method %s(%s) does not exist",
                               slot_method, slot_params));
      }
      return kFALSE;
   }

#if defined(CHECK_ARGS_NUMBER)
   if ((slotMethod->GetNargsOpt() >= 0) &&
       (signalMethod->GetNargs() <
        (slotMethod->GetNargs() - slotMethod->GetNargsOpt()))) {
      sender_class->Error("CheckConnectArgs",
                          "inconsitency in numbers of arguments");
      return kFALSE;
   }
#endif

   // cleaning
   if (slot_method) {
      delete [] slot_method;
      slot_method =  0;
   }

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//    TQConnectionList is the named list of connections,                      //
//    see also TQConnection class.                                            //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////
class TQConnectionList : public TList {

public:
   TQConnectionList(const char *name) : TList() { fName = name; }
   virtual ~TQConnectionList();

   Bool_t Disconnect(void *receiver=0, const char *slot_name=0);
   void  ls(Option_t *option = "") const;
   void  Print(Option_t *option = "") const ;
};

//______________________________________________________________________________
TQConnectionList::~TQConnectionList()
{
   // Destructor.

   TIter next(this);
   TQConnection *connection;

   while ((connection = (TQConnection*)next())) {
      // remove this from feed back reference list
      connection->Remove(this);
      if (connection->IsEmpty()) delete connection;
   }
}

//______________________________________________________________________________
Bool_t TQConnectionList::Disconnect(void *receiver, const char *slot_name)
{
   // Remove connection from the list. For more info see
   // TQObject::Disconnect()

   TQConnection *connection = 0;
   Bool_t return_value = kFALSE;

   TObjLink *lnk = FirstLink();
   TObjLink *savlnk; // savlnk is used when link is deleted

   while (lnk) {
      connection = (TQConnection*)lnk->GetObject();
      const char *name = connection->GetName();
      void *obj = connection->GetReceiver();

      if (!slot_name || !strlen(slot_name)
                     || !strcmp(name,slot_name)) {

         if (!receiver || (receiver == obj)) {
            return_value = kTRUE;
            savlnk = lnk->Next();   // keep next link ..
            Remove(lnk);
            lnk = savlnk;           // current link == saved ...
            connection->Remove(this);      // remove back reference
            if (connection->IsEmpty()) SafeDelete(connection);
            continue;               // .. continue from saved link
         }
      }
      lnk = lnk->Next();
   }
   return return_value;
}

//______________________________________________________________________________
void TQConnectionList::ls(Option_t *option) const
{
   // List signal name and list all connections in this signal list.

   cout <<  "TQConnectionList:" << "\t" << GetName() << endl;
   ((TQConnectionList*)this)->ForEach(TQConnection,Print)(option);
}

//______________________________________________________________________________
void TQConnectionList::Print(Option_t *option) const
{
   // Print signal name.

   cout << "TQConnectionList:" << "\t" << GetName() << endl;
}


//______________________________________________________________________________
TQObject::TQObject()
{
   // TQObject Constructor.
   // Comment:
   //  - In order to minimize memory allocation fListOfSignals and
   //    fListOfConnections are allocated only if it is neccesary
   //  - When fListOfSignals/fListOfConnections are empty they will
   //    be deleted

   fListOfSignals     = 0;
   fListOfConnections = 0;
}

//______________________________________________________________________________
TQObject::~TQObject()
{
   // TQObject Destructor.
   //    - delete all connections and signal list

   Destroyed();   // emit "Destroyed()" signal

   TQConnectionList *list = 0;

   if (fListOfSignals) {
      TIter next(fListOfSignals);

      // delete all signals lists
      while ((list = (TQConnectionList*)next())) {
         SafeDelete(list);
      }
      SafeDelete(fListOfSignals);   // delete list of signals
   }

   // loop over all connections and remove references to this object
   if (fListOfConnections) {
      TIter next_connection(fListOfConnections);
      TQConnection *connection = 0;

      while ((connection = (TQConnection*)next_connection())) {
         TIter next_list(connection);
         while ((list = (TQConnectionList*)next_list())) {
            list->Remove(connection);
            if (list->IsEmpty()) SafeDelete(list);
         }
      }
   }
}

//______________________________________________________________________________
TList *TQObject::GetListOfClassSignals() const
{
   // Returns pointer to list of signals of this class.

   TQClass *qcl = 0;
#ifdef R__RTTI
   qcl = dynamic_cast<TQClass*>(IsA());
#else
   if (IsA()->IsA() == TQClass::Class())
      qcl = (TQClass*) IsA();
#endif
   return qcl ? qcl->fListOfSignals : 0; //!!
}

//______________________________________________________________________________
void TQObject::HighPriority(const char *signal_name, const char *slot_name)
{
   // 1. If slot_name = 0 => makes signal defined by the signal_name
   //    to be the first in the fListOfSignals, this decreases
   //    the time for lookup.
   // 2. If slot_name != 0 => makes slot defined by the slot_name
   //    to be executed first when signal_name is emitted.

   TQConnectionList *clist = 0;
   TIter    next_list(fListOfSignals);
   while ((clist = (TQConnectionList*)next_list())) {
      if (!strcmp(signal_name,clist->GetName()))
         break;
   }

   if (!clist)  return;      // not found
   if (!slot_name)  {        // update list of signal lists
      fListOfSignals->Remove(clist);   // remove and add first
      fListOfSignals->AddFirst(clist);
      return;
   } else {                   // slot_name != 0 , update signal list
      TQConnection *con = 0;
      TIter next_con(clist);
      while ((con = (TQConnection*)next_con())) {
         if (!strcmp(slot_name,con->GetName()))
            break;
      }

      if (!con) return;       // not found
      clist->Remove(con);     // remove and add as first
      clist->AddFirst(con);
   }
}

//______________________________________________________________________________
void TQObject::LowPriority(const char *signal_name, const char *slot_name)
{
   // 1. If slot_name = 0 => makes signal defined by the signal_name
   //    to be the last in the fListOfSignals, this increase the time
   //    for lookup.
   // 2. If slot_name != 0 => makes slot defined by the slot_name
   //    to  be executed last when signal_name is emitted.

   TQConnectionList *clist = 0;
   TIter    next_list(fListOfSignals);
   while ((clist = (TQConnectionList*)next_list())) {
      if (!strcmp(signal_name,clist->GetName()))
         break;
   }

   if (!clist)   return;
   if (!slot_name)  {
      fListOfSignals->Remove(clist);   // remove and add first
      fListOfSignals->AddLast(clist);
      return;
   } else  {                  // slot_name != 0 , update signal list
      TQConnection *con = 0;
      TIter next_con(clist);
      while ((con = (TQConnection*)next_con())) {
         if (!strcmp(slot_name,con->GetName()))  break;
      }

      if (!con) return;
      clist->Remove(con);     // remove and add as last
      clist->AddLast(con);
   }
}

//______________________________________________________________________________
Bool_t TQObject::HasConnection(const char *signal_name) const
{
   // Return true if there is any object connected to this signal.
   // Only checks for object signals.

   if (!fListOfSignals)
      return kFALSE;

   register TQConnectionList *clist  = 0;
   char *signal = CompressName(signal_name);

   // check object signals
   TIter next_list(fListOfSignals);
   while ((clist = (TQConnectionList*)next_list())) {
      if (!strcmp(signal, clist->GetName())) {
         delete [] signal;
         return kTRUE;
      }
   }
   delete [] signal;
   return kFALSE;
}

//______________________________________________________________________________
Int_t TQObject::NumberOfSignals() const
{
   // Return number of signals for this object.

   if (fListOfSignals)
      return fListOfSignals->GetSize();
   return 0;
}

//______________________________________________________________________________
Int_t TQObject::NumberOfConnections() const
{
   // Return number of connections for this object.

   if (fListOfConnections)
      return fListOfConnections->GetSize();
   return 0;
}

//______________________________________________________________________________
void TQObject::Emit(const char *signal_name)
{
   // Acitvate signal without args
   // Example:
   //          theButton->Emit("Clicked()");

   TList *slist = GetListOfClassSignals();

   if (!slist && !fListOfSignals)
      return;

   gTQSender = GetSender();
   register TQConnectionList *clist  = 0;
   register TQConnection *connection = 0;

   char *signal = CompressName(signal_name);

   // execute class signals
   if (slist) {
      TIter nextcl_list(slist);
      while ((clist = (TQConnectionList*)nextcl_list())) {
         if (!strcmp(signal,clist->GetName())) break;
      }

      if (clist) {
         TIter nextcl(clist);
         while ((connection = (TQConnection*)nextcl())) {
            connection->ExecuteMethod();
         }
      }
   }
   if (!fListOfSignals) {
      delete [] signal;
      return;
   }

   // execute object signals
   TIter next_list(fListOfSignals);
   while ((clist = (TQConnectionList*)next_list())) {
      if (!strcmp(signal,clist->GetName())) break;
   }
   if (!clist) {
      delete [] signal;
      return;
   }

   TIter next(clist);
   while ((connection = (TQConnection*)next())) {
      connection->ExecuteMethod();
   }
   delete [] signal;
}

//______________________________________________________________________________
void TQObject::Emit(const char *signal_name, Long_t param)
{
   // Activate signal with single parameter
   // Example:
   //          theButton->Emit("Clicked(int)",id)

   TList *slist = GetListOfClassSignals();

   if (!slist && !fListOfSignals)
      return;

   gTQSender = GetSender();
   register TQConnectionList *clist  = 0;
   register TQConnection *connection = 0;

   char *signal = CompressName(signal_name);

   // execute class signals
   if (slist) {
      TIter nextcl_list(slist);
      while ((clist = (TQConnectionList*)nextcl_list())) {
         if (!strcmp(signal,clist->GetName())) break;
      }

      if (clist) {
         TIter nextcl(clist);
         while ((connection = (TQConnection*)nextcl())) {
            connection->ExecuteMethod(param);
         }
      }
   }
   if (!fListOfSignals) {
      delete [] signal;
      return;
   }

   // execute object signals
   TIter next_list(fListOfSignals);
   while ((clist = (TQConnectionList*)next_list())) {
      if (!strcmp(signal,clist->GetName())) break;
   }
   if (!clist) {
      delete [] signal;
      return;
   }

   TIter next(clist);
   while ((connection = (TQConnection*)next())) {
      connection->ExecuteMethod(param);
   }
   delete [] signal;
}

//______________________________________________________________________________
void TQObject::Emit(const char *signal_name, Double_t param)
{
   // Activate signal with single parameter.

   TList *slist = GetListOfClassSignals();

   if (!slist && !fListOfSignals)
      return;

   gTQSender = GetSender();
   register TQConnectionList *clist  = 0;
   register TQConnection *connection = 0;

   char *signal = CompressName(signal_name);

   // execute class signals
   if (slist) {
      TIter nextcl_list(slist);
      while ((clist = (TQConnectionList*)nextcl_list())) {
         if (!strcmp(signal,clist->GetName())) break;
      }

      if (clist) {
         TIter nextcl(clist);
         while ((connection = (TQConnection*)nextcl())) {
            connection->ExecuteMethod(param);
         }
      }
   }
   if (!fListOfSignals) {
      delete [] signal;
      return;
   }

   // execute object signals
   TIter next_list(fListOfSignals);
   while ((clist = (TQConnectionList*)next_list())) {
      if (!strcmp(signal,clist->GetName())) break;
   }

   if (!clist) {
      delete [] signal;
      return;
   }

   TIter next(clist);
   while ((connection = (TQConnection*)next())) {
      connection->ExecuteMethod(param);
   }
   delete [] signal;
}

//______________________________________________________________________________
void TQObject::Emit(const char *signal_name, const char *params)
{
   // Activate signal with parameter text string.
   // Example:
   //          myObject->Emit("Error(char*)","Fatal error");

   TList *slist = GetListOfClassSignals();

   if (!slist && !fListOfSignals)
      return;

   gTQSender = GetSender();
   register TQConnectionList *clist  = 0;
   register TQConnection *connection = 0;

   char *signal = CompressName(signal_name);

   // execute class signals
   if (slist) {
      TIter nextcl_list(slist);
      while ((clist = (TQConnectionList*)nextcl_list())) {
         if (!strcmp(signal,clist->GetName())) break;
      }

      if (clist) {
         TIter nextcl(clist);
         while ((connection = (TQConnection*)nextcl())) {
            connection->ExecuteMethod(params);
         }
      }
   }
   if (!fListOfSignals) {
      delete [] signal;
      return;
   }

   // execute object signals
   TIter next_list(fListOfSignals);
   while ((clist = (TQConnectionList*)next_list())) {
      if (!strcmp(signal,clist->GetName())) break;
   }
   if (!clist) {
      delete [] signal;
      return;
   }

   TIter next(clist);
   while ((connection = (TQConnection*)next())) {
      connection->ExecuteMethod(params);
   }
   delete [] signal;
}

//______________________________________________________________________________
void TQObject::Emit(const char *signal_name, Long_t *paramArr)
{
   // Emit a signal with a varying number of arguments,
   // paramArr is an array of the parameters.
   // Note: any parameter should be converted to long type.
   // Example:
   //    TQObject *processor; // data processor
   //    TH1F     *hist;      // filled with processor results
   //
   //    processor->Connect("Evaluated(Float_t,Float_t)",
   //                       "TH1F",hist,"Fill12(Axis_t x, Axis_t)");
   //
   //    Long_t args[2];
   //    args[0] = (Long_t)processor->GetValue(1);
   //    args[1] = (Long_t)processor->GetValue(2);
   //
   //    processor->Emit("Evaluated(Float_t,Float_t)",args);

   TList *slist = GetListOfClassSignals();

   if (!slist && !fListOfSignals)
      return;

   gTQSender = GetSender();
   register TQConnectionList *clist  = 0;
   register TQConnection *connection = 0;

   char *signal = CompressName(signal_name);

   // execute class signals
   if (slist) {
      TIter nextcl_list(slist);
      while ((clist = (TQConnectionList*)nextcl_list())) {
         if (!strcmp(signal,clist->GetName())) break;
      }

      if (clist) {
         TIter nextcl(clist);
         while ((connection = (TQConnection*)nextcl())) {
            connection->ExecuteMethod(paramArr);
         }
      }
   }
   if (!fListOfSignals) {
      delete [] signal;
      return;
   }

   // execute object signals
   TIter next_list(fListOfSignals);
   while ((clist = (TQConnectionList*)next_list())) {
      if (!strcmp(signal,clist->GetName())) break;
   }
   if (!clist) {
      delete [] signal;
      return;
   }

   TIter next(clist);
   while ((connection = (TQConnection*)next())) {
      connection->ExecuteMethod(paramArr);
   }
   delete [] signal;
}

//______________________________________________________________________________
Bool_t TQObject::ConnectToClass(TQObject *sender,
                                const char *signal,
                                TClass *cl,
                                void *receiver,
                                const char *slot)
{
   // Create connection between sender and receiver.
   // Receiver class needs to have a dictionary.

   // sender should be TQObject
   if (!sender->IsA()->InheritsFrom(TQObject::Class()))
      return kFALSE;

   // remove "const" and strip blanks
   char *signal_name = CompressName(signal);
   char *slot_name   = CompressName(slot);

   // check consitency of signal/slot methods/args
   if (!CheckConnectArgs(sender->IsA(), signal_name, cl, slot_name))
      return kFALSE;

   if (!sender->fListOfSignals)
      sender->fListOfSignals = new TList();

   TQConnectionList *clist=0;
   TIter next_list(sender->fListOfSignals);

   while ((clist = (TQConnectionList*)next_list())) {
      if (!strcmp(signal_name,clist->GetName())) break;
   }

   if (!clist) {
      clist = new TQConnectionList(signal_name);
      sender->fListOfSignals->Add(clist);
   }

   TIter next(clist);
   TQConnection *connection = 0;

   while ((connection = (TQConnection*)next())) {
      if (!strcmp(slot_name,connection->GetName()) &&
          (receiver == connection->GetReceiver())) break;
   }

   if (!connection)
      connection = new TQConnection(cl, receiver, slot_name);

   // check to prevent multiple entries
   if (!clist->FindObject(connection)) {
      clist->Add(connection);
      if (!connection->FindObject(clist)) connection->Add(clist);
      sender->Connected(signal_name);
   }

   // cleaning
   if (signal_name) { delete [] signal_name; signal_name = 0; }
   if (slot_name) { delete [] slot_name; slot_name = 0; }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TQObject::ConnectToClass(const char *class_name,
                                const char *signal,
                                TClass *cl,
                                void *receiver,
                                const char *slot)
{
   // This method allows to make connection from any object
   // of the same class to the receiver object.
   // Receiver class needs to have a dictionary.

   TClass *sender = gROOT->GetClass(class_name);

   // sender class should be TQObject (i.e. TQClass)
   if (!sender || !sender->IsA()->InheritsFrom(TQObject::Class()))
      return kFALSE;

   TList *slist = ((TQClass*)sender)->fListOfSignals;
   char *signal_name = CompressName(signal);
   char *slot_name   = CompressName(slot);

   // check consitency of signal/slot methods/args
   if (!CheckConnectArgs(sender, signal_name, cl, slot_name))
      return kFALSE;

   TQConnectionList *clist = 0;

   if (!slist)
      ((TQClass*)sender)->fListOfSignals = slist = new TList();

   TIter next_list(slist);
   while ((clist = (TQConnectionList*)next_list())) {
      if (!strcmp(signal_name,clist->GetName())) break;
   }

   if (!clist) {
      clist = new TQConnectionList(signal_name);
      slist->Add(clist);
   }

   TQConnection *connection = 0;
   TIter next(clist);

   while ((connection = (TQConnection*)next())) {
      if (!strcmp(slot_name,connection->GetName()) &&
          (receiver == connection->GetReceiver())) break;
   }

   if (!connection)
      connection = new TQConnection(cl, receiver, slot_name);

   // check to prevent multiple entries
   if (!clist->FindObject(connection)) {
      clist->Add(connection);
      if (!connection->FindObject(clist)) connection->Add(clist);
      ((TQClass*)sender)->Connected(signal_name);
   }

   // cleaning
   if (signal_name)  { delete [] signal_name; signal_name = 0; }
   if (slot_name) { delete [] slot_name; slot_name = 0; }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TQObject::Connect(TQObject *sender,
                         const char *signal,
                         const char *cl,
                         void *receiver,
                         const char *slot)
{
   // Create connection between sender and receiver.
   // Signal and slot string must have a form:
   //    "Draw(char*, Option_t* ,Int_t)"
   // All blanks and "const" words will be removed,
   //
   // cl != 0 - class name, it can be class with or
   //           without dictionary, e.g interpreted class.
   // Example:
   //       TGButton *myButton;
   //       TH2F     *myHist;
   //
   //       TQObject::Connect(myButton,"Clicked()",
   //                         "TH2F", myHist,"Draw(Option_t*)");
   //
   // cl == 0 - corresponds to function (interpereted or global)
   //           the name of the function is defined by the slot string,
   //           parameter receiver should be 0.
   // Example:
   //       TGButton *myButton;
   //       TH2F     *myHist;
   //
   //       TQObject::Connect(myButton,"Clicked()",
   //                         0, 0,"hsimple()");
   //
   // Warning:
   //  If receiver is class not derived from TQObject and going to be
   //  deleted, disconnect all connections to this receiver.
   //  In case of class derived from TQObject it is done automatically.

   if (cl) {
      TClass *rcv_cl = gROOT->GetClass(cl);
      if (rcv_cl) return ConnectToClass(sender, signal, rcv_cl, receiver, slot);
   }

   // the following is the case of receiver class without dictionary
   // e.g. interpreted class or function.

   // sender should be TQObject
   if (!sender->IsA()->InheritsFrom(TQObject::Class()))
      return kFALSE;

   // remove "const" and strip blanks
   char *signal_name = CompressName(signal);
   char *slot_name   = CompressName(slot);

   // Warning! No check on consitency of signal/slot methods/args

   if (!sender->fListOfSignals) sender->fListOfSignals = new TList();

   TQConnectionList *clist = 0;
   TIter next_list(sender->fListOfSignals);

   while ((clist = (TQConnectionList*)next_list())) {
      if (!strcmp(signal_name,clist->GetName())) break;
   }

   if (!clist) {
      clist = new TQConnectionList(signal_name);
      sender->fListOfSignals->Add(clist);
   }

   TQConnection *connection = 0;
   TIter next(clist);

   while ((connection = (TQConnection*)next())) {
      if (!strcmp(slot_name,connection->GetName()) &&
          (receiver == connection->GetReceiver())) break;
   }

   if (!connection)
      connection = new TQConnection(cl, receiver, slot_name);

   // check to prevent multiple entries
   if (!clist->FindObject(connection)) {
      clist->Add(connection);
      if (!connection->FindObject(clist)) connection->Add(clist);
      sender->Connected(signal_name);
   }

   // cleaning
   if (signal_name) { delete [] signal_name; signal_name = 0; }
   if (slot_name) { delete [] slot_name; slot_name = 0; }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TQObject::Connect(const char *class_name,
                         const char *signal,
                         const char *cl,
                         void *receiver,
                         const char *slot)
{
   // This method allows to make a connection from any object
   // of the same class to a single slot.
   // Signal and slot string must have a form:
   //    "Draw(char*, Option_t* ,Int_t)"
   // All blanks and "const" words will be removed,
   //
   // cl != 0 - class name, it can be class with or
   //           without dictionary, e.g interpreted class.
   // Example:
   //       TGButton *myButton;
   //       TH2F     *myHist;
   //
   //       TQObject::Connect("TGButton", "Clicked()",
   //                         "TH2F", myHist, "Draw(Option_t*)");
   //
   // cl == 0 - corresponds to function (interpereted or global)
   //           the name of the function is defined by the slot string,
   //           parameter receiver should be 0.
   // Example:
   //       TGButton *myButton;
   //       TH2F     *myHist;
   //
   //       TQObject::Connect("TGButton", "Clicked()",
   //                         0, 0, "hsimple()");
   //
   // Warning:
   //  If receiver class not derived from TQObject and going to be
   //  deleted, disconnect all connections to this receiver.
   //  In case of class derived from TQObject it is done automatically.

   if (cl) {
      TClass *rcv_cl = gROOT->GetClass(cl);
      if (rcv_cl) return ConnectToClass(class_name, signal, rcv_cl, receiver,
                                        slot);
   }

   // the following is case of receiver class without dictionary
   // e.g. interpreted class or function.

   TClass *sender = gROOT->GetClass(class_name);

   // sender class should be TQObject (i.e. TQClass)
   if (!sender || !sender->IsA()->InheritsFrom(TQObject::Class()))
      return kFALSE;

   TList *slist = ((TQClass*)sender)->fListOfSignals;

   char *signal_name = CompressName(signal);
   char *slot_name   = CompressName(slot);

   // Warning! No check on consitency of the signal/slot methods/args

   TQConnectionList *clist = 0;

   if (!slist) {
      slist = ((TQClass*)sender)->fListOfSignals = new TList();
   }

   TIter next_list(slist);

   while ((clist = (TQConnectionList*)next_list()))  {
      if (!strcmp(signal_name,clist->GetName())) break;
   }

   if (!clist) {
      clist = new TQConnectionList(signal_name);
      slist->Add(clist);
   }

   TQConnection *connection = 0;
   TIter next(clist);

   while ((connection = (TQConnection*)next())) {
      if (!strcmp(slot_name,connection->GetName()) &&
          (receiver == connection->GetReceiver())) break;
   }

   if (!connection)
      connection = new TQConnection(cl, receiver, slot_name);

   // check to prevent multiple entries
   if (!clist->FindObject(connection)) {
      clist->Add(connection);
      if (!connection->FindObject(clist)) connection->Add(clist);
      ((TQClass*)sender)->Connected(signal_name);
   }

   // cleaning
   if (signal_name)  { delete [] signal_name; signal_name = 0; }
   if (slot_name) { delete [] slot_name; slot_name = 0; }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TQObject::Connect(const char *signal,
                         const char *receiver_class,
                         void *receiver,
                         const char *slot)
{
   // Non-static method is used to connect from the signal
   // of this object to the receiver slot.
   //
   // Warning! No check on consistency of sender/receiver
   // classes/methods.
   //
   // This method makes possible to have connection/signals from
   // interpreted class. See also RQ_OBJECT.h.

   // remove "const" and strip blanks
   char *signal_name = CompressName(signal);
   char *slot_name   = CompressName(slot);

   if (!fListOfSignals) fListOfSignals = new TList();

   TQConnectionList *clist = 0;
   TIter next_list(fListOfSignals);

   while ((clist = (TQConnectionList*)next_list())) {
      if (!strcmp(signal_name,clist->GetName())) break;
   }

   if (!clist) {
      clist = new TQConnectionList(signal_name);
      fListOfSignals->Add(clist);
   }

   TIter next(clist);
   TQConnection *connection = 0;

   while ((connection = (TQConnection*)next())) {
      if (!strcmp(slot_name,connection->GetName()) &&
          (receiver == connection->GetReceiver())) break;
   }

   if (!connection)
      connection = new TQConnection(receiver_class, receiver, slot_name);

   // check to prevent multiple entries
   if (!clist->FindObject(connection)) {
      clist->Add(connection);
      if (!connection->FindObject(clist)) connection->Add(clist);
      Connected(signal_name);
   }

   // cleaning
   if (signal_name) { delete [] signal_name; signal_name = 0; }
   if (slot_name) { delete [] slot_name; slot_name = 0; }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TQObject::Disconnect(TQObject *sender,
                            const char *signal,
                            void *receiver,
                            const char *slot)
{
   // Disconnects signal in object sender from slot_method in
   // object receiver. For objects derived from TQObject signal-slot
   // connection is removed when either of the objects involved
   // are destroyed.
   //
   // Disconnect() is typically used in three ways, as the following
   // examples shows:
   //
   //  - Disconnect everything connected to an object's signals:
   //       Disconnect(myObject);
   //  - Disconnect everything connected to a signal:
   //       Disconnect(myObject, "mySignal()");
   //  - Disconnect a specific receiver:
   //       Disconnect(myObject, 0, myReceiver, 0);
   //
   // 0 may be used as a wildcard in three of the four arguments,
   // meaning "any signal", "any receiving object" or
   // "any slot in the receiving object", respectively.
   //
   // The sender has no default and may never be 0
   // (you cannot disconnect signals from more than one object).
   //
   // If signal is 0, it disconnects receiver and slot_method
   // from any signal. If not, only the specified signal is
   // disconnected.
   //
   // If  receiver is 0, it disconnects anything connected to signal.
   // If not, slots in objects other than receiver are not
   // disconnected
   //
   // If slot_method is 0, it disconnects anything that is connected
   // to receiver.  If not, only slots named slot_method will be
   // disconnected, and all other slots are left alone.
   // The slot_method must be 0 if receiver is left out, so you
   // cannot disconnect a specifically-named slot on all objects.

   Bool_t return_value = kFALSE;
   Bool_t next_return  = kFALSE;

   if (!sender->GetListOfSignals()) return kFALSE;

   char *signal_name = CompressName(signal);
   char *slot_name   = CompressName(slot);

   TQConnectionList *slist = 0;
   TIter next_signal(sender->GetListOfSignals());

   while ((slist = (TQConnectionList*)next_signal()))   {
      if (!signal_name) {                // disconnect all signals
         next_return = slist->Disconnect(receiver,slot_name);
         return_value = return_value || next_return;

         if (slist->IsEmpty()) {
            sender->GetListOfSignals()->Remove(slist);
            SafeDelete(slist);            // delete empty list
         }
      } else if (signal && !strcmp(signal_name,slist->GetName())) {
         next_return = slist->Disconnect(receiver,slot_name);
         return_value = return_value || next_return;

         if (slist->IsEmpty()) {
            sender->GetListOfSignals()->Remove(slist);
            SafeDelete(slist);            // delete empty list
            break;
         }
      }
   }

   if (sender->GetListOfSignals() && sender->GetListOfSignals()->IsEmpty()) {
      SafeDelete(sender->fListOfSignals);
   }

   // cleaning
   if (signal_name) { delete [] signal_name; signal_name = 0; }
   if (slot_name) { delete [] slot_name; slot_name = 0; }

   return return_value;
}

//______________________________________________________________________________
Bool_t TQObject::Disconnect(const char *class_name,
                            const char *signal,
                            void *receiver,
                            const char *slot)
{
   // Disconnects "class signal". The class is defined by class_name.
   // See also Connect(class_name,signal,receiver,slot).

   TClass *sender = gROOT->GetClass(class_name);

   // sender should be TQClass (which derives from TQObject)
   if (!sender->IsA()->InheritsFrom(TQObject::Class()))
      return kFALSE;

   TQClass *qcl = (TQClass*)sender;   // cast TClass to TQClass
   return Disconnect(qcl, signal, receiver, slot);
}

//______________________________________________________________________________
Bool_t TQObject::Disconnect(const char *signal,
                            void *receiver,
                            const char *slot)
{
   // Disconnects signal of this object from slot of receiver.
   // Equivalent to Disconnect(this, signal, receiver, slot)

   return Disconnect(this, signal, receiver, slot);
}

//______________________________________________________________________________
void TQObject::Streamer(TBuffer &R__b)
{
   // Stream an object of class TQObject.

   if (R__b.IsReading()) {
      // nothing to read
   } else {
      // nothing to write
   }
}

//______________________________________________________________________________
void TQObject::LoadRQ_OBJECT()
{
   // Load RQ_OBJECT.h which contains the #define RQ_OBJECT needed to
   // let interpreted classes connect to signals of compiled classes.

   char rqh[128];
# ifdef ROOTINCDIR
      sprintf(rqh, "%s/RQ_OBJECT.h", ROOTINCDIR);
# else
      sprintf(rqh, "%s/include/RQ_OBJECT.h", gSystem->Getenv("ROOTSYS"));
# endif
      G__loadfile(rqh);
}

// Global function which simplifies making connection in interpreted
// ROOT session
//
//  ConnectCINT      - connects to interpreter(CINT) command

//______________________________________________________________________________
Bool_t ConnectCINT(TQObject *sender, char *signal, char *slot)
{
   TString str = "ProcessLine(=";
   str += '"';
   str += slot;
   str += '"';
   str += ")";
   return TQObject::Connect(sender,signal, "TInterpreter",
                            gInterpreter, str.Data());
}
