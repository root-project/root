// @(#)root/base:$Name:  $:$Id: TQObject.h,v 1.8 2001/04/20 17:29:57 rdm Exp $
// Author: Valeriy Onuchin & Fons Rademakers   15/10/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TQObject
#define ROOT_TQObject

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// This is the ROOT implementation of the Qt object communication       //
// mechanism (see also http://www.troll.no/qt/metaobjects.html)         //
//                                                                      //
// Signals and slots are used for communication between objects.        //
// When an object has changed in some way that might be interesting     //
// for the outside world, it emits a signal to tell whoever is          //
// listening. All slots that are connected to this signal will be       //
// activated (called).  It is even possible to connect a signal         //
// directly to  another signal (this will emit the second signal        //
// immediately whenever the first is emitted.) There is no limitation   //
// on the number of slots that can be connected to a signal.            //
// The slots will be activated in the order they were connected         //
// to the signal. This mechanism allows objects to be easily reused,    //
// because the object that emits a signal does not need to know         //
// to what the signals are connected to.                                //
// Together, signals and slots make up a powerfull component            //
// programming mechanism.                                               //
//                                                                      //
// This implementation is provided by                                   //
// Valeriy Onuchin (onuchin@sirius.ihep.su).                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TClass
#include "TClass.h"
#endif

class TList;
class TObject;
class TQConnection;
class TQClass;



class TQObject {

friend class TQConnection;

protected:
   TList   *fListOfSignals;        //! list of signals from this object
   TList   *fListOfConnections;    //! list of connections to this object

   TList   *GetListOfClassSignals() const;
   TList   *GetListOfSignals() const { return fListOfSignals; }
   TList   *GetListOfConnections() const { return fListOfConnections; }

   virtual void *GetSender() { return this; }

   static Bool_t ConnectToClass(TQObject *sender,
                                const char *signal,
                                TClass *receiver_class,
                                void *receiver,
                                const char *slot);

   static Bool_t ConnectToClass(const char *sender_class,
                                const char *signal,
                                TClass *receiver_class,
                                void *receiver,
                                const char *slot);
public:
   TQObject();
   virtual ~TQObject();

   void  Emit(const char *signal);
   void  Emit(const char *signal, Double_t param);
   void  Emit(const char *signal, Long_t   param);
   void  Emit(const char *signal, const char *params);
   void  Emit(const char *signal, Long_t  *paramArr);
   void  Emit(const char *signal, Char_t   param)
         { Emit(signal, (Long_t)param); }
   void  Emit(const char *signal, UChar_t param)
         { Emit(signal, (Long_t)param); }
   void  Emit(const char *signal, Short_t param)
         { Emit(signal, (Long_t)param); }
   void  Emit(const char *signal, UShort_t param)
         { Emit(signal, (Long_t)param); }
   void  Emit(const char *signal, Int_t param)
         { Emit(signal, (Long_t)param); }
   void  Emit(const char *signal, UInt_t param)
         { Emit(signal, (Long_t)param); }
   void  Emit(const char *signal, ULong_t param)
         { Emit(signal, (Long_t)param); }
   void  Emit(const char *signal, Float_t param)
         { Emit(signal, (Double_t)param); }

   Bool_t Connect(const char *signal,
                  const char *receiver_class,
                  void *receiver,
                  const char *slot);

   Bool_t Disconnect(const char *signal = 0,
                     void *receiver = 0,
                     const char *slot = 0);

   virtual void   HighPriority(const char *signal_name,
                               const char *slot_name = 0);

   virtual void   LowPriority(const char *signal_name,
                              const char *slot_name = 0);

   virtual Bool_t HasConnection(const char *signal_name) const;
   virtual Int_t  NumberOfSignals() const;
   virtual Int_t  NumberOfConnections() const;
   virtual void   Connected(const char * /*signal_name*/) { }
   virtual void   Disconnected(const char * /*signal_name*/) { }

   virtual void   Destroyed()
                  { Emit("Destroyed()"); }                 //*SIGNAL*
   virtual void   ChangedBy(const char *method)
                  { Emit("ChangedBy(char*)", method); }    //*SIGNAL*
   virtual void   Message(const char *msg)
                  { Emit("Message(char*)", msg); }         //*SIGNAL*

   static Bool_t  Connect(TQObject *sender,
                          const char *signal,
                          const char *receiver_class,
                          void *receiver,
                          const char *slot);

   static Bool_t  Connect(const char *sender_class,
                          const char *signal,
                          const char *receiver_class,
                          void *receiver,
                          const char *slot);

   static Bool_t  Disconnect(TQObject *sender,
                             const char *signal = 0,
                             void *receiver = 0,
                             const char *slot = 0);

   static Bool_t  Disconnect(const char *class_name,
                             const char *signal,
                             void *receiver = 0,
                             const char *slot = 0);

   static void    LoadRQ_OBJECT();

   ClassDef(TQObject,1) //Base class for object communication mechanism
};

R__EXTERN void *gTQSender;   // the latest sender object



class TQObjSender : public TQObject {

protected:
   void   *fSender;    //delegation object

   virtual void *GetSender() { return fSender; }

public:
   TQObjSender() : TQObject() { }
   virtual ~TQObjSender() { Disconnect(); }

   virtual void SetSender(void *sender) { fSender = sender; }

   ClassDef(TQObjSender,0) //Used to "delegate" TQObject functionality
                           //to interpreted classes, see also RQ_OBJECT.h
};



// This class makes it possible to have a single connection from
// all objects of the same class
class TQClass : public TQObject, public TClass {

friend class TQObject;

public:
   TQClass(const char *name, Version_t cversion,
           const char *dfil = 0, const char *ifil = 0,
           Int_t dl = 0, Int_t il = 0) :
           TQObject(), TClass(name, cversion, dfil, ifil, dl, il) { }

   virtual ~TQClass() { Disconnect(); }

   ClassDef(TQClass,0)  // Class with connections
};

// Global function which simplifies making connections in interpreted
// ROOT session
//
//  ConnectCINT      - connects to interpreter(CINT) command

extern Bool_t ConnectCINT(TQObject *sender, char *signal, char *slot);


//---- ClassImpQ macro ----------------------------------------------
//
// This macro corresponds to the ClassImp macro and should be used
// for classes derived from TQObject instead of the ClassImp macro.
// This macro makes it possible to have a single connection from
// all objects of the same class.

#define ClassImpQ(name) \
   void name::Dictionary() { \
      fgIsA = new TQClass(Class_Name(), Class_Version(),  \
                          DeclFileName(), ImplFileName(),  \
                          DeclFileLine(), ImplFileLine()); \
   } \
   _ClassImp_(name)

#endif
