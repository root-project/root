// @(#)root/base:$Name:  $:$Id: TQConnection.cxx,v 1.4 2000/12/13 16:45:36 brun Exp $
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
// TQConnection class is an internal class, used in the object          //
// communication mechanism.                                             //
//                                                                      //
// TQConnection:                                                        //
//    -  is a list of signal_lists containing pointers                  //
//       to this connection                                             //
//    -  receiver is the object to which slot-method is applied         //
//                                                                      //
// This implementation is provided by                                   //
// Valeriy Onuchin (onuchin@sirius.ihep.su).                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TQConnection.h"
#include "TRefCnt.h"
#include "TClass.h"
#include "Api.h"
#include "G__ci.h"
#include <iostream.h>


ClassImpQ(TQConnection)

char *gTQSlotParams;          // used to pass string parameter


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TQSlot = slightly modified TMethodCall class                        //
//           used in the object communication mechanism                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
class TQSlot : public TObject, public TRefCnt {

protected:
   G__CallFunc   *fFunc;      // CINT method invocation environment
   Long_t         fOffset;    // offset added to object pointer
   TString        fName;      // full name of method
   Int_t          fExecuting; // true if one of this slot's ExecuteMethod methods is being called

public:
   TQSlot(TClass *cl, const char *method, const char *funcname);
   TQSlot(const char *class_name, const char *funcname);
   virtual ~TQSlot();

   const char *GetName() const { return fName.Data(); }

   void ExecuteMethod(void *object);
   void ExecuteMethod(void *object, Long_t param);
   void ExecuteMethod(void *object, Double_t param);
   void ExecuteMethod(void *object, const char *params);
   void ExecuteMethod(void *object, Long_t *paramArr);
   void Print(Option_t *opt= "") const;
   void ls(Option_t *opt= "") const { Print(opt); }
};


//______________________________________________________________________________
TQSlot::TQSlot(TClass *cl, const char *method_name,
               const char *funcname) : TObject(), TRefCnt()
{
   // Create the method invocation environment. Necessary input
   // information: the class, full method name with prototype
   // string of the form: method(char*,int,float).
   // To initialize class method with default arguments, method
   // string with default parameters should be of the form:
   // method(=\"ABC\",1234,3.14) (!! parameter string should
   // consists of '=').
   // To execute the method call TQSlot::ExecuteMethod(object,...).

   fFunc      = 0;
   fOffset    = 0;
   fName      = "";
   fExecuting = 0;

   // cl==0, is the case of interpreted function.

   fName = method_name;

   char *method = new char[strlen(method_name)+1];
   if (method) strcpy(method, method_name);

   char *proto;
   char *tmp;
   char *params = 0;

   // separate method and protoype strings

   if ((proto = strchr(method,'('))) {

      // substitute first '(' symbol with '\0'
      *proto++ = '\0';

      // last ')' symbol with '\0'
      if ((tmp = strrchr(proto,')'))) *tmp = '\0';
   }

   if (!proto) proto = "";
   if (proto && (params = strchr(proto,'='))) *params = ' ';

   fFunc = new G__CallFunc;

   // initiate class method (function) with proto
   // or with default params

   if (cl) {
      params ?
      fFunc->SetFunc(cl->GetClassInfo(), (char*)method,
                     (char*)params, &fOffset) :
      fFunc->SetFuncProto(cl->GetClassInfo(), (char*)method,
                          (char*)proto, &fOffset);
   } else {
      G__ClassInfo gcl;
      params ?
      fFunc->SetFunc(&gcl, (char*)funcname,
                     (char*)params, &fOffset) :
      fFunc->SetFuncProto(&gcl, (char*)funcname,
                          (char*)proto, &fOffset);
   }

   // cleaning
   if (method) { delete [] method; method = 0; }
}

//______________________________________________________________________________
TQSlot::TQSlot(const char *class_name, const char *funcname) :
   TObject(), TRefCnt()
{
   // Create the method invocation environment. Necessary input
   // information: the name of class (could be interpreted class),
   // full method name with  prototype or parameter string
   // of the form: method(char*,int,float).
   // To initialize class method with default arguments, method
   // string with default parameters  should be of the form:
   // method(=\"ABC\",1234,3.14) (!! parameter string should
   // consists of '=').
   // To execute the method call TQSlot::ExecuteMethod(object,...).

   fFunc      = 0;
   fOffset    = 0;
   fName      = funcname;
   fExecuting = 0;

   char *method = new char[strlen(funcname)+1];
   if (method) strcpy(method, funcname);

   char *proto;
   char *tmp;
   char *params = 0;

   // separate method and protoype strings

   if ((proto =  strchr(method,'('))) {
      *proto++ = '\0';
      if ((tmp = strrchr(proto,')'))) *tmp  = '\0';
   }

   if (!proto) proto = "";
   if (proto && (params = strchr(proto,'='))) *params = ' ';

   fFunc = new G__CallFunc;

   G__ClassInfo gcl;

   if (!class_name)
      ;                       // function
   else
      gcl.Init(class_name);   // class

   if (params)
      fFunc->SetFunc(&gcl, (char*)method, (char*)params, &fOffset);
   else
      fFunc->SetFuncProto(&gcl, (char*)method, (char*)proto , &fOffset);

   if (method) { delete [] method; method = 0; }
   return;
}

//______________________________________________________________________________
TQSlot::~TQSlot()
{
   // TQSlot dtor.

   // don't delete executing environment of a slot that is being executed
   if (!fExecuting)
      delete fFunc;
}

//______________________________________________________________________________
inline void TQSlot::ExecuteMethod(void *object)
{
   // ExecuteMethod the method (with preset arguments) for
   // the specified object.

   void *address = 0;
   if (object) address = (void*)((Long_t)object + fOffset);
   fExecuting++;
   fFunc->Exec(address);
   fExecuting--;
   if (!TestBit(kNotDeleted) && !fExecuting)
      delete fFunc;
}

//______________________________________________________________________________
inline void TQSlot::ExecuteMethod(void *object, Long_t param)
{
   // ExecuteMethod the method for the specified object and
   // with single argument value.

   void *address = 0;
   fFunc->ResetArg();
   fFunc->SetArg(param);
   if (object) address = (void*)((Long_t)object + fOffset);
   fExecuting++;
   fFunc->Exec(address);
   fExecuting--;
   if (!TestBit(kNotDeleted) && !fExecuting)
      delete fFunc;
}

//______________________________________________________________________________
inline void TQSlot::ExecuteMethod(void *object, Double_t param)
{
   //  ExecuteMethod the method for the specified object and
   // with single argument value.

   void *address = 0;
   fFunc->ResetArg();
   fFunc->SetArg(param);
   if (object) address = (void*)((Long_t)object + fOffset);
   fExecuting++;
   fFunc->Exec(address);
   fExecuting--;
   if (!TestBit(kNotDeleted) && !fExecuting)
      delete fFunc;
}

//______________________________________________________________________________
inline void TQSlot::ExecuteMethod(void *object, const char *param)
{
   // ExecuteMethod the method for the specified object and text param.

   void *address = 0;
   gTQSlotParams = (char*)param;
   fFunc->SetArgs("gTQSlotParams");
   if (object) address = (void*)((Long_t)object + fOffset);
   fExecuting++;
   fFunc->Exec(address);
   fExecuting--;
   if (!TestBit(kNotDeleted) && !fExecuting)
      delete fFunc;
}

//______________________________________________________________________________
inline void TQSlot::ExecuteMethod(void *object, Long_t *paramArr)
{
   //  ExecuteMethod the method for the specified object and with
   //  several argument values.
   //  paramArr is an array containing the addresses
   //  where to take the function parameters.
   //  At least as many pointers should be present in the array as
   //  there are required arguments  (all arguments - default args).

   void *address = 0;
   fFunc->SetArgArray(paramArr);
   if (object) address = (void*)((Long_t)object + fOffset);
   fExecuting++;
   fFunc->Exec(address);
   fExecuting--;
   if (!TestBit(kNotDeleted) && !fExecuting)
      delete fFunc;
}

//______________________________________________________________________________
void TQSlot::Print(Option_t *) const
{
   // Print info about slot.

   cout <<IsA()->GetName() << "\t" << GetName() << "\t"
   << "Number of Connections = " << References() << endl;
}


//______________________________________________________________________________
TQConnection::TQConnection() : TList(), TQObject()
{
   // Default constructor.

   fReceiver = 0;
   fSlot     = 0;
}

//______________________________________________________________________________
TQConnection::TQConnection(TClass *cl, void *receiver, const char *method_name)
   : TList(), TQObject()
{
   // TQConnection ctor.
   //    cl != 0  - connection to object == receiver of class == cl
   //               and method == method_name
   //    cl == 0  - connection to function with name == method_name

   char *funcname = 0;
   fReceiver = receiver;      // fReceiver is pointer to receiver

   if (!cl) {
      funcname = G__p2f2funcname(fReceiver);
      if (!funcname)
         Warning("TQConnection", "%s cannot be compiled", method_name);
   }

   fSlot = new TQSlot(cl, method_name, funcname);
   fSlot->AddReference(); //update counter of references to slot
}

//______________________________________________________________________________
TQConnection::TQConnection(const char *class_name, void *receiver,
                           const char *funcname) : TList(), TQObject()
{
   // TQConnection ctor.
   //    Creates connection to method of class specified by name,
   //    it could be interpreted class and with method == funcname.

   fSlot = new TQSlot(class_name, funcname);  // new slot-method
   fSlot->AddReference();     // update counter of references to slot
   fReceiver = receiver;      // fReceiver is pointer to receiver
}

//______________________________________________________________________________
TQConnection::~TQConnection()
{
   // TQConnection dtor.
   //    - remove this connection from all signal lists
   //    - we do not delete fSlot if it has other connections,
   //      TQSlot::fCounter > 0 .

   TIter next(this);
   register TList *list;

   while ((list = (TList*)next())) {
      list->Remove(this);
      if (list->IsEmpty()) SafeDelete(list);   // delete empty list
   }

   fSlot->RemoveReference();  // decrease references to slot

   if (fSlot->References() <=0) {
      SafeDelete(fSlot);
   }
}

//______________________________________________________________________________
const char *TQConnection::GetName() const
{
   // Returns name of connection

   return fSlot->GetName();
}

//______________________________________________________________________________
void TQConnection::Destroyed()
{
   // Signal Destroyed tells that connection is destroyed.

   MakeZombie();
   Emit("Destroyed()");
}

//______________________________________________________________________________
void TQConnection::ls(Option_t *option) const
{
   // List TQConnection full method name and list all signals
   // connected to this connection.

   cout << "\t" <<  IsA()->GetName() << "\t" << GetName() << endl;
   ((TQConnection*)this)->ForEach(TList,ls)(option);
}

//______________________________________________________________________________
void TQConnection::Print(Option_t *option) const
{
   // Print TQConnection full method name and print all
   // signals connected to this connection.


   cout <<  "\t\t\t" << IsA()->GetName() << "\t" << fReceiver <<
            "\t" << GetName() << endl;
}

//______________________________________________________________________________
void TQConnection::ExecuteMethod()
{
   // Apply slot-method to the fReceiver object without arguments.

   fSlot->ExecuteMethod(fReceiver);
}

//______________________________________________________________________________
void TQConnection::ExecuteMethod(Long_t param)
{
   // Apply slot-method to the fReceiver object with
   // single argument value.

   fSlot->ExecuteMethod(fReceiver, param);
}

//______________________________________________________________________________
void TQConnection::ExecuteMethod(Double_t param)
{
   // Apply slot-method to the fReceiver object with
   // single argument value.

   fSlot->ExecuteMethod(fReceiver, param);
}

//______________________________________________________________________________
void TQConnection::ExecuteMethod(Long_t *params)
{
   // Apply slot-method to the fReceiver object with variable
   // number of argument values.

   fSlot->ExecuteMethod(fReceiver, params);
}

//______________________________________________________________________________
void TQConnection::ExecuteMethod(const char *param)
{
   // Apply slot-method to the fReceiver object and
   // with string parameter.

   fSlot->ExecuteMethod(fReceiver, param);
}
