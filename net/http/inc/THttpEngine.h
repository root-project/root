// $Id$
// Author: Sergey Linev   21/12/2013

#ifndef ROOT_THttpEngine
#define ROOT_THttpEngine

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class THttpServer;
class THttpCallArg;
class TCanvas;

class THttpEngine : public TNamed {
protected:
   friend class THttpServer;

   THttpServer *fServer;    //! object server

   THttpEngine(const char *name, const char *title);

   void SetServer(THttpServer *serv)
   {
      fServer = serv;
   }

   /** Method regularly called in main ROOT context */
   virtual void Process() {}

public:
   virtual ~THttpEngine();

   /** Method to create all components of engine. Called once from by the server */
   virtual Bool_t Create(const char *)
   {
      return kFALSE;
   }

   THttpServer *GetServer() const
   {
      return fServer;
   }

   ClassDef(THttpEngine, 0) // abstract class which should provide http-based protocol for server
};

// ====================================================================


class THttpWSEngine : public TNamed {

protected:

   THttpWSEngine(const char* name, const char* title);

   Bool_t    fReady;     //! indicate if websocket get ready flag to send bigger amount of data
   Bool_t    fModified;  //! true when canvas was modified
   Bool_t    fGetMenu;   //! true when menu was requested
   TCanvas*  fCanv ;     //! canvas associated with websocket

   void  CheckModifiedFlag();

public:
   virtual ~THttpWSEngine();

   virtual void ClearHandle() = 0;

   virtual void Send(const void* buf, int len) = 0;

   virtual void ProcessData(THttpCallArg* arg);

   // --------- method to work with

   virtual void AssignCanvas(TCanvas* canv);

   virtual void CanvasModified();

   ClassDef(THttpWSEngine, 0) // abstract class for working with WebSockets-like protocol
};

#endif
