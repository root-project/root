// $Id$
// Author: Sergey Linev   21/12/2013

#ifndef ROOT_THttpEngine
#define ROOT_THttpEngine

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class THttpServer;

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

#endif
