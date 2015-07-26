// $Id$
// Author: Sergey Linev   21/12/2013

#ifndef ROOT_TCivetweb
#define ROOT_TCivetweb

#ifndef ROOT_THttpEngine
#include "THttpEngine.h"
#endif

class TCivetweb : public THttpEngine {
protected:
   void     *fCtx;           //! civetweb context
   void     *fCallbacks;     //! call-back table for civetweb webserver
   TString   fTopName;       //! name of top item
   Bool_t    fDebug;         //! debug mode

public:
   TCivetweb();
   virtual ~TCivetweb();

   virtual Bool_t Create(const char *args);

   const char *GetTopName() const
   {
      return fTopName.Data();
   }

   Bool_t IsDebugMode() const
   {
      // indicates that

      return fDebug;
   }

   Int_t ProcessLog(const char* message);

   ClassDef(TCivetweb, 0) // http server implementation, based on civetweb embedded server
};


#endif
