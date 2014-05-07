// $Id$
// Author: Sergey Linev   28/12/2013

#ifndef ROOT_TFastCgi
#define ROOT_TFastCgi

#ifndef ROOT_THttpEngine
#include "THttpEngine.h"
#endif

class TThread;

class TFastCgi : public THttpEngine {
protected:
   Int_t  fSocket;     //! socket used by fastcgi
   Bool_t fDebugMode;  //! debug mode, may required for fastcgi debugging in other servers
   TString fTopName;   //! name of top item
   TThread *fThrd;     //! thread which takes requests, can be many later
public:
   TFastCgi();
   virtual ~TFastCgi();

   Int_t GetSocket() const
   {
      return fSocket;
   }

   virtual Bool_t Create(const char *args);

   static void *run_func(void *);

   ClassDef(TFastCgi, 0) // fastcgi engine for THttpServer
};


#endif
