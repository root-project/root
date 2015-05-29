/*************************************************************************
 * Copyright (C) 2013-2014, Omar Andres Zapata Mesa                      *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include<TRObjectProxy.h>
#include<vector>
//______________________________________________________________________________
/* Begin_Html
<center><h2>TRObjectProxy class</h2></center>

<p>
The TRObjectProxy class lets you obtain ROOT's objects from R's objects.<br>
It has some basic template opetarors to convert R's objects into ROOT's datatypes<br>
</p>
A simple example<br>
<p>

</p>
<hr>
End_Html
#include<TRInterface.h>
void Proxy()
{
ROOT::R::TRInterface &r=ROOT::R::TRInterface::Instance();
ROOT::R::TRObjectProxy obj;
obj=r.ParseEval("seq(1,10)");
TVectorD v=obj;
v.Print();
}
*/

using namespace ROOT::R;
ClassImp(TRObjectProxy)

//______________________________________________________________________________
TRObjectProxy::TRObjectProxy(SEXP xx): TObject(), x(xx) { }


//______________________________________________________________________________
void TRObjectProxy::operator=(SEXP xx)
{
   x = xx;
}

//______________________________________________________________________________
TRObjectProxy::TRObjectProxy(SEXP xx, Bool_t status): x(xx), fStatus(status) { }
