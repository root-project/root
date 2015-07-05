/*************************************************************************
 * Copyright (C) 2013-2014, Omar Andres Zapata Mesa                      *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include<TRObject.h>
#include<vector>
//______________________________________________________________________________
/* Begin_Html
<center><h2>TRObject class</h2></center>

<p>
The TRObject class lets you obtain ROOT's objects from R's objects.<br>
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
ROOT::R::TRObject obj;
obj=r.ParseEval("seq(1,10)");
TVectorD v=obj;
v.Print();
}
*/

using namespace ROOT::R;
ClassImp(TRObject)

//______________________________________________________________________________
TRObject::TRObject(SEXP xx): TObject(), fObj(xx),fStatus(kTRUE) { }


//______________________________________________________________________________
void TRObject::operator=(SEXP xx)
{
   fStatus=kTRUE;
   fObj = xx;
}

//______________________________________________________________________________
TRObject::TRObject(SEXP xx, Bool_t status): fObj(xx), fStatus(status) {}
