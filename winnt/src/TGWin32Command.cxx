// @(#)root/winnt:$Name$:$Id$
// Author: Valery Fine   23/01/96

#include "TGWin32Command.h"

//______________________________________________________________________________
   TGWin32Command::TGWin32Command(int code, int type, int master){
           fMasterFlag = master;
           fCodeOP = MAKEWPARAM(code,type);
           fBuffered = 1;   // by default all commands can be buffered
           SetMsgID();
   }
//______________________________________________________________________________
   void  TGWin32Command::SetCOP(int code){ fCodeOP = code;}
//______________________________________________________________________________
   int   TGWin32Command::GetCOP(){return fCodeOP;}

//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
//*-*
//*-*   Here is a set of the special messages to control WIN32 interface
//*-*
//*-*  1. TGWin32Box                     (inline)
//*-*  2. TGWin32Clear                   (inline)
//*-*  3. TGWin32CopyTo
//*-*  4. TGWin32DrawPolyLine
//*-*  5. TGWin32DrawText
//*-*  6. TGWin32GetColor                (inline)
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*TGWin32CopyTo*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

//______________________________________________________________________________
TGWin32CopyTo::TGWin32CopyTo(TGWin32Object *obj, int xpost, int ypost, int xposf, int yposf, int code)
                                     : TGWin32Command(code,ROOT_Pixmap){
  fSourceWinObject = obj;

  fPointFrom.x = xposf;
  fPointFrom.y = yposf;

  fPointTo.x = xpost;
  fPointTo.y = ypost;

}

//______________________________________________________________________________
TGWin32Object *TGWin32CopyTo::GetSource(){
  return fSourceWinObject;
}

//______________________________________________________________________________
POINT *TGWin32CopyTo::GetPointsFrom(){
    return &fPointFrom;
}

//______________________________________________________________________________
POINT *TGWin32CopyTo::GetPointsTo(){
    return &fPointTo;
}
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*TGWin32DrawPolyLine*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

//______________________________________________________________________________
  TGWin32DrawPolyLine::TGWin32DrawPolyLine(int n, POINT *lpp, int code) : TGWin32Command(code){
       fNum = n;
       flpPoint = lpp;
  }

//______________________________________________________________________________
   int TGWin32DrawPolyLine::GetNumber(){return fNum;}
//______________________________________________________________________________
   POINT *TGWin32DrawPolyLine::GetPoints(){return flpPoint;}


//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
