// @(#)root/proof:$Name:  $:$Id: TDrawFeedback.h,v 1.2 2004/11/24 07:41:32 brun Exp $
// Author: Maarten Ballintijn   28/10/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDrawFeedback
#define ROOT_TDrawFeedback


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TDrawFeedback                                                        //
//                                                                      //
// Utility class to draw objects in the feedback list during queries.   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TQObject
#include "TQObject.h"
#endif

class TProof;
class THashList;
class TSeqCollection;


class TDrawFeedback : public TQObject {
private:
   Bool_t         fAll;    //draw all or selected objects
   THashList     *fNames;  //selected objects

protected:
   Option_t      *fOption; //draw option
   TProof        *fProof;  //handle to PROOF session

public:
   TDrawFeedback(TVirtualProof *proof = 0, TSeqCollection *names = 0);
   ~TDrawFeedback();

   void Feedback(TList *objs);
   void SetOption(Option_t *option) { fOption = option; }

   ClassDef(TDrawFeedback,0)  // Present PROOF query feedback
};

#endif
