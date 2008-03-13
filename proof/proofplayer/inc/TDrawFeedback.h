// @(#)root/proofplayer:$Id$
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
#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TProof;
class TProof;
class THashList;
class TSeqCollection;

class TDrawFeedback : public TObject, public TQObject {
private:
   Bool_t         fAll;    //draw all or selected objects
   THashList     *fNames;  //selected objects

protected:
   Option_t      *fOption; //draw option
   TProof        *fProof;  //handle to PROOF session

public:
   TDrawFeedback(TProof *proof = 0, TSeqCollection *names = 0);
   ~TDrawFeedback();

   void Feedback(TList *objs);
   void SetOption(Option_t *option) { fOption = option; }

   ClassDef(TDrawFeedback,0)  // Present PROOF query feedback
};

#endif
