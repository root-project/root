// @(#)root/tree:$Name$:$Id$
// Author: Rene Brun   14/04/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TCut
#define ROOT_TCut

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TCut                                                                 //
//                                                                      //
// A specialized string object used in TTree selections.                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TCut : public TNamed {

public:
   TCut();
   TCut(const char *title);
   TCut(const char *name, const char *title);
   TCut(const TCut &cut);
   virtual ~TCut();

   // Assignment
   TCut&    operator=(const char *rhs);
   TCut&    operator=(const TCut &rhs);
   TCut&    operator+=(const char *rhs);
   TCut&    operator+=(const TCut &rhs);
   TCut&    operator*=(const char *rhs);
   TCut&    operator*=(const TCut &rhs);

   friend TCut operator+(const TCut &lhs, const char *rhs);
   friend TCut operator+(const char *lhs, const TCut &rhs);
   friend TCut operator+(const TCut &lhs, const TCut &rhs);
   friend TCut operator*(const TCut &lhs, const char *rhs);
   friend TCut operator*(const char *lhs, const TCut &rhs);
   friend TCut operator*(const TCut &lhs, const TCut &rhs);
   friend TCut operator&&(const TCut &lhs, const char *rhs);
   friend TCut operator&&(const char *lhs, const TCut &rhs);
   friend TCut operator&&(const TCut &lhs, const TCut &rhs);
   friend TCut operator||(const TCut &lhs, const char *rhs);
   friend TCut operator||(const char *lhs, const TCut &rhs);
   friend TCut operator||(const TCut &lhs, const TCut &rhs);
   friend TCut operator!(const TCut &rhs);

   // Type conversion
   operator const char*() const { return GetTitle(); }

   ClassDef(TCut,1)  //A specialized string object used for TTree selections
};

#endif
