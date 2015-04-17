// @(#)root/tree:$Id$
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
private:
   // Prevent meaningless operator (which otherwise can be reached via
   // the conversion to 'const char*'
   Bool_t operator<(const TCut &rhs); // Intentional left unimplemented
   Bool_t operator<=(const TCut &rhs); // Intentional left unimplemented
   Bool_t operator>(const TCut &rhs); // Intentional left unimplemented
   Bool_t operator>=(const TCut &rhs); // Intentional left unimplemented
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

   // Comparison
   Bool_t   operator==(const char *rhs) const;
   Bool_t   operator==(const TCut &rhs) const;
   Bool_t   operator!=(const char *rhs) const;
   Bool_t   operator!=(const TCut &rhs) const;

   friend TCut operator+(const TCut &lhs, const char *rhs);
   friend TCut operator+(const char *lhs, const TCut &rhs);
   friend TCut operator+(const TCut &lhs, const TCut &rhs);
   friend TCut operator*(const TCut &lhs, const char *rhs);
   friend TCut operator*(const char *lhs, const TCut &rhs);
   friend TCut operator*(const TCut &lhs, const TCut &rhs);
// Preventing warnings with -Weffc++ in GCC since the overloading of the && and || operators was a design choice.
#if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) >= 40600
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#endif
   friend TCut operator&&(const TCut &lhs, const char *rhs);
   friend TCut operator&&(const char *lhs, const TCut &rhs);
   friend TCut operator&&(const TCut &lhs, const TCut &rhs);
   friend TCut operator||(const TCut &lhs, const char *rhs);
   friend TCut operator||(const char *lhs, const TCut &rhs);
   friend TCut operator||(const TCut &lhs, const TCut &rhs);
#if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) >= 40600
#pragma GCC diagnostic pop
#endif
   friend TCut operator!(const TCut &rhs);

   // Type conversion
   operator const char*() const { return GetTitle(); }

   ClassDef(TCut,1)  //A specialized string object used for TTree selections
};

// Declarations.
TCut operator+(const TCut &lhs, const char *rhs);
TCut operator+(const char *lhs, const TCut &rhs);
TCut operator+(const TCut &lhs, const TCut &rhs);
TCut operator*(const TCut &lhs, const char *rhs);
TCut operator*(const char *lhs, const TCut &rhs);
TCut operator*(const TCut &lhs, const TCut &rhs);
// Preventing warnings with -Weffc++ in GCC since the overloading of the && and || operators was a design choice.
#if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) >= 40600
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#endif
TCut operator&&(const TCut &lhs, const char *rhs);
TCut operator&&(const char *lhs, const TCut &rhs);
TCut operator&&(const TCut &lhs, const TCut &rhs);
TCut operator||(const TCut &lhs, const char *rhs);
TCut operator||(const char *lhs, const TCut &rhs);
TCut operator||(const TCut &lhs, const TCut &rhs);
#if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) >= 40600
#pragma GCC diagnostic pop
#endif
TCut operator!(const TCut &rhs);

#endif
