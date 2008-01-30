// @(#) root/glite:$Id$
// Author: Anar Manafov <A.Manafov@gsi.de> 2006-07-30

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/************************************************************************/
/*! \file TGLiteresult.h
*//*

         version number:    $LastChangedRevision: 1678 $
         created by:        Anar Manafov
                            2006-07-30
         last changed by:   $LastChangedBy: manafov $ $LastChangedDate: 2008-01-21 18:22:14 +0100 (Mon, 21 Jan 2008) $

         Copyright (c) 2006 GSI GridTeam. All rights reserved.
*************************************************************************/

#ifndef ROOT_TGLiteResult
#define ROOT_TGLiteResult

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLiteResult                                                         //
//                                                                      //
// Class defining interface to a gLite result set.                      //
// Objects of this class are created by TGrid methods.                  //
//                                                                      //
// Related classes are TGLite.                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGridResult
#include "TGridResult.h"
#endif

class TGLiteResult : public TGridResult
{
public:
   virtual void DumpResult();

   virtual const char* GetFileName(UInt_t i) const;               // returns the file name of list item i
   virtual const char* GetFileNamePath(UInt_t i) const;           // returns the full path + file name of list item i
   virtual const char* GetPath(UInt_t i) const;                   // returns the file path of list item i
   virtual const char* GetKey(UInt_t i, const char *key) const;   // returns the key value of list item i

   virtual Bool_t SetKey(UInt_t i, const char *key, const char *value);   // set the key value of list item i
   virtual TList* GetFileInfoList() const;                 // returns a new allocated List of TFileInfo Objects

   void Print(Option_t *option = "") const;
   void Print(Option_t *wildcard, Option_t *option) const;

   ClassDef(TGLiteResult, 1) // gLite query result set
};

#endif
