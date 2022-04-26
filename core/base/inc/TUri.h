// @(#)root/base:$Id$
// Author: Gerhard E. Bruckner 15/07/07

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TUri
#define ROOT_TUri


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TUri                                                                 //
//                                                                      //
// This class represents a RFC3986 compatible URI.                      //
// See http://rfc.net/rfc3986.html.                                     //
// It provides member functions to return the different parts of        //
// an URI.                                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TObject.h"
#include "TString.h"


class TUri;
Bool_t  operator==(const TUri &u1, const TUri &u2);


class TUri : public TObject {

friend Bool_t operator==(const TUri &u1, const TUri &u2); // comparison operator

private:

   // In order to represent the five basic components of an URI,
   // we use 7 member variables (authority gets split in 3 parts)
   //
   //   foo://user:pass@example.com:8042/over/there?name=ferret#nose
   //   \_/   \________________________/\_________/ \_________/ \__/
   //    |                 |                |            |        |
   //  scheme          authority           path        query   fragment
   //
   // In many cases we have to distinguish between empty
   // TString and undefined value (i.e. delimiter not found).
   // Therefore, we use a TString to hold the string value
   // and a corresponding Bool_t to store if it is defined or not.
   // The Bool_t has precedence.

   TString fScheme;
   TString fUserinfo;     // authority/userinfo: user@password, ...
   TString fHost;         // authority/host: hostname or ip-address
   TString fPort;         // authority/port: port number, normally 1-65535
   TString fPath;
   TString fQuery;
   TString fFragment;

   Bool_t fHasScheme;
   Bool_t fHasUserinfo;
   Bool_t fHasHost;
   Bool_t fHasPort;
   Bool_t fHasPath;
   Bool_t fHasQuery;
   Bool_t fHasFragment;

public:
   TUri(const TUri &uri);
   TUri() { Reset(); }
   TUri(const TString &uri);
   TUri(const char *uri);
   TUri &operator=(const TUri &rhs); //copy ctor
   virtual ~TUri() { }

   const TString GetUri() const;
   const TString GetScheme() const { return fScheme; }
   const TString GetHierPart() const;
   const TString GetRelativePart() const;
   const TString GetAuthority() const;
   const TString GetUserInfo() const { return fUserinfo; }
   const TString GetHost() const { return fHost; }
   const TString GetPort() const { return fPort; }
   const TString GetPath() const { return fPath; }
   const TString GetQuery() const { return fQuery; }
   const TString GetFragment() const { return fFragment; }

   Bool_t HasScheme() const { return fHasScheme; }
   Bool_t HasHierPart() const { return IsHierPart(GetHierPart()); }
   Bool_t HasAuthority() const { return fHasHost; }
   Bool_t HasUserInfo() const { return fHasUserinfo; }
   Bool_t HasHost() const { return fHasHost; }
   Bool_t HasPort() const { return fHasPort; }
   Bool_t HasPath() const { return fHasPath; }
   Bool_t HasQuery() const { return fHasQuery; }
   Bool_t HasFragment() const { return fHasFragment; }
   Bool_t HasRelativePart() const { return IsRelativePart(GetRelativePart()); }

   Bool_t SetUri(const TString &uri);
   Bool_t SetScheme(const TString &scheme);
   Bool_t SetHierPart(const TString &hier);
   Bool_t SetAuthority(const TString &authority);
   Bool_t SetUserInfo(const TString &userinfo);
   Bool_t SetHost(const TString &host);
   Bool_t SetPort(const TString &port);
   Bool_t SetPath(const TString &path);
   Bool_t SetQuery(const TString &path);
   Bool_t SetFragment(const TString &fragment);

   Bool_t SetRelativePart(const TString&);

   void   Print(Option_t *option = "") const override;
   Bool_t IsSortable() const override { return kTRUE; }

   void Normalise();
   void Reset();

   Bool_t IsAbsolute() const;
   Bool_t IsRelative() const;
   Bool_t IsUri() const;
   Bool_t IsReference() const;

   static Bool_t IsUnreserved(const TString &string);

   static const TString PctEncode(const TString &source);
   static const TString PctDecode(const TString &source);
   static const TString PctDecodeUnreserved(const TString &source);
   static const TString PctNormalise(const TString &source);

   static Bool_t IsScheme(const TString&);
   static Bool_t IsHierPart(const TString&);
   static Bool_t IsAuthority(const TString&);
   static Bool_t IsUserInfo(const TString&);
   static Bool_t IsHost(const TString&);
   static Bool_t IsIpv4(const TString&);
   static Bool_t IsRegName(const TString&);
   static Bool_t IsPort(const TString&);
   static Bool_t IsPath(const TString&);
   static Bool_t IsPathAbsolute(const TString&);
   static Bool_t IsPathAbempty(const TString&);
   static Bool_t IsPathNoscheme(const TString&);
   static Bool_t IsPathRootless(const TString&);
   static Bool_t IsPathEmpty(const TString&);
   static Bool_t IsQuery(const TString&);
   static Bool_t IsFragment(const TString&);

   static Bool_t IsRelativePart(const TString&);

   static const TString RemoveDotSegments(const TString&);

   static TUri Transform(const TUri &reference, const TUri &base);
   static const TString MergePaths(const TUri &reference, const TUri &base);

   ClassDefOverride(TUri, 1)  //Represents an URI
};

#endif
