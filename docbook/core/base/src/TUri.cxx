// @(#)root/base:$Id$
// Author: Gerhard E. Bruckner 15/07/07

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TUri                                                                 //
//                                                                      //
// This class represents a RFC 3986 compatible URI.                     //
// See http://rfc.net/rfc3986.html.                                     //
// It provides member functions to set and return the different         //
// the different parts of an URI. The functionality is that of          //
// a validating parser.                                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <ctype.h>    // for tolower()
#include "TUri.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TPRegexp.h"

//RFC3986:
// pchar = unreserved / pct-encoded / sub-delims / ":" / "@"
const char* const kURI_pchar        = "(?:[[:alpha:][:digit:]-._~!$&'()*+,;=:@]|%[0-9A-Fa-f][0-9A-Fa-f])";

//unreserved characters, see chapter 2.3
const char* const kURI_unreserved   = "[[:alpha:][:digit:]-._~]";

// reserved characters, see chapter
// reserved      = gen-delims / sub-delims
const char* const kURI_reserved     = "[:/?#[]@!$&'()*+,;=]";

// gen-delims, see chapter 2.2
// delimiters of the generic URI components
const char* const kURI_gendelims    = "[:/?#[]@]";

// sub-delims, see chapter 2.2
const char* const kURI_subdelims    = "[!$&'()*+,;=]";


ClassImp(TUri)

//______________________________________________________________________________
TUri::TUri(const TString &uri)
{
   // Constructor that calls SetUri with a complete URI.

   SetUri(uri);
}

//______________________________________________________________________________
TUri::TUri(const char *uri)
{
   // Constructor that calls SetUri with a complete URI.

   SetUri(uri);
}

//______________________________________________________________________________
TUri::TUri(const TUri &uri) : TObject(uri)
{
   // TUri copy ctor.

   fScheme = uri.fScheme;
   fUserinfo = uri.fUserinfo;
   fHost = uri.fHost;
   fPort = uri.fPort;
   fPath = uri.fPath;
   fQuery = uri.fQuery;
   fFragment = uri.fFragment;
   fHasScheme = uri.fHasScheme;
   fHasUserinfo = uri.fHasUserinfo;
   fHasHost = uri.fHasHost;
   fHasPort = uri.fHasPort;
   fHasPath = uri.fHasPath;
   fHasQuery = uri.fHasQuery;
   fHasFragment = uri.fHasFragment;
}

//______________________________________________________________________________
TUri &TUri::operator= (const TUri & rhs)
{
   // TUri assignment operator.

   if (this != &rhs) {
      TObject::operator= (rhs);
      fScheme = rhs.fScheme;
      fUserinfo = rhs.fUserinfo;
      fHost = rhs.fHost;
      fPort = rhs.fPort;
      fPath = rhs.fPath;
      fQuery = rhs.fQuery;
      fFragment = rhs.fFragment;
      fHasScheme = rhs.fHasScheme;
      fHasUserinfo = rhs.fHasUserinfo;
      fHasHost = rhs.fHasHost;
      fHasPort = rhs.fHasPort;
      fHasPath = rhs.fHasPath;
      fHasQuery = rhs.fHasQuery;
      fHasFragment = rhs.fHasFragment;
   }
   return *this;
}

//______________________________________________________________________________
Bool_t operator== (const TUri &u1, const TUri &u2)
{
   // Implementation of a TUri Equivalence operator
   // that uses syntax-based normalisation
   // see chapter 6.2.2.

   // make temporary copies of the operands
   TUri u11 = u1;
   TUri u22 = u2;
   // normalise them
   u11.Normalise();
   u22.Normalise();
   // compare them as TStrings
   return u11.GetUri() == u22.GetUri();
}

//______________________________________________________________________________
const TString TUri::GetUri() const
{
   // Returns the whole URI -
   // an implementation of chapter 5.3 component recomposition.
   // The result URI is composed out of the five basic parts.
   //
   // URI         = scheme ":" hier-part [ "?" query ] [ "#" fragment ]
   // hier-part   = "//" authority path-abempty
   //             / path-absolute
   //             / path-rootless
   //             / path-empty

   TString result = "";
   if (fHasScheme)
      result = fScheme + ":";
   result += GetHierPart();
   if (fHasQuery)
      result += TString("?") + fQuery;
   if (fHasFragment)
      result += TString("#") + fFragment;
   return result;
}

//______________________________________________________________________________
const TString TUri::RemoveDotSegments(const TString &inp)
{
   // This functions implements the "remove_dot_segments" routine
   // of chapter 5.2.4 "for interpreting and removing the
   // special '.' and '..' complete path segments from a
   // referenced path".

   TString source = inp;
   TString sink = TString("");  // sink buffer

   // Step 2 "While the source buffer is not empty, loop as follows:"
   while (source.Length() > 0) {
      // Rule 2.A
      if (TPRegexp("^\\.\\.?/(.*)$").Substitute(source, "/$1") > 0)
         continue;

      // Rule 2.B
      if (TPRegexp("^/\\./(.*)$|^/\\.($)").Substitute(source, "/$1") > 0)
         continue;

      // Rule 2.C
      if (TPRegexp("^/\\.\\./(.*)$|^/\\.\\.($)").Substitute(source, "/$1") > 0) {
         Ssiz_t last = sink.Last('/');
         if (last == -1)
            last = 0;
         sink.Remove(last, sink.Length() - last);
         continue;
      }

      // Rule 2.D
      if (source.CompareTo(".") == 0 || source.CompareTo("..") == 0) {
         source.Remove(0, source.Length() - 11);
         continue;
      }

      // Rule 2.E
      TPRegexp regexp = TPRegexp("^(/?[^/]*)(?:/|$)");
      TObjArray *tokens = regexp.MatchS(source);
      TString segment = ((TObjString*) tokens->At(1))->GetString();
      sink += segment;
      source.Remove(0, segment.Length());
      delete tokens;
   }

   // Step 3: return sink buffer
   return sink;
}

//______________________________________________________________________________
Bool_t TUri::IsAbsolute() const
{
   // Returns kTRUE if instance qualifies as absolute-URI
   // absolute-URI  = scheme ":" hier-part [ "?" query ]
   // cf. Appendix A.

   return (HasScheme() && HasHierPart() && !HasFragment());
}

//______________________________________________________________________________
Bool_t TUri::IsRelative() const
{
   // Returns kTRUE if instance qualifies as relative-ref
   // relative-ref  = relative-part [ "?" query ] [ "#" fragment ]
   // cf. Appendix A.

   return (!HasScheme() && HasRelativePart());
}

//______________________________________________________________________________
Bool_t TUri::IsUri() const
{
   // Returns kTRUE if instance qualifies as URI
   // URI = scheme ":" hier-part [ "?" query ] [ "#" fragment ]
   // cf. Appendix A.

   return (HasScheme() && HasHierPart());
}

//______________________________________________________________________________
Bool_t TUri::IsReference() const
{
   // Returns kTRUE if instance qualifies as URI-reference
   // URI-reference = URI / relative-ref
   // cf. Appendix A.

   return (IsUri() || IsRelative());
}

//______________________________________________________________________________
Bool_t TUri::SetScheme(const TString &scheme)
{
   // Set scheme component of URI:
   // scheme      = ALPHA *( ALPHA / DIGIT / "+" / "-" / "." )

   if (!scheme) {
      fHasScheme = kFALSE;
      return kTRUE;
   }
   if (IsScheme(scheme)) {
      fScheme = scheme;
      fHasScheme = kTRUE;
      return kTRUE;
   } else {
      Error("SetScheme", "<scheme> component \"%s\" of URI is not compliant with RFC 3986.", scheme.Data());
      return kFALSE;
   }
}

//______________________________________________________________________________
Bool_t TUri::IsScheme(const TString &string)
{
   // Returns kTRUE if string qualifies as URI scheme:
   // scheme      = ALPHA *( ALPHA / DIGIT / "+" / "-" / "." )

   return TPRegexp(
             "^[[:alpha:]][[:alpha:][:digit:]+-.]*$"
          ).Match(string);
}

//______________________________________________________________________________
const TString TUri::GetAuthority() const
{
   // Returns the authority part of the instance:
   // authority   = [ userinfo "@" ] host [ ":" port ]

   TString authority = fHasUserinfo ? fUserinfo + "@" + fHost : fHost;
   if (fHasPort && !fPort.IsNull())
      // add port only if not empty
      authority += TString(":") + TString(fPort);
   return (authority);
}

//______________________________________________________________________________
Bool_t TUri::SetQuery(const TString &query)
{
   // Set query component of URI:
   // query       = *( pchar / "/" / "?" )

   if (!query) {
      fHasQuery = kFALSE;
      return kTRUE;
   }
   if (IsQuery(query)) {
      fQuery = query;
      fHasQuery = kTRUE;
      return kTRUE;
   } else {
      Error("SetQuery", "<query> component \"%s\" of URI is not compliant with RFC 3986.", query.Data());
      return kFALSE;
   }
}

//______________________________________________________________________________
Bool_t TUri::IsQuery(const TString &string)
{
   // Returns kTRUE if string qualifies as URI query:
   // query       = *( pchar / "/" / "?" )

   return TPRegexp(
             TString("^([/?]|") + kURI_pchar + ")*$"
          ).Match(string);
}

//______________________________________________________________________________
Bool_t TUri::SetAuthority(const TString &authority)
{
   // Set authority part of URI:
   // authority   = [ userinfo "@" ] host [ ":" port ]
   //
   // Split into components {userinfo@, host, :port},
   // remember that according to the RFC, it is necessary to
   // distinguish between missing component (no delimiter)
   // and empty component (delimiter present).

   if (authority.IsNull()) {
      fHasUserinfo = kFALSE;
      fHasHost = kFALSE;
      fHasPort = kFALSE;
      return kTRUE;
   }
   TPRegexp regexp = TPRegexp("^(?:(.*@))?([^:]*)((?::.*)?)$");
   TObjArray *tokens = regexp.MatchS(authority);

   if (tokens->GetEntries() != 4) {
      Error("SetAuthority", "<authority> component \"%s\" of URI is not compliant with RFC 3986.", authority.Data());
      return kFALSE;
   }

   Bool_t valid = kTRUE;

   // handle userinfo
   TString userinfo = ((TObjString*) tokens->At(1))->GetString();
   if (userinfo.EndsWith("@")) {
      userinfo.Remove(TString::kTrailing, '@');
      valid &= SetUserInfo(userinfo);
   }

   // handle host
   TString host = ((TObjString*) tokens->At(2))->GetString();
   valid &= SetHost(host);

   // handle port
   TString port = ((TObjString*) tokens->At(3))->GetString();
   if (port.BeginsWith(":")) {
      port.Remove(TString::kLeading, ':');
      valid &= SetPort(port);
   }

   return valid;
}

//______________________________________________________________________________
Bool_t TUri::IsAuthority(const TString &string)
{
   // Returns kTRUE if string qualifies as valid URI authority:
   // authority   = [ userinfo "@" ] host [ ":" port ]

   // split into parts {userinfo, host, port}
   TPRegexp regexp = TPRegexp("^(?:(.*)@)?([^:]*)(?::(.*))?$");
   TObjArray *tokens = regexp.MatchS(string);
   TString userinfo = ((TObjString*) tokens->At(1))->GetString();
   TString host = ((TObjString*) tokens->At(2))->GetString();
   TString port;
   // port is optional
   if (tokens->GetEntries() == 4)
      port = ((TObjString*) tokens->At(3))->GetString();
   else
      port = "";
   return (IsHost(host) && IsUserInfo(userinfo) && IsPort(port));
}

//______________________________________________________________________________
Bool_t TUri::SetUserInfo(const TString &userinfo)
{
   // Set userinfo component of URI:
   // userinfo    = *( unreserved / pct-encoded / sub-delims / ":" )

   if (userinfo.IsNull()) {
      fHasUserinfo = kFALSE;
      return kTRUE;
   }
   if (IsUserInfo(userinfo)) {
      fUserinfo = userinfo;
      fHasUserinfo = kTRUE;
      return kTRUE;
   } else {
      Error("SetUserInfo", "<userinfo> component \"%s\" of URI is not compliant with RFC 3986.", userinfo.Data());
      return kFALSE;
   }
}

//______________________________________________________________________________
Bool_t TUri::IsUserInfo(const TString &string)
{
   // Return kTRUE is string qualifies as valid URI userinfo:
   // userinfo    = *( unreserved / pct-encoded / sub-delims / ":" )
   // this equals to pchar without the '@' character

   return (TPRegexp(
              "^" + TString(kURI_pchar) + "*$"
           ).Match(string) > 0 && !TString(string).Contains("@"));
}

//______________________________________________________________________________
Bool_t TUri::SetHost(const TString &host)
{
   // Set host component of URI:
   // RFC 3986:    host = IP-literal / IPv4address / reg-name
   // implemented: host =  IPv4address / reg-name

   if (IsHost(host)) {
      fHost = host;
      fHasHost = kTRUE;
      return kTRUE;
   } else {
      Error("SetHost", "<host> component \"%s\" of URI is not compliant with RFC 3986.", host.Data());
      return kFALSE;
   }
}

//______________________________________________________________________________
Bool_t TUri::SetPort(const TString &port)
{
   // Set port component of URI:
   // port        = *DIGIT

   if (IsPort(port)) {
      fPort = port;
      fHasPort = kTRUE;
      return kTRUE;
   }
   Error("SetPort", "<port> component \"%s\" of URI is not compliant with RFC 3986.", port.Data());
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TUri::SetPath(const TString &path)
{
   // Set path component of URI:
   // path          = path-abempty    ; begins with "/" or is empty
   //               / path-absolute   ; begins with "/" but not "//"
   //               / path-noscheme   ; begins with a non-colon segment
   //               / path-rootless   ; begins with a segment
   //               / path-empty      ; zero characters

   if (IsPath(path)) {
      fPath = path;
      fHasPath = kTRUE;
      return kTRUE;
   }
   Error("SetPath", "<path> component \"%s\" of URI is not compliant with RFC 3986.", path.Data());
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TUri::SetFragment(const TString &fragment)
{
   // Set fragment component of URI:
   // fragment    = *( pchar / "/" / "?" )

   if (IsFragment(fragment)) {
      fFragment = fragment;
      fHasFragment = kTRUE;
      return kTRUE;
   } else {
      Error("SetFragment", "<fragment> component \"%s\" of URI is not compliant with RFC 3986.", fragment.Data());
      return kFALSE;
   }
}

//______________________________________________________________________________

Bool_t TUri::IsFragment(const TString &string)
{
   // Returns kTRUE if string qualifies as valid fragment component
   // fragment    = *( pchar / "/" / "?" )

   return (TPRegexp(
              "^(" + TString(kURI_pchar) + "|[/?])*$"
           ).Match(string) > 0);
}

//______________________________________________________________________________
void TUri::Print(Option_t *option) const
{
   // Display function,
   // option "d" .. debug output
   // anything else .. simply print URI.

   if (strcmp(option, "d") != 0) {
      Printf("%s", GetUri().Data());
      return ;
   }
   // debug output
   Printf("URI: <%s>", GetUri().Data());
   Printf("(%c) |--scheme---------<%s>", fHasScheme ? 't' : 'f', fScheme.Data());
   Printf("    |--hier-----------<%s>", GetHierPart().Data());
   Printf("(%c)     |--authority------<%s>", HasAuthority() ? 't' : 'f', GetAuthority().Data());
   Printf("(%c)         |--userinfo---<%s>", fHasUserinfo ? 't' : 'f', fUserinfo.Data());
   Printf("(%c)         |--host-------<%s>", fHasHost ? 't' : 'f', fHost.Data());
   Printf("(%c)         |--port-------<%s>", fHasPort ? 't' : 'f', fPort.Data());
   Printf("(%c)     |--path-------<%s>", fHasPath ? 't' : 'f', fPath.Data());
   Printf("(%c) |--query------<%s>", fHasQuery ? 't' : 'f', fQuery.Data());
   Printf("(%c) |--fragment---<%s>", fHasFragment ? 't' : 'f', fFragment.Data());
   printf("path flags: ");
   if (IsPathAbempty(fPath))
      printf("abempty ");
   if (IsPathAbsolute(fPath))
      printf("absolute ");
   if (IsPathRootless(fPath))
      printf("rootless ");
   if (IsPathEmpty(fPath))
      printf("empty ");
   printf("\nURI flags: ");
   if (IsAbsolute())
      printf("absolute-URI ");
   if (IsRelative())
      printf("relative-ref ");
   if (IsUri())
      printf("URI ");
   if (IsReference())
      printf("URI-reference ");
   printf("\n");
}

//______________________________________________________________________________
void TUri::Reset()
{
   // Initialize this URI object.
   // Set all TString members to empty string,
   // set all Bool_t members to kFALSE.

   fScheme = "";
   fUserinfo = "";
   fHost = "";
   fPort = "";
   fPath = "";
   fQuery = "";
   fFragment = "";

   fHasScheme = kFALSE;
   fHasUserinfo = kFALSE;
   fHasHost = kFALSE;
   fHasPort = kFALSE;
   fHasPath = kFALSE;
   fHasQuery = kFALSE;
   fHasFragment = kFALSE;
}

//______________________________________________________________________________
Bool_t TUri::SetUri(const TString &uri)
{
   // Parse URI and set the member variables accordingly,
   // returns kTRUE if URI validates, and kFALSE otherwise:
   // URI         = scheme ":" hier-part [ "?" query ] [ "#" fragment ]
   // hier-part   = "//" authority path-abempty
   //             / path-absolute
   //             / path-rootless
   //             / path-empty
   //

   // Reset member variables
   Reset();

   // regular expression taken from appendix B
   // reference points          12            3  4          5       6   7        8 9
   TPRegexp regexp = TPRegexp("^(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)([?]([^#]*))?(#(.*))?");
   TObjArray *tokens = regexp.MatchS(uri);

   // collect bool values to see if all setters succeed
   Bool_t valid = kTRUE;
   //tokens->Print();
   switch (tokens->GetEntries()) {
      case 10:
         // URI contains fragment delimiter '#'
         valid &= SetFragment(((TObjString*) tokens->At(9))->GetString());
         // fallthrough

      case 8:
         // URI does not contain a fragment delimiter
         // if there is a query delimiter '?', set query
         if (!((TString)((TObjString*) tokens->At(6))->GetString()).IsNull())
            valid &= SetQuery(((TObjString*) tokens->At(7))->GetString());
         // fallthrough

      case 6:
         // URI does not contain fragment or query delimiters
         valid &= SetPath(((TObjString*) tokens->At(5))->GetString());
         // if there is an authority delimiter '//', set authority
         if (!((TString)((TObjString*) tokens->At(3))->GetString()).IsNull())
            valid &= SetAuthority(((TObjString*) tokens->At(4))->GetString());
         // if there is a scheme delimiter ':', set scheme
         if (!((TString)((TObjString*) tokens->At(1))->GetString()).IsNull())
            valid &= SetScheme(((TObjString*) tokens->At(2))->GetString());
         break;

      default:
         // regular expression did not match
         Error("SetUri", "URI \"%s\" is not is not compliant with RFC 3986.", uri.Data());
         valid = kFALSE;
   }

   // reset member variables once again, if one at least setter failed
   if (!valid)
      Reset();

   delete tokens;
   return valid;
}

//______________________________________________________________________________
const TString TUri::GetHierPart() const
{
   // hier-part   = "//" authority path-abempty
   //             / path-absolute
   //             / path-rootless
   //             / path-empty

   if (HasAuthority() && IsPathAbempty(fPath))
      return (TString("//") + GetAuthority() + fPath);
   else
      return fPath;
}

//______________________________________________________________________________
const TString TUri::GetRelativePart() const
{
   // relative-part = "//" authority path-abempty
   //               / path-absolute
   //               / path-noscheme
   //               / path-empty

   if (HasAuthority() && IsPathAbempty(fPath))
      return (TString("//") + GetAuthority() + fPath);
   else
      return fPath;
}

//______________________________________________________________________________
Bool_t TUri::SetHierPart(const TString &hier)
{
   // returns hier-part component of URI
   // hier-part   = "//" authority path-abempty
   //             / path-absolute
   //             / path-rootless
   //             / path-empty
   //

   /*  if ( IsPathAbsolute(hier) || IsPathRootless(hier) || IsPathEmpty(hier) ) {
     SetPath (hier);
     return kTRUE;
    }
    */

   // reference points:         1  2          3
   TPRegexp regexp = TPRegexp("^(//([^/?#]*))?([^?#]*)$");
   TObjArray *tokens = regexp.MatchS(hier);

   if (tokens->GetEntries() == 0) {
      Error("SetHierPart", "<hier-part> component \"%s\" of URI is not compliant with RFC 3986.", hier.Data());
      delete tokens;
      return false;
   }

   TString delm = ((TObjString*) tokens->At(1))->GetString();
   TString auth = ((TObjString*) tokens->At(2))->GetString();
   TString path = ((TObjString*) tokens->At(3))->GetString();

   Bool_t valid = kTRUE;

   if (!delm.IsNull() && IsPathAbempty(path)) {
      // URI contains an authority delimiter '//' ...
      valid &= SetAuthority(auth);
      valid &= SetPath(path);
   } else {
      // URI does not contain an authority
      if (IsPathAbsolute(path) || IsPathRootless(path) || IsPathEmpty(path))
         valid &= SetPath(path);
      else {
         valid = kFALSE;
         Error("SetHierPart", "<hier-part> component \"%s\" of URI is not compliant with RFC 3986.", hier.Data());
      }
   }
   delete tokens;
   return valid;
}

//______________________________________________________________________________
Bool_t TUri::IsHierPart(const TString &string)
{
   // Returns kTRUE if string qualifies as hier-part:
   //
   // hier-part   = "//" authority path-abempty
   //             / path-absolute
   //             / path-rootless
   //             / path-empty

   // use functionality of SetHierPart
   // in order to avoid duplicate code
   TUri uri;
   return (uri.SetHierPart(string));
}

//______________________________________________________________________________
Bool_t TUri::IsRelativePart(const TString &string)
{
   // Returns kTRUE is string qualifies as relative-part:
   // relative-part = "//" authority path-abempty
   //               / path-absolute
   //               / path-noscheme
   //               / path-empty

   // use functionality of SetRelativePart
   // in order to avoid duplicate code
   TUri uri;
   return (uri.SetRelativePart(string));
}

//______________________________________________________________________________
Bool_t TUri::SetRelativePart(const TString &relative)
{
   // Returns kTRUE is string qualifies as relative-part:
   // relative-part = "//" authority path-abempty
   //               / path-absolute
   //               / path-noscheme
   //               / path-empty

   // reference points:         1  2          3
   TPRegexp regexp = TPRegexp("^(//([^/?#]*))?([^?#]*)$");
   TObjArray *tokens = regexp.MatchS(relative);

   if (tokens->GetEntries() == 0) {
      Error("SetRelativePath", "<relative-part> component \"%s\" of URI is not compliant with RFC 3986.", relative.Data());
      delete tokens;
      return false;
   }
   TString delm = ((TObjString*) tokens->At(1))->GetString();
   TString auth = ((TObjString*) tokens->At(2))->GetString();
   TString path = ((TObjString*) tokens->At(3))->GetString();

   Bool_t valid = kTRUE;

   if (!delm.IsNull() && IsPathAbempty(path)) {
      // URI contains an authority delimiter '//' ...
      valid &= SetAuthority(auth);
      valid &= SetPath(path);
   } else {
      // URI does not contain an authority
      if (IsPathAbsolute(path) || IsPathNoscheme(path) || IsPathEmpty(path))
         valid &= SetPath(path);
      else {
         valid = kFALSE;
         Error("SetRelativePath", "<relative-part> component \"%s\" of URI is not compliant with RFC 3986.", relative.Data());
      }
   }
   delete tokens;
   return valid;
}

//______________________________________________________________________________
const TString TUri::PctEncode(const TString &source)
{
   // Percent-encode and return the given string according to RFC 3986
   // in principle, this function cannot fail or produce an error.

   TString sink = "";
   // iterate through source
   for (Int_t i = 0; i < source.Length(); i++) {
      if (IsUnreserved(TString(source(i)))) {
         // unreserved character -> copy
         sink = sink + source[i];
      } else {
         // reserved character -> encode to 2 digit hex
         // preceded by '%'
         char buffer[4];
         sprintf(buffer, "%%%02X", source[i]);
         sink = sink + buffer;
      }
   }
   return sink;
}

//______________________________________________________________________________
Bool_t TUri::IsHost(const TString &string)
{
   // Returns kTRUE if string qualifies as valid host component:
   // host = IP-literal / IPv4address / reg-name
   // implemented: host =  IPv4address / reg-name

   return (IsRegName(string) || IsIpv4(string));
}

//______________________________________________________________________________
Bool_t TUri::IsPath(const TString &string)
{
   // Retruns kTRUE if string qualifies as valid path component:
   // path          = path-abempty    ; begins with "/" or is empty
   //               / path-absolute   ; begins with "/" but not "//"
   //               / path-noscheme   ; begins with a non-colon segment
   //               / path-rootless   ; begins with a segment
   //               / path-empty      ; zero characters

   return (IsPathAbempty(string) ||
           IsPathAbsolute(string) ||
           IsPathNoscheme(string) ||
           IsPathRootless(string) ||
           IsPathEmpty(string));
}

//______________________________________________________________________________
Bool_t TUri::IsPathAbempty(const TString &string)
{
   // Returns kTRUE if string qualifies as valid path-abempty component:
   //    path-abempty  = *( "/" segment )
   //    segment       = *pchar

   return (TPRegexp(
              TString("^(/") + TString(kURI_pchar) + "*)*$"
           ).Match(string) > 0);
}

//______________________________________________________________________________
Bool_t TUri::IsPathAbsolute(const TString &string)
{
   // Returns kTRUE if string qualifies as valid path-absolute component
   //    path-absolute = "/" [ segment-nz *( "/" segment ) ]
   //    segment-nz    = 1*pchar
   //    segment       = *pchar

   return (TPRegexp(
              TString("^/(") + TString(kURI_pchar) + "+(/" + TString(kURI_pchar) + "*)*)?$"
           ).Match(string) > 0);
}

//______________________________________________________________________________
Bool_t TUri::IsPathNoscheme(const TString &string)
{
   // Returns kTRUE if string qualifies as valid path-noscheme component:
   // path-noscheme = segment-nz-nc *( "/" segment )
   // segment-nz-nc = 1*( unreserved / pct-encoded / sub-delims / "@" )
   // segment       = *pchar

   return (TPRegexp(
              TString("^(([[:alpha:][:digit:]-._~!$&'()*+,;=@]|%[0-9A-Fa-f][0-9A-Fa-f])+)(/") + TString(kURI_pchar) + "*)*$"
           ).Match(string) > 0);
}

//______________________________________________________________________________
Bool_t TUri::IsPathRootless(const TString &string)
{
   // Returns kTRUE if string qualifies as valid path-rootless component:
   // path-rootless = segment-nz *( "/" segment )

   return TPRegexp(
             TString("^") + TString(kURI_pchar) + "+(/" + TString(kURI_pchar) + "*)*$"
          ).Match(string);
}

//______________________________________________________________________________
Bool_t TUri::IsPathEmpty(const TString &string)
{
   // Returns kTRUE if string qualifies as valid path-empty component:
   // path-empty    = 0<pchar>
   return TString(string).IsNull();
}

//______________________________________________________________________________
Bool_t TUri::IsPort(const TString &string)
{
   // Returns kTRUE if string qualifies as valid port component:
   // RFC 3986: port        = *DIGIT

   return (TPRegexp("^[[:digit:]]*$").Match(string) > 0);
}

//______________________________________________________________________________
Bool_t TUri::IsRegName(const TString &string)
{
   // Returns kTRUE if string qualifies as valid reg-name:
   //
   //  reg-name    = *( unreserved / pct-encoded / sub-delims )
   //  sub-delims  = "!" / "$" / "&" / "'" / "(" / ")"
   //                  / "*" / "+" / "," / ";" / "="
   //

   return (TPRegexp(
              "^([[:alpha:][:digit:]-._~!$&'()*+,;=]|%[0-9A-Fa-f][0-9A-Fa-f])*$").Match(string) > 0);
}

//______________________________________________________________________________
Bool_t TUri::IsIpv4(const TString &string)
{
   // Returns kTRUE, if string holds a valid IPv4 address
   // currently only decimal variant supported.
   // Existence of leadig 0s or numeric range remains unchecked
   // IPv4address = dec-octet "." dec-octet "." dec-octet "." dec-octet.
   //
   return (TPRegexp(
              "^([[:digit:]]{1,3}[.]){3}[[:digit:]]{1,3}$").Match(string) > 0);
}

//______________________________________________________________________________
Bool_t TUri::IsUnreserved(const TString &string)
{
   // Returns kTRUE, if the given string does not contain
   // RFC 3986 reserved characters
   // unreserved  = ALPHA / DIGIT / "-" / "." / "_" / "~"

   return (TPRegexp(
              "^" + TString(kURI_unreserved) + "*$").Match(string) > 0);
}

//______________________________________________________________________________
void TUri::Normalise()
{
   // Syntax based normalisation according to
   // RFC chapter 6.2.2.

   // case normalisation of host and scheme
   // cf. chapter 6.2.2.1
   fScheme.ToLower();
   if (fHasHost) {
      TString host = GetHost();
      host.ToLower();
      SetHost(host);
   }
   // percent-encoding normalisation (6.2.2.2) for
   // userinfo, host (reg-name), path, query, fragment
   fUserinfo = PctNormalise(PctDecodeUnreserved(fUserinfo));
   fHost = PctNormalise(PctDecodeUnreserved(fHost));
   fPath = PctNormalise(PctDecodeUnreserved(fPath));
   fQuery = PctNormalise(PctDecodeUnreserved(fQuery));
   fFragment = PctNormalise(PctDecodeUnreserved(fFragment));

   // path segment normalisation (6.2.2.3)
   if (fHasPath)
      SetPath(RemoveDotSegments(GetPath()));
}

//______________________________________________________________________________
TString const TUri::PctDecodeUnreserved(const TString &source)
{
   // Percent-decode the given string according to chapter 2.1
   // we assume a valid pct-encoded string.

   TString sink = "";
   Int_t i = 0;
   while (i < source.Length()) {
      if (source[i] == '%') {
         if (source.Length() < i+2) {
            // abort if out of bounds
            return sink;
         }
         // two hex digits follow -> decode to ASCII
         // upper nibble, bits 4-7
         char c1 = tolower(source[i + 1]) - '0';
         if (c1 > 9) // a-f
            c1 -= 39;
         // lower nibble, bits 0-3
         char c0 = tolower(source[i + 2]) - '0';
         if (c0 > 9) // a-f
            c0 -= 39;
         char decoded = c1 << 4 | c0;
         if (TPRegexp(kURI_unreserved).Match(decoded) > 0) {
            // we have an unreserved character -> store decoded version
            sink = sink + decoded;
         } else {
            // this is a reserved character
            TString pct = source(i,3);
            pct.ToUpper();
            sink = sink + pct;
         }
         // advance 2 characters
         i += 2;
      } else {
         // regular character -> copy
         sink = sink + source[i];
      }
      i++;
   }
   return sink;
}

//______________________________________________________________________________
TString const TUri::PctNormalise(const TString &source)
{
   // Normalise the percent-encoded parts of the string
   // i.e. uppercase the hexadecimal digits
   // %[:alpha:][:alpha:] -> %[:ALPHA:][:ALPHA:]

   TString sink = "";
   Int_t i = 0;
   while (i < source.Length()) {
      if (source[i] == '%') {
         if (source.Length() < i+2) {
            // abort if out of bounds
            return sink;
         }
         TString pct = source(i,3);
         // uppercase the pct part
         pct.ToUpper();
         sink = sink + pct;
         // advance 2 characters
         i += 2;
      } else {
         // regular character -> copy
         sink = sink + source[i];
      }
      i++;
   }
   return sink;
}

//______________________________________________________________________________
TString const TUri::PctDecode(const TString &source)
{
   // Percent-decode the given string according to chapter 2.1
   // we assume a valid pct-encoded string.

   TString sink = "";
   Int_t i = 0;
   while (i < source.Length()) {
      if (source[i] == '%') {
         if (source.Length() < i+2) {
            // abort if out of bounds
            return sink;
         }
         // two hex digits follow -> decode to ASCII
         // upper nibble, bits 4-7
         char c1 = tolower(source[i + 1]) - '0';
         if (c1 > 9) // a-f
            c1 -= 39;
         // lower nibble, bits 0-3
         char c0 = tolower(source[i + 2]) - '0';
         if (c0 > 9) // a-f
            c0 -= 39;
         sink = sink + (char)(c1 << 4 | c0);
         // advance 2 characters
         i += 2;
      } else {
         // regular character -> copy
         sink = sink + source[i];
      }
      i++;
   }
   return sink;
}

//______________________________________________________________________________
TUri TUri::Transform(const TUri &reference, const TUri &base)
{
   // Transform a URI reference into its target URI using
   // given a base URI.
   // This is an implementation of the pseudocode in chapter 5.2.2.

   TUri target;
   if (reference.HasScheme()) {
      target.SetScheme(reference.GetScheme());
      if (reference.HasAuthority())
         target.SetAuthority(reference.GetAuthority());
      if (reference.HasPath())
         target.SetPath(RemoveDotSegments(reference.GetPath()));
      if (reference.HasQuery())
         target.SetQuery(reference.GetQuery());
   } else {
      if (reference.HasAuthority()) {
         target.SetAuthority(reference.GetAuthority());
         if (reference.HasPath())
            target.SetPath(RemoveDotSegments(reference.GetPath()));
         if (reference.HasQuery())
            target.SetQuery(reference.GetQuery());
      } else {
         if (reference.GetPath().IsNull()) {
            target.SetPath(base.GetPath());
            if (reference.HasQuery()) {
               target.SetQuery(reference.GetQuery());
            } else {
               if (base.HasQuery())
                  target.SetQuery(base.GetQuery());
            }
         } else {
            if (reference.GetPath().BeginsWith("/")) {
               target.SetPath(RemoveDotSegments(reference.GetPath()));
            } else {
               target.SetPath(RemoveDotSegments(MergePaths(reference, base)));
            }
            if (reference.HasQuery())
               target.SetQuery(reference.GetQuery());
         }
         if (base.HasAuthority())
            target.SetAuthority(base.GetAuthority());
      }
      if (base.HasScheme())
         target.SetScheme(base.GetScheme());
   }
   if (reference.HasFragment())
      target.SetFragment(reference.GetFragment());
   return target;
}

//______________________________________________________________________________
const TString TUri::MergePaths(const TUri &reference, const TUri &base)
{
   // RFC 3986, 5.3.2.
   // If the base URI has a defined authority component and an empty
   // path, then return a string consisting of "/" concatenated with the
   // reference's path; otherwise,
   // return a string consisting of the reference's path component
   // appended to all but the last segment of the base URI's path (i.e.,
   // excluding any characters after the right-most "/" in the base URI
   // path, or excluding the entire base URI path if it does not contain
   // any "/" characters).

   TString result = "";
   if (base.HasAuthority() && base.GetPath().IsNull()) {
      result = TString("/") + reference.GetPath();
   } else {
      TString basepath = base.GetPath();
      Ssiz_t last = basepath.Last('/');
      if (last == -1)
         result = reference.GetPath();
      else
         result = basepath(0, last + 1) + reference.GetPath();
   }
   return result;
}
