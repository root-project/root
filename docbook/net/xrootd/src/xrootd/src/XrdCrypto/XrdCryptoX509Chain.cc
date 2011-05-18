// $Id$

const char *XrdCryptoX509ChainCVSID = "$Id$";
/******************************************************************************/
/*                                                                            */
/*                  X r d C r y p t o X 5 0 9 C h a i n . c c                 */
/*                                                                            */
/* (c) 2005  G. Ganis, CERN                                                   */
/*                                                                            */
/******************************************************************************/
#include <time.h>
#include <string.h>

#include <XrdCrypto/XrdCryptoX509Chain.hh>
#include <XrdCrypto/XrdCryptoTrace.hh>


// ---------------------------------------------------------------------------//
//                                                                            //
// XrdCryptoX509Chain                                                         //
//                                                                            //
// Light single-linked list for managing stacks of XrdCryptoX509* objects     //
//                                                                            //
// ---------------------------------------------------------------------------//

// For test dumps, to avoid interfering with the trace mutex
#define LOCDUMP(y)    { cerr << epname << ":" << y << endl; }

// Description of errors
static const char *X509ChainErrStr[] = {
   "no error condition occured",         // 0
   "chain is inconsistent",              // 1
   "size exceeds max allowed depth",     // 2
   "invalid or missing CA",              // 3
   "certificate missing",                // 4
   "unexpected certificate type",        // 5
   "names invalid or missing",           // 6
   "certificate has been revoked",       // 7
   "certificate expired",                // 8
   "extension not found",                // 9
   "signature verification failed",      // 10
   "issuer had no signing rights",       // 11
   "CA issued by another CA"             // 12
};

//___________________________________________________________________________
XrdCryptoX509Chain::XrdCryptoX509Chain(XrdCryptoX509 *c)
{
   // Constructor

   previous = 0;
   current = 0;
   begin = 0;
   end = 0;
   size = 0; 
   lastError = "";
   caname = "";
   eecname = "";
   cahash = "";
   eechash = "";
   statusCA = kUnknown;

   if (c) {
      XrdCryptoX509ChainNode *f = new XrdCryptoX509ChainNode(c,0);
      current = begin = end = f;
      size++;
      //
      // If CA verify it and save result
      if (c->type == XrdCryptoX509::kCA) {
         caname = c->Subject();
         cahash = c->SubjectHash();
         EX509ChainErr ecode = kNone;
         if (!Verify(ecode, "CA: ",XrdCryptoX509::kCA, 0, c, c))
            statusCA = kInvalid;
         else
            statusCA = kValid;
      }
   }
} 

//___________________________________________________________________________
XrdCryptoX509Chain::XrdCryptoX509Chain(XrdCryptoX509Chain *ch)
{
   // Copy constructor

   previous = 0;
   current = 0;
   begin = 0;
   end = 0;
   size = 0; 
   lastError = ch->LastError();
   caname = ch->CAname();
   eecname = ch->EECname();
   cahash = ch->CAhash();
   eechash = ch->EEChash();
   statusCA = ch->StatusCA(); 

   XrdCryptoX509 *c = ch->Begin();
   while (c) {
      XrdCryptoX509ChainNode *nc = new XrdCryptoX509ChainNode(c,0);
      if (!begin)
         begin = nc;
      if (end)
         end->SetNext(nc);
      end = nc;
      size++;
      // Go to Next
      c = ch->Next();
   }
}

//___________________________________________________________________________
XrdCryptoX509Chain::~XrdCryptoX509Chain()
{
   // Destructor

   XrdCryptoX509ChainNode *n = 0;
   XrdCryptoX509ChainNode *c = begin;
   while (c) {
      n = c->Next();
      delete (c);
      c = n;
   }
}

//___________________________________________________________________________
void XrdCryptoX509Chain::Cleanup(bool keepCA)
{
   // Destructs content of nodes AND their content
   // If keepCA is true, the top CA is kept

   XrdCryptoX509ChainNode *n = 0;
   XrdCryptoX509ChainNode *c = begin;
   while (c) {
      n = c->Next();
      if (c->Cert() &&
         (!keepCA || (c->Cert()->type != XrdCryptoX509::kCA)))
         delete (c->Cert());
      delete (c);
      c = n;
   }

   // Reset
   previous = 0;
   current = 0;
   begin = 0;
   end = 0;
   size = 0; 
   lastError = "";
   caname = "";
   eecname = "";
   cahash = "";
   eechash = "";
   statusCA = kUnknown;
}

//___________________________________________________________________________
bool XrdCryptoX509Chain::CheckCA(bool checkselfsigned)
{
   // Search the list for a valid CA and set it at top.
   // Search stops when a valid CA is found; an invalid CA is flagged.
   // A second CA is always ignored.
   // Signature check failures are accepted if 'checkselfsigned' is false.
   // Return 1 if found, 0 otherwise; lastError is filled with the reason of
   // failure, if any.

   XrdCryptoX509 *xc = 0;
   XrdCryptoX509ChainNode *n = 0;
   XrdCryptoX509ChainNode *c = begin;
   XrdCryptoX509ChainNode *p = 0;
   lastError = "";
   while (c) {
      n = c->Next();
      xc = c->Cert();
      if (xc && xc->type == XrdCryptoX509::kCA) {
         caname = xc->Subject();
         cahash = xc->SubjectHash();
         EX509ChainErr ecode = kNone;
         bool CAok = Verify(ecode, "CA: ",XrdCryptoX509::kCA, 0, xc, xc);
         if (!CAok && (ecode != kVerifyFail || checkselfsigned)) {
            statusCA = kInvalid;
            lastError += X509ChainError(ecode);
         } else {
            statusCA = kValid;
            if (p) {
               // Move at top
               p->SetNext(c->Next());
               c->SetNext(begin);
               if (end == c) end = p;
               begin = c;
            }
            return 1;
         }
      }
      p = c;  // Previous node
      c = n;
   }

   // Found nothing
   return 0;
}

//___________________________________________________________________________
const char *XrdCryptoX509Chain::X509ChainError(EX509ChainErr e)
{
   // Return error string

   return X509ChainErrStr[e];
}

//___________________________________________________________________________
XrdCryptoX509ChainNode *XrdCryptoX509Chain::Find(XrdCryptoX509 *c)
{
   // Find node containing bucket b

   XrdCryptoX509ChainNode *nd = begin;
   for (; nd; nd = nd->Next()) {
      if (nd->Cert() == c)
         return nd;
   }
   return (XrdCryptoX509ChainNode *)0;
}

//___________________________________________________________________________
void XrdCryptoX509Chain::PutInFront(XrdCryptoX509 *c)
{
   // Add at the beginning of the list
   // Check to avoid duplicates

   if (!Find(c)) {
      XrdCryptoX509ChainNode *nc = new XrdCryptoX509ChainNode(c,begin);
      begin = nc;
      if (!end)
         end = nc;
      size++;
   }
}

//___________________________________________________________________________
void XrdCryptoX509Chain::InsertAfter(XrdCryptoX509 *c, XrdCryptoX509 *cp)
{
   // Add or move certificate 'c' after certificate 'cp'; if 'cp' is not
   // in the list, push-back

   XrdCryptoX509ChainNode *nc = Find(c);
   XrdCryptoX509ChainNode *ncp = Find(cp);
   if (ncp) {
      // Create a new node, if not there
      if (!nc) {
         nc = new XrdCryptoX509ChainNode(c,ncp->Next());
         size++;
      }
      // Update pointers
      ncp->SetNext(nc);
      if (end == ncp)
         end = nc;

   } else {
      // Referebce certificate not in the list
      // If new, add in last position; otherwise leave it where it is
      if (!nc)
         PushBack(c);
   }
}

//___________________________________________________________________________
void XrdCryptoX509Chain::PushBack(XrdCryptoX509 *c)
{
   // Add at the end of the list
   // Check to avoid duplicates

   if (!Find(c)) {
      XrdCryptoX509ChainNode *nc = new XrdCryptoX509ChainNode(c,0);
      if (!begin)
         begin = nc;
      if (end)
         end->SetNext(nc);
      end = nc;
      size++;
   }
}

//___________________________________________________________________________
void XrdCryptoX509Chain::Remove(XrdCryptoX509 *c)
{
   // Remove node containing bucket b

   XrdCryptoX509ChainNode *curr = current;
   XrdCryptoX509ChainNode *prev = previous;

   if (!curr || curr->Cert() != c || (prev && curr != prev->Next())) {
      // We need first to find the address
      curr = begin;
      prev = 0;
      for (; curr; curr = curr->Next()) {
         if (curr->Cert() == c)
            break;
         prev = curr;
      }
   }

   // The certificate is not in the list
   if (!curr)
      return;

   //
   // If this was the top CA update the related information
   if (c->type == XrdCryptoX509::kCA && curr == begin) {
      // There may be other CAs in the chain, but we will
      // check when needed
      statusCA = kUnknown;
      caname = "";
      cahash = "";
   }

   // Now we have all the information to remove
   if (prev) {
      current  = curr->Next();
      prev->SetNext(current);
      previous = curr;
   } else if (curr == begin) {
      // First buffer
      current  = curr->Next();
      begin = current;
      previous = 0;
   }

   // Cleanup and update size
   delete curr;
   size--;
}

//___________________________________________________________________________
XrdCryptoX509 *XrdCryptoX509Chain::Begin()
{ 
   // Iterator functionality: init

   previous = 0;
   current = begin;
   if (current)
      return current->Cert();
   return (XrdCryptoX509 *)0;
}

//___________________________________________________________________________
XrdCryptoX509 *XrdCryptoX509Chain::Next()
{ 
   // Iterator functionality: get next

   previous = current;
   if (current) {
      current = current->Next();
      if (current)
         return current->Cert();
   }
   return (XrdCryptoX509 *)0;
}

//___________________________________________________________________________
XrdCryptoX509 *XrdCryptoX509Chain::SearchByIssuer(const char *issuer,
                                                  ESearchMode mode)
{ 
   // Return first certificate in the chain with issuer
   // Match according to mode.

   XrdCryptoX509ChainNode *cn = FindIssuer(issuer, mode);

   // We are done
   return ((cn) ? cn->Cert() : (XrdCryptoX509 *)0);
}

//___________________________________________________________________________
XrdCryptoX509 *XrdCryptoX509Chain::SearchBySubject(const char *subject,
                                                   ESearchMode mode)
{ 
   // Return first certificate in the chain with subject
   // Match according to mode.

   XrdCryptoX509ChainNode *cn = FindSubject(subject, mode);

   // We are done
   return ((cn) ? cn->Cert() : (XrdCryptoX509 *)0);

}

//___________________________________________________________________________
XrdCryptoX509ChainNode *XrdCryptoX509Chain::FindIssuer(const char *issuer,
                          ESearchMode mode, XrdCryptoX509ChainNode **prev)
{ 
   // Return first chain node with certificate having issuer
   // Match according to mode.

   // Make sure we got something to compare
   if (!issuer) 
      return (XrdCryptoX509ChainNode *)0;

   XrdCryptoX509ChainNode *cp = 0;
   XrdCryptoX509ChainNode *n = 0;
   XrdCryptoX509ChainNode *cn = begin;
   XrdCryptoX509 *c = 0;
   while (cn) {
      n = cn->Next();
      c = cn->Cert();
      const char *pi = c->Issuer();
      if (c && pi) {
         if (mode == kExact) {
            if (!strcmp(pi, issuer))
               break;
         } else if (mode == kBegin) {
            if (strstr(pi, issuer) == c->Issuer())
               break;
         } else if (mode == kEnd) {
            int ibeg = strlen(pi) - strlen(issuer);
            if (!strcmp(pi + ibeg, issuer))
               break;
         }
      }
      c = 0;
      cp = cn;  // previous
      cn = n;
   }
   // return previous, if requested
   if (prev)
      *prev = (cn) ? cp : 0;

   // We are done
   return ((cn) ? cn : (XrdCryptoX509ChainNode *)0);
}

//___________________________________________________________________________
XrdCryptoX509ChainNode *XrdCryptoX509Chain::FindSubject(const char *subject,
                            ESearchMode mode, XrdCryptoX509ChainNode **prev)
{ 
   // Return first chain node with certificate having subject
   // Match according to mode.

   // Make sure we got something to compare
   if (!subject) 
      return (XrdCryptoX509ChainNode *)0;

   XrdCryptoX509ChainNode *cp = 0;
   XrdCryptoX509ChainNode *n = 0;
   XrdCryptoX509ChainNode *cn = begin;
   XrdCryptoX509 *c = 0;
   while (cn) {
      n = cn->Next();
      c = cn->Cert();
      const char *ps = c ? c->Subject() : 0;
      if (c && ps) {
         if (mode == kExact) {
            if (!strcmp(ps, subject))
               break;
         } else if (mode == kBegin) {
            if (strstr(ps, subject) == ps)
               break;
         } else if (mode == kEnd) {
            int sbeg = strlen(ps) - strlen(subject);
            if (!strcmp(ps + sbeg, subject))
               break;
         }
      }
      c = 0;
      cp = cn;  // previous
      cn = n;
   }
   // return previous, if requested
   if (prev)
      *prev = (cn) ? cp : 0;

   // We are done
   return ((cn) ? cn : (XrdCryptoX509ChainNode *)0);
}

//___________________________________________________________________________
void XrdCryptoX509Chain::Dump()
{
   // Dump content
   EPNAME("X509Chain::Dump");

   LOCDUMP("//------------------Dumping X509 chain content ------------------//");
   LOCDUMP("//");
   LOCDUMP("// Chain instance: "<<this);
   LOCDUMP("//");
   LOCDUMP("// Number of certificates: "<<Size());
   LOCDUMP("//");
   if (CAname()) {
      LOCDUMP("// CA:  "<<CAname());
   } else {
      LOCDUMP("// CA: absent");
   }
   if (EECname()) {
      LOCDUMP("// EEC:  "<<EECname());
   } else {
      LOCDUMP("// EEC: absent");
   }
   LOCDUMP("//");
   XrdCryptoX509ChainNode *n = 0;
   XrdCryptoX509ChainNode *c = begin;
   while (c) {
      n = c->Next();
      if (c->Cert()) {
         LOCDUMP("// Issuer: "<<c->Cert()->IssuerHash()<<
               " Subject: "<<c->Cert()->SubjectHash()<<
                  " Type: "<<c->Cert()->Type());
      }
      c = n;
   }
   LOCDUMP("//");
   LOCDUMP("//---------------------------- END ------------------------------//")
}

//___________________________________________________________________________
int XrdCryptoX509Chain::Reorder()
{ 
   // Reorder certificates in such a way that certificate n is the
   // issuer of certificate n+1 .
   // Return -1 if inconsistencies are found.
   EPNAME("X509Chain::Reorder");

   if (size < 2) {
      DEBUG("Nothing to reorder (size: "<<size<<")");
      return 0;
   }

   // Loop over the certificates
   XrdCryptoX509ChainNode *nc = 0, *np = 0, *nn = 0, *nr = 0, *npp = 0;

   // Look for the first one, if needed
   nr = begin;
   np = nr;
   while (nr) {
      //
      if (!(nn = FindSubject(nr->Cert()->Issuer(),kExact,&npp)) ||
            nn == nr)
         break;
      np = nr;
      nr = nr->Next();
   }

   // Move it in first position if not yet there
   if (nr != begin) { 
      np->SetNext(nr->Next()); // short cut old position
      nr->SetNext(begin);      // set our next to present begin
      if (end == nr)           // Update end
         end = np;
      begin = nr;              // set us as begin
      // Flag if not CA: we do not check validity here
      if (nr->Cert()->type != XrdCryptoX509::kCA) {
         statusCA = kAbsent;
      } else if (caname.length() <= 0) {
         // Set the CA properties only if not done already to avoid overwriting
         // the result of previous analysis
         caname = nr->Cert()->Subject();
         cahash = nr->Cert()->SubjectHash();
         statusCA = kUnknown;
      }
   }

   int left = size-1;
   np = begin;
   while (np) {
      if (np->Cert()) {
         const char *pi = np->Cert()->Subject();
         // Set the EEC name, if not yet done
         if (np->Cert()->type == XrdCryptoX509::kEEC && eecname.length() <= 0) {
            eecname = pi;
            eechash = np->Cert()->SubjectHash();
         }
         npp = np;
         nc = np->Next();
         while (nc) {
            if (nc->Cert() && !strcmp(pi, nc->Cert()->Issuer())) {
               left--;
               if (npp != np) {
                  npp->SetNext(nc->Next()); // drop child from previous pos
                  nc->SetNext(np->Next());  // set child next as our present
                  np->SetNext(nc);          // set our next as child
                  if (nc == end)
                     end = npp;
               }
               break;
            }
            npp = nc;
            nc = nc->Next();
         }
      }
      np = np->Next();
   }

   // Check consistency
   if (left > 0) {
      DEBUG("Inconsistency found: "<<left<<
            " certificates could not be correctly enchained!");
      return -1;
   }

   // We are done
   return 0;
} 

//___________________________________________________________________________
bool XrdCryptoX509Chain::Verify(EX509ChainErr &errcode, x509ChainVerifyOpt_t *vopt)
{
   // Verify cross signatures of the chain
   EPNAME("X509Chain::Verify");
   errcode = kNone; 

   // Do nothing if empty 
   if (size < 1) {
      DEBUG("Nothing to verify (size: "<<size<<")");
      return 0;
   }

   //
   // Reorder if needed
   if (Reorder() != 0) {
      errcode = kInconsistent;
      lastError = ":";
      lastError += X509ChainError(errcode);
      return 0;
   }

   //
   // Verification options
   int when = (vopt) ? vopt->when : (int)time(0);
   int plen = (vopt) ? vopt->pathlen : -1;
   bool chkss = (vopt) ? (vopt->opt & kOptsCheckSelfSigned) : 1;

   //
   // Global path depth length consistency check
   if (plen > -1 && plen < size) {
      errcode = kTooMany;
      lastError = "checking path depth: ";
      lastError += X509ChainError(errcode);
   }

   //
   // Check the first certificate: it MUST be of CA type, valid,
   // self-signed
   if (!CheckCA(chkss)) {
      errcode = kNoCA;
      lastError = X509ChainError(errcode);
      return 0;
   }

   //
   // Analyse the rest
   XrdCryptoX509ChainNode *node = begin;
   XrdCryptoX509 *xsig = node->Cert();    // Signing certificate
   XrdCryptoX509 *xcer = 0;               // Certificate under exam
   node = node->Next();
   while (node) {

      // Attache to certificate
      xcer = node->Cert();

      // Standard verification
      if (!Verify(errcode, "cert: ", XrdCryptoX509::kUnknown, when, xcer, xsig))
         return 0;

      // Get next
      xsig = xcer;
      node = node->Next();
   }

   // We are done (successfully!)
   return 1;
}

//___________________________________________________________________________
int XrdCryptoX509Chain::CheckValidity(bool outatfirst, int when)
{
   // Check validity at 'when' of certificates in the chain and return
   // the number of invalid certificates.
   // If 'outatfirst' return after the first invalid has been
   // found.
   EPNAME("X509Chain::CheckValidity");
   int ninv = 0;

   // Do nothing if empty 
   if (size < 1) {
      DEBUG("Nothing to verify (size: "<<size<<")");
      return ninv;
   }

   // Loop over the certificates
   XrdCryptoX509ChainNode *nc = begin;
   while (nc) {
      //
      XrdCryptoX509 *c = nc->Cert();
      if (c) {
         if (!(c->IsValid(when))) {
            ninv++;
            DEBUG("invalid certificate found");
            if (outatfirst)
               return ninv;
         }
      } else {
         ninv++;
         DEBUG("found node without certificate");
         if (outatfirst)
            return ninv;
      }
      // Get next      
      nc = nc->Next();
   }

   // We are done
   return ninv;
}

//___________________________________________________________________________
bool XrdCryptoX509Chain::Verify(EX509ChainErr &errcode, const char *msg,
                                XrdCryptoX509::EX509Type type, int when,
                                XrdCryptoX509 *xcer, XrdCryptoX509 *xsig,
                                XrdCryptoX509Crl *crl)
{
   // Internal verification method

   // Certificate must be defined
   if (!xcer) {
      errcode = kNoCertificate;
      lastError = msg;
      lastError += X509ChainError(errcode);
      return 0;
   }

   // Type should be the one expected
   if (type != XrdCryptoX509::kUnknown && xcer->type != type) {
      errcode = kInvalidType;
      lastError = msg;
      lastError += X509ChainError(errcode);
      return 0;
   }

   // Must not be revoked (check only if required)
   if (crl) {
      // Get certificate serial number
      XrdOucString sn = xcer->SerialNumberString();
      if (crl->IsRevoked(sn.c_str(), when)) {
         errcode = kRevoked;
         lastError = msg;
         lastError += X509ChainError(errcode);
         return 0;
      }
   }

   // Check validity in time
   if (when >= 0 && !(xcer->IsValid(when))) {
      errcode = kExpired;
      lastError = msg;
      lastError += X509ChainError(errcode);
      return 0;
   }

   // Check signature
   if (!xsig || !(xcer->Verify(xsig))) {
      errcode = kVerifyFail;
      lastError = msg;
      lastError += X509ChainError(errcode);
      return 0;
   }

   // We are done
   return 1;
}

//_____________________________________________________________________________
const char *XrdCryptoX509Chain::CAname()
{
   // Return subject name of the CA in the chain
   EPNAME("X509Chain::CAname");

   // If we do not have it already, try extraction
   if (caname.length() <= 0 && statusCA == kUnknown) {

      if (!CheckCA()) {
         DEBUG("CA not found in chain");
         return (const char *)0;
      }
   }

   // return what we have
   return (caname.length() > 0) ? caname.c_str() : (const char *)0;
}

//_____________________________________________________________________________
const char *XrdCryptoX509Chain::EECname()
{
   // Return subject name of the EEC in the chain
   EPNAME("X509Chain::EECname");

   // If we do not have it already, try extraction
   if (eecname.length() <= 0) {

      XrdCryptoX509ChainNode *c = begin;
      while (c) {
         if (c->Cert()->type == XrdCryptoX509::kEEC) {
            eecname = c->Cert()->Subject();
            break;
         }
         c = c->Next();
      }
      if (eecname.length() <= 0) {
         DEBUG("EEC not found in chain");
         return (const char *)0;
      }
   }

   // return what we have
   return (eecname.length() > 0) ? eecname.c_str() : (const char *)0;
}

//_____________________________________________________________________________
const char *XrdCryptoX509Chain::CAhash()
{
   // Return the subject name hash of the CA in the chain
   EPNAME("X509Chain::CAhash");

   // If we do not have it already, try extraction
   if (cahash.length() <= 0 && statusCA == kUnknown) {

      if (!CheckCA()) {
         DEBUG("CA not found in chain");
         return (const char *)0;
      }
   }

   // return what we have
   return (cahash.length() > 0) ? cahash.c_str() : (const char *)0;
}

//_____________________________________________________________________________
const char *XrdCryptoX509Chain::EEChash()
{
   // Return the subject name hash of the EEC in the chain
   EPNAME("X509Chain::EEChash");

   // If we do not have it already, try extraction
   if (eechash.length() <= 0) {

      XrdCryptoX509ChainNode *c = begin;
      while (c) {
         if (c->Cert()->type == XrdCryptoX509::kEEC) {
            eechash = c->Cert()->SubjectHash();
            break;
         }
         c = c->Next();
      }
      if (eechash.length() <= 0) {
         DEBUG("EEC not found in chain");
         return (const char *)0;
      }
   }

   // return what we have
   return (eechash.length() > 0) ? eechash.c_str() : (const char *)0;
}
