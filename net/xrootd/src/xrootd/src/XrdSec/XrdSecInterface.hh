#ifndef __SEC_INTERFACE_H__
#define __SEC_INTERFACE_H__
/******************************************************************************/
/*                                                                            */
/*                    X r d S e c I n t e r f a c e . h h                     */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <errno.h>
#ifndef WIN32
#include <netdb.h>
#include <netinet/in.h>
#include <sys/param.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#if defined(__CYGWIN__) || defined(__FreeBSD__)
#include <sys/socket.h>
#endif

#include "XrdSec/XrdSecEntity.hh"

/******************************************************************************/
/*  X r d S e c C r e d e n t i a l s   &   X r d S e c P a r a m e t e r s   */
/******************************************************************************/
  
// The following structure is used to pass security information back and forth
//
struct XrdSecBuffer
{
       int   size;
       char *buffer;

       XrdSecBuffer(char *bp=0, int sz=0) : size(sz), buffer(bp), membuf(bp) {}
      ~XrdSecBuffer() {if (membuf) free(membuf);}

private:
        char *membuf;
};

// When the buffer is used for credentials, the start of the buffer always
// holds the credential protocol name (e.g., krb4) as a string. The client
// will get credentials and the size will be filled out so that the contents
// of buffer can be easily transmitted to the server.
//
typedef XrdSecBuffer XrdSecCredentials;

// When the buffer is used for parameters, the contents must be interpreted
// in the context that it is used. For instance, the server will send the
// security configuration parameters on the initial login. The format differs
// from. say, the x.500 continuation paremeters that would be sent during
// PKI authentication via an "authmore" return status.
//
typedef XrdSecBuffer XrdSecParameters;
  
/******************************************************************************/
/*                        X r d S e c P r o t o c o l                         */
/******************************************************************************/

// The XrdSecProtocol is used to generate authentication credentials and to
// authenticate those credentials. For example, When a server indicates
// that authentication is needed (i.e., it returns security parameters), the 
// client must call XrdSecgetProtocol() to get an appropriate XrdSecProtocol
// (i.e., one specific to the authentication protocol that needs to be used). 
// Then the client can use the first form getCredentials() to generate the 
// appropriate identification information. On subsequent calls in response to
// "authmore" the client must use the second form, providing the additional
// parameters the the server sends. The server uses Authenticate() to verify
// the credentials. When XrdOucErrInfo is null (as it will usually be), error
// messages are routed to standard error. So, for example, a client would

// 1) Call XrdSecGetProtocol() to get an appropriate XrdSecProtocol
//    (i.e., one specific to the authentication protocol that needs to be used).
//    Note that successive calls to XrdSecGetProtocol() using the same
//    XrdSecParameters will use the subsequent protocol named in the list of
//    protocols that the server returned. Failure is indicated when the list
//    is exhausted or none of the protocols apply (which exhausts the list).


// 2) Call getCredentials() without supplying any parameters so as to
//    generate identification information and send them to the server.

// 3) If the server indicates "authmore", call getCredentials() supplying
//    the additional parameters sent by the server. The returned credentials
//    are then sent to the server using the "authneticate" request code.

// 4) Repeat step (3) as often as "authmore" is requested by the server.

// The server uses Authenticate() to verify the credentials and getParms()
// to generate initial parameters to start the authentication process. 

// When XrdOucErrInfo is null (as it will usually be), error messages are
// are routed to standard error.

// Server-side security is handled by the XrdSecServer object and, while
// it uses XrdSecProtocol objects to perform authentication, the XrdSecServer
// object is used to initialize the security environment and to generate
// the appropriate protocol objects at run-time. See XrdSecServer.hh.

// MT Requirements: Must be MT_Safe.

class XrdOucErrInfo;

class XrdSecProtocol
{
public:

// The following structure holds the entity's identification. It is filled
// in by a successful call to Authenticate().
//
XrdSecEntity               Entity;

// Authenticate credentials supplied by the client or server. Upon success,
// the XrdSecIdentity structure is completed. The method returns:
//
// > 0 -> parms  present (more authentication needed)
// = 0 -> client present (authentication suceeded)
// < 0 -> einfo  present (error has occured)
//
virtual int                Authenticate  (XrdSecCredentials  *cred,
                                          XrdSecParameters  **parms,
                                          XrdOucErrInfo      *einfo=0)=0;

// Generate credentials to be used in the authentication process. Upon
// success, return a credentials object. Upon failure, returns null and
// einfo, if present, has the reason for the failure.
//
virtual XrdSecCredentials *getCredentials(XrdSecParameters   *parm=0,
                                          XrdOucErrInfo      *einfo=0)=0;

// Encrypt data in inbuff and place it in outbuff.
//
// Returns: < 0 Failed, the return value is -errno of the reason. Typically,
//              -EINVAL    - one or more arguments are invalid.
//              -NOTSUP    - encryption not supported by the protocol
//              -EOVERFLOW - outbuff is too small to hold result
//              -ENOENT    - Context not innitialized
//          = 0 Success, outbuff contains a pointer to the encrypted data.
//
virtual int     Encrypt(const char    * /*inbuff*/,  // Data to be encrypted
                              int       /*inlen*/,   // Length of data in inbuff
                        XrdSecBuffer ** /*outbuff*/  // Returns encrypted data
                             ) {return -ENOTSUP;}

// Decrypt data in inbuff and place it in outbuff.
//
// Returns: < 0 Failed,the return value is -errno (see Encrypt).
//          = 0 Success, outbuff contains a pointer to the encrypted data.
//
virtual int     Decrypt(const char  * /*inbuff*/,   // Data to be decrypted
                              int     /*inlen*/,    // Length of data in inbuff
                      XrdSecBuffer ** /*outbuff*/   // Buffer for decrypted data
                              ) {return -ENOTSUP;}

// Sign data in inbuff and place the signiture in outbuff.
//
// Returns: < 0 Failed, returned value is -errno (see Encrypt).
//          = 0 Success, the return value is the length of the signature
//              placed in outbuff.
//
virtual int     Sign(const char  * /*inbuff*/,   // Data to be signed
                           int     /*inlen*/,    // Length of data in inbuff
                   XrdSecBuffer ** /*outbuff*/   // Buffer for the signature
                           ) {return -ENOTSUP;}

// Verify a signature
//
// Returns: < 0 Failed, returned value is -errno (see Encrypt).
//          = 0 Signature matches the value in inbuff.
//          > 0 Failed to verify, signature does not match inbuff data.
//
virtual int     Verify(const char  * /*inbuff*/,   // Data to be decrypted
                             int     /*inlen*/,    // Length of data in inbuff
                       const char  * /*sigbuff*/,  // Buffer for signature
                             int     /*siglen*/)   // Length if signature
                      {return -ENOTSUP;}

// Get the current encryption key
//
// Returns: < 0 Failed, returned value if -errno (see Encrypt)
//         >= 0 The size of the encyption key. The supplied buffer of length
//              size hold the key. If the buffer address is 0, only the 
//              size of the key is returned.
//
virtual int     getKey(char * /*buff*/=0, int /*size*/=0) {return -ENOTSUP;}

// Set the current encryption key
//
// Returns: < 0 Failed, returned value if -errno (see Encrypt)
//            0 The new key has been set.
//
virtual int     setKey(char * /*buff*/, int /*size*/) {return -ENOTSUP;}

// DO NOT use C++ delete() on this object. Since the same compiler may not
// have been used in constructing all shared libraries, you must use the object
// specific Delete() method to insure that the object creator's delete is used.
//
virtual void    Delete()=0; // Normally does "delete this"

              XrdSecProtocol(const char *pName) : Entity(pName) {}
protected:

virtual      ~XrdSecProtocol() {}
};
 
/******************************************************************************/
/*           P r o t o c o l   N a m i n g   C o n v e n t i o n s            */
/******************************************************************************/

// Each specific protocol resides in a shared library named "libXrdSec<p>.so"
// where <p> is the protocol identifier (e.g., krb5, gsi, etc). The library
// contains a class derived from the XrdSecProtocol object. The library must
// also contain a two extern "C" functions:
// 1) XrdSec<p>Init()   - for one-time protocol ininitialization, and
// 2) XrdSec<p>Object() - for protocol object instantiation.
//
// extern "C" {char   *XrdSecProtocol<p>Init  (const char              who,
//                                             const char             *parms,
//                                                   XrdOucErrInfo    *einfo);
//            }
// Is used by the dynamic protocol loader to initialize the protocol when the
// shared library is loaded. Parmater who contains 'c' when called on the
// client side and 's' when called on the server side. For client initialization,
// the parms is null. For server size initialization, parms contains the
// parameters specified in the configuration file. The protocol must return
// the parameters it needs to have sent to the client during the initial
// authentication handshake. If no parameters need to be sent, it must return
// the null string. If initialization fails, null must be returned and einfo
// must contain the reason for the failure. The storage occupied by the returned
// string is not freed by the dynamic loader; therefore, constant strings can 
// be returned.

// MT Requirements: None. Function called once in single-thread mode.

// extern "C" {
//     XrdSecProtocol *XrdSecProtocol<p>Object(const char              who,
//                                             const char             *hostname,
//                                             const struct sockaddr  &netaddr,
//                                             const char             *parms,
//                                                   XrdOucErrInfo    *einfo);
//            }
// Is used by the dynamic protocol loader to obtain an instance of the
// XrdSecProtocol object. Argument who will contain 'c' for client-side calls
// and 's' for server-side calls. When who = 'c' then parms contains the parms
// supplied by the protocol at server-side initialization time (see the
// function Xrdsec<p>Init*(, explained above). When who = 's', parms is null.

// Warning! The protocol *must* allow both 'c' and 's' calls to occur within
// the same execution context. This occurs when a server acts like a client.

// The naming conventions were chosen to avoid platform dependent run-time 
// loaders that resolve all addresses with the same name in all shared libraries 
// to the first address with the same name encountered by the run-time loader.

// MT Requirements: Must be MT_Safe.
  
/******************************************************************************/
/*                     X r d S e c G e t P r o t o c o l                      */
/*                                                                            */
/*                  C l i e n t   S i d e   U S e   O n l y                   */
/******************************************************************************/
  
// The following external routine creates a security context and returns an
// XrdSecProtocol object to be used for authentication purposes. The caller
// provides the host name and IP address of the remote connection along with 
// any parameters provided by the server. A null return means an error occured.
// Error messages are sent to standard error unless and XrdOucErrInfo class is 
// provided to capture the message. There should be one protocol object per
// physical TCP/IP connection. 

// When the connection is closed, the protocol's Delete() method should be 
// called to properly delete the object.
//
extern "C"
{
extern XrdSecProtocol *XrdSecGetProtocol(const char             *hostname,
                                         const struct sockaddr  &netaddr,
                                               XrdSecParameters &parms,
                                               XrdOucErrInfo    *einfo=0);
}

// MT Requirements: Must be MT_Safe.
 
/******************************************************************************/
/*                         X r d S e c S e r v i c e                          */
/*                                                                            */
/*                  S e r v e r   S i d e   U s e   O n l y                   */
/******************************************************************************/
  
// The XrdSecService object is the the object that the server uses to obtain
// parameters to be passed to the client on initial contact and to create the
// appropriate protocol on the initial receipt of the client's credentials.
// Server-side processing is a bit more complicated because the set of valid
// protocols needs to be configured and that configuration needs to be supplied
// to the client so that both can agree on a compatible protocol. This object
// is created via a call to XrdSecgetService, defined later on.
  
class XrdSecService
{
public:

// = 0 -> No security parameters need to be supplied to the client.
//        This implies that authentication need not occur.
// ! 0 -> Address of the parameter string (which may be host-specigfic if hname
//        was supplied). Ths length of the string is returned in size.
//
virtual const char     *getParms(int &size, const char *hname=0) = 0;

// = 0 -> No protocol can be returned (einfo has the reason)
// ! 0 -> Address of protocol object is bing returned. If cred is null,
//        a host protocol object is returned if so allowed.
//
virtual XrdSecProtocol *getProtocol(const char              *host,    // In
                                    const struct sockaddr   &hadr,    // In
                                    const XrdSecCredentials *cred,    // In
                                    XrdOucErrInfo           *einfo)=0;// Out

                        XrdSecService() {}
virtual                ~XrdSecService() {}
};

// MT Requirements: Must be MT_Safe.
  
/******************************************************************************/
/*                      X r d g e t S e c S e r v i c e                       */
/******************************************************************************/

// The XrdSecSgetService function is calle during server initialization to
// obtain the XrdSecService object. This object is used to control server-side
// authentication.
//
class XrdSysLogger;

extern "C"
{
extern XrdSecService *XrdSecgetService(XrdSysLogger *lp, const char *cfn);
}

// MT Requirements: None. Function called once in single-thread mode.
#endif
