/******************************************************************************/
/*                                                                            */
/*                 X r d S e c P r o t o c o l s s l . c c                    */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//       $Id$


const char *XrdSecProtocolsslCVSID = "$Id$";

#include "XrdSecProtocolssl.hh"

char*  XrdSecProtocolssl::sslcadir=0;
char*  XrdSecProtocolssl::sslvomsdir=0;
char*  XrdSecProtocolssl::procdir=(char*)"";
XrdSecProtocolsslProc* XrdSecProtocolssl::proc=(XrdSecProtocolsslProc*)0;
int    XrdSecProtocolssl::verifydepth=10;
int    XrdSecProtocolssl::verifyindex=0;
int    XrdSecProtocolssl::sslselecttimeout=10;
int    XrdSecProtocolssl::sslsessioncachesize=2000;
char*  XrdSecProtocolssl::sslkeyfile=0;
char*  XrdSecProtocolssl::sslserverkeyfile=0;
char   XrdSecProtocolssl::sslserverexportpassword[EXPORTKEYSTRENGTH+1];
char*  XrdSecProtocolssl::sslcertfile=0;
char*  XrdSecProtocolssl::sslproxyexportdir=(char*)0;
bool   XrdSecProtocolssl::sslproxyexportplain=1;
int    XrdSecProtocolssl::debug=0;
time_t XrdSecProtocolssl::sslsessionlifetime=86400;
bool   XrdSecProtocolssl::isServer=0;
bool   XrdSecProtocolssl::forwardProxy=0;
bool   XrdSecProtocolssl::allowSessions=0;
char*  XrdSecProtocolssl::SessionIdContext = (char*)"xrootdssl"; 
char*  XrdSecProtocolssl::gridmapfile = (char*) "/etc/grid-security/grid-mapfile";
bool   XrdSecProtocolssl::mapuser  = false;
bool   XrdSecProtocolssl::mapnobody  = false;
char*  XrdSecProtocolssl::vomsmapfile = (char*) "/etc/grid-security/voms-mapfile";
bool   XrdSecProtocolssl::mapgroup = false;
bool   XrdSecProtocolssl::mapcerncertificates = false;
int    XrdSecProtocolssl::threadsinuse=0;
int    XrdSecProtocolssl::errortimeout=0;
int    XrdSecProtocolssl::errorverify=0;
int    XrdSecProtocolssl::errorqueue=0;
int    XrdSecProtocolssl::erroraccept=0;
int    XrdSecProtocolssl::errorread=0;
int    XrdSecProtocolssl::errorabort=0;
int    XrdSecProtocolssl::forwardedproxies=0;
X509_STORE*    XrdSecProtocolssl::store=0;
X509_LOOKUP*   XrdSecProtocolssl::lookup=0;
SSL_CTX*       XrdSecProtocolssl::ctx=0;
XrdSysError    XrdSecProtocolssl::ssleDest(0, "secssl_");
XrdSysLogger   XrdSecProtocolssl::Logger;
time_t         XrdSecProtocolssl::storeLoadTime;

XrdSysMutex XrdSecsslSessionLock::sessionmutex;
XrdOucHash<XrdOucString> XrdSecProtocolssl::gridmapstore;
XrdOucHash<XrdOucString> XrdSecProtocolssl::vomsmapstore;
XrdOucHash<XrdOucString> XrdSecProtocolssl::stringstore;
XrdSysMutex              XrdSecProtocolssl::StoreMutex;           
XrdSysMutex              XrdSecProtocolssl::GridMapMutex;
XrdSysMutex              XrdSecProtocolssl::VomsMapMutex;
XrdSysMutex*             XrdSecProtocolssl::CryptoMutexPool[PROTOCOLSSL_MAX_CRYPTO_MUTEX];
XrdSysMutex              XrdSecProtocolssl::ThreadsInUseMutex;
XrdSysMutex              XrdSecProtocolssl::ErrorMutex;

XrdSysMutex SSLTRACEMUTEX;

/******************************************************************************/
/*             T h r e a d - S a f e n e s s   F u n c t i o n s              */
/******************************************************************************/
static unsigned long protocolssl_id_callback(void) {
  return (unsigned long)XrdSysThread::ID();
}

void protocolssl_lock(int mode, int n, const char *file, int line)
{
  if (mode & CRYPTO_LOCK) {
    if (XrdSecProtocolssl::CryptoMutexPool[n]) {
      XrdSecProtocolssl::CryptoMutexPool[n]->Lock();
    }
  } else {
    if (XrdSecProtocolssl::CryptoMutexPool[n]) {
      XrdSecProtocolssl::CryptoMutexPool[n]->UnLock();
    }
  }
}


/******************************************************************************/
/*             C l i e n t   O r i e n t e d   F u n c t i o n s              */
/******************************************************************************/

int secprotocolssl_pem_cb(char *buf, int size, int rwflag, void *password)
{
  memset(buf,0,size);
  memcpy(buf,XrdSecProtocolssl::sslserverexportpassword,EXPORTKEYSTRENGTH+1);
  return EXPORTKEYSTRENGTH;
}

void XrdSecProtocolssl::GetEnvironment() {
  EPNAME("GetEnvironment");
  // default the cert/key file to the standard proxy locations
  char proxyfile[1024];
  sprintf(proxyfile,"/tmp/x509up_u%d",(int)geteuid());

  if (sslproxyexportdir) {
    sprintf(proxyfile,"%s/x509up_u%d",sslproxyexportdir,(int)geteuid()); 
  }

  if (XrdSecProtocolssl::sslcertfile) { free (XrdSecProtocolssl::sslcertfile); }
  if (XrdSecProtocolssl::sslkeyfile) { free (XrdSecProtocolssl::sslkeyfile); }

  XrdSecProtocolssl::sslcertfile = strdup(proxyfile);
  XrdSecProtocolssl::sslkeyfile  = strdup(proxyfile);

  char *cenv = getenv("XrdSecDEBUG");
  // debug
  if (cenv)
    if (cenv[0] >= 49 && cenv[0] <= 57) XrdSecProtocolssl::debug = atoi(cenv);

  // directory with CA certificates
  cenv = getenv("XrdSecSSLCADIR");
  if (cenv) {
    if (XrdSecProtocolssl::sslcadir) { free (XrdSecProtocolssl::sslcadir); }
    XrdSecProtocolssl::sslcadir = strdup(cenv);
  }
  else {
    // accept X509_CERT_DIR 
    cenv = getenv("X509_CERT_DIR");
    if (cenv) {
      if (XrdSecProtocolssl::sslcadir) { free (XrdSecProtocolssl::sslcadir); }
      XrdSecProtocolssl::sslcadir = strdup(cenv);
    }
  }
  // directory with VOMS certificates
  cenv = getenv("XrdSecSSLVOMSDIR");
  if (cenv) {
    if (XrdSecProtocolssl::sslvomsdir) { free (XrdSecProtocolssl::sslvomsdir); }
    XrdSecProtocolssl::sslvomsdir = strdup(cenv);
  }


  // file with user cert
  cenv = getenv("XrdSecSSLUSERCERT");
  if (cenv) {
    if (XrdSecProtocolssl::sslcertfile) { free (XrdSecProtocolssl::sslcertfile); }
    XrdSecProtocolssl::sslcertfile = strdup(cenv);  
  } else {
    // accept X509_USER_CERT
    cenv = getenv("X509_USER_CERT");
    if (cenv) {
      if (XrdSecProtocolssl::sslcertfile) { free (XrdSecProtocolssl::sslcertfile); }
      XrdSecProtocolssl::sslcertfile = strdup(cenv);
    } else {
      // accept X509_USER_PROXY
      cenv = getenv("X509_USER_PROXY");
      if (cenv) {
	if (XrdSecProtocolssl::sslcertfile) { free (XrdSecProtocolssl::sslcertfile); }
	XrdSecProtocolssl::sslcertfile = strdup(cenv);
      }
    }
  }

  cenv = getenv("XrdSecSSLSELECTTIMEOUT");
  if (cenv) {
    XrdSecProtocolssl::sslselecttimeout = atoi(cenv);
    if ( XrdSecProtocolssl::sslselecttimeout < 5) {
      XrdSecProtocolssl::sslselecttimeout = 5;
    }
  }

  // file with user key
  cenv = getenv("XrdSecSSLUSERKEY");
  if (cenv) {
    if (XrdSecProtocolssl::sslkeyfile) { free (XrdSecProtocolssl::sslkeyfile); }    
      XrdSecProtocolssl::sslkeyfile = strdup(cenv);
  } else {
    // accept X509_USER_KEY
    cenv = getenv("X509_USER_KEY");
    if (cenv) {
      if (XrdSecProtocolssl::sslkeyfile) { free (XrdSecProtocolssl::sslkeyfile); }    
      XrdSecProtocolssl::sslkeyfile = strdup(cenv);
    } else {
      // accept X509_USER_PROXY
      cenv = getenv("X509_USER_PROXY");
      if (cenv) {
	if (XrdSecProtocolssl::sslkeyfile) { free (XrdSecProtocolssl::sslkeyfile); }    
	XrdSecProtocolssl::sslkeyfile = strdup(cenv);
      }
    }
  }
  // verify depth
  cenv = getenv("XrdSecSSLVERIFYDEPTH");
  if (cenv)
    XrdSecProtocolssl::verifydepth = atoi(cenv);
    
  // proxy forwarding
  cenv = getenv("XrdSecSSLPROXYFORWARD");
  if (cenv)
    XrdSecProtocolssl::forwardProxy = atoi(cenv);

  // ssl session reuse
  cenv = getenv("XrdSecSSLSESSION");
  if (cenv)
    XrdSecProtocolssl::allowSessions = atoi(cenv);
  
  TRACE(Authen,"====> debug         = " << XrdSecProtocolssl::debug);
  TRACE(Authen,"====> cadir         = " << XrdSecProtocolssl::sslcadir);
  TRACE(Authen,"====> keyfile       = " << XrdSecProtocolssl::sslkeyfile);
  TRACE(Authen,"====> certfile      = " << XrdSecProtocolssl::sslcertfile);
  TRACE(Authen,"====> verify depth  = " << XrdSecProtocolssl::verifydepth);
  TRACE(Authen,"====> timeout       = " << XrdSecProtocolssl::sslselecttimeout);
}

int XrdSecProtocolssl::Fatal(XrdOucErrInfo *erp, const char *msg, int rc)
{
  const char *msgv[8];
  int k, i = 0;
  
  msgv[i++] = "Secssl: ";    //0
  msgv[i++] = msg;            //1

  if (erp) erp->setErrInfo(rc, msgv, i);
  else {for (k = 0; k < i; k++) cerr <<msgv[k];
    cerr <<endl;
  }
  
  if (XrdSecProtocolssl::proc) {
    XrdSecProtocolsslProcFile* pf;
    char ErrorInfo[16384];
    sprintf(ErrorInfo,"errortimeout  = %d\nerrorverify   = %d\nerrorqueue    = %d\nerroraccept   = %d\nerrorread     = %d\nerrorabort    = %d", errortimeout, errorverify, errorqueue, erroraccept, errorread, errorabort); 
    pf= XrdSecProtocolssl::proc->Handle("error"); pf && pf->Write(ErrorInfo);
  }

  return -1;
}


int ssl_select(int fd) {

  if (fd<0)
    return -1;

  fd_set read_mask;

  struct timeval timeout;
    
  timeout.tv_sec = 0;
  timeout.tv_usec = 100000;
    
  FD_ZERO(&read_mask);
  FD_SET(fd, &read_mask);

  int result = select(fd + 1, &read_mask, 0, 0, &timeout);
    
  if ( (result < 0 ) && (errno == EINTR || errno == EAGAIN))
    return 0;

  if (result < 0)
    return -1;

  return 1;
}


int ssl_continue(SSL* ssl, int err) {
  switch (SSL_get_error(ssl,err)) {
  case SSL_ERROR_NONE:
    return 0;
  case SSL_ERROR_WANT_WRITE:
    return 1;
  case SSL_ERROR_WANT_READ:
    return 1;
  case SSL_ERROR_WANT_X509_LOOKUP:
    return 1;
  case SSL_ERROR_SYSCALL:
  case SSL_ERROR_SSL:
    if (errno == EAGAIN)
      return 1;
  case SSL_ERROR_ZERO_RETURN:
    return -1;
  }
  return -1;
}

/******************************************************************************/
/*                        g e t C r e d e n t i a l s                         */
/******************************************************************************/


void   
XrdSecProtocolssl::secClient(int theFD, XrdOucErrInfo      *error) {
  
  EPNAME("secClient");

  XrdSecsslThreadInUse ThreadInUse();

  char *nossl = getenv("XrdSecNoSSL");
  if (nossl) {
    error->setErrInfo(ENOENT,"SSL is disabled by force");
    return ;
  }

  XrdSecProtocolssl::GetEnvironment();

  error->setErrInfo(0,"");
  SSLMutex.Lock();

  int err=0;
  char*    str;
  SSL_METHOD *meth;
  SSL_SESSION *session=0;

  SSL_load_error_strings();  
  SSLeay_add_ssl_algorithms();
  meth = (SSL_METHOD*) TLSv1_client_method();

  ERR_load_crypto_strings();

  XrdOucString sslsessionfile="";
  XrdOucString sslsessionid="";
  
  sslsessionfile = "/tmp/xssl_";
  sslsessionid += (int)geteuid();
  sslsessionid += ":";
  sslsessionid += host.c_str();
  sslsessionfile += sslsessionid;

  XrdSecsslSessionLock sessionlock;
  sessionlock.SoftLock();

  if (allowSessions) {
    struct stat sessionstat;

    if (!stat(sslsessionfile.c_str(),&sessionstat)) {
      // session exists ... I try to read it
      FILE* fp = fopen(sslsessionfile.c_str(), "r");
      if (fp) {
	if (sessionlock.HardLock(sslsessionfile.c_str())) {
	  session = PEM_read_SSL_SESSION(fp, NULL, NULL, NULL);
	  fclose(fp);
	  sessionlock.HardUnLock();
	}
      }
      
      if (session) {
	
	DEBUG("Info: ("<<__FUNCTION__<<") Session loaded from " << sslsessionfile.c_str());
	char session_id[1024];
	for (int i=0; i< (int)session->session_id_length; i++) {
	  sprintf(session_id+(i*2),"%02x",session->session_id[i]);
	}
	
	unsigned char buf[5],*p;
	unsigned long l;
	
	p=buf;
	l=session->cipher_id;
	l2n(l,p);

	DEBUG("Info: ("<<__FUNCTION__<<") Session Id: "<< session_id << " Verify: " << session->verify_result << " (" << X509_verify_cert_error_string(session->verify_result) << ")");
      } else {
	DEBUG("Info: ("<<__FUNCTION__<<") Session load failed from " << sslsessionfile.c_str());
	ERR_print_errors_fp(stderr);
      }
    }
  }

  clientctx = SSL_CTX_new (meth);


  SSL_CTX_set_options(clientctx,  SSL_OP_ALL | SSL_OP_NO_SSLv2);

  if (!clientctx) {
    Fatal(error,"Cannot do SSL_CTX_new",-1);
    exit(2);
  }
  
  if (!XrdSecProtocolssl::sslproxyexportplain) {
    // set a password callback here
    SSL_CTX_set_default_passwd_cb(clientctx, secprotocolssl_pem_cb);
    SSL_CTX_set_default_passwd_cb_userdata(clientctx, XrdSecProtocolssl::sslserverexportpassword);
  }

  if (SSL_CTX_use_certificate_chain_file(clientctx, sslcertfile) <= 0) {
    ERR_print_errors_fp(stderr);
    Fatal(error,"Cannot use certificate file",-1);
    if (clientctx) SSL_CTX_free (clientctx);
    SSLMutex.UnLock();
    return;
  }

  if (SSL_CTX_use_PrivateKey_file(clientctx, sslkeyfile, SSL_FILETYPE_PEM) <= 0) {
    ERR_print_errors_fp(stderr);
    Fatal(error,"Cannot use private key file",-1);
    if (clientctx) SSL_CTX_free (clientctx);
    SSLMutex.UnLock();
    return;
  }


  if (!SSL_CTX_check_private_key(clientctx)) {
    fprintf(stderr,"Error: (%s) Private key does not match the certificate public key\n",__FUNCTION__);
    Fatal(error,"Private key does not match the certificate public key",-1);
    if (clientctx) SSL_CTX_free (clientctx);
    SSLMutex.UnLock();
    return;
  } else {
    DEBUG("Private key check passed ...");
  }
  SSL_CTX_load_verify_locations(clientctx, NULL,sslcadir);
  SSL_CTX_set_verify(clientctx, SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT,  GRST_callback_SSLVerify_wrapper);

  SSL_CTX_set_cert_verify_callback(clientctx, GRST_verify_cert_wrapper, (void *) NULL);

  grst_cadir   = sslcadir;
  grst_vomsdir = sslvomsdir;

  grst_depth=verifydepth;
  SSL_CTX_set_verify_depth(clientctx, verifydepth);

  if (session) {
    SSL_CTX_add_session(clientctx,session);
  }


  ssl = SSL_new (clientctx);            
  SSL_set_purpose(ssl,X509_PURPOSE_ANY);
  if (session) {
    SSL_set_session(ssl, session);
  }

  sessionlock.SoftUnLock();
  sessionlock.HardUnLock();

  if (!ssl) {
    Fatal(error,"Cannot do SSL_new",-1);
    exit(6);
  }

  SSL_set_fd (ssl, theFD);

  /* make socket non-blocking */
  int flags;

  /* Set socket to non-blocking */
  if ((flags = fcntl(theFD, F_GETFL, 0)) < 0) {
    /* Handle error */
    fprintf(stderr,"Error: (%s) failed to make socket non-blocking\n",__FUNCTION__);
  } else {
    if (fcntl(theFD, F_SETFL, flags | O_NONBLOCK) < 0) {
      /* Handle error */
      fprintf(stderr,"Error: (%s) failed to make socket non-blocking\n",__FUNCTION__);
    }
  }

  time_t now= time(NULL);
  
  do {
    if ( (time(NULL)-now) > XrdSecProtocolssl::sslselecttimeout ) {
      ErrorMutex.Lock();erroraccept++;errortimeout++; ErrorMutex.UnLock();
      /* timeout */
      Fatal(error,"authenticate - handshake time out",-ETIMEDOUT);
      TRACE(Authen,"Error: ("<<__FUNCTION__<<") handshake timedout in SSL_connect");
      SSLMutex.UnLock();
      return;
    }

    int set = ssl_select(theFD);
    if (set < 1)
      continue;
    err = SSL_connect (ssl);
    if (err>0)
      break;
  } while ( (ssl_continue(ssl,err))==1);



  if (err!=1) {
    // we want to see the error message from the server side
    ERR_print_errors_fp(stderr);
    if (clientctx) SSL_CTX_free (clientctx);
    SSLMutex.UnLock();
    return;
  }

  /* I am not sure, if this is actually needed at all */
  if (!session)
    session = SSL_get1_session(ssl);
  
  /* Get the cipher - opt */
  
  TRACE(Authen,"SSL connection uses cipher: "<<SSL_get_cipher (ssl));
  
  /* Get server's certificate (note: beware of dynamic allocation) - opt */
  
  server_cert = SSL_get_peer_certificate (ssl);       

  if (!server_cert) {
    TRACE(Authen,"Server didn't provide certificate");
  }

  XrdOucString rdn;
  
  str = X509_NAME_oneline (X509_get_subject_name (server_cert),0,0);
  rdn = str;
  TRACE(Authen,"Server certificate subject:\t" << str);
  OPENSSL_free (str);
  
  str = X509_NAME_oneline (X509_get_issuer_name  (server_cert),0,0);
  TRACE(Authen,"Server certificate  issuer: \t" << str);
  OPENSSL_free (str);
  
  X509_free (server_cert);
  server_cert=0;


  /******************************************/
  /* this is called only to cleanup objects */
  /******************************************/

  /* get the grst_chain  */
  GRSTx509Chain *grst_chain = (GRSTx509Chain*) SSL_get_app_data(ssl);
  SSL_set_app_data(ssl,0);
  
  if (grst_chain) {
    GRST_print_ssl_creds((void*) grst_chain);
    char* vr = GRST_get_voms_roles_and_free((void*) grst_chain);
    if (vr) {
      free(vr);
    }
  }


  if (forwardProxy) {
    if (!strcmp(sslkeyfile,sslcertfile)) {
      // this is a cert & key in one file atleast ... looks like proxy
      int fd = open(sslkeyfile,O_RDONLY);
      if (fd>=0) {
	int nread = read(fd,proxyBuff, sizeof(proxyBuff));
	if (nread>=0) {
	  TRACE(Authen,"Uploading my Proxy ...\n");


	  do {
	    if ( (time(NULL)-now) > XrdSecProtocolssl::sslselecttimeout ) {
	      ErrorMutex.Lock();errorread++;errortimeout++; ErrorMutex.UnLock();
	      /* timeout */
	      Fatal(error,"authenticate - handshake time out",-ETIMEDOUT);
	      TRACE(Authen,"Error: ("<<__FUNCTION__<<") handshake timedout in SSL_read");
	      ERR_remove_state(0);
	      SSLMutex.UnLock();
	      return;
	    }
	    
	    int set = ssl_select(theFD);
	    if (set < 1)
	      continue;
	    err = SSL_write(ssl, proxyBuff,nread);
	  } while ( (ssl_continue(ssl,err)) == 1);


	  if (err!= nread) {
	    Fatal(error,"Cannot forward proxy",-1);
	    if (clientctx) SSL_CTX_free (clientctx);
	    if (session) SSL_SESSION_free(session);
	    SSLMutex.UnLock();
	    return;
	  }
	  
	  char ok[16];

	  do {
	    if ( (time(NULL)-now) > XrdSecProtocolssl::sslselecttimeout ) {
	      ErrorMutex.Lock();errorread++;errortimeout++; ErrorMutex.UnLock();
	      /* timeout */
	      Fatal(error,"authenticate - handshake time out",-ETIMEDOUT);
	      TRACE(Authen,"Error: ("<<__FUNCTION__<<") handshake timedout in SSL_read");
	      ERR_remove_state(0);
	      SSLMutex.UnLock();
	      return;
	    }
	    
	    int set = ssl_select(theFD);
	    if (set < 1)
	      continue;
	    err = SSL_read(ssl,ok, 3);
	  } while ( (ssl_continue(ssl,err)) == 1);
	  
	  if (err != 3) {
	    Fatal(error,"Didn't receive OK",-1);
	    if (clientctx) SSL_CTX_free (clientctx);
	    if (session) SSL_SESSION_free(session);
	    SSLMutex.UnLock();
	    return;
	  } 
	} else {
	  close(fd);
	  Fatal(error,"Cannot read proxy file to forward",-1);
	  if (clientctx) SSL_CTX_free (clientctx);
	  if (session) SSL_SESSION_free(session);
	  SSLMutex.UnLock();
	  return;
	}
      } else {
	Fatal(error,"Cannot read proxy file to forward",-1);
	if (clientctx) SSL_CTX_free (clientctx);
	if (session) SSL_SESSION_free(session);
	SSLMutex.UnLock();
	return;
      }
      close(fd);
    }
  }

  if (allowSessions && session) {
    char session_id[1024];
    for (int i=0; i< (int)session->session_id_length; i++) {
      sprintf(session_id+(i*2),"%02x",session->session_id[i]);
    }
    
    if (session->cipher) {
      DEBUG("Info: ("<<__FUNCTION__<<") Session Id: "<< session_id << " Cipher: " << session->cipher->name  << " Verify: " << session->verify_result << " (" << X509_verify_cert_error_string(session->verify_result) << ")");
    } else {
      DEBUG("Info: ("<<__FUNCTION__<<") Session Id: "<< session_id << " Verify: " << session->verify_result << " (" << X509_verify_cert_error_string(session->verify_result) << ")");
    }
    // write out the session
    FILE* fp = fopen((const char*)(sslsessionfile.c_str()),"w+");
    if (fp) {
      if (sessionlock.HardLock(sslsessionfile.c_str())) {
	PEM_write_SSL_SESSION(fp, session);
      }
      fclose(fp);
      sessionlock.HardUnLock();
      DEBUG("Info: ("<<__FUNCTION__<<") Session stored to " << sslsessionfile.c_str());
      if (chmod(sslsessionfile.c_str(),S_IRUSR| S_IWUSR)) {
	Fatal(error,"secure session file (chmod 600 failed) ",-errno);
      }
    }
  }

  do {
    err = SSL_shutdown(ssl);
  } while (ssl_continue(ssl,err) || (!err));
  
  if (ssl) {
    SSL_free(ssl);ssl = 0;
  }

  if (clientctx) SSL_CTX_free (clientctx);

  if (session) {
    SSL_SESSION_free(session);
  }

  SSLMutex.UnLock();
  return;
}

/******************************************************************************/
/*               S e r v e r   O r i e n t e d   M e t h o d s                */
/******************************************************************************/



/*----------------------------------------------------------------------------*/
/* this helps to avoid memory leaks by strdup                                 */
/* we maintain a string hash to keep all used user ids/group ids etc.         */

char* 
STRINGSTORE(const char* __charptr__) {
  XrdOucString* yourstring;
  if (!__charptr__ ) return (char*)"";

  XrdSecProtocolssl::StoreMutex.Lock();
  yourstring = XrdSecProtocolssl::stringstore.Find(__charptr__);
  XrdSecProtocolssl::StoreMutex.UnLock();

  if (yourstring) {
    return (char*)yourstring->c_str();
  } else {
    XrdOucString* newstring = new XrdOucString(__charptr__);
    XrdSecProtocolssl::StoreMutex.Lock();
    XrdSecProtocolssl::stringstore.Add(__charptr__,newstring);
    XrdSecProtocolssl::StoreMutex.UnLock();
    return (char*)newstring->c_str();
  } 
}

/*----------------------------------------------------------------------------*/
void MyGRSTerrorLogFunc (char *lfile, int lline, int llevel, char *fmt, ...) {
  EPNAME("grst");
  va_list args;
  char fullmessage[4096];
  fullmessage[0] = 0;

  va_start(args, fmt);
  vsprintf(fullmessage,fmt,args);
  va_end(args);

  // just remove linefeeds
  XrdOucString sfullmessage = fullmessage;
  sfullmessage.replace("\n","");

  if (llevel <= GRST_LOG_WARNING) {
    TRACE(Authen," ("<< lfile << ":" << lline <<"): " << sfullmessage);    
  } else if (llevel <= GRST_LOG_INFO) {
    TRACE(Authen, " ("<< lfile << ":" << lline <<"): " << sfullmessage);    
  } else {
    DEBUG(" ("<< lfile << ":" << lline <<"): " << sfullmessage);
  }
}

/*----------------------------------------------------------------------------*/
void
XrdSecProtocolssl::secServer(int theFD, XrdOucErrInfo      *error) {
  int err=0;

  char*    str;

  SSLMutex.Lock();


  EPNAME("secServer");

  XrdSecsslThreadInUse ThreadInUse;

  if ((debug>=4)) {
    TRACE(Identity,"Info: having " << threadsinuse << " threads running SSL authentication");
  }

  XrdSecsslSessionLock sessionlock;

  /* check if we should reload the store */
  GridMapMutex.Lock();
  if ((time(NULL)-storeLoadTime) > 3600) {
    if (store) {
      TRACE(Authen,"Reloading X509 Store from " << sslcadir);
      X509_STORE_free(store);
      store = SSL_X509_STORE_create(NULL, sslcadir);
      X509_STORE_set_flags(XrdSecProtocolssl::store,0);
      storeLoadTime = time(NULL);
    }
  }
  GridMapMutex.UnLock();

  if (XrdSecProtocolssl::sslsessioncachesize) {
    SSL_CTX_set_session_cache_mode(ctx, SSL_SESS_CACHE_BOTH); // enable autoclear every 255 connections | SSL_SESS_CACHE_NO_AUTO_CLEAR );
  } else {
    SSL_CTX_set_session_cache_mode(ctx, SSL_SESS_CACHE_OFF);
  }

  ssl = SSL_new (ctx);
  SSL_set_purpose(ssl,X509_PURPOSE_ANY);


  TRACE(Authen,"Info: ("<<__FUNCTION__<<") Session Cache has size: " <<SSL_CTX_sess_get_cache_size(ctx));

  if (!ssl) {
    fprintf(stderr,"Error: (%s) failed to create context\n",__FUNCTION__);
    TRACE(Authen,"Error: ("<<__FUNCTION__<<") failed to create context");
    exit(5);
  }

  SSL_set_app_data(ssl,0);

  SSL_set_fd (ssl, theFD);

  /* make socket non-blocking */
  int flags;

  /* Set socket to non-blocking */
  if ((flags = fcntl(theFD, F_GETFL, 0)) < 0) {
    /* Handle error */
    fprintf(stderr,"Error: (%s) failed to make socket non-blocking\n",__FUNCTION__);
  } else {
    if (fcntl(theFD, F_SETFL, flags | O_NONBLOCK) < 0) {
      /* Handle error */
      fprintf(stderr,"Error: (%s) failed to make socket non-blocking\n",__FUNCTION__);
    }
  }

  TRACE(Authen,"Before:: SSL accept loop");
  time_t now= time(NULL);
  do {
    if (terminate) {
      ErrorMutex.Lock();erroraccept++;errorabort++; ErrorMutex.UnLock();
      /* timeout */
      Fatal(error,"authenticate - handshake abort from client",-ETIMEDOUT);
      TRACE(Authen,"Error: ("<<__FUNCTION__<<") aborted SSL_accept");
      ERR_remove_state(0);
      SSLMutex.UnLock();
      return;
    }
    
    if ( (time(NULL)-now) > XrdSecProtocolssl::sslselecttimeout ) {
      ErrorMutex.Lock();erroraccept++;errortimeout++; ErrorMutex.UnLock();
      /* timeout */
      Fatal(error,"authenticate - handshake time out",-ETIMEDOUT);
      TRACE(Authen,"Error: ("<<__FUNCTION__<<") handshake timedout in SSL_accept");
      SSLMutex.UnLock();
      return;
    }

    int set = ssl_select(theFD);
    if (set < 1)
      continue;
    TRACE(Authen,"Before:: SSL accept");
    err = SSL_accept (ssl);
    TRACE(Authen,"After :: SSL accept");
    if (err>0)
      break;
  } while ( (ssl_continue(ssl,err))==1);

  TRACE(Authen,"After :: SSL accept loop");
  
  if (err!=1) {
    long verifyresult = SSL_get_verify_result(ssl);
    if (verifyresult != X509_V_OK) {
      ErrorMutex.Lock();erroraccept++;errorverify++; ErrorMutex.UnLock();
      Fatal(error,X509_verify_cert_error_string(verifyresult),verifyresult);
      TRACE(Authen,"Error: ("<<__FUNCTION__<<") failed SSL_accept ");
    } else {
      ErrorMutex.Lock();erroraccept++;errorqueue++; ErrorMutex.UnLock();
      Fatal(error,"do SSL_accept",-1);
      unsigned long lerr;
      while ((lerr=ERR_get_error())) {TRACE(Authen,"SSL Queue error: err=" << lerr << " msg=" <<
					  ERR_error_string(lerr, NULL));Fatal(error,ERR_error_string(lerr,NULL),-1);}
    }

    GRSTx509Chain *grst_chain = (GRSTx509Chain*) SSL_get_app_data(ssl);
    SSL_set_app_data(ssl,0);
    
    if (grst_chain) {
      GRST_free_chain((void*)grst_chain);
    }
    ERR_remove_state(0);
    SSLMutex.UnLock();
    return;
  }

  TRACE(Authen,"Before:: SSL get session");

  SSL_SESSION* session = SSL_get1_session(ssl);

  TRACE(Authen,"After :: SSL get session");
  if (session) {
    char session_id[1024];
    TRACE(Authen,"Doing :: SSL Print session");
    for (int i=0; i< (int)session->session_id_length; i++) {
      sprintf(session_id+(i*2),"%02x",session->session_id[i]);
    }
    
    DEBUG("Info: ("<<__FUNCTION__<<") Session Id: "<< session_id << " Verify: " << session->verify_result << " (" << X509_verify_cert_error_string(session->verify_result) << ")");
    DEBUG("Info: ("<<__FUNCTION__<<") cache items             : " << SSL_CTX_sess_number(ctx));
    DEBUG("Info: ("<<__FUNCTION__<<") client connects         : " << SSL_CTX_sess_connect(ctx));
    DEBUG("Info: ("<<__FUNCTION__<<") client renegotiates     : " << SSL_CTX_sess_connect_renegotiate(ctx));
    DEBUG("Info: ("<<__FUNCTION__<<") client connect finished : " << SSL_CTX_sess_connect_good(ctx));
    DEBUG("Info: ("<<__FUNCTION__<<") server accepts          : " << SSL_CTX_sess_accept(ctx));
    DEBUG("Info: ("<<__FUNCTION__<<") server renegotiates     : " << SSL_CTX_sess_accept_renegotiate(ctx));
    DEBUG("Info: ("<<__FUNCTION__<<") server accepts finished : " << SSL_CTX_sess_accept_good(ctx));
    DEBUG("Info: ("<<__FUNCTION__<<") session cache hits      : " << SSL_CTX_sess_hits(ctx));
    DEBUG("Info: ("<<__FUNCTION__<<") session cache misses    : " << SSL_CTX_sess_misses(ctx));
    DEBUG("Info: ("<<__FUNCTION__<<") session cache timeouts  : " << SSL_CTX_sess_timeouts(ctx));
    DEBUG("Info: ("<<__FUNCTION__<<") callback cache hits     : " << SSL_CTX_sess_cb_hits(ctx));
    DEBUG("Info: ("<<__FUNCTION__<<") cache full overflows    : " << SSL_CTX_sess_cache_full(ctx) << " allowed: " << SSL_CTX_sess_get_cache_size(ctx));

    if (XrdSecProtocolssl::proc) {
      XrdSecProtocolsslProcFile* pf;
      char CacheInfo[16384];
      sprintf(CacheInfo,"items                 = %ld\nclientconnects        = %ld\nclientrenegotiates    = %ld\nclientconnectfinished = %ld\nserveraccept          = %ld\nserverrenegotiates    = %ld\nserveracceptfinished  = %ld\nsessioncachehits      = %ld\nsessioncachemisses    = %ld\nsessioncachetimeouts  = %ld\ncallbackcachehits     = %ld\ncachefulloverflows    = %ld\ncachesize             = %ld\nhandshakethreads      = %d\nforwardedproxies      = %d\n",  SSL_CTX_sess_number(ctx),  SSL_CTX_sess_connect(ctx),SSL_CTX_sess_connect_renegotiate(ctx),SSL_CTX_sess_connect_good(ctx), SSL_CTX_sess_accept(ctx), SSL_CTX_sess_accept_renegotiate(ctx), SSL_CTX_sess_accept_good(ctx), SSL_CTX_sess_hits(ctx), SSL_CTX_sess_misses(ctx), SSL_CTX_sess_timeouts(ctx),SSL_CTX_sess_cb_hits(ctx), SSL_CTX_sess_cache_full(ctx),  SSL_CTX_sess_get_cache_size(ctx), XrdSecProtocolssl::threadsinuse, XrdSecProtocolssl::forwardedproxies);

      pf= XrdSecProtocolssl::proc->Handle("cache"); pf && pf->Write(CacheInfo);
      
      char ErrorInfo[16384];
      sprintf(ErrorInfo,"errortimeout  = %d\nerrorverify   = %d\nerrorqueue    = %d\nerroraccept   = %d\nerrorread     = %d\nerrorabort    = %d", errortimeout, errorverify, errorqueue, erroraccept, errorread, errorabort); 
      pf= XrdSecProtocolssl::proc->Handle("error"); pf && pf->Write(ErrorInfo);
    }
  }

  
  SSL_SESSION_free(session);

  /* get the grst_chain  */
  GRSTx509Chain *grst_chain = (GRSTx509Chain*) SSL_get_app_data(ssl);
  SSL_set_app_data(ssl,0);
  
  XrdOucString vomsroles="";
  XrdOucString clientdn="";
  
  if (grst_chain) {
    GRST_print_ssl_creds((void*) grst_chain);
    char* vr = GRST_get_voms_roles_and_free((void*) grst_chain);
    if (vr) {
      vomsroles = vr;
      free(vr);
    }
  }
  
  TRACE(Authen,"Authenticated with VOMS roles: "<<vomsroles);
  
  long verifyresult = SSL_get_verify_result(ssl);
  
  TRACE(Authen,"Verify result is = "<<verifyresult);
  
  /* Get the cipher - opt */
  
  DEBUG("SSL connection uses cipher " << SSL_get_cipher(ssl));
  
  /* Get client's certificate (note: beware of dynamic allocation) - opt */
  
  client_cert = SSL_get_peer_certificate (ssl);


  if (client_cert != NULL) {
    str = X509_NAME_oneline (X509_get_subject_name (client_cert), 0, 0);
    if (str) {
      TRACE(Authen,"client certificate subject: "<< str);
      clientdn = str;
      OPENSSL_free (str);
    } else {
      TRACE(Authen,"client certificate subject: none");
    }
    
    str = X509_NAME_oneline (X509_get_issuer_name  (client_cert), 0, 0);
    
    if (str) {
      TRACE(Authen,"client certificate issuer : "<<str);
      TRACE(Authen,"Setting dn="<<clientdn<<" roles="<<vomsroles);
      OPENSSL_free (str);
    } else {
      TRACE(Authen,"client certificate issuer : none");
      Fatal(error,"no client issuer",-1);
      ERR_remove_state(0);
      SSLMutex.UnLock();
      return;
    }
  } else {
    TRACE(Authen,"Client does not have certificate.");
    Fatal(error,"no client certificate",-1);
    ERR_remove_state(0);
    SSLMutex.UnLock();
    return;
  }

  /* receive client proxy - if he send's one */
  
  do {
    if (terminate) {
      ErrorMutex.Lock();errorabort++; ErrorMutex.UnLock();
      /* timeout */
      Fatal(error,"authenticate - handshake abort from client",-ECONNABORTED);
      TRACE(Authen,"Error: ("<<__FUNCTION__<<") aborted SSL_read client proxy");
      ERR_remove_state(0);
      SSLMutex.UnLock();
      return;
    }

    if ( (time(NULL)-now) > XrdSecProtocolssl::sslselecttimeout ) {
      ErrorMutex.Lock();errorread++;errortimeout++; ErrorMutex.UnLock();
      /* timeout */
      Fatal(error,"authenticate - handshake time out",-ETIMEDOUT);
      TRACE(Authen,"Error: ("<<__FUNCTION__<<") handshake timedout in SSL_read");
      ERR_remove_state(0);
      SSLMutex.UnLock();
      return;
    }

    int set = ssl_select(theFD);
    if (set < 1)
      continue;
    err = SSL_read(ssl,proxyBuff, sizeof(proxyBuff));
  } while ( (ssl_continue(ssl,err)) == 1);

  if (err>0) {
    ErrorMutex.Lock();forwardedproxies++; ErrorMutex.UnLock();
    TRACE(Authen,"Received proxy buffer with " << err << " bytes");
    proxyBuff[err] = 0; //0 terminate the proxy buffer
    Entity.endorsements = proxyBuff;
    err = SSL_write(ssl,"OK\n",3);
    if (err!=3) {
      ErrorMutex.Lock();errorread++; ErrorMutex.UnLock();
      Fatal(error,"could not send end of handshake OK",-1);
      ERR_remove_state(0);
      SSLMutex.UnLock();
      return;
    }
    //    err = SSL_read(ssl,dummy, sizeof(dummy));
    //    ;
  } else {
    TRACE(Authen,"Received no proxy");
  }

  struct timeval tv1, tv2;
  struct timezone tz;

  gettimeofday(&tv1,&tz);

  do {
    if (terminate) {
      break;
    }
    gettimeofday(&tv2,&tz);
    if ( ((((tv2.tv_sec-tv1.tv_sec)*1000)) + (((tv2.tv_usec-tv1.tv_usec))/1000)) > 500) {
      // old client versions don't shut down the ssl connection, so we leave only 500ms grace time to shutdown
      TRACE(Authen,"Warning: ("<<__FUNCTION__<<") shutdown timed out");
      SSL_set_shutdown(ssl, SSL_SENT_SHUTDOWN | SSL_RECEIVED_SHUTDOWN);
      break;
    }
    err = SSL_shutdown(ssl);
  } while (ssl_continue(ssl,err) || (!err));

  strncpy(Entity.prot,"ssl", sizeof(Entity.prot));

  /*----------------------------------------------------------------------------*/
  /* mapping interface                                                          */
  /*----------------------------------------------------------------------------*/
     
  if (!mapuser && !mapcerncertificates) { 
    // no mapping put the DN
    Entity.name = strdup(clientdn.c_str());
  } else {
    bool mapped=false;
    // map user from grid map file
    if (mapcerncertificates) {
      // map from CERN DN
      if ( (mapcerncertificates) && (clientdn.beginswith("/DC=ch/DC=cern/OU=Organic Units/OU=Users/CN="))) {
	XrdOucString certsubject = clientdn;
	certsubject.erasefromstart(strlen("/DC=ch/DC=cern/OU=Organic Units/OU=Users/CN="));
	int pos=certsubject.find('/');                               
	if (pos != STR_NPOS)                                         
	  certsubject.erase(pos);			  	        
	Entity.name = strdup(certsubject.c_str());
	mapped=true;
	TRACE(Authen,"Found CERN certificate - mapping to AFS account " << certsubject);
      }
    }
    if (!mapped) {
      if (mapuser) {
	// treatment of old proxy
	XrdOucString certsubject = clientdn;
	certsubject.replace("/CN=proxy","");                           
	// treatment of new proxy - leave only the first CN=, cut the rest
	int pos = certsubject.find("CN=");
	int pos2 = certsubject.find("/",pos);
	if (pos2>0) certsubject.erase(pos2);
	XrdOucString* gridmaprole;                                     
	ReloadGridMapFile();
	GridMapMutex.Lock();                             
	
	if ((gridmaprole = gridmapstore.Find(certsubject.c_str()))) { 
	  Entity.name = strdup(gridmaprole->c_str());      
	  Entity.role = 0;
	}  else {
	  Entity.name = strdup((char*)"nobody");
	  Entity.role = 0;
	  if (!XrdSecProtocolssl::mapnobody) {
	    Fatal(error,"user cannot be mapped",-1);
	  }
	}
	GridMapMutex.UnLock();
      } else {
	Entity.name = strdup((char*)"nobody");
	Entity.role = 0;
	if (!XrdSecProtocolssl::mapnobody) {
	  Fatal(error,"user cannot be mapped",-1);
	}
      }
    }
  }
  
  
  if (!mapgroup) {
    if (vomsroles.length()) {
      // no mapping put the VOMS groups and role
      Entity.grps = strdup(vomsroles.c_str());
      
      XrdOucString vomsrole = vomsroles.c_str();
      
      if (vomsroles.length()) {
	int dp = vomsrole.find(":");
	if (dp != STR_NPOS) {
	  vomsrole.assign(vomsroles,0,dp-1);
	}
	Entity.role = strdup(vomsrole.c_str());
      } else {
	Entity.role = strdup("");
      }
    } else {
      // map the group from the passwd/group file
      struct passwd* pwd;
      struct group*  grp;
      StoreMutex.Lock();
      if ( (pwd = getpwnam(Entity.name)) && (grp = getgrgid(pwd->pw_gid))) {
	Entity.grps   = strdup(grp->gr_name);
	Entity.role   = strdup(grp->gr_name);
      }
      StoreMutex.UnLock();
    }
  } else {
    // map groups & role from VOMS mapfile
    XrdOucString defaultgroup="";                                     
    XrdOucString allgroups="";  

    // map the group from the passwd/group file at first place
    struct passwd* pwd;
    struct group*  grp;
    StoreMutex.Lock();
    if ( (pwd = getpwnam(Entity.name)) && (grp = getgrgid(pwd->pw_gid))) {
      Entity.grps   = strdup(grp->gr_name);
      Entity.role   = strdup(grp->gr_name);
    }
    StoreMutex.UnLock();

    if (vomsroles.length()) {
      if (VomsMapGroups(vomsroles.c_str(), allgroups,defaultgroup)) {
	// allow mapping from VOMS role to uid
	if (defaultgroup.beginswith("uid:")) {
	  defaultgroup.erase(0,4);
	  if (Entity.name) {
	    free(Entity.name);
	  }
	  Entity.name = strdup(defaultgroup.c_str());
	  allgroups=":";
	}
	if (!strcmp(allgroups.c_str(),":")) {
	  // map the group from the passwd/group file
	  struct passwd* pwd;
	  struct group*  grp;
	  StoreMutex.Lock();
	  if ( (pwd = getpwnam(Entity.name)) && (grp = getgrgid(pwd->pw_gid))) {
	    allgroups    = grp->gr_name;
	    defaultgroup = grp->gr_name;
	  }
	  StoreMutex.UnLock();
	}
	Entity.grps   = strdup(allgroups.c_str());
	Entity.role   = strdup(defaultgroup.c_str());
      }
    }
  }


  /*----------------------------------------------------------------------------*/
  /* proxy forwarding                                                           */
  /*----------------------------------------------------------------------------*/

  if (sslproxyexportdir && Entity.endorsements) {
    StoreMutex.Lock();
    // get the UID of the entity name
    struct passwd* pwd;
    XrdOucString outputproxy = sslproxyexportdir; outputproxy+="/x509up_u"; 
    if ( (pwd = getpwnam(Entity.name)) ) {
      outputproxy += (int)pwd->pw_uid;
    } else {
      outputproxy += Entity.name;
    }
    XrdOucString outputproxytmp = outputproxy;
    outputproxytmp += (int) rand();

    if (XrdSecProtocolssl::sslproxyexportplain) {
      int fd = open (outputproxytmp.c_str(),O_CREAT| O_TRUNC | O_RDWR, S_IRUSR | S_IWUSR);
      if (fd>0) {
	if ( ((int)write(fd,Entity.endorsements,strlen(Entity.endorsements))) != (int)strlen(Entity.endorsements)) {
	  unlink(outputproxytmp.c_str());
	  Fatal(error,"cannot export(write) user proxy",-1);
	} else {
	  TRACE(Identity,"Exported proxy buffer of " << Entity.name << " to file " << outputproxy.c_str());
	}
	if ( rename(outputproxytmp.c_str(),outputproxy.c_str()) ) {
	  unlink(outputproxytmp.c_str());
	  Fatal(error,"cannot rename temporary export proxy",-1);
	}
	close(fd);
      } else {
	Fatal(error,"cannot export(open) user proxy",-1);
      }
    } else {
      EVP_PKEY* pkey=NULL;
      X509* x509=NULL;
      // we protect the private key with our session password
      BIO* bp = BIO_new_mem_buf( (void *)Entity.endorsements, strlen(Entity.endorsements)+1);
      FILE* fout = fopen (outputproxytmp.c_str(),"w+");
      if (!fout) {
	Fatal(error,"cannot export user proxy - unable to open proxy file",-1);
      } else {
	if (bp) {
	  pkey = PEM_read_bio_PrivateKey(bp, &pkey,0,0);
	  BIO_free(bp);
	  if (!pkey) {
	    Fatal(error,"cannot export user proxy - unable to read key/cert from BIO",-1);
	  } else {
	    int wk = PEM_write_PrivateKey(fout, pkey, EVP_des_ede3_cbc(),(unsigned char*)XrdSecProtocolssl::sslserverexportpassword,EXPORTKEYSTRENGTH,0,0);
	    EVP_PKEY_free(pkey);
	    
	    if (!wk) {
	      Fatal(error,"cannot export user proxy - unable to write private key",-1);
	    } else {
	      // deal with the certificates
	      char* certificatebuffer = 0;
	      certificatebuffer = Entity.endorsements;
	      while ((certificatebuffer = strstr(certificatebuffer,"-----BEGIN CERTIFICATE-----"))) {
		// we point to the next certificate to export in memory
		BIO* bp = BIO_new_mem_buf( (void *)certificatebuffer, strlen(certificatebuffer)+1);
		if (bp) {
		  x509 = NULL;
		  x509 = PEM_read_bio_X509(bp, &x509,0,0);
		  BIO_free(bp);
		  if (x509)  {
		    int wc = PEM_write_X509(fout,x509);
		    X509_free(x509);
		    if (!wc) {
		      Fatal(error,"cannto export user proxy - unable to write certificate",-1);
		      break;
		    }
		  }
		} else {
		  Fatal(error,"cannot export user proxy - unable to allocate BIO to read private key",-1);
		}
		certificatebuffer++;
	      }
	    }
	  }
	} else {
	  Fatal(error,"cannot export user proxy - unable to allocate BIO to read private key",-1);
	}
	
	fclose(fout);
	if ( rename(outputproxytmp.c_str(),outputproxy.c_str()) ) {
	  unlink(outputproxytmp.c_str());
	  Fatal(error,"cannot rename temporary export proxy",-1);
	}
      }
    }
    
    StoreMutex.UnLock();
  }


  TRACE(Identity,"[usermapping] name=|" << Entity.name << "| role=|" << (Entity.role?Entity.role:"-") << "| grps=|"<< (Entity.grps?Entity.grps:"-") << "| DN=|" << clientdn.c_str() << "| VOMS=|" << vomsroles.c_str() << "|");

  if (ssl) {
    SSL_free(ssl);ssl = 0;
  }

  ERR_remove_state(0);
  SSLMutex.UnLock();
  return;
}

int 
XrdSecProtocolssl::GenerateSession(const SSL* ssl, unsigned char *id, unsigned int *id_len) {
  EPNAME("GenerateSession");
  unsigned int count = 0;
  do      {
    RAND_pseudo_bytes(id, *id_len);
    /* Prefix the session_id with the required prefix. NB: If our
     * prefix is too long, clip it - but there will be worse effects
     * anyway, eg. the server could only possibly create 1 session
     * ID (ie. the prefix!) so all future session negotiations will
     * fail due to conflicts. */
    memcpy(id, "xrootdssl",
	   (strlen("xrootdssl") < *id_len) ?
	   strlen("xrootdssl") : *id_len);
   TRACE(Authen,"Generated SSID **********************");
  }
  while(SSL_has_matching_session_id(ssl, id, *id_len) &&
	(++count < MAX_SESSION_ID_ATTEMPTS));
  if(count >= MAX_SESSION_ID_ATTEMPTS)
    return 0;
  return 1;
}

int 
XrdSecProtocolssl::NewSession(SSL* ssl, SSL_SESSION *session) {
  EPNAME("NewSession");
  TRACE(Authen,"Creating new Session");
  char session_id[1024];
  for (int i=0; i< (int)session->session_id_length; i++) {
    sprintf(session_id+(i*2),"%02x",session->session_id[i]);
  }
  DEBUG("Info: ("<<__FUNCTION__<<") Session Id: "<< session_id << " Verify: " << session->verify_result << " (" << X509_verify_cert_error_string(session->verify_result) << ")");
  
  SSL_set_timeout(session, sslsessionlifetime);
  return 0;
}

/*----------------------------------------------------------------------------*/
void 
XrdSecProtocolssl::ReloadGridMapFile()
{ 
  EPNAME("ReloadGridMapFile");

  static time_t         GridMapMtime=0;
  static time_t         GridMapCheckTime=0;
  int now = time(NULL);

  if ((!GridMapCheckTime) || ((now >GridMapCheckTime + 60)) ) {
    // load it for the first time or again
    struct stat buf;
    if (!::stat(gridmapfile,&buf)) {
      if (buf.st_mtime != GridMapMtime) {
	GridMapMutex.Lock();
	// store the last modification time
	GridMapMtime = buf.st_mtime;
	// store the current time of the check
	GridMapCheckTime = now;
	// dump the current table
	gridmapstore.Purge();
	// open the gridmap file
	FILE* mapin = fopen(gridmapfile,"r");
	if (!mapin) {
	  // error no grid map possible
	  TRACE(Authen,"Unable to open gridmapfile " << XrdOucString(gridmapfile) << " - no mapping!");
	} else {
	  char userdnin[4096];
	  char usernameout[4096];
	  int nitems;
	  // parse it
	  while ( (nitems = fscanf(mapin,"\"%[^\"]\" %s\n", userdnin,usernameout)) == 2) {
	    XrdOucString dn = userdnin;
	    dn.replace("\"","");
	    // leave only the first CN=, cut the rest
	    int pos = dn.find("CN=");
	    int pos2 = dn.find("/",pos);
	    if (pos2>0) dn.erase(pos2);

	    if (!gridmapstore.Find(dn.c_str())) {
	      gridmapstore.Add(dn.c_str(), new XrdOucString(usernameout));
	      TRACE(Authen, "gridmapfile Mapping Added: " << dn.c_str() << " |=> " << usernameout);
	    }
	  }
	  fclose(mapin);
	}
	GridMapMutex.UnLock();
      } else {
	// the file didn't change, we don't do anything
      }
    } else {
      TRACE(Authen,"Unable to stat gridmapfile " << XrdOucString(gridmapfile) << " - no mapping!");
    }
  }
}

/*----------------------------------------------------------------------------*/

void 
XrdSecProtocolssl::ReloadVomsMapFile()
{
  EPNAME("ReloadVomsMapFile");

  static time_t         VomsMapMtime=0;
  static time_t         VomsMapCheckTime=0;
  int now = time(NULL);

  if ((!VomsMapCheckTime) || ((now >VomsMapCheckTime + 60 )) ) {
    // load it for the first time or again
    struct stat buf;
    if (!::stat(vomsmapfile,&buf)) {
      if (buf.st_mtime != VomsMapMtime) {
	VomsMapMutex.Lock();
	// store the last modification time
	VomsMapMtime = buf.st_mtime;
	// store the current time of the check
	VomsMapCheckTime = now;
	// dump the current table
	vomsmapstore.Purge();
	// open the vomsmap file
	FILE* mapin = fopen(vomsmapfile,"r");
	if (!mapin) {
	  // error no voms map possible
	  TRACE(Authen,"Unable to open vomsmapfile " << XrdOucString(vomsmapfile) << " - no mapping!");
	} else {
	  char userdnin[4096];
	  char usernameout[4096];
	  int nitems;
	  // parse it
	  while ( (nitems = fscanf(mapin,"\"%[^\"]\" %s\n", userdnin,usernameout)) == 2) {
	    XrdOucString dn = userdnin;
	    dn.replace("\"","");
	    if (!vomsmapstore.Find(dn.c_str())) {
	      vomsmapstore.Add(dn.c_str(), new XrdOucString(usernameout));
	      TRACE(Authen,"vomsmapfile Mapping Added: " << dn.c_str() << " |=> " << usernameout);
	    }
	  }
	  fclose(mapin);
	}
	VomsMapMutex.UnLock();
      } else {
	// the file didn't change, we don't do anything
      }
    } else {
      TRACE(Authen,"Unable to stat vomsmapfile " << XrdOucString(vomsmapfile) << " - no mapping!");
    }
  }
}

/*----------------------------------------------------------------------------*/

bool
XrdSecProtocolssl::VomsMapGroups(const char* groups, XrdOucString& allgroups, XrdOucString& defaultgroup) 
{
  EPNAME("VomsMapGroups");
  ReloadVomsMapFile();
  // loop over all VOMS groups and replace them according to the mapping
  XrdOucString vomsline = groups;
  allgroups = ":";
  defaultgroup = "";
  vomsline.replace(":","\n");
  XrdOucTokenizer vomsgroups((char*)vomsline.c_str());
  const char* stoken;
  int ntoken=0;
  XrdOucString* vomsmaprole;                                     
  while( (stoken = vomsgroups.GetLine())) {
    if ((vomsmaprole = XrdSecProtocolssl::vomsmapstore.Find(stoken))) { 
      allgroups += vomsmaprole->c_str();
      allgroups += ":";
      if (ntoken == 0) {
	defaultgroup = vomsmaprole->c_str();
      }
      ntoken++;
    } else {
      // scan for a wildcard rule
      XrdOucString vomsattribute = stoken;
      int rpos=STR_NPOS;
      while ((rpos = vomsattribute.rfind("/",rpos))!=STR_NPOS) {
	rpos--;
	XrdOucString wildcardattribute = vomsattribute;
	wildcardattribute.erase(rpos+2);
	wildcardattribute += "*";
	if ((vomsmaprole = XrdSecProtocolssl::vomsmapstore.Find(wildcardattribute.c_str()))) {
	  allgroups += vomsmaprole->c_str();
	  allgroups += ":";
	  if (ntoken == 0) {
	    defaultgroup = vomsmaprole->c_str();
	  }
	  ntoken++;
	  break; // leave the wildcard loop
	}
	if ( rpos < 0) {
	  break;
	}
      }
    }
  }

  if (allgroups == ":") {
    TRACE(Authen,"No VOMS mapping found for " << XrdOucString(stoken) << " using default group");
    return false;
  }
  return true;
}

/******************************************************************************/
/*                X r d S e c p r o t o c o l u n i x I n i t                 */
/******************************************************************************/
  
extern "C"
{
char  *XrdSecProtocolsslInit(const char     mode,
                              const char    *parms,
                              XrdOucErrInfo *erp)
{
  EPNAME("ProtocolsslInit");
  // Initiate error logging and tracing
  XrdSecProtocolssl::ssleDest.logger(&XrdSecProtocolssl::Logger);
  
  GRSTerrorLogFunc = &MyGRSTerrorLogFunc;
  static bool serverinitialized = false;

  // create the tracer
  if (!SSLxTrace)
    SSLxTrace = new XrdOucTrace(&XrdSecProtocolssl::ssleDest);

  for (int i=0; i< PROTOCOLSSL_MAX_CRYPTO_MUTEX; i++) {
    XrdSecProtocolssl::CryptoMutexPool[i] = new XrdSysMutex();
  }


  // read the configuration options
  if ( (mode == 's') && (!serverinitialized) )  {
    XrdSecProtocolssl::sslcertfile = strdup("/etc/grid-security/hostcert.pem");
    XrdSecProtocolssl::sslkeyfile  = strdup("/etc/grid-security/hostkey.pem");
    XrdSecProtocolssl::sslcadir    = strdup("/etc/grid-security/certificates");
    XrdSecProtocolssl::sslvomsdir  = (char*)"/etc/grid-security/vomsdir";
    
    XrdSecProtocolssl::isServer = 1;
    serverinitialized = true;
    if (parms){
      // Duplicate the parms
      char parmbuff[1024];
      strlcpy(parmbuff, parms, sizeof(parmbuff));
      //
      // The tokenizer
      XrdOucTokenizer inParms(parmbuff);
      char *op;

      while (inParms.GetLine()) { 
	while ((op = inParms.GetToken())) {
	  if (!strncmp(op, "-d:",3)) {
	    XrdSecProtocolssl::debug = atoi(op+3);
	  } else if (!strncmp(op, "-cadir:",7)) {
	    XrdSecProtocolssl::sslcadir = strdup(op+7);
	  } else if (!strncmp(op, "-vomsdir:",6)) {
	    XrdSecProtocolssl::sslvomsdir = strdup(op+6);
	  } else if (!strncmp(op, "-cert:",6)) {
	    XrdSecProtocolssl::sslcertfile = strdup(op+6);
	  } else if (!strncmp(op, "-key:",5)) {
	    XrdSecProtocolssl::sslkeyfile = strdup(op+5);
	  } else if (!strncmp(op, "-ca:",4)) {
	    XrdSecProtocolssl::verifydepth = atoi(op+4);
	  } else if (!strncmp(op, "-t:",3)) {
	    XrdSecProtocolssl::sslsessionlifetime = atoi(op+3);
	  } else if (!strncmp(op, "-export:",8)) {
	    XrdSecProtocolssl::sslproxyexportdir = strdup(op+8);
	  } else if (!strncmp(op, "-gridmapfile:",13)) {
	    XrdSecProtocolssl::gridmapfile = strdup(op+13);
	  } else if (!strncmp(op, "-vomsmapfile:",13)) {
	    XrdSecProtocolssl::vomsmapfile = strdup(op+13);
	  } else if (!strncmp(op, "-mapuser:",9)) {
	    XrdSecProtocolssl::mapuser = (bool) atoi(op+9);
	  } else if (!strncmp(op, "-mapnobody:",11)) {
	    XrdSecProtocolssl::mapnobody = (bool) atoi(op+11);
	  } else if (!strncmp(op, "-mapgroup:",10)) {
	    XrdSecProtocolssl::mapgroup = (bool) atoi(op+10);
	  } else if (!strncmp(op, "-mapcernuser:",13)) {
	    XrdSecProtocolssl::mapcerncertificates = (bool) atoi(op+13);
	  } else if (!strncmp(op, "-sessioncachesize:", 18)) {
	    XrdSecProtocolssl::sslsessioncachesize = atoi(op+18);
	  } else if (!strncmp(op, "-selecttimeout:", 15)) {
	    XrdSecProtocolssl::sslselecttimeout = atoi(op + 15);
	    if ( XrdSecProtocolssl::sslselecttimeout < 5) {
	      XrdSecProtocolssl::sslselecttimeout = 5;
	    }
	  } else if (!strncmp(op, "-procdir:",9)) {
	    XrdSecProtocolssl::procdir = strdup(op+9);
	    XrdSecProtocolssl::proc = new XrdSecProtocolsslProc(XrdSecProtocolssl::procdir, false);
	    if (XrdSecProtocolssl::proc) {
	      if (!XrdSecProtocolssl::proc->Open()) {
		delete XrdSecProtocolssl::proc;
		XrdSecProtocolssl::proc = NULL;
	      }
	    }
	    time_t now = time(NULL);

	    if (XrdSecProtocolssl::proc) {
	      XrdSecProtocolsslProcFile* pf;
	      XrdOucString ID = XrdSecProtocolsslCVSID;
	      ID+="\n";
	      pf= XrdSecProtocolssl::proc->Handle("version"); pf && pf->Write(ID.c_str());
	      pf= XrdSecProtocolssl::proc->Handle("start"); pf && pf->Write(ctime(&now));
	    }
	  }
	}
      }
    }
  } else {
    if ( (mode == 'c') || (serverinitialized)) {
      if (mode == 'c') {
	for (int i=0; i< PROTOCOLSSL_MAX_CRYPTO_MUTEX; i++) {
	  XrdSecProtocolssl::CryptoMutexPool[i] = 0;
	}
	XrdSecProtocolssl::sslcertfile = strdup("/etc/grid-security/hostcert.pem");
	XrdSecProtocolssl::sslkeyfile  = strdup("/etc/grid-security/hostkey.pem");
	XrdSecProtocolssl::sslcadir    = strdup("/etc/grid-security/certificates");
	XrdSecProtocolssl::sslvomsdir  = (char*)"/etc/grid-security/vomsdir";
      }
      XrdSecProtocolssl::GetEnvironment();
      XrdSecProtocolssl::isServer = 0;
      if (serverinitialized) {
	XrdSecProtocolssl::sslproxyexportplain = 0;
      }
    }
  }

  if (XrdSecProtocolssl::debug >= 4) {
    SSLxTrace->What = TRACE_ALL | TRACE_Debug;
  } else if (XrdSecProtocolssl::debug == 3 ) {
    SSLxTrace->What |= TRACE_Authen;
    SSLxTrace->What |= TRACE_Debug;
    SSLxTrace->What |= TRACE_Identity;
  } else if (XrdSecProtocolssl::debug == 2) {
    SSLxTrace->What = TRACE_Debug;
  } else if (XrdSecProtocolssl::debug == 1) {
    SSLxTrace->What = TRACE_Identity;
  } else SSLxTrace->What = 0;

  // thread-saftey
  if (PROTOCOLSSL_MAX_CRYPTO_MUTEX < CRYPTO_num_locks() ) {
    fprintf(stderr,"Error: (%s) I don't have enough crypto mutexes as required by crypto_ssl [recompile increasing PROTOCOLSSL_MAX_CRYPTO_MUTEX to %d] \n",__FUNCTION__,CRYPTO_num_locks());
    TRACE(Authen,"Error: I don't have enough crypto mutexes as required by crypto_ssl [recompile increasing PROTOCOLSSL_MAX_CRYPTO_MUTEX to " << (int)CRYPTO_num_locks() << "]");
  } else {
    TRACE(Authen,"====> SSL requires " << (int)CRYPTO_num_locks() << " mutexes for thread-safety");
  }

#if defined(OPENSSL_THREADS)
  // thread support enabled
  TRACE(Authen,"====> SSL with thread support!");
#else
  fprintf(stderr,"Error: (%s) SSL lacks thread support: Abort!");
  TRACE(Authen,"Error: SSL lacks thread support: Abort!");
#endif

  // set callback functions
  CRYPTO_set_locking_callback(protocolssl_lock);
  CRYPTO_set_id_callback(protocolssl_id_callback);




  if (XrdSecProtocolssl::isServer) {
    TRACE(Authen,"====> debug            = " << XrdSecProtocolssl::debug);
    TRACE(Authen,"====> cadir            = " << XrdSecProtocolssl::sslcadir);
    TRACE(Authen,"====> keyfile          = " << XrdSecProtocolssl::sslkeyfile);
    TRACE(Authen,"====> certfile         = " << XrdSecProtocolssl::sslcertfile);
    TRACE(Authen,"====> verify depth     = " << XrdSecProtocolssl::verifydepth);
    TRACE(Authen,"====> sess.lifetime    = " << XrdSecProtocolssl::sslsessionlifetime);
    TRACE(Authen,"====> gridmapfile      = " << XrdSecProtocolssl::gridmapfile);
    TRACE(Authen,"====> vomsmapfile      = " << XrdSecProtocolssl::vomsmapfile);
    TRACE(Authen,"====> mapuser          = " << XrdSecProtocolssl::mapuser);
    TRACE(Authen,"====> mapnobody        = " << XrdSecProtocolssl::mapnobody);
    TRACE(Authen,"====> mapgroup         = " << XrdSecProtocolssl::mapgroup);
    TRACE(Authen,"====> mapcernuser      = " << XrdSecProtocolssl::mapcerncertificates);
    TRACE(Authen,"====> selecttimeout    = " << XrdSecProtocolssl::sslselecttimeout);
    TRACE(Authen,"====> sessioncachesize = " << XrdSecProtocolssl::sslsessioncachesize);
    TRACE(Authen,"====> procdir       = " << XrdSecProtocolssl::procdir);
    char Info[16384];
    sprintf(Info,"debug         = %d\ncadir         = %s\nkeyfile       = %s\ncertfile      = %s\nverify depth  = %d\nsess.lifetime = %ld\ngridmapfile   = %s\nvomsmapfile   = %s\nmapuser       = %d\nmapnobody     = %d\nmapgroup      = %d\nmapcernuser   = %d\nselecttimeout = %d\nprocdir       = %s\nsessioncachesz = %d\n",XrdSecProtocolssl::debug,XrdSecProtocolssl::sslcadir, XrdSecProtocolssl::sslkeyfile, XrdSecProtocolssl::sslcertfile, XrdSecProtocolssl::verifydepth, XrdSecProtocolssl::sslsessionlifetime, XrdSecProtocolssl::gridmapfile, XrdSecProtocolssl::vomsmapfile, XrdSecProtocolssl::mapuser,  XrdSecProtocolssl::mapnobody, XrdSecProtocolssl::mapgroup, XrdSecProtocolssl::mapcerncertificates, XrdSecProtocolssl::sslselecttimeout,  XrdSecProtocolssl::procdir, XrdSecProtocolssl::sslsessioncachesize);
    if (XrdSecProtocolssl::proc) {
      XrdSecProtocolsslProcFile* pf;
      pf= XrdSecProtocolssl::proc->Handle("info"); pf && pf->Write(Info);
    }
  } else {
    if (XrdSecProtocolssl::debug) {
      TRACE(Authen,"====> debug         = " << XrdSecProtocolssl::debug);
      TRACE(Authen,"====> cadir         = " << XrdSecProtocolssl::sslcadir);
      TRACE(Authen,"====> keyfile       = " << XrdSecProtocolssl::sslkeyfile);
      TRACE(Authen,"====> certfile      = " << XrdSecProtocolssl::sslcertfile);
      TRACE(Authen,"====> verify depth  = " << XrdSecProtocolssl::verifydepth);
    }
  }

  if (XrdSecProtocolssl::isServer) {
    XrdSecProtocolssl::sslproxyexportplain=0; // for security reasons a server should not export plain private keys
    // check if we can map with a grid map file
    if (XrdSecProtocolssl::mapuser && access(XrdSecProtocolssl::gridmapfile,R_OK)) {
      fprintf(stderr,"Error: (%s) cannot access gridmapfile %s\n",__FUNCTION__,XrdSecProtocolssl::gridmapfile);
      TRACE(Authen,"Error: cannot access gridmapfile "<< XrdOucString(XrdSecProtocolssl::gridmapfile));
      return 0;
    }
    // check if we can map with a voms map file
    if (XrdSecProtocolssl::mapgroup && access(XrdSecProtocolssl::vomsmapfile,R_OK)) {
      fprintf(stderr,"Error: (%s) cannot access vomsmapfile %s\n",__FUNCTION__,XrdSecProtocolssl::vomsmapfile);
      TRACE(Authen,"Error: cannot access vomsmapfile "<< XrdOucString(XrdSecProtocolssl::vomsmapfile));
      return 0;
    }
    // check if we can export proxies
    XrdOucString exportplain=XrdSecProtocolssl::sslproxyexportdir;
    // if the export file starts with plain: we don't write the proxy with a passphrase
    if (exportplain.beginswith("plain:")) {
      XrdSecProtocolssl::sslproxyexportdir+=6;
      XrdSecProtocolssl::sslproxyexportplain=true;
      TRACE(Authen,"====> export plain proxy (warning: can be re-used out of daemon context) to dir: " << XrdSecProtocolssl::sslproxyexportdir);
    }
    if (XrdSecProtocolssl::sslproxyexportdir && access(XrdSecProtocolssl::sslproxyexportdir,R_OK | W_OK)) {
      fprintf(stderr,"Error: (%s) cannot read/write proxy export directory %s\n",__FUNCTION__,XrdSecProtocolssl::sslproxyexportdir);
      TRACE(Authen,"Error: cannot access proxyexportdir "<< XrdOucString(XrdSecProtocolssl::sslproxyexportdir));
      return 0;
    }
  }

  if (XrdSecProtocolssl::isServer) {
    SSL_METHOD *meth;
    // initialize openssl until the context is created 
    SSL_load_error_strings();
    SSLeay_add_ssl_algorithms();
    
    meth = (SSL_METHOD*)SSLv23_server_method();

    XrdSecProtocolssl::ctx = SSL_CTX_new (meth);
    if (!XrdSecProtocolssl::ctx) {
      ERR_print_errors_fp(stderr);
      return 0;
    }
      
    if (SSL_CTX_use_certificate_file(XrdSecProtocolssl::ctx, XrdSecProtocolssl::sslcertfile, SSL_FILETYPE_PEM) <= 0) {
      ERR_print_errors_fp(stderr);
      return 0;
    }
    
    if (SSL_CTX_use_PrivateKey_file(XrdSecProtocolssl::ctx,XrdSecProtocolssl::sslkeyfile, SSL_FILETYPE_PEM) <= 0) {
      ERR_print_errors_fp(stderr);
      return 0;
    }
      
    if (!SSL_CTX_check_private_key(XrdSecProtocolssl::ctx)) {
      fprintf(stderr,"Private key does not match the certificate public key\n");
      return 0;
    }

    XrdSecProtocolssl::sslserverkeyfile=XrdSecProtocolssl::sslkeyfile;
    // use the private server key as password for proxy private key export
    memset(XrdSecProtocolssl::sslserverexportpassword,0,EXPORTKEYSTRENGTH); 

    unsigned int seed = (unsigned int) (time(NULL) + (unsigned int) random());
    srand(seed);
    char rexportkey[16384];
    rexportkey[0]=0;
    for (int i=0; i < EXPORTKEYSTRENGTH; i++) {
      XrdSecProtocolssl::sslserverexportpassword[i] = (unsigned char)(rand()%256);
      if (!XrdSecProtocolssl::sslserverexportpassword[i]) XrdSecProtocolssl::sslserverexportpassword[i]++;
      sprintf(rexportkey,"%s%x",rexportkey,XrdSecProtocolssl::sslserverexportpassword[i]);
    }
    XrdSecProtocolssl::sslserverexportpassword[EXPORTKEYSTRENGTH] = 0;
    sprintf((char*)XrdSecProtocolssl::sslserverexportpassword,"1234567890");
    // debug 
    DEBUG("Created random export key: "<< rexportkey);
    SSL_CTX_load_verify_locations(XrdSecProtocolssl::ctx, NULL,XrdSecProtocolssl::sslcadir);  
    
    // create the store
    if (!XrdSecProtocolssl::store) {
      DEBUG("Created SSL CRL store: " << XrdSecProtocolssl::store);
      XrdSecProtocolssl::store = SSL_X509_STORE_create(NULL,XrdSecProtocolssl::sslcadir);
      X509_STORE_set_flags(XrdSecProtocolssl::store,0);
      XrdSecProtocolssl::storeLoadTime = time(NULL);
    }
    
    XrdSecProtocolssl::ctx->verify_mode = SSL_VERIFY_PEER | SSL_VERIFY_CLIENT_ONCE|SSL_VERIFY_FAIL_IF_NO_PEER_CERT;

    grst_cadir   = XrdSecProtocolssl::sslcadir;
    grst_vomsdir = XrdSecProtocolssl::sslvomsdir;
    grst_depth   = XrdSecProtocolssl::verifydepth;
    
    SSL_CTX_set_cert_verify_callback(XrdSecProtocolssl::ctx,
				     GRST_verify_cert_wrapper,
				     (void *) NULL);
    
    SSL_CTX_set_verify(XrdSecProtocolssl::ctx, XrdSecProtocolssl::ctx->verify_mode,GRST_callback_SSLVerify_wrapper);
    SSL_CTX_set_verify_depth(XrdSecProtocolssl::ctx, XrdSecProtocolssl::verifydepth + 1);

    if(!SSL_CTX_set_generate_session_id(XrdSecProtocolssl::ctx, XrdSecProtocolssl::GenerateSession)) {
      TRACE(Authen,"Cannot set session generator");
      return 0;
    }

    SSL_CTX_set_options(XrdSecProtocolssl::ctx,  SSL_OP_ALL | SSL_OP_NO_SSLv2);
    
    SSL_CTX_sess_set_cache_size(XrdSecProtocolssl::ctx,XrdSecProtocolssl::sslsessioncachesize);

    if (XrdSecProtocolssl::sslsessioncachesize) {
      SSL_CTX_set_session_cache_mode(XrdSecProtocolssl::ctx, SSL_SESS_CACHE_BOTH); // | SSL_SESS_CACHE_NO_AUTO_CLEAR );
    } else {
      SSL_CTX_set_session_cache_mode(XrdSecProtocolssl::ctx, SSL_SESS_CACHE_OFF | SSL_SESS_CACHE_NO_INTERNAL );
      
    }

    SSL_CTX_set_session_id_context(XrdSecProtocolssl::ctx,(const unsigned char*) XrdSecProtocolssl::SessionIdContext,  strlen(XrdSecProtocolssl::SessionIdContext));
    SSL_CTX_sess_set_new_cb(XrdSecProtocolssl::ctx, XrdSecProtocolssl::NewSession);
  }
  return (char *)"";
}
}

/******************************************************************************/
/*              X r d S e c P r o t o c o l u n i x O b j e c t               */
/******************************************************************************/
  
extern "C"
{
XrdSecProtocol *XrdSecProtocolsslObject(const char              mode,
                                         const char             *hostname,
                                         const struct sockaddr  &netaddr,
                                         const char             *parms,
                                               XrdOucErrInfo    *erp)
{
   XrdSecProtocolssl *prot;

// Return a new protocol object
//
   if (!(prot = new XrdSecProtocolssl(hostname, &netaddr)))
      {const char *msg = "Secssl: Insufficient memory for protocol.";
       if (erp) erp->setErrInfo(ENOMEM, msg);
          else cerr <<msg <<endl;
       return (XrdSecProtocol *)0;
      }
   
// All done
//
   return prot;
}
}

