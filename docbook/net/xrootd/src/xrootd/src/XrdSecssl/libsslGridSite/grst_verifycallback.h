#ifndef GRST_VERIFYCALLBACK__H
#define GRST_VERIFYCALLBACK__H

#include <openssl/ssl.h>

extern X509_STORE* grst_store;

#define GRST_VERIFY_OPTIONAL_NO_CA 1

extern int grst_verify; // relax the verify setting if you want to GRST_VERIFY_OPTIONAL_NO_CA
extern int grst_depth;

extern char* grst_vomsdir;
extern char* grst_cadir;

#ifdef __cplusplus
extern "C"
{
#endif
  int GRST_X509_set_log_fd(FILE* fd);
  int GRST_verify_cert_wrapper(X509_STORE_CTX *ctx, void *p);
  int GRST_X509_check_issued_wrapper(X509_STORE_CTX *ctx, X509 *x, X509 *issuer);
  int GRST_callback_SSLVerify_wrapper(int ok, X509_STORE_CTX *ctx);
  void GRST_print_ssl_creds(void *grst_chain);
  char* GRST_get_voms_roles_and_free(void *grst_chain);
  void GRST_free_chain(void *grst_chain);

  X509_STORE *SSL_X509_STORE_create(char *cpFile, char *cpPath);
  int SSL_X509_STORE_lookup(X509_STORE *pStore, int nType,
			    X509_NAME *pName, X509_OBJECT *pObj);

#ifdef __cplusplus
}
#endif

#endif
