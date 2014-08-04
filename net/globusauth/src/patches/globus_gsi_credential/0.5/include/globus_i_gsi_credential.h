#ifndef GLOBUS_DONT_DOCUMENT_INTERNAL
/**
 * @file globus_i_gsi_credential.h
 * Globus GSI Credential Library
 * @author Sam Lang, Sam Meder
 *
 * $RCSfile: globus_i_gsi_credential.h,v $
 * $Revision: 1.7 $
 * $Date: 2002/08/28 03:03:59 $
 */

#ifndef GLOBUS_I_INCLUDE_GSI_CREDENTIAL_H
#define GLOBUS_I_INCLUDE_GSI_CREDENTIAL_H

#include "globus_gsi_credential.h"
#include "proxycertinfo.h"

/* DEBUG MACROS */

#ifdef BUILD_DEBUG

extern int                              globus_i_gsi_cred_debug_level;
extern FILE *                           globus_i_gsi_cred_debug_fstream;

#define GLOBUS_I_GSI_CRED_DEBUG(_LEVEL_) \
    (globus_i_gsi_cred_debug_level >= (_LEVEL_))

#define GLOBUS_I_GSI_CRED_DEBUG_FPRINTF(_LEVEL_, _MESSAGE_) \
    { \
        if (GLOBUS_I_GSI_CRED_DEBUG(_LEVEL_)) \
        { \
           globus_libc_fprintf _MESSAGE_; \
        } \
    }


#define GLOBUS_I_GSI_CRED_DEBUG_FNPRINTF(_LEVEL_, _MESSAGE_) \
    { \
        if (GLOBUS_I_GSI_CRED_DEBUG(_LEVEL_)) \
        { \
           char *                          _tmp_str_ = \
               globus_gsi_cert_utils_create_nstring _MESSAGE_; \
           globus_libc_fprintf(globus_i_gsi_cred_debug_fstream, \
                               _tmp_str_); \
           globus_libc_free(_tmp_str_); \
        } \
    }

#define GLOBUS_I_GSI_CRED_DEBUG_PRINT(_LEVEL_, _MESSAGE_) \
    { \
        if (GLOBUS_I_GSI_CRED_DEBUG(_LEVEL_)) \
        { \
           globus_libc_fprintf(globus_i_gsi_cred_debug_fstream, _MESSAGE_); \
        } \
    }

#define GLOBUS_I_GSI_CRED_DEBUG_PRINT_OBJECT(_LEVEL_, _OBJ_NAME_, _OBJ_) \
    { \
        if (GLOBUS_I_GSI_CRED_DEBUG(_LEVEL_)) \
        { \
           _OBJ_NAME_##_print_fp(globus_i_gsi_cred_debug_fstream, _OBJ_); \
        } \
    }

#else

#define GLOBUS_I_GSI_CRED_DEBUG_FPRINTF(_LEVEL_, _MESSAGE_) {}
#define GLOBUS_I_GSI_CRED_DEBUG_FNPRINTF(_LEVEL_, _MESSAGE_) {}
#define GLOBUS_I_GSI_CRED_DEBUG_PRINT(_LEVEL_, _MESSAGE_) {}
#define GLOBUS_I_GSI_CRED_DEBUG_PRINT_OBJECT(_LEVEL_, _OBJ_NAME_, _OBJ_) {}

#endif

#define GLOBUS_I_GSI_CRED_DEBUG_ENTER \
            GLOBUS_I_GSI_CRED_DEBUG_FPRINTF( \
                2, (globus_i_gsi_cred_debug_fstream, \
                    "%s entering\n", _function_name_))

#define GLOBUS_I_GSI_CRED_DEBUG_EXIT \
            GLOBUS_I_GSI_CRED_DEBUG_FPRINTF( \
                2, (globus_i_gsi_cred_debug_fstream, \
                    "%s exiting\n", _function_name_))

/* ERROR MACROS */

#define GLOBUS_GSI_CRED_OPENSSL_ERROR_RESULT(_RESULT_, _ERRORTYPE_, _ERRSTR_) \
    { \
        char *                          _tmp_str_ = \
            globus_gsi_cert_utils_create_string _ERRSTR_; \
        _RESULT_ = globus_i_gsi_cred_openssl_error_result(_ERRORTYPE_, \
                                                          __FILE__, \
                                                          _function_name_, \
                                                          __LINE__, \
                                                          _tmp_str_, \
                                                          NULL); \
        globus_libc_free(_tmp_str_); \
    }

#define GLOBUS_GSI_CRED_ERROR_RESULT(_RESULT_, _ERRORTYPE_, _ERRSTR_) \
    { \
        char *                          _tmp_str_ = \
            globus_gsi_cert_utils_create_string _ERRSTR_; \
        _RESULT_ = globus_i_gsi_cred_error_result(_ERRORTYPE_, \
                                                  __FILE__, \
                                                  _function_name_, \
                                                  __LINE__, \
                                                  _tmp_str_, \
                                                  NULL); \
        globus_libc_free(_tmp_str_); \
    }

#define GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(_TOP_RESULT_, _ERRORTYPE_) \
    _TOP_RESULT_ = globus_i_gsi_cred_error_chain_result(_TOP_RESULT_, \
                                                        _ERRORTYPE_, \
                                                        __FILE__, \
                                                        _function_name_, \
                                                        __LINE__, \
                                                        NULL, \
                                                        NULL)

#define GLOBUS_GSI_CRED_OPENSSL_LONG_ERROR_RESULT(_RESULT_, \
                                                  _ERRORTYPE_, \
                                                  _ERRSTR_, \
                                                  _LONG_DESC_) \
    { \
        char *                          _tmp_str_ = \
            globus_gsi_cert_utils_create_string _ERRSTR_; \
        _RESULT_ = globus_i_gsi_cred_openssl_error_result(_ERRORTYPE_, \
                                                          __FILE__, \
                                                          _function_name_, \
                                                          __LINE__, \
                                                          _tmp_str_, \
                                                          _LONG_DESC_); \
        globus_libc_free(_tmp_str_); \
    }

#define GLOBUS_GSI_CRED_LONG_ERROR_RESULT(_RESULT_, \
                                          _ERRORTYPE_, \
                                          _ERRSTR_) \
    { \
        char *                          _tmp_str_ = \
            globus_gsi_cert_utils_create_string _ERRSTR_; \
        _RESULT_ = globus_i_gsi_cred_error_result(_ERRORTYPE_, \
                                                  __FILE__, \
                                                  _function_name_, \
                                                  __LINE__, \
                                                  _tmp_str_, \
                                                  _LONG_DESC_); \
        globus_libc_free(_tmp_str_); \
    }

#define GLOBUS_GSI_CRED_LONG_ERROR_CHAIN_RESULT(_TOP_RESULT_, \
                                                _ERRORTYPE_, \
                                                _LONG_DESC_) \
    _TOP_RESULT_ = globus_i_gsi_cred_error_chain_result(_TOP_RESULT_, \
                                                        _ERRORTYPE_, \
                                                        __FILE__, \
                                                        _function_name_, \
                                                        __LINE__, \
                                                        NULL, \
                                                        _LONG_DESC_)

extern char *                    globus_l_gsi_cred_error_strings[];

/**
 * Handle attributes.
 * @ingroup globus_gsi_credential_handle_attrs
 */

/**
 * GSI Credential handle attributes implementation
 * @ingroup globus_gsi_credential_handle
 * @internal
 *
 * This structure contains immutable attributes
 * of a credential handle
 */
typedef struct globus_l_gsi_cred_handle_attrs_s
{
    /* the filename of the CA certificate directory */
    char *                              ca_cert_dir;
    /* the order to search in for a certificate */
    globus_gsi_cred_type_t *            search_order; /*{PROXY,USER,HOST}*/
} globus_i_gsi_cred_handle_attrs_t;

/**
 * GSI Credential handle implementation
 * @ingroup globus_gsi_credential_handle
 * @internal
 *
 * Contains all the state associated with a credential handle, including
 *
 * @see globus_credential_handle_init(), globus_credential_handle_destroy()
 */
typedef struct globus_l_gsi_cred_handle_s
{
    /** The credential's signed certificate */
    X509 *                              cert;
    /** The private key of the credential */
    EVP_PKEY *                          key;
    /** The chain of signing certificates */
    STACK_OF(X509) *                    cert_chain;
    /** The immutable attributes of the credential handle */
    globus_gsi_cred_handle_attrs_t      attrs;
    /** The amout of time the credential is valid for */
    time_t                              goodtill;
} globus_i_gsi_cred_handle_t;


globus_result_t
globus_i_gsi_cred_goodtill(
    globus_gsi_cred_handle_t            cred_handle,
    time_t *                            goodtill);

globus_result_t globus_i_gsi_cred_get_proxycertinfo(
    X509 *                              cert,
    PROXYCERTINFO **                    proxycertinfo);

int
globus_i_gsi_cred_password_callback_no_prompt(
    char *                              buffer,
    int                                 size,
    int                                 w);

globus_result_t
globus_i_gsi_cred_openssl_error_result(
    int                                 error_type,
    const char *                        filename,
    const char *                        function_name,
    int                                 line_number,
    const char *                        short_desc,
    const char *                        long_desc);

globus_result_t
globus_i_gsi_cred_error_result(
    int                                 error_type,
    const char *                        filename,
    const char *                        function_name,
    int                                 line_number,
    const char *                        short_desc,
    const char *                        long_desc);

globus_result_t
globus_i_gsi_cred_error_chain_result(
    globus_result_t                     chain_result,
    int                                 error_type,
    const char *                        filename,
    const char *                        function_name,
    int                                 line_number,
    const char *                        short_desc,
    const char *                        long_desc);

globus_result_t
globus_i_gsi_cred_error_join_chains_result(
    globus_result_t                     outter_error,
    globus_result_t                     inner_error);

EXTERN_C_END

#endif /* GLOBUS_I_INCLUDE_GSI_CREDENTIAL_H */

#endif /* GLOBUS_DONT_DOCUMENT_INTERNAL */
