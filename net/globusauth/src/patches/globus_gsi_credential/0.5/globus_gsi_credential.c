#ifndef GLOBUS_DONT_DOCUMENT_INTERNAL
/**
 * @file globus_gsi_credential.c
 * @author Sam Lang, Sam Meder
 *
 * $RCSfile: globus_gsi_credential.c,v $
 * $Revision: 1.23 $
 * $Date: 2002/11/15 01:27:04 $
 */
#endif

#include "globus_i_gsi_credential.h"
#include "globus_gsi_system_config.h"
#include "globus_gsi_cert_utils.h"
#include "version.h"
#include <openssl/pem.h>
#include <openssl/x509.h>
#include <openssl/pkcs12.h>
#include <openssl/err.h>

#ifndef GLOBUS_DONT_DOCUMENT_INTERNAL

static int globus_l_gsi_credential_activate(void);
static int globus_l_gsi_credential_deactivate(void);

int                                     globus_i_gsi_cred_debug_level = 0;
FILE *                                  globus_i_gsi_cred_debug_fstream = NULL;

/**
 * Module descriptor static initializer.
 */
globus_module_descriptor_t globus_i_gsi_credential_module =
{
    "globus_credential",
    globus_l_gsi_credential_activate,
    globus_l_gsi_credential_deactivate,
    GLOBUS_NULL,
    GLOBUS_NULL,
    &local_version
};

/**
 * Module activation
 */
static
int
globus_l_gsi_credential_activate(void)
{
    int                                 result = (int) GLOBUS_SUCCESS;
    char *                              tmp_string;
    static char *                       _function_name_ =
        "globus_l_gsi_credential_activate";

    tmp_string = globus_module_getenv("GLOBUS_GSI_CRED_DEBUG_LEVEL");
    if(tmp_string != GLOBUS_NULL)
    {
        globus_i_gsi_cred_debug_level = atoi(tmp_string);

        if(globus_i_gsi_cred_debug_level < 0)
        {
            globus_i_gsi_cred_debug_level = 0;
        }
    }

    tmp_string = globus_module_getenv("GLOBUS_GSI_CRED_DEBUG_FILE");
    if(tmp_string != GLOBUS_NULL)
    {
        globus_i_gsi_cred_debug_fstream = fopen(tmp_string, "a");
        if(globus_i_gsi_cred_debug_fstream == NULL)
        {
            result = (int) GLOBUS_FAILURE;
            goto exit;
        }
    }
    else
    {
        /* if the env. var. isn't set, use stderr */
        globus_i_gsi_cred_debug_fstream = stderr;
    }

    GLOBUS_I_GSI_CRED_DEBUG_ENTER;

    result = globus_module_activate(GLOBUS_COMMON_MODULE);

    if(result != GLOBUS_SUCCESS)
    {
        goto exit;
    }

    result = globus_module_activate(GLOBUS_GSI_SYSCONFIG_MODULE);

    if(result != GLOBUS_SUCCESS)
    {
        goto exit;
    }

    result = globus_module_activate(GLOBUS_GSI_CALLBACK_MODULE);

    if(result != GLOBUS_SUCCESS)
    {
        goto exit;
    }

    OpenSSL_add_all_algorithms();

    GLOBUS_I_GSI_CRED_DEBUG_EXIT;

 exit:

    return result;
}

/**
 * Module deactivation
 *
 */
static
int
globus_l_gsi_credential_deactivate(void)
{
    int                                 result = (int) GLOBUS_SUCCESS;
    static char *                       _function_name_ =
        "globus_l_gsi_credential_deactivate";

    GLOBUS_I_GSI_CRED_DEBUG_ENTER;

    EVP_cleanup();

    globus_module_deactivate(GLOBUS_GSI_CALLBACK_MODULE);

    globus_module_deactivate(GLOBUS_GSI_SYSCONFIG_MODULE);

    globus_module_deactivate(GLOBUS_COMMON_MODULE);

    GLOBUS_I_GSI_CRED_DEBUG_EXIT;

    if(globus_i_gsi_cred_debug_fstream != stderr)
    {
        fclose(globus_i_gsi_cred_debug_fstream);
    }
    return result;
}
/* globus_l_gsi_credential_deactivate() */

static globus_result_t
globus_l_gsi_cred_get_service(
    X509_NAME *                         subject,
    char **                             service);

static globus_result_t
globus_l_gsi_cred_subject_cmp(
    X509_NAME *                   actual_subject,
    X509_NAME *                   desired_subject);

#endif

/**
 * Read Credential
 * @ingroup globus_gsi_cred_operation
 */
/* @{ */
/**
 * Read a Credential from a filesystem location.  The credential
 * to read will be determined by the search order of the handle
 * attributes.
 * NOTE:  This function always searches for the desired credential.
 *        If you don't want to perform a search, then don't use this
 *        function.  The search goes in the order of the handle
 *        attributes' search order.
 *
 * @param handle
 *        The credential handle to set.  This credential handle
 *        should already be initialized using globus_gsi_cred_handle_init.
 * @param desired_subject
 *        The subject to check for when reading in a credential.  The
 *        desired_subject should be either a exact match of the read cert's
 *        subject or should just contain the /CN entry. If null, the
 *        credential read in is the first match based on the system
 *        configuration (paths and environment variables)
 * @return
 *        GLOBUS_SUCCESS if no errors occured, otherwise, an error object
 *        identifier is returned.
 *
 * @see globus_gsi_cred_read_proxy
 * @see globus_gsi_cred_read_cert_and_key
 */
globus_result_t globus_gsi_cred_read(
    globus_gsi_cred_handle_t            handle,
    X509_NAME *                         desired_subject)
{
    time_t                              lifetime = 0;
    int                                 index = 0;
    int                                 result_index = 0;
    int                                 result_count = 0;
    globus_result_t                     result = GLOBUS_SUCCESS;
    globus_result_t                     results[4];
    X509_NAME *                         found_subject = NULL;
    char *                              cert = NULL;
    char *                              key = NULL;
    char *                              proxy = NULL;
    char *                              service_name = NULL;

    static char *                       _function_name_ =
        "globus_gsi_cred_read";

    GLOBUS_I_GSI_CRED_DEBUG_ENTER;

    for(result_index = 0; result_index < 4; ++result_index)
    {
        results[result_index] = GLOBUS_SUCCESS;
    }
    result_index = 0;

    if(handle == NULL)
    {
        GLOBUS_GSI_CRED_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_READING_CRED,
            ("Null handle passed to function: %s", _function_name_));
        goto exit;
    }

    /* search for the credential of choice */

    do
    {
        switch(handle->attrs->search_order[index])
        {
        case GLOBUS_PROXY:

            results[result_index] = GLOBUS_GSI_SYSCONFIG_GET_PROXY_FILENAME(
                &proxy,
                GLOBUS_PROXY_FILE_INPUT);
            if(results[result_index] != GLOBUS_SUCCESS)
            {
                proxy = NULL;
                GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
                    results[result_index],
                    GLOBUS_GSI_CRED_ERROR_READING_PROXY_CRED);
                break;
            }

            results[result_index] = globus_gsi_cred_read_proxy(handle, proxy);
            if(results[result_index] != GLOBUS_SUCCESS)
            {
                GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
                    results[result_index],
                    GLOBUS_GSI_CRED_ERROR_READING_PROXY_CRED);
                goto exit;
            }

            if(desired_subject != NULL)
            {
                results[result_index] = globus_gsi_cred_get_X509_subject_name(
                    handle,
                    &found_subject);
                if(results[result_index] != GLOBUS_SUCCESS)
                {
                    GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
                        results[result_index],
                        GLOBUS_GSI_CRED_ERROR_READING_PROXY_CRED);
                    goto exit;
                }

                results[result_index] = globus_l_gsi_cred_subject_cmp(found_subject,
                                                                      desired_subject);

                X509_NAME_free(found_subject);
                found_subject = NULL;

                if(results[result_index] != GLOBUS_SUCCESS)
                {
                    GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
                        results[result_index],
                        GLOBUS_GSI_CRED_ERROR_READING_PROXY_CRED);
                    goto exit;
                }
            }

            results[result_index] = globus_gsi_cred_get_lifetime(
                handle,
                &lifetime);
            if(results[result_index] != GLOBUS_SUCCESS)
            {
                GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
                    results[result_index],
                    GLOBUS_GSI_CRED_ERROR_WITH_CRED);
                goto exit;
            }

            if(lifetime <= 0)
            {
                char *                          subject = NULL;

                subject = X509_NAME_oneline(
                    X509_get_subject_name(handle->cert),
                    NULL, 0);

                GLOBUS_GSI_CRED_ERROR_RESULT(
                    results[result_index],
                    GLOBUS_GSI_CRED_ERROR_WITH_CRED,
                    ("The proxy credential: %s\n      with subject: %s\n"
                     "      expired %d minutes ago.\n",
                     proxy,
                     subject,
                     (-lifetime)/60));

                free(subject);
                goto exit;
            }

            goto exit;

        case GLOBUS_USER:

            results[result_index] =
                GLOBUS_GSI_SYSCONFIG_GET_USER_CERT_FILENAME(&cert, &key);
            if(results[result_index] != GLOBUS_SUCCESS)
            {
                cert = NULL;
                key = NULL;
                GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
                    results[result_index],
                    GLOBUS_GSI_CRED_ERROR_READING_CRED);
                break;
            }

            results[result_index] = globus_gsi_cred_read_cert(handle, cert);
            if(results[result_index] != GLOBUS_SUCCESS)
            {
                GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
                    results[result_index],
                    GLOBUS_GSI_CRED_ERROR_READING_CRED);
                goto exit;
            }

            results[result_index] = globus_gsi_cred_read_key(
                handle,
                key,
                globus_i_gsi_cred_password_callback_no_prompt);
            if(results[result_index] != GLOBUS_SUCCESS)
            {
                globus_object_t *       error_obj;
                error_obj = globus_error_get(results[result_index]);
                if(globus_error_get_type(error_obj) ==
                   GLOBUS_GSI_CRED_ERROR_KEY_IS_PASS_PROTECTED)
                {
                    results[result_index] = globus_error_put(error_obj);
                    GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
                        results[result_index],
                        GLOBUS_GSI_CRED_ERROR_READING_CRED);
                    break;
                }

                results[result_index] = globus_error_put(error_obj);

                GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
                    results[result_index],
                    GLOBUS_GSI_CRED_ERROR_READING_CRED);
                goto exit;
            }

            results[result_index] = globus_i_gsi_cred_goodtill(
                handle,
                &(handle->goodtill));
            if(results[result_index] != GLOBUS_SUCCESS)
            {
                GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
                    results[result_index],
                    GLOBUS_GSI_CRED_ERROR_READING_CRED);
                goto exit;
            }

            if(desired_subject != NULL)
            {
                results[result_index] = globus_gsi_cred_get_X509_subject_name(
                    handle,
                    &found_subject);
                if(results[result_index] != GLOBUS_SUCCESS)
                {
                    GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
                        results[result_index],
                        GLOBUS_GSI_CRED_ERROR_READING_CRED);
                    goto exit;
                }

                results[result_index] = globus_l_gsi_cred_subject_cmp(
                    found_subject,
                    desired_subject);

                X509_NAME_free(found_subject);
                found_subject = NULL;

                if(results[result_index] != GLOBUS_SUCCESS)
                {
                    GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
                        results[result_index],
                        GLOBUS_GSI_CRED_ERROR_READING_CRED);
                    goto exit;
                }
            }

            results[result_index] = globus_gsi_cred_get_lifetime(
                handle,
                &lifetime);
            if(results[result_index] != GLOBUS_SUCCESS)
            {
                GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
                    results[result_index],
                    GLOBUS_GSI_CRED_ERROR_WITH_CRED);
                goto exit;
            }

            if(lifetime <= 0)
            {
                char *                          subject = NULL;

                subject = X509_NAME_oneline(
                    X509_get_subject_name(handle->cert),
                    NULL, 0);

                GLOBUS_GSI_CRED_ERROR_RESULT(
                    results[result_index],
                    GLOBUS_GSI_CRED_ERROR_WITH_CRED,
                    ("The user credential: %s\n      with subject: %s\n"
                     "     has expired %d minutes ago.\n",
                     cert,
                     subject,
                     (-lifetime)/60));

                free(subject);
                goto exit;
            }

            goto exit;

        case GLOBUS_HOST:

            results[result_index] =
                GLOBUS_GSI_SYSCONFIG_GET_HOST_CERT_FILENAME(&cert, &key);
            if(results[result_index] != GLOBUS_SUCCESS)
            {
                cert = NULL;
                key = NULL;
                GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
                    results[result_index],
                    GLOBUS_GSI_CRED_ERROR_READING_HOST_CRED);
                break;
            }

            results[result_index] = globus_gsi_cred_read_cert(handle, cert);
            if(results[result_index] != GLOBUS_SUCCESS)
            {
                GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
                    results[result_index],
                    GLOBUS_GSI_CRED_ERROR_READING_HOST_CRED);
                goto exit;
            }

            results[result_index] = globus_gsi_cred_read_key(
                handle,
                key,
                globus_i_gsi_cred_password_callback_no_prompt);
            if(results[result_index] != GLOBUS_SUCCESS)
            {
                globus_object_t *       error_obj;
                error_obj = globus_error_get(results[result_index]);
                if(globus_error_get_type(error_obj) ==
                   GLOBUS_GSI_CRED_ERROR_KEY_IS_PASS_PROTECTED)
                {
                    results[result_index] = globus_error_put(error_obj);
                    GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
                        results[result_index],
                        GLOBUS_GSI_CRED_ERROR_READING_HOST_CRED);
                    break;
                }

                results[result_index] = globus_error_put(error_obj);

                GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
                    results[result_index],
                    GLOBUS_GSI_CRED_ERROR_READING_HOST_CRED);
                goto exit;
            }

            results[result_index] = globus_i_gsi_cred_goodtill(
                handle,
                &(handle->goodtill));
            if(results[result_index] != GLOBUS_SUCCESS)
            {
                GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
                    results[result_index],
                    GLOBUS_GSI_CRED_ERROR_READING_HOST_CRED);
                goto exit;
            }

            if(desired_subject != NULL)
            {
                results[result_index] = globus_gsi_cred_get_X509_subject_name(
                    handle,
                    &found_subject);

                if(results[result_index] != GLOBUS_SUCCESS)
                {
                    GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
                        results[result_index],
                        GLOBUS_GSI_CRED_ERROR_READING_HOST_CRED);
                    goto exit;
                }

                results[result_index] = globus_l_gsi_cred_subject_cmp(found_subject,
                                                                      desired_subject);

                X509_NAME_free(found_subject);
                found_subject = NULL;

                if(results[result_index] != GLOBUS_SUCCESS)
                {
                    GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
                        results[result_index],
                        GLOBUS_GSI_CRED_ERROR_READING_HOST_CRED);
                    goto exit;
                }
            }

            results[result_index] = globus_gsi_cred_get_lifetime(
                handle,
                &lifetime);
            if(results[result_index] != GLOBUS_SUCCESS)
            {
                GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
                    results[result_index],
                    GLOBUS_GSI_CRED_ERROR_WITH_CRED);
                goto exit;
            }

            if(lifetime <= 0)
            {
                char *                          subject = NULL;

                subject = X509_NAME_oneline(
                    X509_get_subject_name(handle->cert),
                    NULL, 0);

                GLOBUS_GSI_CRED_ERROR_RESULT(
                    results[result_index],
                    GLOBUS_GSI_CRED_ERROR_WITH_CRED,
                    ("The host credential: %s\n     with subject: %s\n     "
                     "has expired %d minutes ago.\n",
                     cert,
                     subject,
                     (-lifetime)/60));

                free(subject);
                goto exit;
            }

            goto exit;

        case GLOBUS_SERVICE:

            if(desired_subject != NULL)
            {
                results[result_index] =
                    globus_l_gsi_cred_get_service(desired_subject,
                                                  &service_name);

                if(results[result_index] != GLOBUS_SUCCESS)
                {
                    service_name = NULL;
                    GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
                        results[result_index],
                        GLOBUS_GSI_CRED_ERROR_READING_SERVICE_CRED);
                    break;
                }

                results[result_index] =
                    GLOBUS_GSI_SYSCONFIG_GET_SERVICE_CERT_FILENAME(
                        service_name, &cert, &key);
                if(results[result_index] != GLOBUS_SUCCESS)
                {
                    cert = NULL;
                    key = NULL;
                    GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
                        results[result_index],
                        GLOBUS_GSI_CRED_ERROR_READING_SERVICE_CRED);
                    break;
                }

                results[result_index] =
                    globus_gsi_cred_read_cert(handle, cert);
                if(results[result_index] != GLOBUS_SUCCESS)
                {
                    GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
                        results[result_index],
                        GLOBUS_GSI_CRED_ERROR_READING_SERVICE_CRED);
                    goto exit;
                }

                results[result_index] = globus_gsi_cred_read_key(
                    handle,
                    key,
                    globus_i_gsi_cred_password_callback_no_prompt);
                if(results[result_index] != GLOBUS_SUCCESS)
                {
                    globus_object_t *   error_obj;
                    error_obj = globus_error_get(results[result_index]);
                    if(globus_error_get_type(error_obj) ==
                       GLOBUS_GSI_CRED_ERROR_KEY_IS_PASS_PROTECTED)
                    {
                        results[result_index] = globus_error_put(error_obj);
                        GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
                            results[result_index],
                            GLOBUS_GSI_CRED_ERROR_READING_SERVICE_CRED);
                        break;
                    }

                    results[result_index] = globus_error_put(error_obj);
                    GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
                        results[result_index],
                        GLOBUS_GSI_CRED_ERROR_READING_SERVICE_CRED);
                    goto exit;
                }

                results[result_index] = globus_i_gsi_cred_goodtill(
                    handle,
                    &(handle->goodtill));
                if(results[result_index] != GLOBUS_SUCCESS)
                {
                    GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
                        results[result_index],
                        GLOBUS_GSI_CRED_ERROR_WITH_CRED);
                    goto exit;
                }

                if(desired_subject != NULL)
                {
                    results[result_index] = globus_gsi_cred_get_X509_subject_name(
                        handle,
                        &found_subject);
                    if(results[result_index] != GLOBUS_SUCCESS)
                    {
                        GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
                            results[result_index],
                            GLOBUS_GSI_CRED_ERROR_READING_SERVICE_CRED);
                        goto exit;
                    }

                    results[result_index] = globus_l_gsi_cred_subject_cmp(found_subject,
                                                                          desired_subject);

                    X509_NAME_free(found_subject);
                    found_subject = NULL;

                    if(results[result_index] != GLOBUS_SUCCESS)
                    {
                        GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
                            results[result_index],
                            GLOBUS_GSI_CRED_ERROR_READING_SERVICE_CRED);
                        break;
                    }
                }

                results[result_index] = globus_gsi_cred_get_lifetime(
                    handle,
                    &lifetime);
                if(results[result_index] != GLOBUS_SUCCESS)
                {
                    GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
                        results[result_index],
                        GLOBUS_GSI_CRED_ERROR_WITH_CRED);
                    goto exit;
                }

                if(lifetime <= 0)
                {
                    char *                          subject = NULL;

                    subject = X509_NAME_oneline(
                        X509_get_subject_name(handle->cert),
                        NULL, 0);

                    GLOBUS_GSI_CRED_ERROR_RESULT(
                        results[result_index],
                        GLOBUS_GSI_CRED_ERROR_WITH_CRED,
                        ("The service credential: %s\n     with subject:\n%s\n"
                         "     has expired %d minutes ago.\n",
                         cert,
                         subject,
                         (-lifetime)/60));

                    free(subject);
                    goto exit;
                }

                goto exit;
            }
            else
            {
                result_index--;
                break;
            }

        case GLOBUS_SO_END:

            result_count = result_index;
            for(result_index = (result_count - 2);
                result_index >= 0;
                --result_index)
            {
                results[result_index] =
                    globus_i_gsi_cred_error_join_chains_result(
                        results[result_index],
                        results[result_index + 1]);
                results[result_index + 1] = GLOBUS_SUCCESS;
            }

            result_index = 0;
            GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
                results[result_index],
                GLOBUS_GSI_CRED_ERROR_NO_CRED_FOUND);
            goto exit;
        }

        if(proxy)
        {
            free(proxy);
            proxy = NULL;
        }

        if(cert)
        {
            free(cert);
            cert = NULL;
        }

        if(key)
        {
            free(key);
            key = NULL;
        }

        if(service_name)
        {
            free(service_name);
            service_name = NULL;
        }

        result_index++;
    } while(++index);

 exit:

    result = results[result_index];
    for(index = 0; index < result_index; ++index)
    {
        globus_object_t *               result_obj;
        if(results[index] != GLOBUS_SUCCESS)
        {
            result_obj = globus_error_get(results[index]);
            globus_object_free(result_obj);
        }
    }

    if(proxy)
    {
        free(proxy);
    }

    if(cert)
    {
        free(cert);
    }

    if(key)
    {
        free(key);
    }

    GLOBUS_I_GSI_CRED_DEBUG_EXIT;
    return result;
}
/* @} */

/**
 * Read Proxy
 * @ingroup globus_gsi_cred_operation
 */
/* @{ */
/**
 * Read a proxy from a PEM file.  Assumes that the handle
 * attributes contain the filename of the proxy to read
 *
 * @param handle
 *        The credential handle to set based on the proxy
 *        parameters read from the file
 * @return
 *        GLOBUS_SUCCESS or an error object identifier
 */
globus_result_t globus_gsi_cred_read_proxy(
    globus_gsi_cred_handle_t            handle,
    char *                              proxy_filename)
{
    BIO *                               proxy_bio = NULL;
    globus_result_t                     result;

    static char *                       _function_name_ =
        "globus_gsi_cred_read_proxy";

    GLOBUS_I_GSI_CRED_DEBUG_ENTER;

    if(handle == NULL)
    {
        GLOBUS_GSI_CRED_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_READING_PROXY_CRED,
            ("NULL handle passed to function: %s", _function_name_));
        goto exit;
    }

    /* create the bio to read the proxy in from */

    if((proxy_bio = BIO_new_file(proxy_filename, "r")) == NULL)
    {
        GLOBUS_GSI_CRED_OPENSSL_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_READING_PROXY_CRED,
            ("Can't open proxy file: %s for reading", proxy_filename));
        goto exit;
    }

    result = globus_gsi_cred_read_proxy_bio(handle, proxy_bio);
    if(result != GLOBUS_SUCCESS)
    {
        GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_READING_PROXY_CRED);
        goto exit;
    }

 exit:

    if(proxy_bio)
    {
        BIO_free(proxy_bio);
    }

    GLOBUS_I_GSI_CRED_DEBUG_EXIT;
    return result;
}
/* @} */

/**
 * Read Credential
 * @ingroup globus_gsi_cred_operations
 */
/* @{ */
/**
 * Read a Credential from a BIO stream and set the
 * credential handle to represent the read credential.
 * The values read from the stream, in order, will be
 * the signed certificate, the private key,
 * and the certificate chain
 *
 * @param handle
 *        The credential handle to set.  The credential
 *        should not be initialized (i.e. NULL).
 * @param bio
 *        The stream to read the credential from
 * @return
 *        GLOBUS_SUCCESS unless an error occurred, in which
 *        case an error object is returned
 */
globus_result_t
globus_gsi_cred_read_proxy_bio(
    globus_gsi_cred_handle_t            handle,
    BIO *                               bio)
{
    int                                 i = 0;
    globus_result_t                     result;
    X509 *                              tmp_cert = NULL;

    static char *                       _function_name_ =
        "globus_gsi_cred_read_proxy_bio";

    GLOBUS_I_GSI_CRED_DEBUG_ENTER;

    if(handle == NULL)
    {
        GLOBUS_GSI_CRED_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_READING_PROXY_CRED,
            ("Null handle passed to function: %s", _function_name_));
        goto exit;
    }

    if(bio == NULL)
    {
        GLOBUS_GSI_CRED_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_READING_PROXY_CRED,
            ("Null bio variable passed to function: %s", _function_name_));
        goto exit;
    }

    /* read in the certificate of the handle */

    if(handle->cert != NULL)
    {
        X509_free(handle->cert);
        handle->cert = NULL;
    }

    if(!PEM_read_bio_X509(bio, & handle->cert, NULL, NULL))
    {
        GLOBUS_GSI_CRED_OPENSSL_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_READING_PROXY_CRED,
            ("Couldn't read X509 proxy cert from bio"));
        goto exit;
    }

    /* read in the private key of the handle */

    if(handle->key != NULL)
    {
        EVP_PKEY_free(handle->key);
        handle->key = NULL;
    }

    handle->key = PEM_read_bio_PrivateKey(
        bio,
        NULL,
        (int (*) ()) globus_i_gsi_cred_password_callback_no_prompt,
        NULL);
    if(!handle->key)
    {
        if(ERR_GET_REASON(ERR_peek_error()) == PEM_R_BAD_PASSWORD_READ)
        {
            GLOBUS_GSI_CRED_OPENSSL_ERROR_RESULT(
                result,
                GLOBUS_GSI_CRED_ERROR_KEY_IS_PASS_PROTECTED,
                ("The proxy certificate's private key "
                 "is password protected.\n"));
            goto exit;
        }

        GLOBUS_GSI_CRED_OPENSSL_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_READING_PROXY_CRED,
            ("Couldn't read proxy's private key from bio"));
        goto exit;
    }

    /* read in the certificate chain of the handle */

    if(handle->cert_chain != NULL)
    {
        sk_X509_pop_free(handle->cert_chain, X509_free);
        handle->cert_chain = NULL;
    }

    if((handle->cert_chain = sk_X509_new_null()) == NULL)
    {
        GLOBUS_GSI_CRED_OPENSSL_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_READING_PROXY_CRED,
            ("Can't initialize cert chain"));
        goto exit;
    }

    while(!BIO_eof(bio))
    {
        tmp_cert = NULL;
        if(!PEM_read_bio_X509(bio, &tmp_cert, NULL, NULL))
        {
            /* appears to continue reading after EOF and
             * so an error occurs here
             */
            break;
        }

        if(!sk_X509_insert(handle->cert_chain, tmp_cert, i))
        {
            X509_free(tmp_cert);
            GLOBUS_GSI_CRED_OPENSSL_ERROR_RESULT(
                result,
                GLOBUS_GSI_CRED_ERROR_READING_PROXY_CRED,
                ("Error adding certificate to proxy's cert chain"));
            goto exit;
        }
        ++i;
    }

    result = globus_i_gsi_cred_goodtill(handle, &(handle->goodtill));

    if(result != GLOBUS_SUCCESS)
    {
        GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_WITH_CRED);
        goto exit;
    }

    result = GLOBUS_SUCCESS;

 exit:

    GLOBUS_I_GSI_CRED_DEBUG_EXIT;
    return result;
}
/* @} */

/**
 * Read Key
 * @ingroup globus_gsi_cred_operations
 */
/* @{ */
/**
 * Read a key from a the file locations specified in the
 * handle attributes.  Cert and key should be in PEM format.
 *
 * @param handle
 *        the handle to set based on the key that is read
 * @param key_filename
 *        the filename of the key to read
 * @param pw_cb
 *        the callback for the password to read in the key
 * @return
 *        GLOBUS_SUCCESS or an error object identifier
 */
globus_result_t
globus_gsi_cred_read_key(
    globus_gsi_cred_handle_t            handle,
    char *                              key_filename,
    int                                 (*pw_cb)())
{
    BIO *                               key_bio = NULL;
    globus_result_t                     result;

    static char *                       _function_name_ =
        "globus_gsi_cred_read_key";

    GLOBUS_I_GSI_CRED_DEBUG_ENTER;

    if(handle == NULL)
    {
        GLOBUS_GSI_CRED_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_READING_CRED,
            ("NULL handle passed to function: %s", _function_name_));
       goto exit;
    }

    if(!(key_bio = BIO_new_file(key_filename, "r")))
    {
        GLOBUS_GSI_CRED_OPENSSL_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_READING_CRED,
            ("Can't open bio stream for "
             "key file: %s for reading", key_filename));
        goto exit;
    }

    /* read in the key */

    if(handle->key != NULL)
    {
        EVP_PKEY_free(handle->key);
        handle->key = NULL;
    }

    if(!PEM_read_bio_PrivateKey(key_bio, & handle->key, pw_cb, NULL))
    {
        if(ERR_GET_REASON(ERR_peek_error()) == PEM_R_BAD_PASSWORD_READ)
        {
            GLOBUS_GSI_CRED_OPENSSL_ERROR_RESULT(
                result,
                GLOBUS_GSI_CRED_ERROR_KEY_IS_PASS_PROTECTED,
                ("GSI does not currently support password protected "
                 "private keys."));
            goto exit;
        }
        GLOBUS_GSI_CRED_OPENSSL_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_READING_CRED,
            ("Can't read credential's private key from PEM"));
        goto exit;
    }

    result = GLOBUS_SUCCESS;

 exit:

    if(key_bio)
    {
        BIO_free(key_bio);
    }

    GLOBUS_I_GSI_CRED_DEBUG_EXIT;
    return result;
}
/* @} */

/**
 * Read Cert
 * @ingroup globus_gsi_cred_operations
 */
/* @{ */
/**
 * Read a cert from a the file locations specified in the
 * handle attributes.  Cert should be in PEM format.
 *
 * @param handle
 *        the handle to set based on the certificate that is read
 * @param cert_filename
 *        the filename of the certificate to read
 * @return
 *        GLOBUS_SUCCESS or an error object identifier
 */
globus_result_t globus_gsi_cred_read_cert(
    globus_gsi_cred_handle_t            handle,
    char *                              cert_filename)
{
    BIO *                               cert_bio = NULL;
    globus_result_t                     result;
    int                                 i = 0;
    STACK_OF(X509) *                    tmp_cert_chain = NULL;
    static char *                       _function_name_ =
        "globus_gsi_cred_read_cert";

    GLOBUS_I_GSI_CRED_DEBUG_ENTER;

    if(handle == NULL)
    {
        GLOBUS_GSI_CRED_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_READING_CRED,
            ("NULL handle passed to function: %s", _function_name_));
       goto exit;
    }

    if(!(cert_bio = BIO_new_file(cert_filename, "r")))
    {
        GLOBUS_GSI_CRED_OPENSSL_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_READING_CRED,
            ("Can't open cert file: %s for reading", cert_filename));
        goto exit;
    }

    /* read in the cert */

    if(handle->cert != NULL)
    {
        X509_free(handle->cert);
        handle->cert = NULL;
    }

    if(!PEM_read_bio_X509(cert_bio, & handle->cert, NULL, NULL))
    {
        GLOBUS_GSI_CRED_OPENSSL_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_READING_CRED,
            ("Can't read credential cert from bio stream"));
        goto exit;
    }

    if(handle->cert_chain != NULL)
    {
        sk_X509_pop_free(handle->cert_chain, X509_free);
        handle->cert_chain = NULL;
    }

    if((tmp_cert_chain = sk_X509_new_null()) == NULL)
    {
        GLOBUS_GSI_CRED_OPENSSL_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_READING_CRED,
            ("Can't initialize cert chain\n"));
        goto exit;
    }

    while(!BIO_eof(cert_bio))
    {
        X509 *                          tmp_cert = NULL;
        if(!PEM_read_bio_X509(cert_bio, &tmp_cert, NULL, NULL))
        {
            ERR_clear_error();
            break;
        }

        if(!sk_X509_insert(tmp_cert_chain, tmp_cert, i))
        {
            X509_free(tmp_cert);
            GLOBUS_GSI_CRED_OPENSSL_ERROR_RESULT(
                result,
                GLOBUS_GSI_CRED_ERROR_READING_CRED,
                ("Error adding cert: %s\n to issuer cert chain\n",
                 X509_NAME_oneline(X509_get_subject_name(tmp_cert), 0, 0)));
            goto exit;
        }
        ++i;
    }

    if(sk_X509_num(tmp_cert_chain) > 0)
    {
        result = globus_gsi_cred_set_cert_chain(handle, tmp_cert_chain);
        if(result != GLOBUS_SUCCESS)
        {
            GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
                result,
                GLOBUS_GSI_CRED_ERROR_WITH_CRED);
            goto exit;
        }
    }

    sk_X509_pop_free(tmp_cert_chain, X509_free);

    result = globus_i_gsi_cred_goodtill(handle, &(handle->goodtill));

    if(result != GLOBUS_SUCCESS)
    {
        GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_WITH_CRED);
        goto exit;
    }

    result = GLOBUS_SUCCESS;

 exit:

    if(cert_bio)
    {
        BIO_free(cert_bio);
    }

    GLOBUS_I_GSI_CRED_DEBUG_EXIT;
    return result;
}
/* @} */

globus_result_t globus_gsi_cred_read_pkcs12(
    globus_gsi_cred_handle_t            handle,
    char *                              pkcs12_filename)
{
    globus_result_t                     result = GLOBUS_SUCCESS;
    char                                password[100];
    STACK_OF(X509) *                    pkcs12_certs = NULL;
    PKCS12 *                            pkcs12 = NULL;
    PKCS12_SAFEBAG *                    bag = NULL;
    STACK_OF(PKCS12_SAFEBAG) *          pkcs12_safebags = NULL;
    PKCS7 *                             pkcs7 = NULL;
    STACK_OF(PKCS7) *                   auth_safes = NULL;
    PKCS8_PRIV_KEY_INFO *               pkcs8 = NULL;
    BIO *                               pkcs12_bio = NULL;
    int                                 i, j, bag_NID;
    static char *                       _function_name_ =
        "globus_gsi_cred_read_pkcs12";
    GLOBUS_I_GSI_CRED_DEBUG_ENTER;

    if(handle == NULL)
    {
        GLOBUS_GSI_CRED_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_READING_CRED,
            ("NULL handle passed to function: %s", _function_name_));
       goto exit;
    }

    pkcs12_bio = BIO_new_file(pkcs12_filename, "r");
    if(!pkcs12_bio)
    {
        GLOBUS_GSI_CRED_OPENSSL_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_READING_CRED,
            ("Couldn't create BIO for file: %s", pkcs12_filename));
        goto exit;
    }

    d2i_PKCS12_bio(pkcs12_bio, &pkcs12);
    if(!pkcs12)
    {
        GLOBUS_GSI_CRED_OPENSSL_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_READING_CRED,
            ("Couldn't read in PKCS12 credential from BIO"));
        goto exit;
    }

    EVP_read_pw_string(password, 100, NULL, 0);

    if(!PKCS12_verify_mac(pkcs12, password, -1))
    {
        GLOBUS_GSI_CRED_OPENSSL_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_READING_CRED,
            ("Couldn't verify the PKCS12 MAC using the specified password"));
        goto exit;
    }

    auth_safes = M_PKCS12_unpack_authsafes(pkcs12);

    if(!auth_safes)
    {
        GLOBUS_GSI_CRED_OPENSSL_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_READING_CRED,
            ("Couldn't dump cert and key from PKCS12 credential"));
        goto exit;
    }

    pkcs12_certs = sk_X509_new_null();

    for (i = 0; i < sk_PKCS7_num(auth_safes); i++)
    {
        pkcs7 = sk_PKCS7_value(auth_safes, i);

        bag_NID = OBJ_obj2nid(pkcs7->type);

        if(bag_NID == NID_pkcs7_data)
        {
            pkcs12_safebags = M_PKCS12_unpack_p7data(pkcs7);
        }
        else if(bag_NID == NID_pkcs7_encrypted)
        {
            pkcs12_safebags = M_PKCS12_unpack_p7encdata (pkcs7, password, -1);
        }
        else
        {
            GLOBUS_GSI_CRED_OPENSSL_ERROR_RESULT(
                result,
                GLOBUS_GSI_CRED_ERROR_READING_CRED,
                ("Couldn't get NID from PKCS7 that matched "
                 "{NID_pkcs7_data, NID_pkcs7_encrypted}"));
            goto exit;
        }

        if(!pkcs12_safebags)
        {
            GLOBUS_GSI_CRED_OPENSSL_ERROR_RESULT(
                result,
                GLOBUS_GSI_CRED_ERROR_READING_CRED,
                ("Couldn't unpack the PKCS12 safebags from "
                 "the PKCS7 credential"));
            goto exit;
        }

        for (j = 0; j < sk_PKCS12_SAFEBAG_num(pkcs12_safebags); j++)
        {
            bag = sk_PKCS12_SAFEBAG_value(pkcs12_safebags, j);

            if(M_PKCS12_bag_type(bag) == NID_certBag &&
               M_PKCS12_cert_bag_type(bag) == NID_x509Certificate)
            {
                sk_X509_push(pkcs12_certs,
                             M_PKCS12_certbag2x509(bag));
            }
            else if(M_PKCS12_bag_type(bag) == NID_keyBag &&
                    handle->key == NULL)
            {
                pkcs8 = bag->value.keybag;
                handle->key = EVP_PKCS82PKEY(pkcs8);
                if (!handle->key)
                {
                    GLOBUS_GSI_CRED_OPENSSL_ERROR_RESULT(
                        result,
                        GLOBUS_GSI_CRED_ERROR_READING_CRED,
                        ("Couldn't get the private key from the"
                         "PKCS12 credential"));
                    goto exit;
                }
            }
            else if(M_PKCS12_bag_type(bag) ==
                    NID_pkcs8ShroudedKeyBag &&
                    handle->key == NULL)
            {
                pkcs8 = M_PKCS12_decrypt_skey(bag,
                                              password,
                                              strlen(password));
                if(!pkcs8)
                {
                    GLOBUS_GSI_CRED_OPENSSL_ERROR_RESULT(
                        result,
                        GLOBUS_GSI_CRED_ERROR_READING_CRED,
                        ("Couldn't get PKCS8 key from PKCS12 credential"));
                    goto exit;
                }

                handle->key = EVP_PKCS82PKEY(pkcs8);
                if (!handle->key)
                {
                    GLOBUS_GSI_CRED_OPENSSL_ERROR_RESULT(
                        result,
                        GLOBUS_GSI_CRED_ERROR_READING_CRED,
                        ("Couldn't get private key from PKCS12 credential"));
                    goto exit;
                }

                PKCS8_PRIV_KEY_INFO_free(pkcs8);
            }
        }
    }

    if(!handle->key)
    {
        GLOBUS_GSI_CRED_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_READING_CRED,
            ("Couldn't read private key from PKCS12 credential "
             "for unknown reason"));
        goto exit;
    }

    for(i = 0 ; i < sk_X509_num(pkcs12_certs); i++)
    {
        handle->cert = sk_X509_pop(pkcs12_certs);

        if(X509_check_private_key(handle->cert, handle->key))
        {
            sk_X509_pop_free(pkcs12_certs, X509_free);
            pkcs12_certs = NULL;
            break;
        }
        else
        {
            X509_free(handle->cert);
            handle->cert = NULL;
        }
    }

    if(!handle->cert)
    {
        GLOBUS_GSI_CRED_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_READING_CRED,
            ("Couldn't read X509 certificate from PKCS12 credential"));
        goto exit;
    }

    result = globus_i_gsi_cred_goodtill(handle, &(handle->goodtill));
    if(result != GLOBUS_SUCCESS)
    {
        GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_WITH_CRED);
        goto exit;
    }

 exit:

    if(pkcs12_bio)
    {
        BIO_free(pkcs12_bio);
    }

    if(pkcs12)
    {
        PKCS12_free(pkcs12);
    }

    if(pkcs12_certs)
    {
        sk_X509_pop_free(pkcs12_certs, X509_free);
    }

    GLOBUS_I_GSI_CRED_DEBUG_EXIT;
    return result;
}

/**
 * Write Credential
 * @ingroup globus_gsi_cred_operations
 */
/* @{ */
/**
 * Write out a credential to a BIO.  The credential parameters written,
 * in order, are the signed certificate, the RSA private key,
 * and the certificate chain (a set of X509 certificates).
 * the credential is written out in PEM format.
 *
 * @param handle
 *        The credential to write out
 * @param bio
 *        The BIO stream to write out to
 * @return
 *        GLOBUS_SUCCESS unless an error occurred, in which
 *        case an error object ID is returned.
 */
globus_result_t globus_gsi_cred_write(
    globus_gsi_cred_handle_t            handle,
    BIO *                               bio)
{
    int                                 i;
    globus_result_t                     result = GLOBUS_SUCCESS;
    static char *                       _function_name_ =
        "globus_gsi_cred_write";

    GLOBUS_I_GSI_CRED_DEBUG_ENTER;

    if(handle == NULL)
    {
        GLOBUS_GSI_CRED_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_WRITING_CRED,
            ("NULL handle passed to function: %s", _function_name_));
        goto error_exit;
    }

    if(bio == NULL)
    {
        GLOBUS_GSI_CRED_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_WRITING_CRED,
            ("NULL bio variable passed to function: %s", _function_name_));
        goto error_exit;
    }

    if(!PEM_write_bio_X509(bio, handle->cert))
    {
        GLOBUS_GSI_CRED_OPENSSL_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_WRITING_CRED,
            ("Can't write PEM formatted X509 cert to BIO stream"));
        goto error_exit;
    }

    if(!PEM_ASN1_write_bio(i2d_PrivateKey, PEM_STRING_RSA,
                           bio, (char *) handle->key,
                           NULL, NULL, 0, NULL, NULL))
    {
        GLOBUS_GSI_CRED_OPENSSL_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_WRITING_CRED,
            ("Can't write PEM formatted private key to BIO stream"));
        goto error_exit;
    }

    for(i = 0; i < sk_X509_num(handle->cert_chain); ++i)
    {
        if(!PEM_write_bio_X509(bio, sk_X509_value(handle->cert_chain, i)))
        {
            GLOBUS_GSI_CRED_OPENSSL_ERROR_RESULT(
                result,
                GLOBUS_GSI_CRED_ERROR_WRITING_CRED,
                ("Can't write PEM formatted X509 cert"
                 " in cert chain to BIO stream"));
            goto error_exit;
        }
    }

 error_exit:

    GLOBUS_I_GSI_CRED_DEBUG_EXIT;
    return result;
}
/* @} */


/* Utility function that will write the credential to the standard
 * proxy file.
 */

globus_result_t globus_gsi_cred_write_proxy(
    globus_gsi_cred_handle_t            handle,
    char *                              proxy_filename)
{
    globus_result_t                     result = GLOBUS_SUCCESS;
    BIO *                               proxy_bio = NULL;

    static char *                       _function_name_ =
        "globus_gsi_cred_write_proxy";

    GLOBUS_I_GSI_CRED_DEBUG_ENTER;

    if(handle == NULL)
    {
        GLOBUS_GSI_CRED_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_WRITING_PROXY_CRED,
            ("NULL handle passed to function: %s", _function_name_));
        goto exit;
    }

    if(!(proxy_bio = BIO_new_file(proxy_filename, "w")))
    {
        GLOBUS_GSI_CRED_OPENSSL_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_WRITING_PROXY_CRED,
            ("Can't open bio stream for writing to file: %s", proxy_filename));
        goto exit;
    }

    result = globus_gsi_cred_write(handle, proxy_bio);
    if(result != GLOBUS_SUCCESS)
    {
        GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_WRITING_PROXY_CRED);
        goto close_proxy_bio;
    }

    if(proxy_bio)
    {
        BIO_free(proxy_bio);
        proxy_bio = NULL;
    }

    result = GLOBUS_GSI_SYSCONFIG_SET_KEY_PERMISSIONS(proxy_filename);
    if(result != GLOBUS_SUCCESS)
    {
        GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_WRITING_PROXY_CRED);
        goto exit;
    }

    goto exit;

 close_proxy_bio:

    if(proxy_bio != NULL)
    {
        BIO_free(proxy_bio);
    }

 exit:

    GLOBUS_I_GSI_CRED_DEBUG_EXIT;
    return result;
}

globus_result_t
globus_gsi_cred_check_proxy(
    globus_gsi_cred_handle_t               handle,
    globus_gsi_cert_utils_proxy_type_t *   type)
{
    globus_result_t                     result;
    static char *                       _function_name_ =
        "globus_gsi_cred_check_proxy";
    GLOBUS_I_GSI_CRED_DEBUG_ENTER;

    result = globus_gsi_cert_utils_check_proxy_name(handle->cert, type);
    if(result != GLOBUS_SUCCESS)
    {
        GLOBUS_GSI_CRED_ERROR_CHAIN_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_WITH_CRED_CERT);
    }

    GLOBUS_I_GSI_CRED_DEBUG_EXIT;
    return result;
}


#ifndef GLOBUS_DONT_DOCUMENT_INTERNAL

/**
 * Get PROXYCERTINFO Struct
 * @ingroup globus_i_gsi_cred
 */
/* @{ */
/**
 * Get the PROXYCERTINFO struct from the X509 struct.
 * The PROXYCERTINFO struct that gets set must be freed
 * with a call to PROXYCERTINFO_free.
 *
 * @param cert
 *        The X509 struct containing the PROXYCERTINFO struct
 *        in its extensions
 * @param proxycertinfo
 *        The resulting PROXYCERTINFO struct.  This variable
 *        should be freed with a call to PROXYCERTINFO_free when
 *        no longer in use.  It will have a value of NULL if no
 *        proxycertinfo extension exists in the X509 certificate
 * @return
 *        GLOBUS_SUCCESS (even if no proxycertinfo extension was found)
 *        or an globus error object id if an error occurred
 */
globus_result_t
globus_i_gsi_cred_get_proxycertinfo(
    X509 *                              cert,
    PROXYCERTINFO **                    proxycertinfo)
{
    globus_result_t                     result;
    int                                 pci_NID;
    X509_EXTENSION *                    pci_extension = NULL;
    ASN1_OCTET_STRING *                 ext_data;
    int                                 extension_loc;
    static char *                       _function_name_ =
        "globus_i_gsi_cred_get_proxycertinfo";

    GLOBUS_I_GSI_CRED_DEBUG_ENTER;

    pci_NID = OBJ_sn2nid(PROXYCERTINFO_SN);
    if(pci_NID == NID_undef)
    {
        GLOBUS_GSI_CRED_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_WITH_CRED,
            ("Couldn't get numeric ID for PROXYCERTINFO extension"));
        goto exit;
    }

    if(cert == NULL)
    {
        GLOBUS_GSI_CRED_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_WITH_CRED,
            ("NULL X509 cert parameter passed to function: %s",
             _function_name_));
        goto exit;
    }

    if((extension_loc = X509_get_ext_by_NID(
            cert,
            pci_NID, -1)) == -1)
    {
        /* no proxycertinfo extension found in cert */
        *proxycertinfo = NULL;
        result = GLOBUS_SUCCESS;
        goto exit;
    }

    if((pci_extension = X509_get_ext(cert,
                                     extension_loc)) == NULL)
    {
        GLOBUS_GSI_CRED_OPENSSL_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_WITH_CRED,
            ("Can't find PROXYCERTINFO extension in X509 cert at "
             "expected location: %d in extension stack", extension_loc));
        goto free_ext;
    }

    if((ext_data = X509_EXTENSION_get_data(pci_extension)) == NULL)
    {
        GLOBUS_GSI_CRED_OPENSSL_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_WITH_CRED,
            ("Can't get DER encoded extension "
             "data from X509 extension object"));
        goto free_ext_data;
    }

    if((d2i_PROXYCERTINFO(
        proxycertinfo,
        & ext_data->data,
        ext_data->length)) == NULL)
    {
        GLOBUS_GSI_CRED_OPENSSL_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_WITH_CRED,
            ("Can't convert DER encoded PROXYCERTINFO "
             "extension to internal form"));
        goto free_pci;
    }

    result = GLOBUS_SUCCESS;

 free_pci:
    PROXYCERTINFO_free(*proxycertinfo);
 free_ext_data:
    ASN1_OCTET_STRING_free(ext_data);
 free_ext:
    X509_EXTENSION_free(pci_extension);
 exit:

    GLOBUS_I_GSI_CRED_DEBUG_EXIT;
    return result;
}
/* @} */

int
globus_i_gsi_cred_password_callback_no_prompt(
    char *                              buffer,
    int                                 size,
    int                                 w)
{
    static char *                       _function_name_ =
        "globus_i_gsi_cred_password_callback_no_prompt";
    GLOBUS_I_GSI_CRED_DEBUG_ENTER;

    /* current gsi implementation does not allow for a password
     * encrypted certificate to be used for authentication
     */

    GLOBUS_I_GSI_CRED_DEBUG_EXIT;
    return -1;
}

static globus_result_t
globus_l_gsi_cred_subject_cmp(
    X509_NAME *                         actual_subject,
    X509_NAME *                         desired_subject)
{
    int                                 cn_index;
    char *                              desired_cn = NULL;
    char *                              actual_cn = NULL;
    char *                              desired_service;
    char *                              actual_service;
    char *                              desired_host;
    char *                              actual_host;
    char *                              desired_str = NULL;
    char *                              actual_str = NULL;
    globus_result_t                     result = GLOBUS_SUCCESS;
    int                                 length;
    static char *                       _function_name_ =
        "globus_l_gsi_cred_subject_cmp";

    GLOBUS_I_GSI_CRED_DEBUG_ENTER;


    /* if desired subject is NULL return success */

    if(!desired_subject)
    {
        goto exit;
    }

    /* check for single /CN entry */

    if(X509_NAME_entry_count(desired_subject) == 1)
    {
        /* make sure we actually got a common name */

        cn_index = X509_NAME_get_index_by_NID(desired_subject, NID_commonName, -1);

        if(cn_index < 0)
        {
            desired_str = X509_NAME_oneline(desired_subject, NULL, 0);
            GLOBUS_GSI_CRED_ERROR_RESULT(
                result,
                GLOBUS_GSI_CRED_ERROR_SUBJECT_CMP,
                ("No Common Name found in desired subject %s.\n", desired_str));
            goto exit;
        }

        /* find /CN entry in actual subject */

        cn_index = X509_NAME_get_index_by_NID(actual_subject, NID_commonName, -1);

        /* error if no common name was found */

        if(cn_index < 0)
        {
            actual_str = X509_NAME_oneline(actual_subject, NULL, 0);
            GLOBUS_GSI_CRED_ERROR_RESULT(
                result,
                GLOBUS_GSI_CRED_ERROR_SUBJECT_CMP,
                ("No Common Name found in subject %s.\n", actual_str));
            goto exit;
        }

        /* check that actual subject only has one CN entry */

        if(X509_NAME_get_index_by_NID(actual_subject, NID_commonName, cn_index) != -1)
        {
            actual_str = X509_NAME_oneline(actual_subject, NULL, 0);
            GLOBUS_GSI_CRED_ERROR_RESULT(
                result,
                GLOBUS_GSI_CRED_ERROR_SUBJECT_CMP,
                ("More than one Common Name found in subject %s.\n", actual_str));
            goto exit;
        }

        /* get CN text for desired subject */

        length = X509_NAME_get_text_by_NID(desired_subject, NID_commonName,
                                           NULL, 1024) + 1;

        desired_cn = malloc(length);

        X509_NAME_get_text_by_NID(desired_subject, NID_commonName,
                                  desired_cn, length);

        /* get CN text for actual subject */

        length = X509_NAME_get_text_by_NID(actual_subject, NID_commonName,
                                           NULL, 1024) + 1;

        actual_cn = malloc(length);

        X509_NAME_get_text_by_NID(actual_subject, NID_commonName,
                                  actual_cn, length);

        /* straight comparison */

        if(!strcmp(desired_cn,actual_cn))
        {
            goto exit;
        }

        actual_host = strchr(actual_cn,'/');

        if(actual_host == NULL)
        {
            actual_host = actual_cn;
            actual_service = NULL;
        }
        else
        {
            *actual_host = '\0';
            actual_service = actual_cn;
            actual_host++;
        }

        desired_host = strchr(desired_cn,'/');

        if(desired_host == NULL)
        {
            desired_host = desired_cn;
            desired_service = NULL;
        }
        else
        {
            *desired_host = '\0';
            desired_service = desired_cn;
            desired_host++;
        }

        if(desired_service == NULL &&
           actual_service == NULL)
        {
            actual_str = X509_NAME_oneline(actual_subject, NULL, 0);
            desired_str = X509_NAME_oneline(desired_subject, NULL, 0);

            GLOBUS_GSI_CRED_ERROR_RESULT(
                result,
                GLOBUS_GSI_CRED_ERROR_SUBJECT_CMP,
                ("Desired subject and actual subject of certificate"
                 " do not match.\n"
                 "     Desired subject: %s\n"
                 "     Actual subject: %s\n",
                 desired_str,
                 actual_str));

            goto exit;
        }
        else if(desired_service == NULL)
        {
            if(strcmp("host",actual_service))
            {
                actual_str = X509_NAME_oneline(actual_subject, NULL, 0);
                desired_str = X509_NAME_oneline(desired_subject, NULL, 0);

                GLOBUS_GSI_CRED_ERROR_RESULT(
                    result,
                    GLOBUS_GSI_CRED_ERROR_SUBJECT_CMP,
                    ("Desired subject and actual subject of certificate"
                     " do not match.\n"
                     "     Desired subject: %s\n"
                     "     Actual subject: %s\n",
                     desired_str,
                     actual_str));
            }

            goto exit;
        }
        else if(actual_service == NULL)
        {
            if(strcmp("host",desired_service))
            {
                actual_str = X509_NAME_oneline(actual_subject, NULL, 0);
                desired_str = X509_NAME_oneline(desired_subject, NULL, 0);

                GLOBUS_GSI_CRED_ERROR_RESULT(
                    result,
                    GLOBUS_GSI_CRED_ERROR_SUBJECT_CMP,
                    ("Desired subject and actual subject of certificate"
                     " do not match.\n"
                     "     Desired subject: %s\n"
                     "     Actual subject: %s\n",
                     desired_str,
                     actual_str));
            }

            goto exit;
        }
        else
        {
            if(strcmp(desired_service,actual_service))
            {
                actual_str = X509_NAME_oneline(actual_subject, NULL, 0);
                desired_str = X509_NAME_oneline(desired_subject, NULL, 0);

                GLOBUS_GSI_CRED_ERROR_RESULT(
                    result,
                    GLOBUS_GSI_CRED_ERROR_SUBJECT_CMP,
                    ("Desired subject and actual subject of certificate"
                     " do not match.\n"
                     "     Desired subject: %s\n"
                     "     Actual subject: %s\n",
                     desired_str,
                     actual_str));
            }

            goto exit;

        }
    }
    else
    {
        /* full subject name, don't care about equivalence classes */

        if(X509_NAME_cmp(desired_subject, actual_subject))
        {
            actual_str = X509_NAME_oneline(actual_subject, NULL, 0);
            desired_str = X509_NAME_oneline(desired_subject, NULL, 0);

            GLOBUS_GSI_CRED_ERROR_RESULT(
                result,
                GLOBUS_GSI_CRED_ERROR_SUBJECT_CMP,
                ("Desired subject and actual subject of certificate"
                 " do not match.\n"
                 "     Desired subject: %s\n"
                 "     Actual subject: %s\n",
                 desired_str,
                 actual_str));
        }
        goto exit;
    }

 exit:

    if(actual_cn)
    {
        free(actual_cn);
    }

    if(desired_cn)
    {
        free(desired_cn);
    }

    if(actual_str)
    {
        free(actual_str);
    }

    if(desired_str)
    {
        free(desired_str);
    }

    GLOBUS_I_GSI_CRED_DEBUG_EXIT;

    return result;
}

static globus_result_t
globus_l_gsi_cred_get_service(
    X509_NAME *                         subject,
    char **                             service)
{
    int                                 cn_index;
    int                                 length;
    char *                              cn = NULL;
    char *                              host;
    char *                              subject_str = NULL;
    globus_result_t                     result = GLOBUS_SUCCESS;
    static char *                       _function_name_ =
        "globus_l_gsi_cred_get_service";

    GLOBUS_I_GSI_CRED_DEBUG_ENTER;

    *service = NULL;

    /* if desired subject is NULL return success */

    if(!subject)
    {
        goto exit;
    }

    /* find /CN entry in subject */

    cn_index = X509_NAME_get_index_by_NID(subject, NID_commonName, -1);

    /* error if no common name was found */

    if(cn_index < 0)
    {
        subject_str = X509_NAME_oneline(subject, NULL, 0);
        GLOBUS_GSI_CRED_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_GETTING_SERVICE_NAME,
            ("No Common Name found in subject %s.\n", subject_str));
        goto exit;
    }

    /* check that subject only has one CN entry */

    if(X509_NAME_get_index_by_NID(subject, NID_commonName, cn_index) != -1)
    {
        subject_str = X509_NAME_oneline(subject, NULL, 0);
        GLOBUS_GSI_CRED_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_GETTING_SERVICE_NAME,
            ("More than one Common Name found in subject %s.\n", subject_str));
        goto exit;
    }

    /* get CN text for subject */

    length = X509_NAME_get_text_by_NID(subject, NID_commonName,
                                       NULL, 1024) + 1;

    cn = malloc(length);

    X509_NAME_get_text_by_NID(subject, NID_commonName,
                              cn, length);

    host = strchr(cn,'/');

    if(host == NULL)
    {
        subject_str = X509_NAME_oneline(subject, NULL, 0);
        GLOBUS_GSI_CRED_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_GETTING_SERVICE_NAME,
            ("No service name found in subject %s.\n", subject_str));
        goto exit;
    }

    *host = '\0';

    if(strcmp("host",cn))
    {
        *service = strdup(cn);
    }
    else
    {
        subject_str = X509_NAME_oneline(subject, NULL, 0);
        GLOBUS_GSI_CRED_ERROR_RESULT(
            result,
            GLOBUS_GSI_CRED_ERROR_GETTING_SERVICE_NAME,
            ("No service name found in subject %s.\n", subject_str));
    }

    goto exit;

 exit:

    if(cn)
    {
        free(cn);
    }

    if(subject_str)
    {
        free(subject_str);
    }

    GLOBUS_I_GSI_CRED_DEBUG_EXIT;

    return result;
}


#endif
