#ifndef GLOBUS_DONT_DOCUMENT_INTERNAL
/**
 * @file globus_gsi_sysconfig_system_config.c
 * @author Sam Lang, Sam Meder
 *
 * $RCSfile: globus_gsi_system_config.c,v $
 * $Revision: 1.40.2.2 $
 * $Date: 2003/06/12 22:02:41 $
 */
#endif

#include "globus_common.h"
#include "globus_gsi_system_config.h"
#include "globus_i_gsi_system_config.h"
#include <openssl/rand.h>
#include <pwd.h>
#include <time.h>
#include <errno.h>
#include <sys/times.h>
#include "version.h"

#ifndef DEFAULT_SECURE_TMP_DIR
#ifndef WIN32
#define DEFAULT_SECURE_TMP_DIR          "/tmp"
#else
#define DEFAULT_SECURE_TMP_DIR          "c:\\tmp"
#endif
#endif

#ifndef DEFAULT_EGD_PATH
#ifndef WIN32
#define DEFAULT_EGD_PATH                "/tmp"
#else
#define DEFAULT_EGD_PATH                "c:\\tmp"
#endif
#endif

#ifndef DEFAULT_RANDOM_FILE
#ifndef WIN32
#define DEFAULT_RANDOM_FILE             "/tmp"
#else
#define DEFAULT_RANDOM_FILE             UNDEFINED_VALUE
#endif
#endif

#ifdef WIN32
#include "winglue.h"
#include <io.h>
#else
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#include <dirent.h>
#endif

#define X509_CERT_DIR                   "X509_CERT_DIR"
#define X509_CERT_FILE                  "X509_CERT_FILE"
#define X509_USER_PROXY                 "X509_USER_PROXY"
#define X509_USER_CERT                  "X509_USER_CERT"
#define X509_USER_KEY                   "X509_USER_KEY"
#define X509_UNIQUE_PROXY_FILE          "x509up_p"
#define X509_USER_PROXY_FILE            "x509up_u"
#define SIGNING_POLICY_FILE_EXTENSION   ".signing_policy"

/* This is added after the CA name hash to make the policy filename */
#define SIGNING_POLICY_FILE_EXTENSION   ".signing_policy"

#ifdef WIN32
#define FILE_SEPERATOR "\\"
#define GSI_REGISTRY_DIR                "software\\Globus\\GSI"
#define X509_DEFAULT_USER_CERT          ".globus\\usercert.pem"
#define X509_DEFAULT_USER_KEY           ".globus\\userkey.pem"
#define X509_DEFAULT_PKCS12_FILE        ".globus\\usercred.p12"
#define X509_DEFAULT_TRUSTED_CERT_DIR   "SLANG: NEEDS TO BE DETERMINED"
#define X509_INSTALLED_TRUSTED_CERT_DIR "SLANG: NEEDS TO BE DETERMINED"
#define X509_LOCAL_TRUSTED_CERT_DIR     ".globus\\certificates"
#define X509_DEFAULT_CERT_DIR           "SLANG: NEEDS TO BE DETERMINED"
#define X509_INSTALLED_CERT_DIR         "etc"
#define X509_LOCAL_CERT_DIR             ".globus"
#define DEFAULT_GRIDMAP                 "SLANG: NEEDS TO BE DETERMINED"
#define LOCAL_GRIDMAP                   "SLANG: NEEDS TO BE DETERMINED"
#define DEFAULT_AUTHZ_FILE              "Sam: TBD"
#define INSTALLED_AUTHZ_FILE            "Sam: TBD"
#define LOCAL_AUTHZ_FILE                "Sam: TBD"
#else
#define FILE_SEPERATOR                  "/"
#define X509_DEFAULT_USER_CERT          ".globus/usercert.pem"
#define X509_DEFAULT_USER_KEY           ".globus/userkey.pem"
#define X509_DEFAULT_PKCS12_FILE        ".globus/usercred.p12"
#define X509_DEFAULT_TRUSTED_CERT_DIR   "/etc/grid-security/certificates"
#define X509_INSTALLED_TRUSTED_CERT_DIR "share/certificates"
#define X509_LOCAL_TRUSTED_CERT_DIR     ".globus/certificates"
#define X509_DEFAULT_CERT_DIR           "/etc/grid-security"
#define X509_INSTALLED_CERT_DIR         "etc"
#define X509_LOCAL_CERT_DIR             ".globus"
#define DEFAULT_GRIDMAP                 "/etc/grid-security/grid-mapfile"
#define INSTALLED_GRIDMAP               "etc/grid-mapfile"
#define LOCAL_GRIDMAP                   ".gridmap"
#define DEFAULT_AUTHZ_FILE              "/etc/grid-security/gsi-authz.conf"
#define INSTALLED_AUTHZ_FILE            "etc/gsi-authz.conf"
#define LOCAL_AUTHZ_FILE                ".gsi-authz.conf"
#endif

#define X509_HOST_PREFIX                "host"
#define X509_CERT_SUFFIX                "cert.pem"
#define X509_KEY_SUFFIX                 "key.pem"

#define X509_HASH_LENGTH                8

#ifndef GLOBUS_DONT_DOCUMENT_INTERNAL

#define GLOBUS_GSI_SYSTEM_CONFIG_MALLOC_ERROR \
    globus_error_put(globus_error_wrap_errno_error( \
        GLOBUS_GSI_SYSCONFIG_MODULE, \
        errno, \
        GLOBUS_GSI_SYSCONFIG_ERROR_ERRNO, \
        "%s:%d: Could not allocate enough memory", \
        __FILE__, __LINE__))


int                                     globus_i_gsi_sysconfig_debug_level;
FILE *                                  globus_i_gsi_sysconfig_debug_fstream;

static int globus_l_gsi_sysconfig_activate(void);
static int globus_l_gsi_sysconfig_deactivate(void);

int globus_i_gsi_sysconfig_debug_level = 0;

/**
 * Module descriptor static initializer.
 */
globus_module_descriptor_t globus_i_gsi_sysconfig_module =
{
    "globus_sysconfig",
    globus_l_gsi_sysconfig_activate,
    globus_l_gsi_sysconfig_deactivate,
    GLOBUS_NULL,
    GLOBUS_NULL,
    &local_version
};

/**
 * Module activation
 */
static
int
globus_l_gsi_sysconfig_activate(void)
{
    int                                 result = (int) GLOBUS_SUCCESS;
    const char *                              random_file = NULL;
    char *                              egd_path = NULL;
    clock_t                             uptime;
    struct tms                          proc_times;
    char                                buffer[200];
    char *                              tmp_string;
    static char *                       _function_name_ =
        "globus_l_gsi_sysconfig_activate";

    tmp_string = getenv("GLOBUS_GSI_SYSCONFIG_DEBUG_LEVEL");
    if(tmp_string != GLOBUS_NULL)
    {
        globus_i_gsi_sysconfig_debug_level = atoi(tmp_string);
        
        if(globus_i_gsi_sysconfig_debug_level < 0)
        {
            globus_i_gsi_sysconfig_debug_level = 0;
        }
    }

    tmp_string = getenv("GLOBUS_GSI_SYSCONFIG_DEBUG_FILE");
    if(tmp_string != GLOBUS_NULL)
    {
        globus_i_gsi_sysconfig_debug_fstream = fopen(tmp_string, "w");
        if(globus_i_gsi_sysconfig_debug_fstream == NULL)
        {
            result = (int) GLOBUS_FAILURE;
            goto exit;
        }
    }
    else
    {
        /* if the env. var. isn't set, use stderr */
        globus_i_gsi_sysconfig_debug_fstream = stderr;
    }

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    /* OpenSSL's random generator is fed with random
     * information, which requires system dependant information
     * (path names)
     */

    random_file = RAND_file_name(buffer, 200);
    if(random_file)
    {
        RAND_load_file(random_file, 1024L * 1024L);
    }

    egd_path = getenv("EGD_PATH");
    if(egd_path == NULL)
    {
        egd_path = DEFAULT_EGD_PATH;
    }
    RAND_egd(egd_path);
    
    if(RAND_status() == 0)
    {
        globus_gsi_statcheck_t          status;

        /* this function does a RAND_add based on the
         * filename - provides platform independence
         */
        GLOBUS_GSI_SYSCONFIG_FILE_EXISTS(DEFAULT_RANDOM_FILE, &status);

        /* probably overestimating the entropy in the below */
        
        uptime = times(&proc_times);
        
        RAND_add((void *) &uptime, sizeof(clock_t), 2);
        RAND_add((void *) &proc_times, sizeof(struct tms), 8);
    }

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_FPRINTF(
        2, (globus_i_gsi_sysconfig_debug_fstream,
            "RAND_status = %d", RAND_status()));

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;

 exit:
    return result;
}

/**
 * Module deactivation
 *
 */
static
int
globus_l_gsi_sysconfig_deactivate(void)
{
    int                                 result = (int) GLOBUS_SUCCESS;
    static char *                       _function_name_ =
        "globus_l_gsi_sysconfig_deactivate";

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    RAND_cleanup();

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;

    if(globus_i_gsi_sysconfig_debug_fstream != stderr)
    {
        fclose(globus_i_gsi_sysconfig_debug_fstream);
    }

    return result;
}
/* globus_l_gsi_sysconfig_deactivate() */


globus_result_t
globus_i_gsi_sysconfig_create_cert_dir_string(
    char **                             cert_dir,
    char **                             cert_dir_value,
    globus_gsi_statcheck_t *            status,              
    const char *                        format,
    ...)
{
    va_list                             ap;
    globus_result_t                     result;

    static char *                       _function_name_ =
        "globus_i_gsi_sysconfig_create_cert_dir_string";

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    *cert_dir = NULL;

    va_start(ap, format);

    *cert_dir_value = globus_common_v_create_string(format, ap);

    va_end(ap);

    if(*cert_dir_value == NULL)
    {
        result = GLOBUS_GSI_SYSTEM_CONFIG_MALLOC_ERROR;
        goto exit;
    }
     
    result = GLOBUS_GSI_SYSCONFIG_FILE_EXISTS(*cert_dir_value, status);
    if(result != GLOBUS_SUCCESS)
    {
        GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_DIR);
        free(*cert_dir_value);
        *cert_dir_value = NULL;
        goto exit;
    }

    if(format && *status == GLOBUS_FILE_DIR)
    {
        *cert_dir = *cert_dir_value;
    }
    else if((*status) != GLOBUS_FILE_DOES_NOT_EXIST)
    {
        GLOBUS_GSI_SYSCONFIG_ERROR_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_DIR,
            ("%s %s\n",
             *cert_dir_value, 
             globus_l_gsi_sysconfig_status_strings[*status]));
        free(*cert_dir_value);
        *cert_dir_value = NULL;
        goto exit;
    }

    result = GLOBUS_SUCCESS;

 exit:

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;
}
    

globus_result_t
globus_i_gsi_sysconfig_create_cert_string(
    char **                             cert_string,
    char **                             cert_string_value,
    globus_gsi_statcheck_t *            status,
    const char *                        format,
    ...)
{
    va_list                             ap;
    globus_result_t                     result;
    
    static char *                       _function_name_ =
        "globus_i_gsi_sysconfig_create_cert_string";

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    *cert_string = NULL;

    va_start(ap, format);

    *cert_string_value = globus_common_v_create_string(format, ap);

    va_end(ap);

    if(*cert_string_value == NULL)
    {
        result = GLOBUS_GSI_SYSTEM_CONFIG_MALLOC_ERROR;
        goto exit;
    }

    result = GLOBUS_GSI_SYSCONFIG_CHECK_CERTFILE(*cert_string_value, status);
    if(result != GLOBUS_SUCCESS)
    {
        GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_STRING);
        free(*cert_string_value);
        *cert_string_value = NULL;
        goto exit;
    }

    if(format && (*status) == GLOBUS_FILE_VALID)
    {
        *cert_string = *cert_string_value;
    }
    else if((*status) != GLOBUS_FILE_DOES_NOT_EXIST)
    {
        GLOBUS_GSI_SYSCONFIG_ERROR_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_STRING,
            ("%s %s\n",
             *cert_string_value, 
             globus_l_gsi_sysconfig_status_strings[*status]));
        free(*cert_string_value);
        *cert_string_value = NULL;
        goto exit;
    }

    result = GLOBUS_SUCCESS;

 exit:

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;
}

globus_result_t
globus_i_gsi_sysconfig_create_key_string(
    char **                             key_string,
    char **                             key_string_value,
    globus_gsi_statcheck_t *            status,
    const char *                        format,
    ...)
{
    va_list                             ap;
    globus_result_t                     result;

    static char *                       _function_name_ =
        "globus_i_gsi_sysconfig_create_key_string";

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    *key_string = NULL;

    va_start(ap, format);

    *key_string_value = globus_common_v_create_string(format, ap);
    
    va_end(ap);
    
    if(*key_string_value == NULL)
    {
        result = GLOBUS_GSI_SYSTEM_CONFIG_MALLOC_ERROR;
        goto exit;
    }

    result = GLOBUS_GSI_SYSCONFIG_CHECK_KEYFILE(*key_string_value, status);
    if(result != GLOBUS_SUCCESS)
    {
        GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_KEY_STRING);
        free(*key_string_value);
        *key_string_value = NULL;
        goto exit;
    }

    if(format && (*status) == GLOBUS_FILE_VALID)
    {
        *key_string = *key_string_value;
    }
    else if((*status) != GLOBUS_FILE_DOES_NOT_EXIST)
    {
        GLOBUS_GSI_SYSCONFIG_ERROR_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_KEY_STRING,
            ("%s %s\n",
             *key_string_value,
             globus_l_gsi_sysconfig_status_strings[*status]));
        free(*key_string_value);
        *key_string_value = NULL;
        goto exit;
    }

    result = GLOBUS_SUCCESS;

 exit:

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;
}

#endif /* GLOBUS_DONT_DOCUMENT_INTERNAL */

#ifdef WIN32  /* define all the *_win32 functions */


/**
 * @name Win32 - Set Key Permissions
 * @ingroup globus_gsi_sysconfig_win32
 */
/* @{ */
/**
 * Set the file permissions of a file to read only by the user
 * which are the permissions that should be set for all private keys.
 *
 * @param filename
 *
 * @return
 *        GLOBUS_SUCCESS or an error object id
 */
globus_result_t
globus_gsi_sysconfig_set_key_permissions_win32(
    char *                              filename)
{
    globus_result_t                     result = GLOBUS_SUCCESS;
    globus_gsi_statcheck_t              status;
    static char *                       _function_name_ =
        "globus_gsi_sysconfig_set_key_permissions_win32";
    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    result = globus_gsi_sysconfig_file_exists_win32(
        filename,
        &status);
    if(result != GLOBUS_SUCCESS)
    {
        GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_SETTING_PERMS);
        goto exit;
    }

    if(status != GLOBUS_FILE_VALID ||
       status != GLOBUS_FILE_DIR)
    {
        GLOBUS_GSI_SYSCONFIG_ERROR_RESULT(
            result,
            GLOBUS_GIS_SYSCONFIG_ERROR_SETTING_PERMS,
            ("Invalid file: %s", filename));
        goto exit;
    }

#error need to fill this in

 exit:

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;
}
/* @} */

#ifndef GLOBUS_DONT_DOCUMENT_INTERNAL

/**
 * @name Win32 - Get HOME Directory
 * @ingroup globus_i_gsi_sysconfig_win32
 */
/* @{ */
/**
 * Get the HOME directory, currently c:\windows
 * 
 * @param home_dir
 *        The home directory of the current user
 * @return
 *        GLOBUS_SUCCESS if no error occured, otherwise
 *        an error object is returned.
 */
globus_result_t
globus_i_gsi_sysconfig_get_home_dir_win32(
    char **                             home_dir)
{
    globus_result_t                     result;

    const char *                        _function_name_ =
        "globus_i_gsi_sysconfig_get_home_dir_win32";

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    *home_dir = "c:\\windows";

    if((*home_dir) == NULL)
    {
        result = GLOBUS_GSI_SYSCONFIG_ERROR_RESULT(
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_HOME_DIR,
            ("Could not get a home directory for this machine"));
        goto error_exit;
    }

    result = GLOBUS_SUCCESS;

 error_exit:

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;
}
/* @} */

#endif

/**
 * @name Win32 - File Exists
 * @ingroup globus_gsi_sysconfig_win32
 */
/* @{ */
/**
 * Check that the file exists
 *
 * @param filename the file to check
 * @param status   the status of the file
 *
 * @return 
 *        GLOBUS_SUCCESS (even if the file doesn't exist) - in some
 *        abortive cases an error object identifier is returned
 */
globus_result_t
globus_gsi_sysconfig_file_exists_win32(
    const char *                        filename,
    globus_gsi_statcheck_t *            status)
{
    globus_result_t                     result;
    struct stat                         stx;

    static char *                       _function_name_ =
        "globus_i_gsi_sysconfig_file_exists_win32";

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    if (stat(filename,&stx) == -1)
    {
        switch (errno)
        {
          case ENOENT:
          case ENOTDIR:
            *status = GLOBUS_DOES_NOT_EXIST;
            result = GLOBUS_SUCCESS;
            goto exit;

          case EACCES:

            *status = GLOBUS_BAD_PERMISSIONS;
            result = GLOBUS_SUCCESS;
            goto exit;

          default:
            result = globus_error_put(
                globus_error_wrap_errno_error(
                    GLOBUS_GSI_SYSCONFIG_MODULE,
                    errno,
                    GLOBUS_GSI_SYSCONFIG_ERROR_ERRNO,
                    __FILE__":%d:%s: Error getting status of keyfile\n",
                    __LINE__,
                    _function_name_));
            goto exit;
        }
    }

    /*
     * use any stat output as random data, as it will 
     * have file sizes, and last use times in it. 
     */
    RAND_add((void*)&stx, sizeof(stx), 2);

    if (stx.st_size == 0)
    {
        *status = GLOBUS_ZERO_LENGTH;
        result = GLOBUS_SUCCESS;
        goto exit;
    }

    *status = GLOBUS_VALID;
    
 exit:

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;
}    
/* @} */


/**
 * @name Win32 - Check File Status for Key
 * @ingroup globus_i_gsi_sysconfig_win32
 */
/* @{ */
/**
 * This is a convenience function used to check the status of a 
 * private key file.  The desired status is only the current user has
 * ownership and read permissions, everyone else should not be able
 * to access it.
 * 
 * @param filename
 *        The name of the file to check the status of
 * @param status
 *        The status of the file being checked
 *        see @ref globus_gsi_statcheck_t for possible values
 *        of this variable 
 *
 * @return 
 *        GLOBUS_SUCCESS if the status of the file was able
 *        to be determined.  Otherwise, an error object
 *        identifier
 *
 * @see globus_gsi_statcheck_t
 */
globus_result_t
globus_i_gsi_sysconfig_check_keyfile_win32(
    const char *                        filename,
    globus_gsi_statcheck_t *            status)
{
    struct stat                         stx;
    globus_result_t                     result;
    static char *                       _function_name_ =
        "globus_i_gsi_sysconfig_check_keyfile_win32";

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    if (stat(filename,&stx) == -1)
    {
        switch (errno)
        {
          case ENOENT:
          case ENOTDIR:
            *status = GLOBUS_DOES_NOT_EXIST;
            result = GLOBUS_SUCCESS;
            goto exit;

          case EACCES:

            *status = GLOBUS_BAD_PERMISSIONS;
            result = GLOBUS_SUCCESS;
            goto exit;

          default:
            result = globus_error_put(
                globus_error_wrap_errno_error(
                    GLOBUS_GSI_SYSCONFIG_MODULE,
                    errno,
                    GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_KEY_STRING,
                    __FILE__":%d:%s: Error getting status of keyfile\n",
                    __LINE__,
                    _function_name_));
            goto exit;
        }
    }

    /*
     * use any stat output as random data, as it will 
     * have file sizes, and last use times in it. 
     */
    RAND_add((void*)&stx,sizeof(stx),2);

    if (stx.st_size == 0)
    {
        *status = GLOBUS_ZERO_LENGTH;
        result = GLOBUS_SUCCESS;
    }

    *status = GLOBUS_VALID;

 exit:

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;
}
/* @} */


#ifndef GLOBUS_DONT_DOCUMENT_INTERNAL

/**
 * @name Win32 - Check File Status for Cert
 * @ingroup globus_i_gsi_sysconfig_win32
 */
/* @{ */
/**
 * This is a convenience function used to check the status of a 
 * certificate file.  The desired status is the current user has
 * ownership and read/write permissions, while group and others only
 * have read permissions.
 * 
 * @param filename
 *        The name of the file to check the status of
 * @param status
 *        The status of the file being checked
 *        see @ref globus_gsi_statcheck_t for possible values
 *        of this variable 
 *
 * @return 
 *        GLOBUS_SUCCESS if the status of the file was able
 *        to be determined.  Otherwise, an error object
 *        identifier
 *
 * @see globus_gsi_statcheck_t
 */
globus_result_t
globus_i_gsi_sysconfig_check_certfile_win32(
    const char *                        filename,
    globus_gsi_statcheck_t *            status)
{
    globus_result_t                     result;
    struct stat                         stx;

    static char *                       _function_name_ =
        "globus_i_gsi_sysconfig_check_certfile_win32";

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;
 
    if (stat(filename,&stx) == -1)
    {
        switch (errno)
        {
          case ENOENT:
          case ENOTDIR:
            *status = GLOBUS_DOES_NOT_EXIST;
            result = GLOBUS_SUCCESS;
            goto exit;

          case EACCES:

            *status = GLOBUS_BAD_PERMISSIONS;
            result = GLOBUS_SUCCESS;
            goto exit;

          default:
            result = globus_error_put(
                globus_error_wrap_errno_error(
                    GLOBUS_GSI_SYSCONFIG_MODULE,
                    errno,
                    GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_STRING,
                    __FILE__":%d:%s: Error getting status of keyfile\n",
                    __LINE__,
                    _function_name_));
            goto exit;
        }
    }

    /*
     * use any stat output as random data, as it will 
     * have file sizes, and last use times in it. 
     */
    RAND_add((void*)&stx,sizeof(stx),2);

    if (stx.st_size == 0)
    {
        *status = GLOBUS_ZERO_LENGTH;
        result = GLOBUS_SUCCESS;
    }

    *status = GLOBUS_VALID;

 exit:

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;

    return result;
}
/* @} */

#endif

/**
 * @name Win32 - Get Current Working Directory
 * @ingroup globus_gsi_sysconfig_win32
 */
/* @{ */
/**
 * Get the current working directory on a windows system
 *
 * @param working_dir
 *        The working directory to get
 * @return
 *        GLOBUS_SUCCESS if no error occurred, otherwise an error object
 *        ID is returned
 */
globus_result_t
globus_gsi_sysconfig_get_current_working_dir_win32(
    char **                             working_dir)
{
    globus_result_t                     result = GLOBUS_SUCCESS;
    static char *                       _function_name_ = 
        "globus_gsi_sysconfig_get_current_working_dir_win32";
    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

#error /* this needs to be filled in */
    
    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;
}
/* @} */

/**
 * @name Win32 - Make Absolute Path
 * @ingroup globus_gsi_sysconfig_win32
 */
/* @{ */
/**
 * Make the filename into an absolute path string based
 * on the current working directory.
 *
 * @param filename
 *        the filename to get the absolute path of.  
 * @param absolute_path
 *        The resulting absolute path
 * @return
 *        GLOBUS_SUCCESS if no error occurred, otherwise
 *        an error object ID is returned
 */
globus_result_t
globus_gsi_sysconfig_make_absolute_path_for_filename_win32(
    char *                              filename,
    char **                             absolute_path)
{
    globus_result_t                     result = GLOBUS_SUCCESS;
    static char *                       _function_name_ =
        "globus_gsi_sysconfig_make_absolute_path_for_filename_win32";
    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;
    
#error /* this needs to be filled in */

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;
}
/* @} */

/**
 * @name Win32 - Split Directory and Filename
 * @ingroup globus_gsi_sysconfig_win32
 */
/* @{ */
/**
 * Split the directory and filename portions of a filename string
 * into two separate strings
 *
 * @param full_filename
 * @param dir_string
 * @param filename_string
 *
 * @return
 */
globus_result_t
globus_gsi_sysconfig_split_dir_and_filename_win32(
    char *                              full_filename,
    char **                             dir_string,
    char **                             filename_string)
{
    globus_result_t                     result = GLOBUS_SUCCESS;
    static char *                       _function_name_ =
        "globus_gsi_sysconfig_split_dir_and_filename_win32";
    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

#error /* this needs to be filled in */
    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;
}
/* @} */

/**
 * @name Win32 - Get User ID
 * @ingroup globus_i_gsi_sysconfig_win32
 */
/* @{ */
/**
 * Get a unique string representing the current user.  
 * On Windows, SLANG: NOT DETERMINED
 */
globus_result_t
globus_gsi_sysconfig_get_user_id_string_win32(
    char **                             user_id_string)
{
    int                                 uid;

    static char *                       _function_name_ =
        "globus_gsi_sysconfig_get_user_id_string_win32";

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    result = globus_gsi_sysconfig_get_username_win32(user_id_string);

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    
    return GLOBUS_SUCCESS;
}
/* @} */

/**
 * @name Win32 - Get Username
 * @ingroup globus_i_gsi_sysconfig_win32
 */
/* @{ */
/**
 * Get the username of the current user.  
 * On Windows, SLANG: NOT DETERMINED
 */
globus_result_t
globus_gsi_sysconfig_get_username_win32(
    char **                             username)
{
    static char *                       _function_name_ =
        "globus_gsi_sysconfig_get_username_win32";

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

#error /* SLANG: need to set the string to the username or whatever */

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    
    return GLOBUS_SUCCESS;
}
/* @} */

/**
 * @name Win32 - Get Process ID
 * @ingroup globus_i_gsi_sysconfig_win32
 */
/* @{ */
/**
 * Get a unique string representing the current process.  
 * On Windows, SLANG: NOT DETERMINED
 */
globus_result_t
globus_gsi_sysconfig_get_proc_id_string_win32(
    char **                             proc_id_string)
{
    int                                 uid;

    static char *                       _function_name_ =
        "globus_gsi_sysconfig_get_proc_id_string_win32";

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

#error /* SLANG: need to set the string to the process name or whatever */

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    
    return GLOBUS_SUCCESS;
}
/* @} */

/**
 * @name Win32 - Get Trusted CA Cert Dir
 * @ingroup globus_gsi_sysconfig_win32
 */
/* @{ */
/**
 * Get the Trusted Certificate Directory containing the trusted
 * Certificate Authority certificates.  This directory is determined
 * in the order shown below.  Failure in one method results in attempting
 * the next.
 *
 * <ol>
 * <li> <b>X509_CERT_DIR environment variable</b> - if this is set, the
 * trusted certificates will be searched for in that directory.  This
 * variable allows the end user to specify the location of trusted
 * certificates.
 * <li> <b>"x509_cert_dir" registry key</b> - If
 * this registry key is set on windows, the directory it points to should
 * contain the trusted certificates.  The path to the registry key is
 * software\Globus\GSI
 * <li> <b>\<user home directory\>\.globus\certificates</b> - If this
 * directory exists, and the previous methods of determining the trusted
 * certs directory failed, this directory will be used.  
 * <li> <b>Host Trusted Cert Dir</b> - This location is intended
 * to be independant of the globus installation ($GLOBUS_LOCATION), and 
 * is generally only writeable by the host system administrator.  
 * SLANG: This value is not currently set for WINDOWS
 * <li> <b>Globus Install Trusted Cert Dir</b> - this
 * is $GLOBUS_LOCATION\share\certificates.  
 * </ol>
 *
 * @param cert_dir
 *        The trusted certificates directory
 * @return
 *        GLOBUS_SUCCESS if no error occurred, and a sufficient trusted
 *        certificates directory was found.  Otherwise, an error object 
 *        identifier returned.
 */
globus_result_t
globus_gsi_sysconfig_get_cert_dir_win32(
    char **                             cert_dir)
{
    char *                              env_cert_dir = NULL;
    char *                              val_cert_dir[512];
    char *                              reg_cert_dir = NULL;
    char *                              local_cert_dir = NULL;
    char *                              default_cert_dir = NULL;
    char *                              installed_cert_dir = NULL;
    int                                 len;    
    HKEY                                hkDir = NULL;
    globus_result_t                     result;
    char *                              home;
    char *                              globus_location;

    static char *                       _function_name_ =
        "globus_gsi_sysconfig_get_cert_dir_win32";

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    *cert_dir = NULL;

    if(getenv(X509_CERT_DIR))
    {
        result = globus_i_gsi_sysconfig_create_cert_dir_string(
            cert_dir, 
            & env_cert_dir,
            getenv(X509_CERT_DIR));
        if(result != GLOBUS_SUCCESS)
        {
            GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                result,
                GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_DIR);
            goto error_exit;
        }
    }

    if (!(*cert_dir))
    {
        RegOpenKey(HKEY_CURRENT_USER,GSI_REGISTRY_DIR,&hkDir);
        lval = sizeof(val_cert_dir)-1;
        if (hkDir && (RegQueryValueEx(hkDir,"x509_cert_dir",0,&type,
                                      val_cert_dir,&lval) == ERROR_SUCCESS))
        {
            if((result = globus_i_gsi_sysconfig_create_cert_dir_string(
                    cert_dir, 
                    & reg_cert_dir,
                    val_cert_dir)) != GLOBUS_SUCCESS)
            {
                goto error_exit;
            }
        }
        RegCloseKey(hkDir);
    }

    /* now check for a trusted CA directory in the user's home directory */
    if(!(*cert_dir))
    {
        if((result = globus_i_gsi_sysconfig_get_home_dir(&home)) != GLOBUS_SUCCESS)
        {
            goto error_exit;
        }
            
        if (home) 
        {
            if((result = globus_i_gsi_sysconfig_create_cert_dir_string(
                    cert_dir, 
                    & local_cert_dir,
                    "%s%s%s",
                    home,
                    FILE_SEPERATOR,
                    X509_LOCAL_TRUSTED_CERT_DIR)) != GLOBUS_SUCCESS)
            {
                goto error_exit;
            }
        }
    }

    /* now look in $GLOBUS_LOCATION/share/certificates */
    if (!(*cert_dir))
    {
        if((result = globus_i_gsi_sysconfig_create_cert_dir_string(
                cert_dir,
                & installed_cert_dir,
                X509_INSTALLED_TRUSTED_CERT_DIR)) != GLOBUS_SUCCESS)
        {
            goto error_exit;
        }
    }

    /* now check for host based default directory */
    if (!(*cert_dir))
    {
        globus_location = getenv("GLOBUS_LOCATION");
        
        if (globus_location)
        {
            if((result = globus_i_gsi_sysconfig_create_cert_dir_string(
                    cert_dir,
                    & default_cert_dir,
                    "%s%s%s",
                    globus_location,
                    FILE_SEPERATOR,
                    X509_DEFAULT_TRUSTED_CERT_DIR)) != GLOBUS_SUCCESS)
            {
                goto error_exit;
            }
        }
    }

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_FPRINTF(
        1, (stderr, "Using cert_dir = %s\n", 
            (*cert_dir ? *cert_dir : "null")));

    if(!(*cert_dir))
    {
        result = globus_error_put(globus_error_construct_string(
                                      GLOBUS_GSI_SYSCONFIG_MODULE,
                                      NULL,
                                      "The trusted certificates directory could not be"
                                      "found in any of the following locations: \n"
                                      "1) env. var. X509_CERT_DIR=%s\n"
                                      "2) registry key x509_cert_dir: %s\n"
                                      "3) %s\n4) %s\n5) %s\n",
                                      env_cert_dir,
                                      reg_cert_dir,
                                      local_cert_dir,
                                      installed_cert_dir,
                                      default_cert_dir));

        goto error_exit;
    }

    result = GLOBUS_SUCCESS;
    goto done:

 error_exit:
    
    if(*cert_dir)
    {
        globus_libc_free(*cert_dir);
        *cert_dir = NULL;
    }

 done:

    if(env_cert_dir && (env_cert_dir != (*cert_dir)))
    {
        globus_libc_free(env_cert_dir);
    }
    if(reg_cert_dir && (reg_cert_dir != (*cert_dir)))
    {
        globus_libc_free(reg_cert_dir);
    }
    if(local_cert_dir && (local_cert_dir != (*cert_dir)))
    {
        globus_libc_free(local_cert_dir);
    }
    if(installed_cert_dir && (installed_cert_dir != (*cert_dir)))
    {
        globus_libc_free(installed_cert_dir);
    }
    if(default_cert_dir && (default_cert_dir != (*cert_dir)))
    {
        globus_libc_free(default_cert_dir);
    }

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;

    return result;
}
/* @} */

/**
 * @name Win32 - Get User Certificate Filename
 * @ingroup globus_gsi_sysconfig_win32
 */
/* @{ */
/**
 * Get the User Certificate Filename based on the current user's
 * environment.  The following locations are searched for cert and key
 * files in order:
 * 
 * <ol>
 * <li>environment variables X509_USER_CERT and X509_USER_KEY
 * <li>registry keys x509_user_cert and x509_user_key in software\Globus\GSI
 * <li><users home directory>\.globus\usercert.pem and 
 *     <users home directory>\.globus\userkey.pem
 * <li><users home directory\.globus\usercred.p12 - this is a PKCS12 credential
 * </ol>
 *
 * @param user_cert
 *        pointer the filename of the user certificate
 * @param user_key
 *        pointer to the filename of the user key
 * @return
 *        GLOBUS_SUCCESS if the cert and key files were found in one
 *        of the possible locations, otherwise an error object identifier
 *        is returned
 */
globus_result_t
globus_gsi_sysconfig_get_user_cert_filename_win32(
    char **                             user_cert,
    char **                             user_key)
{
    int                                 len;
    char *                              home = NULL;
    char *                              env_user_cert = NULL;
    char *                              env_user_key = NULL;
    char *                              reg_user_cert = NULL;
    char *                              reg_user_key = NULL;
    char *                              default_user_cert = NULL;
    char *                              default_user_key = NULL;
    char *                              default_pkcs12_user_cred = NULL;
    globus_result_t                     result;
    HKEY                                hkDir = NULL;
    char                                val_user_cert[512];
    char                                val_user_key[512];

    static char *                       _function_name_ =
        "globus_gsi_sysconfig_get_user_cert_filename_win32";

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    *user_cert = NULL;
    *user_key = NULL;

    /* first, check environment variables for valid filenames */

    if(getenv(X509_USER_CERT))
    {
        result = globus_i_gsi_sysconfig_create_cert_string(
            user_cert,
            &env_user_cert,
            getenv(X509_USER_CERT));
        if(result != GLOBUS_SUCCESS)
        {
            GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                result,
                GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_STRING);
            goto error_exit;
        }
    }

    if(getenv(X509_USER_KEY))
    {
        result = globus_i_gsi_sysconfig_create_cert_string(
            user_key,
            &env_user_key,
            getenv(X509_USER_KEY));
        if(result != GLOBUS_SUCCESS)
        {
            GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                result,
                GLOBUS_GSI_SYSCONFIG_ERROR_GETTIN_KEY_STRING);
            goto error_exit;
        }
    }
       
    /* next, check windows registry keys for valid filenames */

    if(!(*user_cert) || !(*user_key))
    {
        RegOpenKey(HKEY_CURRENT_USER,GSI_REGISTRY_DIR,&hkDir);
        lval = sizeof(val_user_cert)-1;
        if (hkDir && (RegQueryValueEx(
                          hkDir,
                          "x509_user_cert",
                          0,
                          &type,
                          val_user_cert,&lval) == ERROR_SUCCESS))
        {
            if((result = globus_i_gsi_sysconfig_create_cert_string(
                    user_cert,
                    & reg_user_cert,
                    val_user_cert)) != GLOBUS_SUCCESS ||
               (result = globus_i_gsi_sysconfig_create_key_string(
                   user_key,
                   & reg_user_key,
                   val_user_key)) != GLOBUS_SUCCESS)
            {
                goto error_exit;
            }
        }
        RegCloseKey(hkDir);
    }


    /* next, check default locations */
    if(!(*user_cert) || !(*user_key))
    {
        result = GLOBUS_I_GSI_SYSCONFIG_GET_HOME_DIR(&home, &status);
        if(result == GLOBUS_SUCCESS && status == GLOBUS_FILE_DIR)
        {
            result = globus_i_gsi_sysconfig_create_cert_string(
                user_cert,
                & default_user_cert,
                "%s%s%s",
                home,
                DEFEAULT_SEPERATOR,
                X509_DEFAULT_USER_CERT);
            if(result != GLOBUS_SUCCESS)
            {
                GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                    result,
                    GLOBUS_GSI_SYSCONFIG_ERROR_WITH_USER_CERT_FILENAME);
                goto error_exit;
            }

            result = globus_i_gsi_sysconfig_create_key_string(
                key_cert,
                & default_key_cert,
                "%s%s%s",
                home,
                DEFAULT_SEPERATOR,
                X509_DEFAULT_USER_KEY);
            if(result != GLOBUS_SUCCESS)
            {
                GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                    result,
                    GLOBUS_GSI_SYSCONFIG_ERROR_WITH_USER_KEY_FILENAME);
                goto error_exit;
            }
        }
    }

    /* if the cert & key don't exist in the default locations
     * or those specified by the environment variables, a
     * pkcs12 cert will be searched for
     */
    if(!(*user_cert) || !(*user_key))
    {
        if((result = globus_i_gsi_sysconfig_get_home_dir(&home)) == GLOBUS_SUCCESS)
        {
            if((result = globus_i_gsi_sysconfig_create_key_string(
                    user_key,
                    & default_pkcs12_user_cred,
                    "%s%s%s",
                    home,
                    FILE_SEPERATOR,
                    X509_DEFAULT_PKCS12_FILE)) != GLOBUS_SUCCESS)
            {
                goto error_exit;
            }
            *user_cert = *user_key;
        }
    }

    if(!(*user_cert) || !(*user_key))
    {
        result = GLOBUS_GSI_SYSCONFIG_ERROR_RESULT(
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_FILENAME,
            ("The user cert could not be found in: \n"
             "1) env. var. X509_USER_CERT=%s\n"
             "2) registry key x509_user_cert: %s\n"
             "3) %s\n4) %s\n\n"
             "The user key could not be found in:\n,"
             "1) env. var. X509_USER_KEY=%s\n"
             "2) registry key x509_user_key: %s\n"
             "3) %s\n4) %s\n",
             env_user_cert,
             reg_user_cert,
             default_user_cert,
             default_pkcs12_user_cred,
             env_user_key,
             reg_user_key,
             default_user_key,
             default_pkcs12_user_cred));

        goto error_exit;
    }

#ifdef DEBUG
    fprintf(stderr,"Using x509_user_cert=%s\n      x509_user_key =%s\n",
            (*user_cert) ? (*user_cert) : NULL, 
            (*user_key) ? (*user_key) : NULL);
#endif

    result = GLOBUS_SUCCESS;
    goto done;

 error_exit:
    
    if(*user_cert)
    {
        globus_libc_free(*user_cert);
        *user_cert = NULL;
    }
    if(*user_key)
    {
        globus_libc_free(*user_key);
        *user_key = NULL;
    }

 done:

    if(env_user_cert && env_user_cert != (*user_cert))
    {
        globus_libc_free(env_user_cert);
    }
    if(env_user_key && env_user_key != (*user_key))
    {
        globus_libc_free(env_user_key);
    }
    if(default_user_cert && default_user_cert != (*user_cert))
    {
        globus_libc_free(default_user_cert);
    }
    if(default_user_key && default_user_key != (*user_key))
    {
        globus_libc_free(default_user_key);
    }
    
    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    
    return result;
}
/* @} */

/**
 * @name Win32 - Get Host Certificate and Key Filenames
 * @ingroup globus_gsi_sysconfig_win32
 */
/* @{ */
/**
 * Get the Host Certificate and Key Filenames based on the current user's
 * environment.  The host cert and key are searched for in the following 
 * locations (in order):
 *
 * <ol>
 * <li>X509_USER_CERT and X509_USER_KEY environment variables
 * <li>registry keys x509_user_cert and x509_user_key in software\Globus\GSI
 * <li>SLANG: NOT DETERMINED - this is the default location
 * <li><GLOBUS_LOCATION>\etc\host[cert|key].pem
 * <li><users home directory>\.globus\host[cert|key].pem
 * </ol>
 * 
 * @param host_cert
 *        pointer to the host certificate filename
 * @param host_key
 *        pointer to the host key filename
 *
 * @return
 *        GLOBUS_SUCCESS if the host cert and key were found, otherwise
 *        an error object identifier is returned 
 */
globus_result_t
globus_gsi_sysconfig_get_host_cert_filename_win32(
    char **                             host_cert,
    char **                             host_key)
{
    int                                 len;
    char *                              home = NULL;
    char *                              host_cert = NULL;
    char *                              host_key = NULL;
    char *                              env_host_cert = NULL;
    char *                              env_host_key = NULL;
    char *                              reg_host_cert = NULL;
    char *                              reg_host_key = NULL;
    char *                              default_host_cert = NULL;
    char *                              default_host_key = NULL;
    char *                              installed_host_cert = NULL;
    char *                              installed_host_key = NULL;
    char *                              local_host_cert = NULL;
    char *                              local_host_key = NULL;
    globus_result_t                     result;

    HKEY                                hkDir = NULL;
    char                                val_host_cert[512];
    char                                val_host_key[512];

    static char *                       _function_name_ =
        "globus_gsi_sysconfig_host_cert_filename_win32";

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    *host_cert = NULL;
    *host_key = NULL;

    /* first check environment variables for valid filenames */

    if(getenv(X509_USER_CERT))
    {
        result = globus_i_gsi_sysconfig_create_cert_string(
            host_cert,
            & env_host_cert,
            getenv(X509_USER_CERT));
        if(result != GLOBUS_SUCCESS)
        {
            GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                result,
                GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_STRING);
            goto error_exit;
        }
    }

    if(getenv(X509_USER_KEY))
    {
        result = globus_i_gsi_sysconfig_create_key_string(
            host_key,
            & env_host_key,
            getenv(X509_USER_KEY));
        if(result != GLOBUS_SUCCESS)
        {
            GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                result,
                GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_KEY_STRING);
            goto error_exit;
        }
    }

    /* now check the windows registry for valid filenames */
    if(!(*host_cert) || !(*host_key))
    {
        RegOpenKey(HKEY_CURRENT_USER,GSI_REGISTRY_DIR,&hkDir);
        lval = sizeof(val_host_cert)-1;
        if (hkDir && (RegQueryValueEx(hkDir,
                                      "x509_user_cert",
                                      0,
                                      &type,
                                      val_host_cert,
                                      &lval) == ERROR_SUCCESS))
        {
            if((result = globus_i_gsi_sysconfig_create_cert_string(
                    host_cert,
                    & reg_host_cert,
                    val_host_cert)) != GLOBUS_SUCCESS ||
               (result = globus_i_gsi_sysconfig_create_cert_string(
                   host_key,
                   & reg_host_key,
                   val_host_key)) != GLOBUS_SUCCESS)
            {
                goto error_exit;
            }
        }
        RegCloseKey(hkDir);
    }

    /* now check default locations for valid filenames */
    if(!(*host_cert) || !(*host_key))
    {
        if((result = globus_i_gsi_sysconfig_get_home_dir(&home)) == GLOBUS_SUCCESS)
        {
            if((result = globus_i_gsi_sysconfig_create_cert_string(
                    host_cert,
                    & default_host_cert,
                    "%s%s%s%s",
                    X509_DEFAULT_CERT_DIR,
                    FILE_SEPERATOR,
                    X509_HOST_PREFIX,
                    X509_CERT_SUFFIX)) != GLOBUS_SUCCESS ||
               (result = globus_i_gsi_sysconfig_create_key_string(
                   host_key,
                   & default_key_cert,
                   "%s%s%s%s",
                   X509_DEFAULT_CERT_DIR,
                   FILE_SEPERATOR,
                   X509_HOST_PREFIX,
                   X509_KEY_SUFFIX)) != GLOBUS_SUCCESS)
            {
                goto error_exit;
            }
        }
    }

    /* now check intstalled location for host cert */
    if(!(*host_cert) || !(*host_key))
    {
        globus_location = getenv("GLOBUS_LOCATION");

        if(globus_location)
        {
            if((result = globus_i_gsi_sysconfig_create_cert_string(
                    host_cert,
                    & installed_host_cert,
                    "%s%s%s%s%s%s",
                    globus_location,
                    FILE_SEPERATOR,
                    X509_INSTALLED_CERT_DIR,
                    FILE_SEPERATOR,
                    X509_HOST_PREFIX,
                    X509_CERT_SUFFIX)) != GLOBUS_SUCCESS ||
               (result = globus_i_gsi_sysconfig_create_key_string(
                   host_key,
                   & installed_host_key,
                   "%s%s%s%s%s%s",
                   globus_location,
                   FILE_SEPERATOR,
                   X509_INSTALLED_CERT_DIR,
                   FILE_SEPERATOR,
                   X509_HOST_PREFIX,
                   X509_KEY_SUFFIX)) != GLOBUS_SUCCESS)
            {
                goto error_exit;
            }
        }
    }

    if(!(*host_cert) || !(*host_key))
    {
        if(GLOBUS_I_GSI_SYSCONFIG_GET_HOME_DIR(&home) == GLOBUS_SUCCESS)
        {
            if((result = globus_i_gsi_sysconfig_create_cert_string(
                    host_cert,
                    & local_host_cert,
                    "%s%s%s%s%s%s",
                    home,
                    FILE_SEPERATOR,
                    X509_LOCAL_CERT_DIR,
                    FILE_SEPERATOR,
                    X509_HOST_PREFIX,
                    X509_CERT_SUFFIX)) != GLOBUS_SUCCESS ||
               (result = globus_i_gsi_sysconfig_create_key_string(
                   host_key,
                   & local_key_cert,
                   "%s%s%s%s%s%s",
                   home,
                   FILE_SEPERATOR,
                   X509_LOCAL_CERT_DIR,
                   FILE_SEPERATOR,
                   X509_HOST_PREFIX,
                   X509_KEY_SUFFIX)) != GLOBUS_SUCCESS)
            {
                goto error_exit;
            }
        }
    }

#ifdef DEBUG
    fprintf(stderr,"Using x509_user_cert=%s\n      x509_user_key =%s\n",
            host_cert, host_key);
#endif

    if(!(*host_cert) || !(*host_key))
    {
        result = GLOBUS_GSI_SYSCONFIG_ERROR_RESULT(
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_FILENAME,
            ("The user cert could not be found in: \n"
             "1) env. var. X509_USER_CERT=%s\n"
             "2) registry key x509_user_cert: %s\n"
             "3) %s\n4) %s5) %s\n\n"
             "The user key could not be found in:\n,"
             "1) env. var. X509_USER_KEY=%s\n"
             "2) registry key x509_user_key: %s\n"
             "3) %s\n4) %s5) %s\n",
             env_host_cert,
             reg_host_cert,
             default_host_cert,
             installed_host_cert,
             local_host_cert,
             env_host_key,
             reg_host_key,
             default_host_key,
             installed_host_key,
             local_host_key));

        goto error_exit;
    }

    result = GLOBUS_SUCCESS;
    goto done;

 error_exit:

    if(*host_cert)
    {
        globus_libc_free(*host_cert);
        *host_cert = NULL;
    }
    if(*host_key)
    {
        globus_libc_free(*host_key);
        *host_key = NULL;
    }

 done:

    if(env_host_cert && env_host_cert != *host_cert)
    {
        globus_libc_free(env_host_cert);
    }
    if(env_host_key && env_host_key != *host_key)
    {
        globus_libc_free(env_host_key);
    }
    if(reg_host_cert && reg_host_cert != *host_cert)
    {
        globus_libc_free(reg_host_cert);
    }
    if(reg_host_key && reg_host_key != *host_key)
    {
        globus_libc_free(reg_host_key);
    }
    if(installed_host_cert && installed_host_cert != *host_cert)
    {
        globus_libc_free(installed_host_cert);
    }
    if(installed_host_key && installed_host_key != *host_key)
    {
        globus_libc_free(installed_host_key);
    }
    if(local_host_cert && local_host_cert != *host_cert)
    {
        globus_libc_free(local_host_cert);
    }
    if(local_host_key && local_host_key != *host_key)
    {
        globus_libc_free(local_host_key);
    }
    if(default_host_cert && default_host_cert != host_cert)
    {
        globus_libc_free(default_host_cert);
    }
    if(default_host_key && default_host_key != host_key)
    {
        globus_libc_free(default_host_key);
    }

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;

    return result;
}
/* @} */

/**
 * @name Win32 - Get Service Certificate and Key Filenames
 * @ingroup globus_gsi_sysconfig_win32
 */
/* @{ */
/**
 * Get the Service Certificate Filename based on the current user's
 * environment.  The host cert and key are searched for in the following 
 * locations (in order):
 *
 * <ol>
 * <li>X509_USER_CERT and X509_USER_KEY environment variables
 * <li>registry keys x509_user_cert and x509_user_key in software\Globus\GSI
 * <li>SLANG: NOT DETERMINED - this is the default location
 * <li>GLOBUS_LOCATION\etc\{service_name}\{service_name}[cert|key].pem
 *     So for example, if my service was named: myservice, the location
 *     of the certificate would be: 
 *     GLOBUS_LOCATION\etc\myservice\myservicecert.pem
 * <li><users home>\.globus\{service_name}\{service_name}[cert|key].pem
 * </ol>
 * 
 * @param service_name
 *        The name of the service which allows us to determine the
 *        locations of cert and key files to look for
 * @param service_cert
 *        pointer to the host certificate filename
 * @param service_key
 *        pointer to the host key filename
 *
 * @return
 *        GLOBUS_SUCCESS if the service cert and key were found, otherwise
 *        an error object identifier 
 */
globus_result_t
globus_gsi_sysconfig_get_service_cert_filename_win32(
    char *                              service_name,
    char **                             service_cert_filename,
    char **                             service_key_filename)
{
    int                                 len;
    char *                              home = NULL;
    char *                              service_cert = NULL;
    char *                              service_key = NULL;
    char *                              env_service_cert = NULL;
    char *                              env_service_key = NULL;
    char *                              reg_service_cert = NULL;
    char *                              reg_service_key = NULL;
    char *                              default_service_cert = NULL;
    char *                              default_service_key = NULL;
    char *                              installed_service_cert = NULL;
    char *                              installed_service_key = NULL;
    char *                              local_service_cert = NULL;
    char *                              local_service_key = NULL;
    globus_result_t                     result;

    HKEY                                hkDir = NULL;
    char                                val_service_cert[512];
    char                                val_service_key[512];

    static char *                       _function_name_ =
        "globus_gsi_sysconfig_get_service_cert_filename_win32";

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    *service_cert = NULL;
    *service_key = NULL;

    /* first check environment variables for valid filenames */

    if(getenv(X509_USER_CERT))
    {
        result = globus_i_gsi_sysconfig_create_cert_string(
            service_cert,
            &env_service_cert,
            getenv(X509_USER_CERT));
        if(result != GLOBUS_SUCCESS)
        {
            GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                result,
                GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_STRING);
            goto error_exit;
        }
    }

    if(getenv(X509_USER_KEY))
    {
        result = globus_i_gsi_sysconfig_create_key_string(
            service_key,
            &env_service_key,
            getenv(X509_USER_KEY));
        if(result != GLOBUS_SUCCESS)
        {
            GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                result,
                GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_KEY_STRING);
            goto error_exit;
        }
    }

    /* now check the windows registry for valid filenames */
    if(!(*service_cert) || !(*service_key))
    {
        RegOpenKey(HKEY_CURRENT_USER,GSI_REGISTRY_DIR,&hkDir);
        lval = sizeof(val_service_cert)-1;
        if (hkDir && (RegQueryValueEx(hkDir,
                                      "x509_user_cert",
                                      0,
                                      &type,
                                      val_service_cert,
                                      &lval) == ERROR_SUCCESS))
        {
            if((result = globus_i_gsi_sysconfig_create_cert_string(
                    service_cert,
                    & reg_service_cert,
                    val_service_cert)) != GLOBUS_SUCCESS ||
               (result = globus_i_gsi_sysconfig_create_cert_string(
                   service_key,
                   & reg_service_key,
                   val_service_key)) != GLOBUS_SUCCESS)
            {
                goto error_exit;
            }
        }
        RegCloseKey(hkDir);
    }


    /* now check default locations for valid filenames */
    if(!(*service_cert) || !(*service_key))
    {
        if((result = globus_i_gsi_sysconfig_get_home_dir(&home)) == GLOBUS_SUCCESS)
        {
            if((result = globus_i_gsi_sysconfig_create_cert_string(
                    service_cert,
                    & default_service_cert,
                    "%s%s%s%s%s%s",
                    X509_DEFAULT_CERT_DIR,
                    FILE_SEPERATOR,
                    service_name,
                    FILE_SEPERATOR,
                    service_name,
                    X509_CERT_SUFFIX)) != GLOBUS_SUCCESS ||
               (result = globus_i_gsi_sysconfig_create_key_string(
                   service_key,
                   & default_key_cert,
                   "%s%s%s%s%s%s",
                   X509_DEFAULT_CERT_DIR,
                   FILE_SEPERATOR,
                   service_name,
                   FILE_SEPERATOR,
                   service_name,
                   X509_KEY_SUFFIX)) != GLOBUS_SUCCESS)
            {
                goto error_exit;
            }
        }
    }

    /* now check intstalled location for service cert */
    if(!(*service_cert) || !(*service_key))
    {
        globus_location = getenv("GLOBUS_LOCATION");

        if(globus_location)
        {
            if((result = globus_i_gsi_sysconfig_create_cert_string(
                    service_cert,
                    & installed_service_cert,
                    "%s%s%s%s%s%s%s%s",
                    globus_location,
                    FILE_SEPERATOR,
                    X509_INSTALLED_CERT_DIR,
                    FILE_SEPERATOR,
                    service_name,
                    FILE_SEPERATOR,
                    service_name,
                    X509_CERT_SUFFIX)) != GLOBUS_SUCCESS ||
               (result = globus_i_gsi_sysconfig_create_key_string(
                   service_key,
                   & installed_service_key,
                   "%s%s%s%s%s%s%s%s",
                   globus_location,
                   FILE_SEPERATOR,
                   X509_INSTALLED_CERT_DIR,
                   FILE_SEPERATOR,
                   service_name,
                   FILE_SEPERATOR,
                   service_name,
                   X509_KEY_SUFFIX)) != GLOBUS_SUCCESS)
            {
                goto error_exit;
            }
        }
    }

    if(!(*service_cert) || !(*service_key))
    {
        if (home) {
            free(home);
            home = NULL;
        }
        if(GLOBUS_I_GSI_SYSCONFIG_GET_HOME_DIR(&home) == GLOBUS_SUCCESS)
        {
            if((result = globus_i_gsi_sysconfig_create_cert_string(
                    service_cert,
                    & local_service_cert,
                    "%s%s%s%s%s%s%s",
                    home,
                    FILE_SEPERATOR,
                    X509_LOCAL_CERT_DIR,
                    FILE_SEPERATOR,
                    service_name,
                    FILE_SEPERATOR,
                    service_name,
                    X509_CERT_SUFFIX)) != GLOBUS_SUCCESS ||
               (result = globus_i_gsi_sysconfig_create_key_string(
                   service_key,
                   & local_key_cert,
                   "%s%s%s%s%s%s%s%s",
                   home,
                   FILE_SEPERATOR,
                   X509_LOCAL_CERT_DIR,
                   FILE_SEPERATOR,
                   service_name,
                   FILE_SEPERATOR,
                   service_name,
                   X509_KEY_SUFFIX)) != GLOBUS_SUCCESS)
            {
                goto error_exit;
            }
        }
    }

#ifdef DEBUG
    fprintf(stderr,"Using x509_user_cert=%s\n      x509_user_key =%s\n",
            service_cert, service_key);
#endif

    if(!(*service_cert) || !(*service_key))
    {
        result = GLOBUS_GSI_SYSCONFIG_ERROR_RESULT(
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_FILENAME,
            ("The user cert could not be found in: \n"
             "1) env. var. X509_USER_CERT=%s\n"
             "2) registry key x509_user_cert: %s\n"
             "3) %s\n4) %s5) %s\n\n"
             "The user key could not be found in:\n,"
             "1) env. var. X509_USER_KEY=%s\n"
             "2) registry key x509_user_key: %s\n"
             "3) %s\n4) %s5) %s\n",
             env_service_cert,
             reg_service_cert,
             default_service_cert,
             installed_service_cert,
             local_service_cert,
             env_service_key,
             reg_service_key,
             default_service_key,
             installed_service_key,
             local_service_key));

        goto error_exit;
    }

    result = GLOBUS_SUCCESS;
    goto done;

 error_exit:

    if(*service_cert)
    {
        globus_libc_free(*service_cert);
        *service_cert = NULL;
    }
    if(*service_key)
    {
        globus_libc_free(*service_key);
        *service_key = NULL;
    }

 done:

    if(env_service_cert && env_service_cert != *service_cert)
    {
        globus_libc_free(env_service_cert);
    }
    if(env_service_key && env_service_key != *service_key)
    {
        globus_libc_free(env_service_key);
    }
    if(reg_service_cert && reg_service_cert != *service_cert)
    {
        globus_libc_free(reg_service_cert);
    }
    if(reg_service_key && reg_service_key != *service_key)
    {
        globus_libc_free(reg_service_key);
    }
    if(installed_service_cert && installed_service_cert != *service_cert)
    {
        globus_libc_free(installed_service_cert);
    }
    if(installed_service_key && installed_service_key != *service_key)
    {
        globus_libc_free(installed_service_key);
    }
    if(local_service_cert && local_service_cert != *service_cert)
    {
        globus_libc_free(local_service_cert);
    }
    if(local_service_key && local_service_key != *service_key)
    {
        globus_libc_free(local_service_key);
    }
    if(default_service_cert && default_service_cert != service_cert)
    {
        globus_libc_free(default_service_cert);
    }
    if(default_service_key && default_service_key != service_key)
    {
        globus_libc_free(default_service_key);
    }
    if (home) {
        free(home);
        home = NULL;
    }

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;

    return result;
}
/* @} */

/**
 * @name Win32 - Get Proxy Filename
 * @ingroup globus_gsi_sysconfig_win32
 */
/* @{ */
/**
 * Get the proxy cert filename based on the following
 * search order:
 * 
 * <ol>
 * <li> X509_USER_PROXY environment variable - This environment variable
 * is set by the at run time for the specific application.  If
 * the proxy_file_type variable is set to GLOBUS_PROXY_OUTPUT
 *  (a proxy filename for writing is requested), 
 * and the X509_USER_PROXY is set, this will be the 
 * resulting value of the user_proxy filename string passed in.  If the
 * proxy_file_type is set to GLOBUS_PROXY_INPUT and X509_USER_PROXY is 
 * set, but the file it points to does not exist, 
 * or has some other readability issues, the 
 * function will continue checking using the other methods available.
 * 
 * <li> check the registry key: x509_user_proxy.  Just as with
 * the environment variable, if the registry key is set, and proxy_file_type
 * is GLOBUS_PROXY_OUTPUT, the string set to be the proxy 
 * filename will be this registry
 * key's value.  If proxy_file_type is GLOBUS_PROXY_INPUT, and the 
 * file doesn't exist, the function will check the next method 
 * for the proxy's filename.
 * 
 * <li> Check the default location for the proxy file.  The default
 * location should be 
 * set to reside in the temp directory on that host, with the filename
 * taking the format:  x509_u<user id>
 * where <user id> is some unique string for that user on the host
 * </ol>
 *
 * @param user_proxy
 *        the proxy filename of the user
 *
 * @return
 *        GLOBUS_SUCCESS or an error object identifier
 */
globus_result_t
globus_gsi_sysconfig_get_proxy_filename_win32(
    char **                             user_proxy,
    globus_gsi_proxy_file_type_t        proxy_file_type)
{
    char *                              env_user_proxy = NULL;
    char *                              env_value = NULL;
    char *                              default_user_proxy = NULL;
    char *                              reg_user_proxy = NULL;
    HKEY                                hkDir = NULL;
    char                                val_user_proxy[512];
    int                                 len;
    globus_result_t                     result;
    char *                              user_id_string;

    static char *                       _function_name_ =
        "globus_gsi_sysconfig_get_proxy_filename_win32";
    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    *user_proxy = NULL;

    if((env_value = getenv(X509_USER_PROXY)) != NULL &&
       (result = globus_i_gsi_sysconfig_create_key_string(
           user_proxy,
           & env_user_proxy,
           getenv(X509_USER_PROXY))) != GLOBUS_SUCCESS)
    {
        goto error_exit;
    }
    
    /* check if the proxy file type is for writing */
    if(!(*user_proxy) && env_user_proxy && proxy_file == GLOBUS_PROXY_OUTPUT)
    {
        *user_proxy = env_user_proxy;
    }

    if (!(*user_proxy))
    {
        RegOpenKey(HKEY_CURRENT_USER,GSI_REGISTRY_DIR,&hkDir);
        lval = sizeof(val_user_proxy)-1;
        if (hkDir && (RegQueryValueEx(hkDir, "x509_user_proxy", 0, &type,
                                      val_user_proxy, &lval) == ERROR_SUCCESS))
        {
            if((result = globus_i_gsi_sysconfig_create_key_string(
                    proxy_cert,
                    & reg_user_proxy,
                    val_user_proxy)) != GLOBUS_SUCCESS)
            {
                goto error_exit;
            }
        }
        RegCloseKey(hkDir);
    }

    if(!(*user_proxy) && reg_user_proxy && proxy_file == GLOBUS_PROXY_OUTPUT)
    {
        *user_proxy = reg_user_proxy;
    }

    if (!user_proxy)
    {
        if((result = GLOBUS_GSI_SYSCONFIG_GET_USER_ID_STRING(
                &user_id_string))
           != GLOBUS_SUCCESS)
        {
            goto error_exit;
        }
        if((result = globus_i_gsi_sysconfig_create_key_string(
                user_proxy,
                & default_user_proxy,
                "%s%s%s%s",
                DEFAULT_SECURE_TMP_DIR,
                FILE_SEPERATOR,
                X509_USER_PROXY_FILE,
                user_id_string)) != GLOBUS_SUCCESS)
        {
            goto error_exit;
        }
    }

    if(!(*user_proxy) && 
       default_user_proxy && 
       proxy_file_type == GLOBUS_PROXY_FILE_OUTPUT)
    {
        *user_proxy = default_user_proxy;
    }

    if(!(*user_proxy))
    {            
        result = GLOBUS_GSI_SYSCONFIG_ERROR_RESULT( 
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_PROXY_FILENAME,
            ("A file location for%s the proxy cert could be found in: \n"
             "1) env. var. X509_USER_PROXY=%s\n"
             "2) registry key x509_user_proxy: %s\n"
             "3) %s\n",
             (proxy_file_type == GLOBUS_PROXY_FILE_INPUT) ? "" : " writing",
             env_user_proxy,
             reg_user_proxy,
             default_user_proxy));
        
        goto error_exit;
    }

    result = GLOBUS_SUCCESS;
    goto done;

 error_exit:
    
    if(*user_proxy)
    {
        globus_libc_free(*user_proxy);
        *user_proxy = NULL;
    }

 done:

    if(reg_user_proxy && (reg_user_proxy != (*user_proxy)))
    {
        globus_libc_free(reg_user_proxy);
    }
    if(default_user_proxy && (default_user_proxy != (*default_user_proxy)))
    {
        globus_libc_free(default_user_proxy);
    }
    
    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;
}
/* @} */

/**
 * @name Win32 - Get CA Cert Filenames
 * @ingroup globus_gsi_sysconfig_win32
 */
/* @{ */
/**
 * Gets a list of trusted CA certificate filenames in 
 * a trusted CA certificate directory.  
 *
 * @param ca_cert_dir
 *        The trusted CA certificate directory to get the filenames from
 * @param ca_cert_list
 *        The resulting list of CA certificate filenames.  This is a
 *        a globus list structure.  
 *        @see globus_fifo_t
 * @return
 *        GLOBUS_SUCCESS if no error occurred, otherwise an error object ID
 *        is returned
 */
globus_result_t
globus_gsi_sysconfig_get_ca_cert_files_win32(
    char *                              ca_cert_dir,
    globus_fifo_t *                     ca_cert_list)
{
    globus_result_t                     result = GLOBUS_SUCCESS;
    static char *                       _function_name_ =
        "globus_gsi_sysconfig_get_ca_cert_files_win32";

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

#error SLANG: need to fill this in

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;
}
/* @} */

/**
 * @name Win32 - Remove all proxies owned by current uid
 * @ingroup globus_gsi_sysconfig_win32
 */
/* @{ */
/**
 * Removes all proxies (ie. all delegated and grid-proxy-init generated
 * proxies) found in the secure tmp directory that are owned by the
 * current user.
 *
 * @param default_filename
 *        The filename of the default proxy
 * @return
 *        GLOBUS_SUCCESS if no error occurred, otherwise an error object ID
 *        is returned
 */
globus_result_t
globus_gsi_sysconfig_remove_all_owned_files_win32(
    char *                              default_filename)
{
    globus_result_t                     result = GLOBUS_SUCCESS;
    static char *                       _function_name_ =
        "globus_gsi_sysconfig_remove_all_owned_files_win32";

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;
    
#error SLANG: need to fill this in

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;
}
/* @} */

/**
 * @name Win32 - Get the path and file name of the grid map file
 * @ingroup globus_gsi_sysconfig_win32
 */
/* @{ */
/**
 * Get the path and file name of the grid map file.
 *
 * @param filename
 *        Contains the location of the grid map file upon successful return
 * @return
 *        GLOBUS_SUCCESS if no error occurred, otherwise an error object ID
 *        is returned
 */
globus_result_t
globus_gsi_sysconfig_get_gridmap_filename_win32(
    char **                             filename)
{
    globus_result_t                     result = GLOBUS_SUCCESS;
    static char *                       _function_name_ =
        "globus_gsi_sysconfig_get_gridmap_filename_win32";
    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

#error SLANG: need to fill this in

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;
}
/* @} */

/* END WIN32 SYSCONFIG DEFINITIONS */

#else

/* BEGIN UNIX SYSCONFIG DEFINITIONS */

/**
 * @name UNIX - Set Key Permissions
 * @ingroup globus_gsi_sysconfig_unix
 */
/* @{ */
/**
 * Set the file permissions of a file to read only by the user
 * which are the permissions that should be set for all private keys.
 *
 * @param filename
 *
 * @return
 *        GLOBUS_SUCCESS or an error object id
 */
globus_result_t
globus_gsi_sysconfig_set_key_permissions_unix(
    char *                              filename)
{
    globus_result_t                     result = GLOBUS_SUCCESS;
    globus_gsi_statcheck_t              status;
    static char *                       _function_name_ =
        "globus_gsi_sysconfig_set_key_permissions_unix";
    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    result = globus_gsi_sysconfig_file_exists_unix(
        filename,
        &status);
    if(result != GLOBUS_SUCCESS)
    {
        GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_SETTING_PERMS);
        goto exit;
    }

    if(status != GLOBUS_FILE_VALID &&
       status != GLOBUS_FILE_DIR)
    {
        GLOBUS_GSI_SYSCONFIG_ERROR_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_SETTING_PERMS,
            ("Error setting permissions of file: %s %s", 
             filename,
             globus_l_gsi_sysconfig_status_strings[status]));
        goto exit;
    }

    if(chmod(filename, S_IRUSR|S_IWUSR) < 0)
    {
        result = globus_error_put(
            globus_error_wrap_errno_error(
                GLOBUS_GSI_SYSCONFIG_MODULE,
                errno,
                GLOBUS_GSI_SYSCONFIG_ERROR_SETTING_PERMS,
                __FILE__":%d:%s: Error setting permissions to "
                "user read only of file: %s\n", 
                __LINE__,
                _function_name_,
                filename));
        goto exit;
    }

 exit:

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;
}
/* @} */

/**
 * @name UNIX - Get User ID
 * @ingroup globus_gsi_sysconfig_unix
 */
/* @{ */
/**
 * Get a unique string representing the current user.  This is just
 * the uid converted to a string.  
 *
 * @param user_id_string
 *        A unique string representing the user
 *
 * @return
 *        GLOBUS_SUCCESS unless an error occurred
 */
globus_result_t
globus_gsi_sysconfig_get_user_id_string_unix(
    char **                             user_id_string)
{
    uid_t                               uid;
    int                                 len;
    globus_result_t                     result;

    static char *                       _function_name_ =
        "globus_gsi_sysconfig_get_user_id_string_unix";

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    uid = getuid();
    
    len = globus_libc_printf_length("%d",uid);

    len++;

    if((*user_id_string = malloc(len)) == NULL)
    {
        result = GLOBUS_GSI_SYSTEM_CONFIG_MALLOC_ERROR;
        goto exit;
    }
    
    globus_libc_snprintf(*user_id_string,len,"%d",uid);

    result = GLOBUS_SUCCESS;

 exit:

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;
}
/* @} */

/**
 * @name UNIX - Get Username
 * @ingroup globus_gsi_sysconfig_unix
 */
/* @{ */
/**
 * Get the username of the current user.
 *
 * @param username
 *        This parameter will contain the current user name upon a successful
 *        return. It is the users responsibility to free memory allocated for
 *        this return value.
 * @return
 *        GLOBUS_SUCCESS unless an error occurred
 */
globus_result_t
globus_gsi_sysconfig_get_username_unix(
    char **                             username)
{
    globus_result_t                     result = GLOBUS_SUCCESS;
    struct passwd                       pwd;
    struct passwd *                     pwd_result;
    char *                              buf;
    int                                 buf_len;
    static char *                       _function_name_ =
        "globus_gsi_sysconfig_get_username_unix";

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    /* the below seems to be fairly portable */
#ifdef _SC_GETPW_R_SIZE_MAX
    buf_len = sysconf(_SC_GETPW_R_SIZE_MAX) + 1;
#else
    buf_len = 1024;
#endif

    buf = malloc(buf_len);

    if(buf == NULL)
    {
        result = GLOBUS_GSI_SYSTEM_CONFIG_MALLOC_ERROR;
        goto exit;
    }
    
    if(globus_libc_getpwuid_r(geteuid(),
                              &pwd,
                              buf,
                              buf_len,
                              &pwd_result) != 0)
    {
        GLOBUS_GSI_SYSCONFIG_ERROR_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_PW_ENTRY,
            ("Error occured for uid: %d",geteuid()));        
        goto exit;
    }

    if(pwd_result == NULL || pwd_result->pw_name == NULL)
    {
        GLOBUS_GSI_SYSCONFIG_ERROR_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_PW_ENTRY,
            ("Error occured for uid: %d",geteuid()));        
        goto exit;        
    }

    *username = malloc(strlen(pwd_result->pw_name) + 1);

    if(!*username)
    {
        result = GLOBUS_GSI_SYSTEM_CONFIG_MALLOC_ERROR;
        goto exit;
    }
        
    strncpy(*username, pwd_result->pw_name, 
            strlen(pwd_result->pw_name) + 1);
    
 exit:

    if(buf != NULL)
    {
        free(buf);
    }
    
    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;
}
/* @} */

/**
 * @name UNIX - Get Process ID
 * @ingroup globus_gsi_sysconfig_unix
 */
/* @{ */
/**
 * Get a unique string representing the current process.  This is just
 * the pid converted to a string.  
 *
 * @param proc_id_string
 *        A unique string representing the process
 *
 * @return
 *        GLOBUS_SUCCESS unless an error occurred
 */
globus_result_t
globus_gsi_sysconfig_get_proc_id_string_unix(
    char **                             proc_id_string)
{
    pid_t                               pid;
    int                                 len;
    globus_result_t                     result;

    static char *                       _function_name_ =
        "globus_gsi_sysconfig_get_proc_id_string_unix";

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    pid = getpid();
    
    len = globus_libc_printf_length("%d",pid);

    len++;

    if((*proc_id_string = malloc(len)) == NULL)
    {
        result = GLOBUS_GSI_SYSTEM_CONFIG_MALLOC_ERROR;
        goto exit;
    }

    globus_libc_snprintf(*proc_id_string,len,"%d",pid);
    
    result = GLOBUS_SUCCESS;

 exit:

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;
}
/* @} */


/**
 * @name UNIX - Make Absolute Path
 * @ingroup globus_gsi_sysconfig_unix
 */
/* @{ */
/**
 * Make the filename into an absolute path string based
 * on the current working directory.
 *
 * @param filename
 *        the filename to get the absolute path of.  
 * @param absolute_path
 *        The resulting absolute path.  This needs to 
 *        be freed when no longer needed.
 * @return
 *        GLOBUS_SUCCESS if no error occurred, otherwise
 *        an error object ID is returned
 */
globus_result_t
globus_gsi_sysconfig_make_absolute_path_for_filename_unix(
    char *                              filename,
    char **                             absolute_path)
{
    int                                 length;
    char *                              cwd = NULL;
    globus_result_t                     result = GLOBUS_SUCCESS;
    static char *                       _function_name_ =
        "globus_gsi_sysconfig_make_absolute_path_for_filename_unix";
    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    if(filename[0] != '/')
    {
        result = GLOBUS_GSI_SYSCONFIG_GET_CURRENT_WORKING_DIR(&cwd);
        if(result != GLOBUS_SUCCESS)
        {
            cwd = NULL;
            GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                result,
                GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CWD);
            goto exit;
        }

        length = strlen(cwd) + strlen(filename) + 2;

        *absolute_path = malloc(length);
        if(!*absolute_path)
        {
            result = GLOBUS_GSI_SYSTEM_CONFIG_MALLOC_ERROR;
            goto exit;
        }
        globus_libc_snprintf(*absolute_path, length, "%s/%s", cwd, filename);
    }
    else
    {
        length = strlen(filename) + 1;

        *absolute_path = malloc(length);
        if(!*absolute_path)
        {
            result = GLOBUS_GSI_SYSTEM_CONFIG_MALLOC_ERROR;
            goto exit;
        }
        globus_libc_snprintf(*absolute_path, length, "%s", filename);
    }

 exit:

    if(cwd != NULL)
    {
        free(cwd);
    }
    
    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;
}
/* @} */


/**
 * @name UNIX - Split Directory and Filename
 * @ingroup globus_gsi_sysconfig_unix
 */
/* @{ */
/**
 * Split the directory and filename portions of a filename string
 * into two separate strings
 *
 * @param full_filename
 *        The filename to split.  Splits on the last occurrance of '/'
 *        where the directory is everything before the last '/', and
 *        the filename is everything after.
 * @param dir_string  
 *        The directory portion of the filename string.  If no '/' is found
 *        throughout the string, this variable points to NULL.
 *        This needs to be freed when no longer needed.
 * @param filename_string
 *        The filename portion of the filename string.  If no '/' is found
 *        throughout, this variable is a duplicate of the full_filename 
 *        parameter.  This needs to be freed when no longer needed.
 *
 * @return
 *        GLOBUS_SUCCESS if no error occurred.  Otherwise an error object ID
 *        is returned.
 */
globus_result_t
globus_gsi_sysconfig_split_dir_and_filename_unix(
    char *                              full_filename,
    char **                             dir_string,
    char **                             filename_string)
{
    int                                 dir_string_length;
    int                                 filename_string_length;
    char *                              split_index = NULL;
    globus_result_t                     result = GLOBUS_SUCCESS;
    static char *                       _function_name_ =
        "globus_gsi_sysconfig_split_dir_and_filename_unix";
    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    *dir_string = NULL;
    *filename_string = NULL;

    split_index = strrchr(full_filename, '/');
    if(!split_index)
    {
        *dir_string = NULL;
        filename_string_length = strlen(full_filename) + 1;
        *filename_string = malloc(filename_string_length);
        if(!*filename_string)
        {
            result = GLOBUS_GSI_SYSTEM_CONFIG_MALLOC_ERROR;
            goto exit;
        }

        globus_libc_snprintf(*filename_string, filename_string_length, 
                             "%s", full_filename); 
    }
    else
    {
        dir_string_length = split_index - full_filename + 1;
        
        *dir_string = malloc(dir_string_length);
        
        if(!*dir_string)
        {
            result = GLOBUS_GSI_SYSTEM_CONFIG_MALLOC_ERROR;
            goto exit;
        }
        
        globus_libc_snprintf(*dir_string,
                             dir_string_length, "%s", full_filename);
        
        filename_string_length = strlen(full_filename) - dir_string_length + 1;
        
        *filename_string = malloc(filename_string_length);
        
        if(!*filename_string)
        {
            result = GLOBUS_GSI_SYSTEM_CONFIG_MALLOC_ERROR;
            if(*dir_string)
            {
                free(*dir_string);
            }
            goto exit;
        }
        
        globus_libc_snprintf(*filename_string,
                             filename_string_length, "%s",
                             &full_filename[dir_string_length]);
    }

 exit:
    
    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;
}
/* @} */


/**
 * @name UNIX - Get Current Working Directory
 * @ingroup globus_gsi_sysconfig_unix
 */
/* @{ */
/**
 * Get the current working directory on the system.  
 *
 * @param working_dir
 *        The current working directory
 * @return
 *        GLOBUS_SUCCESS or an error object identifier
 */
globus_result_t
globus_gsi_sysconfig_get_current_working_dir_unix(
    char **                             working_dir)
{
    globus_result_t                     result = GLOBUS_SUCCESS;
    char *                              buffer = NULL;
    char *                              result_buffer = NULL;
    int                                 length = 128;
    static char *                       _function_name_ =
        "globus_gsi_sysconfig_get_current_working_dir_unix";
    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    buffer = malloc(length);
    if(!buffer)
    {
        result = GLOBUS_GSI_SYSTEM_CONFIG_MALLOC_ERROR;
        goto exit;
    }

    while(1)
    {
        result_buffer = getcwd(buffer, length);
        if(!result_buffer && errno == ERANGE)
        {
            length *= 2;
            if(!(result_buffer = realloc(buffer, length)))
            {
                free(buffer);
                result = GLOBUS_GSI_SYSTEM_CONFIG_MALLOC_ERROR;
                goto exit;
            }

            buffer = result_buffer;
        }
        else if(!result_buffer)
        {
            result = 
                globus_error_put(globus_error_wrap_errno_error(
                    GLOBUS_GSI_SYSCONFIG_MODULE,
                    errno,
                    GLOBUS_GSI_SYSCONFIG_ERROR_ERRNO,
                    "%s:%d: Couldn't get the current working directory",
                    __FILE__, __LINE__));
        }
        else
        {
            break;
        }
    }

    *working_dir = result_buffer;

 exit:

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;
}
/* @} */

/**
 * @name UNIX - Get HOME Directory
 * @ingroup globus_i_gsi_sysconfig_unix
 */
/* @{ */
/**
 * Get the HOME Directory of the current user.  Should
 * be the $HOME environment variable.
 *
 * @param home_dir
 *        The home directory of the current user
 * @return
 *        GLOBUS_SUCCESS if no error occured, otherwise
 *        an error object is returned.
 */
globus_result_t
globus_gsi_sysconfig_get_home_dir_unix(
    char **                             home_dir,
    globus_gsi_statcheck_t *            status)
{
    char *                              temp_home_dir;
    struct passwd                       pwd;
    struct passwd *                     pwd_result;
    char *                              buf;
    int                                 buf_len;
    globus_result_t                     result;
    static char *                        _function_name_ =
        "globus_i_gsi_sysconfig_get_home_dir_unix";

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    *home_dir = NULL;

    /* the below seems to be fairly portable */
    
#ifdef _SC_GETPW_R_SIZE_MAX
    buf_len = sysconf(_SC_GETPW_R_SIZE_MAX) + 1;
#else
    buf_len = 1024;
#endif

    buf = malloc(buf_len);

    if(buf == NULL)
    {
        result = GLOBUS_GSI_SYSTEM_CONFIG_MALLOC_ERROR;
        goto exit;
    }
    
    if(globus_libc_getpwuid_r(geteuid(),
                              &pwd,
                              buf,
                              buf_len,
                              &pwd_result) != 0)
    {
        GLOBUS_GSI_SYSCONFIG_ERROR_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_PW_ENTRY,
            ("Error occured for uid: %d",geteuid()));        
        goto exit;
    }

    if(pwd_result == NULL || pwd_result->pw_dir == NULL)
    {
        GLOBUS_GSI_SYSCONFIG_ERROR_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_PW_ENTRY,
            ("Error occured for uid: %d",geteuid()));        
        goto exit;        
    }

    temp_home_dir = malloc(strlen(pwd_result->pw_dir) + 1);
    strncpy(temp_home_dir, pwd_result->pw_dir, 
            strlen(pwd_result->pw_dir) + 1);

    if(temp_home_dir)
    {
        result = GLOBUS_GSI_SYSCONFIG_FILE_EXISTS(temp_home_dir, status);
        if(result != GLOBUS_SUCCESS)
        {
            globus_object_t *               error_obj;
            free(temp_home_dir);
            error_obj = globus_error_get(result);
            globus_object_free(error_obj);
            result = GLOBUS_SUCCESS;
            *status = GLOBUS_FILE_INVALID;
            goto exit;
        }

        *home_dir = temp_home_dir;
    }
    else
    {
        GLOBUS_GSI_SYSCONFIG_ERROR_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_HOME_DIR,
            ("Could not get a defined HOME directory for user id: %d\n",
             geteuid()));
        goto exit;
    }

    result = GLOBUS_SUCCESS;

 exit:

    if(buf != NULL)
    {
        free(buf);
    }
    
    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;
}
/* @} */

/**
 * @name UNIX - File Exists
 * @ingroup globus_i_gsi_sysconfig_unix
 */
/* @{ */
/**
 * Check if the file exists
 *
 * @param filename
 *        The filename of the file to check for
 * @param status
 *        The resulting status of the file
 *
 * @return
 *        GLOBUS_SUCCESS for almost all cases (even if the file
 *        doesn't exist), otherwise an error object identifier
 *        wrapping the system errno is returned
 */
globus_result_t
globus_gsi_sysconfig_file_exists_unix(
    const char *                        filename,
    globus_gsi_statcheck_t *            status)
{
    struct stat                         stx;
    globus_result_t                     result;

    static char *                       _function_name_ =
        "globus_i_gsi_sysconfig_file_exists_unix";

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    if (stat(filename,&stx) == -1)
    {
        switch(errno)
        {
          case ENOENT:
          case ENOTDIR:
            *status = GLOBUS_FILE_DOES_NOT_EXIST;
            result = GLOBUS_SUCCESS;
            goto exit;
            
          case EACCES:
            
            *status = GLOBUS_FILE_BAD_PERMISSIONS;
            result = GLOBUS_SUCCESS;
            goto exit;

          default:
            result = globus_error_put(
                globus_error_wrap_errno_error(
                    GLOBUS_GSI_SYSCONFIG_MODULE,
                    errno,
                    GLOBUS_GSI_SYSCONFIG_ERROR_CHECKING_FILE_EXISTS,
                    __FILE__":%d:%s: Error getting status "
                    "of certificate directory: %s\n",
                    __LINE__, 
                    _function_name_,
                    filename));
            goto exit;
        
        }
    }

    /*
     * use any stat output as random data, as it will 
     * have file sizes, and last use times in it. 
     */
    RAND_add((void*)&stx,sizeof(stx),2);

    if (stx.st_size == 0)
    {
        *status = GLOBUS_FILE_ZERO_LENGTH;
        result = GLOBUS_SUCCESS;
        goto exit;
    }

    if(stx.st_mode & S_IFDIR)
    {
        *status = GLOBUS_FILE_DIR;
    }
    else if(stx.st_mode & (S_IFREG | S_IFLNK))
    {
        *status = GLOBUS_FILE_VALID;
    }
    else
    {
        *status = GLOBUS_FILE_INVALID;
    }

    result = GLOBUS_SUCCESS;

 exit:

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;
}    
/* @} */


/**
 * @name UNIX - Check File Status for Key
 * @ingroup globus_i_gsi_sysconfig_unix
 */
/* @{ */
/**
 * This is a convenience function used to check the status of a 
 * private key file.  The desired status is only the current user has
 * ownership and read permissions, everyone else should not be able
 * to access it.
 * 
 * @param filename
 *        The name of the file to check the status of
 * @param status
 *        The status of the file being checked
 *        see @ref globus_gsi_statcheck_t for possible values
 *        of this variable 
 *
 * @return 
 *        GLOBUS_SUCCESS if the status of the file was able
 *        to be determined.  Otherwise, an error object
 *        identifier
 *
 * @see globus_gsi_statcheck_t
 */
globus_result_t
globus_gsi_sysconfig_check_keyfile_unix(
    const char *                        filename,
    globus_gsi_statcheck_t *            status)
{
    struct stat                         stx;
    globus_result_t                     result;
    static char *                       _function_name_ =
        "globus_i_gsi_sysconfig_check_keyfile_unix";

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    if (stat(filename,&stx) == -1)
    {
        switch (errno)
        {
          case ENOENT:
          case ENOTDIR:
            *status = GLOBUS_FILE_DOES_NOT_EXIST;
            result = GLOBUS_SUCCESS;
            goto exit;

          case EACCES:

            *status = GLOBUS_FILE_BAD_PERMISSIONS;
            result = GLOBUS_SUCCESS;
            goto exit;

          default:
            result = globus_error_put(
                globus_error_wrap_errno_error(
                    GLOBUS_GSI_SYSCONFIG_MODULE,
                    errno,
                    GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_KEY_STRING,
                    __FILE__":%d:%s: Error getting status of keyfile: %s\n",
                    __LINE__,
                    _function_name_,
                    filename));
            goto exit;
        }
    }

    /*
     * use any stat output as random data, as it will 
     * have file sizes, and last use times in it. 
     */
    RAND_add((void*)&stx,sizeof(stx),2);

    if (stx.st_uid != getuid())
    {
        *status = GLOBUS_FILE_NOT_OWNED;
        result = GLOBUS_SUCCESS;
        goto exit;
    }

    /* check that the key file is not x by user, or rwx by group or others */
    if (stx.st_mode & (S_IXUSR | 
                       S_IRGRP | S_IWGRP | S_IXGRP |
                       S_IROTH | S_IWOTH | S_IXOTH))
    {
        GLOBUS_I_GSI_SYSCONFIG_DEBUG_FPRINTF(
            2, (stderr, "checkstat:%s:mode:%o\n", filename, stx.st_mode)); 

        *status = GLOBUS_FILE_BAD_PERMISSIONS;
        result = GLOBUS_SUCCESS;
        goto exit;
    }

    if (stx.st_size == 0)
    {
        *status = GLOBUS_FILE_ZERO_LENGTH;
        result = GLOBUS_SUCCESS;
        goto exit;
    }

    if(stx.st_mode & S_IFDIR)
    {
        *status = GLOBUS_FILE_DIR;
    }
    else if(stx.st_mode & (S_IFLNK | S_IFREG))
    {
        *status = GLOBUS_FILE_VALID;
    }
    else
    {
        *status = GLOBUS_FILE_INVALID;
    }

    result = GLOBUS_SUCCESS;

 exit:

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;

    return result;
}
/* @} */

/**
 * @name UNIX - Check File Status for Cert
 * @ingroup globus_i_gsi_sysconfig_unix
 */
/* @{ */
/**
 * This is a convenience function used to check the status of a 
 * certificate file.  The desired status is the current user has
 * ownership and read/write permissions, while group and others only
 * have read permissions.
 * 
 * @param filename
 *        The name of the file to check the status of
 * @param status
 *        The status of the file being checked
 *        see @ref globus_gsi_statcheck_t for possible values
 *        of this variable 
 *
 * @return 
 *        GLOBUS_SUCCESS if the status of the file was able
 *        to be determined.  Otherwise, an error object
 *        identifier
 *
 * @see globus_gsi_statcheck_t
 */
globus_result_t
globus_gsi_sysconfig_check_certfile_unix(
    const char *                        filename,
    globus_gsi_statcheck_t *            status)
{
    struct stat                         stx;
    globus_result_t                     result;
    static char *                       _function_name_ =
        "globus_i_gsi_sysconfig_check_certfile_unix";

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;
    
    if (stat(filename,&stx) == -1)
    {
        switch (errno)
        {
          case ENOENT:
          case ENOTDIR:
            *status = GLOBUS_FILE_DOES_NOT_EXIST;
            result = GLOBUS_SUCCESS;
            goto exit;

          case EACCES:

            *status = GLOBUS_FILE_BAD_PERMISSIONS;
            result = GLOBUS_SUCCESS;
            goto exit;

          default:
            result = globus_error_put(
                globus_error_wrap_errno_error(
                    GLOBUS_GSI_SYSCONFIG_MODULE,
                    errno,
                    GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_FILENAME,
                    __FILE__":%d:%s: Error getting status of cert file %s\n",
                    __LINE__,
                    _function_name_,
                    filename));
            goto exit;
        }
    }

    /*
     * use any stat output as random data, as it will 
     * have file sizes, and last use times in it. 
     */
    RAND_add((void*)&stx,sizeof(stx),2);

    if (stx.st_uid != getuid())
    {
        *status = GLOBUS_FILE_NOT_OWNED;
        result = GLOBUS_SUCCESS;
        goto exit;
    }

    /* check that the cert file is not x by user, or wx by group or others */
    if (stx.st_mode & (S_IXUSR |
                       S_IWGRP | S_IXGRP |
                       S_IWOTH | S_IXOTH))
    {
        GLOBUS_I_GSI_SYSCONFIG_DEBUG_FPRINTF(
            2, (stderr, "checkstat:%s:mode:%o\n",filename,stx.st_mode));

        *status = GLOBUS_FILE_BAD_PERMISSIONS;
        result = GLOBUS_SUCCESS;
        goto exit;
    }
    
    if (stx.st_size == 0)
    {
        *status = GLOBUS_FILE_ZERO_LENGTH;
        result = GLOBUS_SUCCESS;
        goto exit;
    }

    if(stx.st_mode & S_IFDIR)
    {
        *status = GLOBUS_FILE_DIR;
    }
    else if(stx.st_mode & (S_IFREG | S_IFLNK))
    {
        *status = GLOBUS_FILE_VALID;
    }
    else
    {
        *status = GLOBUS_FILE_INVALID;
    }

    *status = GLOBUS_FILE_VALID;
    result = GLOBUS_SUCCESS;

 exit:
    
    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;
}
/* @} */

/**
 * @name UNIX - Get Trusted CA Cert Dir
 * @ingroup globus_gsi_sysconfig_unix
 */
/* @{ */
/**
 * Get the Trusted Certificate Directory containing the trusted
 * Certificate Authority certificates.  This directory is determined
 * in the order shown below.  Failure in one method results in attempting
 * the next.
 *
 * <ol>
 * <li> <b>X509_CERT_DIR environment variable</b> - if this is set, the
 * trusted certificates will be searched for in that directory.  This
 * variable allows the end user to specify the location of trusted
 * certificates.
 * <li> <b>$HOME/.globus/certificates</b> - If this
 * directory exists, and the previous methods of determining the trusted
 * certs directory failed, this directory will be used.  
 * <li> <b>/etc/grid-security/certificates</b> - This location is intended
 * to be independant of the globus installation ($GLOBUS_LOCATION), and 
 * is generally only writeable by the host system administrator.  
 * <li> <b>$GLOBUS_LOCATION/share/certificates</b>
 * </ol>
 *
 * @param cert_dir
 *        The trusted certificates directory
 * @return
 *        GLOBUS_SUCCESS if no error occurred, and a sufficient trusted
 *        certificates directory was found.  Otherwise, an error object 
 *        identifier returned.
 */
globus_result_t
globus_gsi_sysconfig_get_cert_dir_unix(
    char **                             cert_dir)
{
    char *                              env_cert_dir = NULL;
    char *                              local_cert_dir = NULL;
    char *                              default_cert_dir = NULL;
    char *                              installed_cert_dir = NULL;
    globus_result_t                     result;
    char *                              home = NULL;
    char *                              globus_location;
    globus_gsi_statcheck_t              status;

    static char *                       _function_name_ =
        "globus_gsi_sysconfig_get_cert_dir_unix";

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;
    
    *cert_dir = NULL;

    if(getenv(X509_CERT_DIR))
    {
        result = globus_i_gsi_sysconfig_create_cert_dir_string(
            cert_dir, 
            & env_cert_dir,
            & status,
            getenv(X509_CERT_DIR));
        if(result != GLOBUS_SUCCESS)
        {
            GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                result,
                GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_DIR);
            goto done;
        }
    }

    /* now check for a trusted CA directory in the user's home directory */
    if(!(*cert_dir))
    {
        result = GLOBUS_GSI_SYSCONFIG_GET_HOME_DIR(&home, &status);
        if(result != GLOBUS_SUCCESS)
        {
	    home = NULL;
            GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                result,
                GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_DIR);
            goto done;
        }
        
        if(home && status == GLOBUS_FILE_DIR)
        {
            result = globus_i_gsi_sysconfig_create_cert_dir_string(
                cert_dir, 
                & local_cert_dir,
                & status,
                "%s%s%s",
                home,
                FILE_SEPERATOR,
                X509_LOCAL_TRUSTED_CERT_DIR);
            if(result != GLOBUS_SUCCESS)
            {
                GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                    result,
                    GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_DIR);
                goto done;
            }
        }
    }

    /* now look in /etc/grid-security/certificates */
    if (!(*cert_dir))
    {
        result = globus_i_gsi_sysconfig_create_cert_dir_string(
            cert_dir,
            &installed_cert_dir,
            &status,
            X509_DEFAULT_TRUSTED_CERT_DIR);
        if(result != GLOBUS_SUCCESS)
        {
            if(status != GLOBUS_FILE_BAD_PERMISSIONS)
            {
                GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                    result,
                    GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_DIR);
                goto done;
            }
            else
            {
                globus_object_t *       error_obj;
                error_obj = globus_error_get(result);
                globus_object_free(error_obj);
            }
        }
    }

    /* now look in  $GLOBUS_LOCATION/share/certificates */
    if (!(*cert_dir))
    {
        globus_location = getenv("GLOBUS_LOCATION");
        
        if (globus_location)
        {
            result = globus_i_gsi_sysconfig_create_cert_dir_string(
                cert_dir,
                &default_cert_dir,
                &status,
                "%s%s%s",
                globus_location,
                FILE_SEPERATOR,
                X509_INSTALLED_TRUSTED_CERT_DIR);
            if(result != GLOBUS_SUCCESS)
            {
                GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                    result,
                    GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_DIR);
                goto done;
            }
        }
    }

    if(!(*cert_dir))
    {
        GLOBUS_GSI_SYSCONFIG_ERROR_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_DIR,
            ("The trusted certificates directory could not be "
             "found in any of the following locations: \n"
             "1) env. var. X509_CERT_DIR=%s\n"
             "2) %s\n3) %s\n4) %s\n",
             env_cert_dir ? env_cert_dir : "NULL",
             local_cert_dir ? local_cert_dir : "NULL",
             installed_cert_dir ? installed_cert_dir : "NULL",
             default_cert_dir ? default_cert_dir : "NULL"));

        goto done;
    }

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_FPRINTF(
        2, (stderr, "Using cert_dir = %s\n", 
            (*cert_dir ? *cert_dir : "null")));
    
    result = GLOBUS_SUCCESS;

 done:

    if(result != GLOBUS_SUCCESS)
    {
        *cert_dir = NULL;
    }
    if(home != NULL)
    {
	free(home);
    }
    
    if(env_cert_dir && (env_cert_dir != (*cert_dir)))
    {
        globus_libc_free(env_cert_dir);
    }
    if(local_cert_dir && (local_cert_dir != (*cert_dir)))
    {
        globus_libc_free(local_cert_dir);
    }
    if(installed_cert_dir && (installed_cert_dir != (*cert_dir)))
    {
        globus_libc_free(installed_cert_dir);
    }
    if(default_cert_dir && (default_cert_dir != (*cert_dir)))
    {
        globus_libc_free(default_cert_dir);
    }

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;

    return result;
}
/* @} */

/**
 * @name UNIX - Get User Certificate and Key Filenames
 * @ingroup globus_gsi_sysconfig_unix
 */
/* @{ */
/**
 * Get the User Certificate Filename based on the current user's
 * environment.  The following locations are searched for cert and key
 * files in order:
 * 
 * <ol>
 * <li>environment variables X509_USER_CERT and X509_USER_KEY
 * <li>$HOME/.globus/usercert.pem and 
 *     $HOME/.globus/userkey.pem
 * <li>$HOME/.globus/usercred.p12 - this is a PKCS12 credential
 * </ol>
 *
 * @param user_cert
 *        pointer the filename of the user certificate
 * @param user_key
 *        pointer to the filename of the user key
 * @return
 *        GLOBUS_SUCCESS if the cert and key files were found in one
 *        of the possible locations, otherwise an error object identifier
 *        is returned
 */
globus_result_t
globus_gsi_sysconfig_get_user_cert_filename_unix(
    char **                             user_cert,
    char **                             user_key)
{
    char *                              home = NULL;
    char *                              env_user_cert = NULL;
    char *                              env_user_key = NULL;
    char *                              default_user_cert = NULL;
    char *                              default_user_key = NULL;
    char *                              default_pkcs12_user_cred = NULL;
    globus_gsi_statcheck_t              status;
    globus_gsi_statcheck_t              home_status;
    globus_result_t                     result;

    static char *                       _function_name_ =
        "globus_gsi_sysconfig_get_user_cert_filename_unix";

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;


    /* first, check environment variables for valid filenames */

    if(user_cert)
    {
        *user_cert = NULL;
        if(getenv(X509_USER_CERT))
        {
            result = globus_i_gsi_sysconfig_create_cert_string(
                user_cert,
                & env_user_cert,
                & status,
                getenv(X509_USER_CERT));
            if(result != GLOBUS_SUCCESS)
            {
                GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                    result,
                    GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_STRING);
                goto done;
            }            
        }

        if(!(*user_cert))
        {
            result = GLOBUS_GSI_SYSCONFIG_GET_HOME_DIR(&home,
                                                       &home_status);
            if(result != GLOBUS_SUCCESS)
            {
                home = NULL;
                GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                    result,
                    GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_STRING);
                goto done;
            }

            if(home && home_status == GLOBUS_FILE_DIR)
            {
                result = globus_i_gsi_sysconfig_create_cert_string(
                    user_cert,
                    & default_user_cert,
                    & status,
                    "%s%s%s",
                    home,
                    FILE_SEPERATOR,
                    X509_DEFAULT_USER_CERT);

                if(result != GLOBUS_SUCCESS)
                {
                    GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                        result,
                        GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_STRING);
                    goto done;
                }   
            }
        }
    }

    if(user_key)
    { 
        *user_key = NULL;
        if(getenv(X509_USER_KEY))
        {
            result = globus_i_gsi_sysconfig_create_key_string(
                user_key,
                & env_user_key,
                & status,
                getenv(X509_USER_KEY));
            if(result != GLOBUS_SUCCESS)
            {
                GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                    result,
                    GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_KEY_STRING);
                goto done;
            }
        }

        if(!(*user_key))
        {
            if(!home)
            {
                result = GLOBUS_GSI_SYSCONFIG_GET_HOME_DIR(&home,
                                                           &home_status);
                if(result != GLOBUS_SUCCESS)
                {
                    home = NULL;
                    GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                        result,
                        GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_STRING);
                    goto done;
                }
            }
            
            if(home && home_status == GLOBUS_FILE_DIR)
            {
                result = globus_i_gsi_sysconfig_create_key_string(
                    user_key,
                    & default_user_key,
                    & status,
                    "%s%s%s",
                    home,
                    FILE_SEPERATOR,
                    X509_DEFAULT_USER_KEY);
                if(result != GLOBUS_SUCCESS)
                {
                    GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                        result,
                        GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_KEY_STRING);
                    goto done;
                }
            }
        }
    }

    /* if the cert & key don't exist in the default locations
     * or those specified by the environment variables, a
     * pkcs12 cert will be searched for
     */
    if(user_cert && user_key && !(*user_cert) && !(*user_key))
    {
        if(!home)
        { 
            result = GLOBUS_GSI_SYSCONFIG_GET_HOME_DIR(&home,
                                                       &home_status);
            if(result != GLOBUS_SUCCESS)
            {
                home = NULL;
                GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                    result,
                    GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_KEY_STRING);
                goto done;
            }
        }

        if(home && home_status == GLOBUS_FILE_DIR)
        {
            result = globus_i_gsi_sysconfig_create_key_string(
                user_key,
                & default_pkcs12_user_cred,
                & status,
                "%s%s%s",
                home,
                FILE_SEPERATOR,
                X509_DEFAULT_PKCS12_FILE);
            if(result != GLOBUS_SUCCESS)
            {
                GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                    result,
                    GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_KEY_STRING);
                goto done;
            }
            *user_cert = *user_key;
        }
    }

    if(user_cert && !(*user_cert))
    {
        GLOBUS_GSI_SYSCONFIG_ERROR_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_STRING,
            ("The user cert could not be found in: \n"
             "1) env. var. X509_USER_CERT=%s\n"
             "2) %s\n3) %s\n\n",
             env_user_cert ? env_user_cert : "NULL",
             default_user_cert ? default_user_cert : "NULL",
             default_pkcs12_user_cred ? default_pkcs12_user_cred : "NULL"));
        goto done;
    }

    if(user_key && !(*user_key))
    {
        GLOBUS_GSI_SYSCONFIG_ERROR_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_KEY_STRING,
            ("The user key could not be found in:\n,"
             "1) env. var. X509_USER_KEY=%s\n"
             "2) %s\n3) %s\n\n",
             env_user_key ? env_user_key : "NULL",
             default_user_key ? default_user_key : "NULL",
             default_pkcs12_user_cred ? default_pkcs12_user_cred : "NULL"));
        goto done;
    }

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_FPRINTF(
        2, (stderr,"Using x509_user_cert=%s\n      x509_user_key =%s\n",
            user_cert ? *user_cert : "NULL",
            user_key ? *user_key : "NULL"));

    result = GLOBUS_SUCCESS;

 done:
    if(result != GLOBUS_SUCCESS && user_cert)
    {
        *user_cert = NULL;
    }
    if(env_user_cert && env_user_cert != (*user_cert))
    {
        globus_libc_free(env_user_cert);
    }
    if(env_user_key && env_user_key != (*user_key))
    {
        globus_libc_free(env_user_key);
    }
    if(default_user_cert && default_user_cert != (*user_cert))
    {
        globus_libc_free(default_user_cert);
    }
    if(default_user_key && default_user_key != (*user_key))
    {
        globus_libc_free(default_user_key);
    }
    if(default_pkcs12_user_cred && default_pkcs12_user_cred != (*user_key))
    {
        globus_libc_free(default_pkcs12_user_cred);
    }
    if(home)
    {
        free(home);
    }

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;
}
/* @} */

/**
 * @name UNIX - Get Host Certificate and Key Filenames
 * @ingroup globus_gsi_sysconfig_unix
 */
/* @{ */
/**
 * Get the Host Certificate and Key Filenames based on the current user's
 * environment.  The host cert and key are searched for in the following 
 * locations (in order):
 *
 * <ol>
 * <li>X509_USER_CERT and X509_USER_KEY environment variables</li>
 * <li>registry keys x509_user_cert and x509_user_key in software\\Globus\\GSI</li>
 * <li>SLANG: NOT DETERMINED - this is the default location</li>
 * <li>\\<GLOBUS_LOCATION\\>\\etc\\host[cert|key].pem</li>
 * <li>\\<users home directory\\>\\.globus\\host[cert|key].pem</li>
 * </ol>
 * 
 * @param host_cert
 *        pointer to the host certificate filename
 * @param host_key
 *        pointer to the host key filename
 *
 * @return
 *        GLOBUS_SUCCESS if the host cert and key were found, otherwise
 *        an error object identifier is returned 
 */
globus_result_t
globus_gsi_sysconfig_get_host_cert_filename_unix(
    char **                             host_cert,
    char **                             host_key)
{
    char *                              home = NULL;
    char *                              env_host_cert = NULL;
    char *                              env_host_key = NULL;
    char *                              default_host_cert = NULL;
    char *                              default_host_key = NULL;
    char *                              installed_host_cert = NULL;
    char *                              installed_host_key = NULL;
    char *                              local_host_cert = NULL;
    char *                              local_host_key = NULL;
    char *                              globus_location = NULL;
    globus_gsi_statcheck_t              status;
    globus_result_t                     result;

    static char *                       _function_name_ =
        "globus_gsi_sysconfig_get_host_cert_filename_unix";
    
    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;
    
    *host_cert = NULL;
    *host_key = NULL;
    
    /* first check environment variables for valid filenames */
    
    if(getenv(X509_USER_CERT) && getenv(X509_USER_KEY))
    {
        result = globus_i_gsi_sysconfig_create_cert_string(
            host_cert,
            & env_host_cert,
            & status,
            getenv(X509_USER_CERT));
        if(result != GLOBUS_SUCCESS)
        {
            GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                result,
                GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_STRING);
            goto done;
        }

        result = globus_i_gsi_sysconfig_create_key_string(
            host_key,
            & env_host_key,
            & status,
            getenv(X509_USER_KEY));
        if(result != GLOBUS_SUCCESS)
        {
            GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                result,
                GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_KEY_STRING);
            goto done;
        }
    }

/* G.Ganis 16/10/2003 begin */
    /* now check validity of files found (if any) */
    /* Perform separate check (G.Ganis) */
    if (*host_cert )
    {
        result = globus_i_gsi_sysconfig_create_cert_string(
            host_cert,
            & default_host_cert,
            & status,
            "%s%s%s%s",
            X509_DEFAULT_CERT_DIR,
            FILE_SEPERATOR,
            X509_HOST_PREFIX,
            X509_CERT_SUFFIX);
        if(result != GLOBUS_SUCCESS)
        {
            GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                result,
                GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_STRING);
            goto done;
        }
    }
    if (*host_key )
    {
        result = globus_i_gsi_sysconfig_create_key_string(
            host_key,
            & default_host_key,
            & status,
            "%s%s%s%s",
            X509_DEFAULT_CERT_DIR,
            FILE_SEPERATOR,
            X509_HOST_PREFIX,
            X509_KEY_SUFFIX);
        if(result != GLOBUS_SUCCESS)
        {
            GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                result,
                GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_KEY_STRING);
            goto done;
        }
    }        
/* G.Ganis 16/10/2003 end */


    /* now check default locations for valid filenames */
    if(!(*host_cert) && !(*host_key))
    {
        result = globus_i_gsi_sysconfig_create_cert_string(
            host_cert,
            & default_host_cert,
            & status,
            "%s%s%s%s",
            X509_DEFAULT_CERT_DIR,
            FILE_SEPERATOR,
            X509_HOST_PREFIX,
            X509_CERT_SUFFIX);
        if(result != GLOBUS_SUCCESS)
        {
            GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                result,
                GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_STRING);
            goto done;
        }
        
        result = globus_i_gsi_sysconfig_create_key_string(
            host_key,
            & default_host_key,
            & status,
            "%s%s%s%s",
            X509_DEFAULT_CERT_DIR,
            FILE_SEPERATOR,
            X509_HOST_PREFIX,
            X509_KEY_SUFFIX);
        if(result != GLOBUS_SUCCESS)
        {
            GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                result,
                GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_KEY_STRING);
            goto done;
        }        
    }

    /* now check installed location for host cert */
    if(!(*host_cert) && !(*host_key)) 
    {
        globus_location = getenv("GLOBUS_LOCATION");

        if(globus_location)
        {
            result = globus_i_gsi_sysconfig_create_cert_string(
                host_cert,
                & installed_host_cert,
                & status,
                "%s%s%s%s%s%s",
                globus_location,
                FILE_SEPERATOR,
                X509_INSTALLED_CERT_DIR,
                FILE_SEPERATOR,
                X509_HOST_PREFIX,
                X509_CERT_SUFFIX);
            if(result != GLOBUS_SUCCESS)
            {
                GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                    result,
                    GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_STRING);
                goto done;
            }

            result = globus_i_gsi_sysconfig_create_key_string(
                host_key,
                & installed_host_key,
                & status,
                "%s%s%s%s%s%s",
                globus_location,
                FILE_SEPERATOR,
                X509_INSTALLED_CERT_DIR,
                FILE_SEPERATOR,
                X509_HOST_PREFIX,
                X509_KEY_SUFFIX);
            if(result != GLOBUS_SUCCESS)
            {
                GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                    result,
                    GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_KEY_STRING);
                goto done;
            }
        }
    }
    
    if(!(*host_cert) && !(*host_key)) 
    {
        result = GLOBUS_GSI_SYSCONFIG_GET_HOME_DIR(&home, &status);
        if(result != GLOBUS_SUCCESS)
        {
            GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                result,
                GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_STRING);
            home = NULL;
            goto done;
        }

        if(home && status == GLOBUS_FILE_DIR)
        {
            result = globus_i_gsi_sysconfig_create_cert_string(
                host_cert,
                & local_host_cert,
                & status,
                "%s%s%s%s%s%s",
                home,
                FILE_SEPERATOR,
                X509_LOCAL_CERT_DIR,
                FILE_SEPERATOR,
                X509_HOST_PREFIX,
                X509_CERT_SUFFIX);
            if(result != GLOBUS_SUCCESS)
            {
                GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                    result,
                    GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_STRING);
                goto done;
            }
            
            result = globus_i_gsi_sysconfig_create_key_string(
                host_key,
                & local_host_key,
                & status,
                "%s%s%s%s%s%s",
                home,
                FILE_SEPERATOR,
                X509_LOCAL_CERT_DIR,
                FILE_SEPERATOR,
                X509_HOST_PREFIX,
                X509_KEY_SUFFIX);
            if(result != GLOBUS_SUCCESS)
            {
                GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                    result,
                    GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_KEY_STRING);
                goto done;
            }
        }
    }
    
    if(!(*host_cert) || !(*host_key))
    {
        GLOBUS_GSI_SYSCONFIG_ERROR_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_FILENAME,
            ("The host cert could not be found in: \n"
             "1) env. var. X509_USER_CERT=%s\n"
             "2) %s\n3) %s\n4) %s\n\n"
             "The host key could not be found in:\n"
             "1) env. var. X509_USER_KEY=%s\n"
             "2) %s\n3) %s\n4) %s\n",
             env_host_cert ? env_host_cert : "NULL",
             default_host_cert ? default_host_cert : "NULL",
             installed_host_cert ? installed_host_cert : "NULL",
             local_host_cert ? local_host_cert : "NULL",
             env_host_key ? env_host_key : "NULL",
             default_host_key ? default_host_key : "NULL",
             installed_host_key ? installed_host_key : "NULL",
             local_host_key ? local_host_key : "NULL"));
        goto done;
    }
    
    GLOBUS_I_GSI_SYSCONFIG_DEBUG_FPRINTF(
        2, (stderr, "Using x509_user_cert=%s\n      x509_user_key =%s\n",
            *host_cert , *host_key));
    
    result = GLOBUS_SUCCESS;
    
 done:
    if(result != GLOBUS_SUCCESS)
    {
        *host_cert = NULL;
        *host_key = NULL;
    }
    if(env_host_cert && env_host_cert != *host_cert)
    {
        globus_libc_free(env_host_cert);
    }
    if(env_host_key && env_host_key != *host_key)
    {
        globus_libc_free(env_host_key);
    }
    if(installed_host_cert && installed_host_cert != *host_cert)
    {
        globus_libc_free(installed_host_cert);
    }
    if(installed_host_key && installed_host_key != *host_key)
    {
        globus_libc_free(installed_host_key);
    }
    if(local_host_cert && local_host_cert != *host_cert)
    {
        globus_libc_free(local_host_cert);
    }
    if(local_host_key && local_host_key != *host_key)
    {
        globus_libc_free(local_host_key);
    }
    if(default_host_cert && default_host_cert != *host_cert)
    {
        globus_libc_free(default_host_cert);
    }
    if(default_host_key && default_host_key != *host_key)
    {
        globus_libc_free(default_host_key);
    }

    if(home)
    {
        free(home);
    }

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;
}
/* @} */

/**
 * @name UNIX - Get Service Certificate and Key Filenames
 * @ingroup globus_gsi_sysconfig_unix
 */
/* @{ */
/**
 * Get the Service Certificate Filename based on the current user's
 * environment.  The host cert and key are searched for in the following 
 * locations (in order):
 *
 * <ol>
 * <li>X509_USER_CERT and X509_USER_KEY environment variables
 * <li>\/etc\/grid-security\/{service_name}\/{service_name}[cert|key].pem
 * <li>GLOBUS_LOCATION\/etc\/{service_name}\/{service_name}[cert|key].pem
 *     So for example, if my service was named: myservice, the location
 *     of the certificate would be: 
 *     GLOBUS_LOCATION\/etc\/myservice\/myservicecert.pem
 * <li>\\<users home\\>\/.globus\/{service_name}\/{service_name}[cert|key].pem
 * </ol>
 * 
 * @param service_name
 *        The name of the service which allows us to determine the
 *        locations of cert and key files to look for
 * @param service_cert
 *        pointer to the host certificate filename
 * @param service_key
 *        pointer to the host key filename
 *
 * @return
 *        GLOBUS_SUCCESS if the service cert and key were found, otherwise
 *        an error object identifier 
 */
globus_result_t
globus_gsi_sysconfig_get_service_cert_filename_unix(
    char *                              service_name,
    char **                             service_cert,
    char **                             service_key)
{
    char *                              home = NULL;
    char *                              env_service_cert = NULL;
    char *                              env_service_key = NULL;
    char *                              default_service_cert = NULL;
    char *                              default_service_key = NULL;
    char *                              installed_service_cert = NULL;
    char *                              installed_service_key = NULL;
    char *                              local_service_cert = NULL;
    char *                              local_service_key = NULL;
    char *                              globus_location = NULL;
    globus_gsi_statcheck_t              status;
    globus_result_t                     result;

    static char *                       _function_name_ =
        "globus_gsi_sysconfig_get_service_cert_filename_unix";

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    *service_cert = NULL;
    *service_key = NULL;

    /* first check environment variables for valid filenames */

    if(getenv(X509_USER_CERT) && getenv(X509_USER_KEY))
    {
        result = globus_i_gsi_sysconfig_create_cert_string(
            service_cert,
            & env_service_cert,
            & status,
            getenv(X509_USER_CERT));
        if(result != GLOBUS_SUCCESS)
        {
            GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                result,
                GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_STRING);
            goto done;
        }

        result = globus_i_gsi_sysconfig_create_key_string(
            service_key,
            & env_service_key,
            & status,
            getenv(X509_USER_KEY));
        if(result != GLOBUS_SUCCESS)
        {
            GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                result,
                GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_KEY_STRING);
            goto done;
        }
    }

    /* now check default locations for valid filenames */
    if(!(*service_cert) && !(*service_key))
    {
        result = GLOBUS_GSI_SYSCONFIG_GET_HOME_DIR(&home, &status);
        if(result != GLOBUS_SUCCESS)
        {
            GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                result,
                GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_STRING);
            home = NULL;
            goto done;
        }

        if(home && status == GLOBUS_FILE_DIR)
        {
            result = globus_i_gsi_sysconfig_create_cert_string(
                service_cert,
                & default_service_cert,
                & status,
                "%s%s%s%s%s%s",
                X509_DEFAULT_CERT_DIR,
                FILE_SEPERATOR,
                service_name,
                FILE_SEPERATOR,
                service_name,
                X509_CERT_SUFFIX);
            if(result != GLOBUS_SUCCESS)
            {
                GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                    result,
                    GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_STRING);
                goto done;
            }

            result = globus_i_gsi_sysconfig_create_key_string(
                service_key,
                & default_service_key,
                & status,
                "%s%s%s%s%s%s",
                X509_DEFAULT_CERT_DIR,
                FILE_SEPERATOR,
                service_name,
                FILE_SEPERATOR,
                service_name,
                X509_KEY_SUFFIX);
            if(result != GLOBUS_SUCCESS)
            {
                GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                    result,
                    GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_KEY_STRING);
                goto done;
            }
        }
    }

    /* now check intstalled location for service cert */
    if(!(*service_cert) && !(*service_key))
    {
        globus_location = getenv("GLOBUS_LOCATION");

        if(globus_location)
        {
            result = globus_i_gsi_sysconfig_create_cert_string(
                service_cert,
                & installed_service_cert,
                & status,
                "%s%s%s%s%s%s%s%s",
                globus_location,
                FILE_SEPERATOR,
                X509_INSTALLED_CERT_DIR,
                FILE_SEPERATOR,
                service_name,
                FILE_SEPERATOR,
                service_name,
                X509_CERT_SUFFIX);
            if(result != GLOBUS_SUCCESS)
            {
                GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                    result,
                    GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_STRING);
                goto done;
            }
               
            result = globus_i_gsi_sysconfig_create_key_string(
                service_key,
                & installed_service_key,
                & status,
                "%s%s%s%s%s%s%s%s",
                globus_location,
                FILE_SEPERATOR,
                X509_INSTALLED_CERT_DIR,
                FILE_SEPERATOR,
                service_name,
                FILE_SEPERATOR,
                service_name,
                X509_KEY_SUFFIX);
            if(result != GLOBUS_SUCCESS)
            {
                GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                    result,
                    GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_KEY_STRING);
                goto done;
            }
        }
    }
    
    if(!(*service_cert) && !(*service_key))
    {
        /* need to change this if I ever fix the status mess */
        if(home)
        {
            free(home);
            home = NULL;
        }

        result = GLOBUS_GSI_SYSCONFIG_GET_HOME_DIR(&home, &status);
        if(result != GLOBUS_SUCCESS)
        {
            GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                result,
                GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_STRING);
            home = NULL;
            goto done;
        }
        
        if(home && status == GLOBUS_FILE_DIR)
        {
            result = globus_i_gsi_sysconfig_create_cert_string(
                service_cert,
                & local_service_cert,
                & status,
                "%s%s%s%s%s%s%s",
                home,
                FILE_SEPERATOR,
                X509_LOCAL_CERT_DIR,
                FILE_SEPERATOR,
                service_name,
                FILE_SEPERATOR,
                service_name,
                X509_CERT_SUFFIX);
            if(result != GLOBUS_SUCCESS)
            {
                GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                    result,
                    GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_STRING);
                goto done;
            }
            
            result = globus_i_gsi_sysconfig_create_key_string(
                service_key,
                & local_service_key,
                & status,
                "%s%s%s%s%s%s%s%s",
                home,
                FILE_SEPERATOR,
                X509_LOCAL_CERT_DIR,
                FILE_SEPERATOR,
                service_name,
                FILE_SEPERATOR,
                service_name,
                X509_KEY_SUFFIX);
            if(result != GLOBUS_SUCCESS)
            {
                GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                    result,
                    GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_STRING);
                goto done;
            }
        }
    }

    if(!(*service_cert) || !(*service_key))
    {
        GLOBUS_GSI_SYSCONFIG_ERROR_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CERT_FILENAME,
            ("\nThe service cert could not be found in: \n"
             "1) env. var. X509_USER_CERT=%s\n"
             "2) %s\n3) %s\n4) %s\n\n"
             "The service key could not be found in:\n"
             "1) env. var. X509_USER_KEY=%s\n"
             "2) %s\n3) %s\n4) %s\n",
             env_service_cert ? env_service_cert : "NULL",
             default_service_cert ? default_service_cert : "NULL",
             installed_service_cert ? installed_service_cert : "NULL",
             local_service_cert ? local_service_cert : "NULL",
             env_service_key ? env_service_key : "NULL",
             default_service_key ? default_service_key : "NULL",
             installed_service_key ? installed_service_key : "NULL",
             local_service_key ? local_service_key : "NULL"));
        goto done;
    }

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_FPRINTF(
        2, (stderr, "Using x509_user_cert=%s\n      x509_user_key =%s\n",
            *service_cert , *service_key));

    result = GLOBUS_SUCCESS;

 done:
    if(result != GLOBUS_SUCCESS)
    {
        *service_cert = NULL;
        *service_key = NULL;
    }
    if(env_service_cert && env_service_cert != *service_cert)
    {
        globus_libc_free(env_service_cert);
    }
    if(env_service_key && env_service_key != *service_key)
    {
        globus_libc_free(env_service_key);
    }
    if(installed_service_cert && installed_service_cert != *service_cert)
    {
        globus_libc_free(installed_service_cert);
    }
    if(installed_service_key && installed_service_key != *service_key)
    {
        globus_libc_free(installed_service_key);
    }
    if(local_service_cert && local_service_cert != *service_cert)
    {
        globus_libc_free(local_service_cert);
    }
    if(local_service_key && local_service_key != *service_key)
    {
        globus_libc_free(local_service_key);
    }
    if(default_service_cert && default_service_cert != *service_cert)
    {
        globus_libc_free(default_service_cert);
    }
    if(default_service_key && default_service_key != *service_key)
    {
        globus_libc_free(default_service_key);
    }

    if(home)
    {
        free(home);
    }

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;
}
/* @} */

/**
 * @name UNIX - Get Proxy Filename
 * @ingroup globus_gsi_sysconfig_unix
 */
/* @{ */
/**
 * Get the proxy cert filename based on the following
 * search order:
 * 
 * <ol>
 * <li> X509_USER_PROXY environment variable - This environment variable
 * is set by the at run time for the specific application.  If
 * the proxy_file_type variable is set to GLOBUS_PROXY_OUTPUT
 *  (a proxy filename for writing is requested), 
 * and the X509_USER_PROXY is set, this will be the 
 * resulting value of the user_proxy filename string passed in.  If the
 * proxy_file_type is set to GLOBUS_PROXY_INPUT and X509_USER_PROXY is 
 * set, but the file it points to does not exist, 
 * or has some other readability issues, the 
 * function will continue checking using the other methods available.
 * 
 * <li> Check the default location for the proxy file of
 * \/tmp\/x509_u\\<user_id\\> where \\<user id\\> is some unique string for
 * that user on the host 
 * </ol>
 *
 * @param user_proxy
 *        the proxy filename of the user
 * @param proxy_file_type
 *        Switch for determining whether to return a existing proxy filename or
 *        if a filename suitable for creating a proxy should be returned
 *
 * @return
 *        GLOBUS_SUCCESS or an error object identifier
 */
globus_result_t
globus_gsi_sysconfig_get_proxy_filename_unix(
    char **                             user_proxy,
    globus_gsi_proxy_file_type_t        proxy_file_type)
{
    char *                              env_user_proxy = NULL;
    char *                              env_value = NULL;
    char *                              default_user_proxy = NULL;
    globus_result_t                     result;
    char *                              user_id_string = NULL;
    globus_gsi_statcheck_t              status;
    static char *                       _function_name_ =
        "globus_gsi_sysconfig_get_proxy_filename_unix";
    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    *user_proxy = NULL;
    
    if((env_value = getenv(X509_USER_PROXY)))
    {
        if(proxy_file_type == GLOBUS_PROXY_FILE_OUTPUT)
        {
            *user_proxy = strdup(env_value);
            if(*user_proxy == NULL)
            {
                result = GLOBUS_GSI_SYSTEM_CONFIG_MALLOC_ERROR;
                goto done;
            }
        }
        else
        { 
            result = globus_i_gsi_sysconfig_create_key_string(
                user_proxy,
                & env_user_proxy,
                & status,
                env_value);
            if(result != GLOBUS_SUCCESS)
            {
                GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                    result,
                    GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_PROXY_FILENAME);
                goto done;
            }
        }
    }
    
    if (!*user_proxy)
    {
        result = GLOBUS_GSI_SYSCONFIG_GET_USER_ID_STRING(&user_id_string);
        if(result != GLOBUS_SUCCESS)
        {
            GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                result,
                GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_PROXY_FILENAME);
            goto done;
        }

        if(proxy_file_type == GLOBUS_PROXY_FILE_OUTPUT)
        {
            *user_proxy = globus_common_create_string(
                "%s%s%s%s",
                DEFAULT_SECURE_TMP_DIR,
                FILE_SEPERATOR,
                X509_USER_PROXY_FILE,
                user_id_string);
            if(*user_proxy == NULL)
            {
                result = GLOBUS_GSI_SYSTEM_CONFIG_MALLOC_ERROR;
                goto done;
            }
        }
        else
        {
            result = globus_i_gsi_sysconfig_create_key_string(
                user_proxy,
                & default_user_proxy,
                & status,
                "%s%s%s%s",
                DEFAULT_SECURE_TMP_DIR,
                FILE_SEPERATOR,
                X509_USER_PROXY_FILE,
                user_id_string);
            
            if(result != GLOBUS_SUCCESS)
            {
                GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                    result,
                    GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_PROXY_FILENAME);
                goto done;
            }
        }
    }

    if(!(*user_proxy))
    {
        GLOBUS_GSI_SYSCONFIG_ERROR_RESULT( 
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_PROXY_FILENAME,
            ("A file location for%s the proxy cert could not be found in: \n"
             "1) env. var. X509_USER_PROXY=%s\n"
             "2) %s\n",
             (proxy_file_type == GLOBUS_PROXY_FILE_INPUT) ? "" : " writing",
             env_user_proxy ? env_user_proxy : "NULL",
             default_user_proxy ? default_user_proxy : "NULL"));
        
        goto done;
    }
    
    result = GLOBUS_SUCCESS;

 done:
    if(result != GLOBUS_SUCCESS)
    {
        *user_proxy = NULL;
    }
    if(user_id_string)
    {
        free(user_id_string);
    }
    if(default_user_proxy && (default_user_proxy != (*user_proxy)))
    {
        globus_libc_free(default_user_proxy);
    }
    
    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;
}
/* @} */

/**
 * @name UNIX - Get Signing Policy Filename
 * @ingroup globus_gsi_sysconfig_unix
 */
/* @{ */
/**
 * Get the Signing Policy Filename on the current system,
 * based on the CA's subject name, and the trusted certificates
 * directory
 *
 * @param ca_name
 *        The X509 subject name of the CA to get the signing policy of.
 *        The hash of the CA is generated from this
 *
 * @param cert_dir
 *        The trusted CA certificates directory, containing the singing_policy
 *        files of the trusted CA's.
 *
 * @param signing_policy_filename
 *        The resulting singing_policy filename
 * @return 
 *        GLOBUS_SUCCESS if no error occurred, otherwise an error object ID
 */
globus_result_t
globus_gsi_sysconfig_get_signing_policy_filename_unix(
    X509_NAME *                         ca_name,
    char *                              cert_dir,
    char **                             signing_policy_filename)
{
    char *                              signing_policy = NULL;
    globus_gsi_statcheck_t              status;
    globus_result_t                     result = GLOBUS_SUCCESS;
    unsigned long                       hash;
    char *                              ca_cert_dir = NULL;
    
    static char *                       _function_name_ =
        "globus_gsi_sysconfig_get_signing_policy_filename_unix";

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    *signing_policy_filename = NULL;

    if (cert_dir == NULL)
    {
        result = GLOBUS_GSI_SYSCONFIG_GET_CERT_DIR(&ca_cert_dir);
        if(result != GLOBUS_SUCCESS)
        {
            GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                result,
                GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_SIGNING_POLICY);
            goto exit;
        }
    }
    else
    {
        ca_cert_dir = cert_dir;
    }

    if(ca_name == NULL)
    {
        GLOBUS_GSI_SYSCONFIG_ERROR_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_SIGNING_POLICY,
            ("NULL parameter ca_name passed to: %s", _function_name_));
        goto exit;
    }

    hash = X509_NAME_hash(ca_name);

    signing_policy = globus_common_create_string(
        "%s%s%08lx%s", 
        ca_cert_dir, FILE_SEPERATOR, hash, SIGNING_POLICY_FILE_EXTENSION);
    
    if(signing_policy == NULL)
    {
        result = GLOBUS_GSI_SYSTEM_CONFIG_MALLOC_ERROR;
        goto exit;
    }

    result = GLOBUS_GSI_SYSCONFIG_FILE_EXISTS(signing_policy, &status);
    if(result != GLOBUS_SUCCESS)
    {
        GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_SIGNING_POLICY);
        goto exit;
    }

    if(status == GLOBUS_FILE_VALID)
    {
        *signing_policy_filename = signing_policy;
    }

 exit:

    if(ca_cert_dir != NULL &&
       cert_dir == NULL)
    {
        free(ca_cert_dir);
    }

    if(signing_policy != NULL &&
       result != GLOBUS_SUCCESS)
    {
        free(signing_policy);
        *signing_policy_filename = NULL;
    }
    
    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;
}
/* @} */

/**
 * @name UNIX - Get CA Cert Filenames
 * @ingroup globus_gsi_sysconfig_unix
 */
/* @{ */
/**
 * Gets a list of trusted CA certificate filenames in 
 * a trusted CA certificate directory.  
 *
 * @param ca_cert_dir
 *        The trusted CA certificate directory to get the filenames from
 * @param ca_cert_list
 *        The resulting list of CA certificate filenames.  This is a
 *        a globus list structure.  
 *        @see globus_fifo_t
 * @return
 *        GLOBUS_SUCCESS if no error occurred, otherwise an error object ID
 *        is returned
 */ 
globus_result_t
globus_gsi_sysconfig_get_ca_cert_files_unix(
    char *                              ca_cert_dir,
    globus_fifo_t *                     ca_cert_list)
{
    DIR *                               dir_handle = NULL;
    struct dirent *                     tmp_entry = NULL;
    int                                 file_length;
    char *                              full_filename_path = NULL;
    globus_result_t                     result = GLOBUS_SUCCESS;
    globus_gsi_statcheck_t              status;
    static char *                       _function_name_ =
        "globus_gsi_sysconfig_get_ca_cert_file_unix";

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    if(!ca_cert_dir)
    {
        GLOBUS_GSI_SYSCONFIG_ERROR_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CA_CERT_FILENAMES,
            ("NULL parameter ca_cert_dir passed to function: %s",
             _function_name_));
        goto exit;
    }

    if(!ca_cert_list)
    {
        GLOBUS_GSI_SYSCONFIG_ERROR_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CA_CERT_FILENAMES,
            ("NULL parameter ca_cert_list passed to function: %s",
             _function_name_));
        goto exit;
    }

    dir_handle = globus_libc_opendir(ca_cert_dir);
    if(dir_handle == NULL)
    {
        result = globus_error_put(
            globus_error_wrap_errno_error(
                GLOBUS_GSI_SYSCONFIG_MODULE,
                errno,
                GLOBUS_GSI_SYSCONFIG_ERROR_ERRNO,
                "Error opening directory: %s", ca_cert_dir));
        goto exit;
    }

    while(globus_libc_readdir_r(dir_handle,&tmp_entry) == 0 &&
          tmp_entry != NULL)
    {
        file_length = strlen(tmp_entry->d_name);
        /* check the following:
         * 
         * - file length is greater than or equal to 10
         * - first 8 characters are alpha-numeric
         * - 9th character is '.'
         * - characters after the '.' are numeric
         */

        full_filename_path = 
            globus_common_create_string(
                "%s%s%s", ca_cert_dir, FILE_SEPERATOR, tmp_entry->d_name);
        
        if(full_filename_path == NULL)
        {
            while((full_filename_path =
                   (char *) globus_fifo_dequeue(ca_cert_list)) != NULL)
            {
                free(full_filename_path);
            }
            GLOBUS_GSI_SYSCONFIG_ERROR_RESULT(
                result,
                GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CA_CERT_FILENAMES,
                ("Couldn't get full pathname for CA cert"));
            goto exit;
        }

        if((result = globus_gsi_sysconfig_file_exists_unix(
                full_filename_path,
                &status)) != GLOBUS_SUCCESS)
        {
            free(full_filename_path);
            while((full_filename_path =
                   (char *) globus_fifo_dequeue(ca_cert_list)) != NULL)
            {
                free(full_filename_path);
            }
            GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                result,
                GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_CA_CERT_FILENAMES);
            goto exit;
        }
        
        if(file_length >= (X509_HASH_LENGTH + 2) &&
           (*(tmp_entry->d_name + X509_HASH_LENGTH) == '.') &&
           (strspn(tmp_entry->d_name, "0123456789abcdefABCDEF") 
            == X509_HASH_LENGTH) &&
           (strspn((tmp_entry->d_name + (X509_HASH_LENGTH + 1)), 
                   "0123456789") == (file_length - 9)) &&
           (status == GLOBUS_FILE_VALID))
        {
            globus_fifo_enqueue(ca_cert_list, (void *)full_filename_path);
        }
        else
        {
            free(full_filename_path);
        }

        globus_free(tmp_entry);    
    }

 exit:

    if(dir_handle != NULL)
    {
        globus_libc_closedir(dir_handle);
    }

    if(tmp_entry != NULL)
    {
	globus_libc_free(tmp_entry);
    }

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;

}
/* @} */

/**
 * @name UNIX - Remove all proxies owned by current uid
 * @ingroup globus_gsi_sysconfig_unix
 */
/* @{ */
/**
 * Removes all proxies (ie. all delegated and grid-proxy-init generated
 * proxies) found in the secure tmp directory that are owned by the
 * current user.
 *
 * @param default_filename
 *        The filename of the default proxy
 * @return
 *        GLOBUS_SUCCESS if no error occurred, otherwise an error object ID
 *        is returned
 */ 
globus_result_t
globus_gsi_sysconfig_remove_all_owned_files_unix(
    char *                              default_filename)
{
    struct stat                         stx;
    char *                              full_filename = NULL;
    globus_result_t                     result = GLOBUS_SUCCESS;
    DIR *                               secure_tmp_dir = NULL;
    struct dirent *                     dir_entry = NULL;
    static char *                       _function_name_ =
        "globus_gsi_sysconfig_remove_all_owned_files_unix";
    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    secure_tmp_dir = globus_libc_opendir(DEFAULT_SECURE_TMP_DIR);
    if(!secure_tmp_dir)
    {
        result = globus_error_put(
            globus_error_wrap_errno_error(
                GLOBUS_GSI_SYSCONFIG_MODULE,
                errno,
                GLOBUS_GSI_SYSCONFIG_ERROR_ERRNO,
                __FILE__":%d:%s: Error opening directory: %s\n",
                __LINE__,
                _function_name_, 
                DEFAULT_SECURE_TMP_DIR));
        goto exit;
    }

    while(globus_libc_readdir_r(secure_tmp_dir, &dir_entry) == 0 &&
          dir_entry != NULL)
    {
        if((default_filename && 
            !strcmp(dir_entry->d_name, default_filename)) ||
           !strncmp(dir_entry->d_name,
                    X509_UNIQUE_PROXY_FILE,
                    strlen(X509_UNIQUE_PROXY_FILE)))
        {
            full_filename = globus_common_create_string(
                "%s%s%s",
                DEFAULT_SECURE_TMP_DIR,
                FILE_SEPERATOR,
                dir_entry->d_name);

            if(stat(full_filename, &stx) == -1)
            {
                globus_free(dir_entry);
                continue;
            }

            RAND_add((void *) &stx, sizeof(stx), 2);
                    
            if(stx.st_uid == getuid())
            {
                static char             msg[65]
                    = "DESTROYED BY GLOBUS\r\n";
                int                     f = open(full_filename, O_RDWR);
                int                     size, rec, left;
                if (f) 
                {
                    size = lseek(f, 0L, SEEK_END);
                    lseek(f, 0L, SEEK_SET);
                    if (size > 0) 
                    {
                        rec = size / 64;
                        left = size - rec * 64;
                        while (rec)
                        {
                            write(f, msg, 64);
                            rec--;
                        }
                        if (left)
                        { 
                            write(f, msg, left);
                        }
                    }
                    close(f);
                }
                remove(full_filename);
            }

            free(full_filename);
        }
        globus_free(dir_entry);
    }

 exit:

    if(secure_tmp_dir != NULL)
    {
        globus_libc_closedir(secure_tmp_dir);
    }

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;
}
/* @} */


/**
 * @name UNIX - Check if the current user is root
 * @ingroup globus_gsi_sysconfig_unix
 */
/* @{ */
/**
 * Checks whether the current user is root.
 *
 * @param is_superuser
 *        1 if the user is the superuser
 *        0 if not
 * @return
 *        GLOBUS_SUCCESS if no error occurred, otherwise an error object ID
 *        is returned
 */ 
globus_result_t
globus_gsi_sysconfig_is_superuser_unix(
    int *                               is_superuser)
{
    static char *                       _function_name_ =
        "globus_gsi_sysconfig_is_superuser_unix";
    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    if(getuid() == 0)
    {
        *is_superuser = 1;
    }
    else
    {
        *is_superuser = 0;
    }

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return GLOBUS_SUCCESS;
}
/* @} */

/**
 * @name UNIX - Get the path and file name of the grid map file
 * @ingroup globus_gsi_sysconfig_unix
 */
/* @{ */
/**
 * Get the path and file name of the grid map file.
 *
 * @param filename
 *        Contains the location of the grid map file upon successful return
 * @return
 *        GLOBUS_SUCCESS if no error occurred, otherwise an error object ID
 *        is returned
 */ 
globus_result_t
globus_gsi_sysconfig_get_gridmap_filename_unix(
    char **                             filename)
{
    char *                              home_dir = NULL;
    char *                              gridmap_env = NULL;
    char *                              gridmap_filename = NULL;
    globus_gsi_statcheck_t              status;
    globus_result_t                     result = GLOBUS_SUCCESS;
    static char *                       _function_name_ =
        "globus_gsi_sysconfig_get_gridmap_filename_unix";
    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    if(((gridmap_env = (char *) getenv("GRIDMAP"))   != NULL) ||
       ((gridmap_env = (char *) getenv("GLOBUSMAP")) != NULL) ||
       ((gridmap_env = (char *) getenv("globusmap")) != NULL) ||
       ((gridmap_env = (char *) getenv("GlobusMap")) != NULL))
    {
        gridmap_filename = globus_common_create_string(
            "%s",
            gridmap_env);
        if(!gridmap_filename)
        {
            GLOBUS_GSI_SYSTEM_CONFIG_MALLOC_ERROR;
            goto exit;
        }
    }

    if(!gridmap_filename)
    {
        if(getuid() == 0)
        {
            /* being run as root */
            
            gridmap_filename = globus_common_create_string(
                "%s",
                DEFAULT_GRIDMAP);
            if(!gridmap_filename)
            {
                GLOBUS_GSI_SYSTEM_CONFIG_MALLOC_ERROR;
                goto exit;
            }
        }
        else
        {
            result = GLOBUS_GSI_SYSCONFIG_GET_HOME_DIR(&home_dir, &status);
            if(result != GLOBUS_SUCCESS)
            {
                GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                    result,
                    GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_GRIDMAP_FILENAME);
                goto exit;
            }
            
            if(home_dir && status == GLOBUS_FILE_DIR)
            {
                gridmap_filename = globus_common_create_string(
                    "%s%s%s",
                    home_dir,
                    FILE_SEPERATOR,
                    LOCAL_GRIDMAP);
                if(!gridmap_filename)
                {
                    GLOBUS_GSI_SYSTEM_CONFIG_MALLOC_ERROR;
                    goto exit;
                }
            }
        }
    }

    if(!gridmap_filename)
    {
        GLOBUS_GSI_SYSCONFIG_ERROR_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_GRIDMAP_FILENAME,
            ("A valid gridmap file could not be found."));
        goto exit;
    }

    *filename = gridmap_filename;

 exit:

    if(home_dir != NULL)
    {
        free(home_dir);
    }
    
    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;
}
/* @} */

/**
 * @name UNIX - Get the path and file name of the authorization callback configuration file 
 * @ingroup globus_gsi_sysconfig_unix
 */
/* @{ */
/**
 * Get the path and file name of the authorization callback
 * configuration file 
 *
 * @param filename
 *        Contains the location of the authorization callback configuration
 *        file upon successful return 
 * @return
 *        GLOBUS_SUCCESS if no error occurred, otherwise an error object ID
 *        is returned
 */ 
globus_result_t
globus_gsi_sysconfig_get_authz_conf_filename_unix(
    char **                             filename)
{
    char *                              home_dir = NULL;
    char *                              authz_env = NULL;
    char *                              authz_filename = NULL;
    globus_gsi_statcheck_t              status;
    globus_result_t                     result = GLOBUS_SUCCESS;
    static char *                       _function_name_ =
        "globus_gsi_sysconfig_get_authz_conf_filename_unix";
    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    if((authz_env = (char *) getenv("GSI_AUTHZ_CONF"))   != NULL)
    {
        authz_filename = globus_common_create_string(
            "%s",
            authz_env);
        if(!authz_filename)
        {
            GLOBUS_GSI_SYSTEM_CONFIG_MALLOC_ERROR;
            goto exit;
        }
    }
    
    if(!authz_filename)
    {
        if(getuid() == 0)
        {
            /* being run as root */
            
            authz_filename = globus_common_create_string(
                "%s",
                DEFAULT_AUTHZ_FILE);
            if(!authz_filename)
            {
                GLOBUS_GSI_SYSTEM_CONFIG_MALLOC_ERROR;
                goto exit;
            }
        }
        else
        {
            result = GLOBUS_GSI_SYSCONFIG_GET_HOME_DIR(&home_dir, &status);
            if(result != GLOBUS_SUCCESS)
            {
                GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
                    result,
                    GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_AUTHZ_FILENAME);
                goto exit;
            }
            
            if(home_dir && status == GLOBUS_FILE_DIR)
            {
                authz_filename = globus_common_create_string(
                    "%s%s%s",
                    home_dir,
                    FILE_SEPERATOR,
                    LOCAL_AUTHZ_FILE);
                if(!authz_filename)
                {
                    GLOBUS_GSI_SYSTEM_CONFIG_MALLOC_ERROR;
                    goto exit;
                }
            }
        }
    }

    if(!authz_filename)
    {
        GLOBUS_GSI_SYSCONFIG_ERROR_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_AUTHZ_FILENAME,
            ("A valid authz file could not be found."));
        goto exit;
    }

    result = globus_gsi_sysconfig_check_certfile_unix(
        authz_filename,
        &status);

    if(result != GLOBUS_SUCCESS)
    {
        GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_AUTHZ_FILENAME);
        goto exit;
    }

    /* work around file check result idiocy */
    
    switch(status)
    {
      case GLOBUS_FILE_NOT_OWNED:
      case GLOBUS_FILE_ZERO_LENGTH:
      case GLOBUS_FILE_VALID:
        break;
      case GLOBUS_FILE_INVALID:
        GLOBUS_GSI_SYSCONFIG_ERROR_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_FILE_NOT_REGULAR,
            ("%s is not a valid authorization callout config file",
             authz_filename));
        GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_AUTHZ_FILENAME);
        goto exit;
      case GLOBUS_FILE_DOES_NOT_EXIST:
        GLOBUS_GSI_SYSCONFIG_ERROR_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_FILE_DOES_NOT_EXIST,
            ("%s is not a valid authorization callout config file",
             authz_filename));
        GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_AUTHZ_FILENAME);
        goto exit;
      case GLOBUS_FILE_BAD_PERMISSIONS:
        GLOBUS_GSI_SYSCONFIG_ERROR_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_FILE_BAD_PERMISSIONS,
            ("%s is not a valid authorization callout config file",
             authz_filename));
        GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_AUTHZ_FILENAME);
        goto exit;
      case GLOBUS_FILE_DIR:
        GLOBUS_GSI_SYSCONFIG_ERROR_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_FILE_IS_DIR,
            ("%s is not a valid authorization callout config file",
             authz_filename));
        GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_AUTHZ_FILENAME);
        goto exit;
    }
    
    *filename = authz_filename;

    authz_filename = NULL;

 exit:

    if(home_dir != NULL)
    {
        free(home_dir);
    }

    if(authz_filename != NULL)
    {
        free(authz_filename);
    }
    
    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;
}
/* @} */

#endif /* done defining *_unix functions */

/**
 * @name Get Unique Proxy Filename
 * @ingroup globus_gsi_sysconfig_shared
 */
/* @{ */
/**
 * Get a unique proxy cert filename.  This is mostly used
 * for delegated proxy credentials.  Each filename returned
 * is going to be unique for each time the function is called.
 * 
 * @param unique_filename
 *        the unique filename for a delegated proxy cert
 *
 * @return
 *        GLOBUS_SUCCESS or an error object identifier
 */
globus_result_t
globus_gsi_sysconfig_get_unique_proxy_filename(
    char **                             unique_filename)
{
    char *                              default_unique_filename = NULL;
    globus_result_t                     result;
    char *                              proc_id_string = NULL;
    char                                unique_tmp_name[L_tmpnam];
    char *                              unique_postfix = NULL;
    static int                          i = 0;
    globus_gsi_statcheck_t              status;
    static char *                       _function_name_ =
        "globus_gsi_sysconfig_get_unique_proxy_filename";

    GLOBUS_I_GSI_SYSCONFIG_DEBUG_ENTER;

    memset(unique_tmp_name, 0, L_tmpnam);
    
    *unique_filename = NULL;

    result = GLOBUS_GSI_SYSCONFIG_GET_PROC_ID_STRING(&proc_id_string);
    if(result != GLOBUS_SUCCESS)
    {
        proc_id_string = NULL;
        GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_DELEG_FILENAME);
        goto done;
    }

    if(tmpnam(unique_tmp_name) == NULL)
    {
        GLOBUS_GSI_SYSCONFIG_ERROR_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_DELEG_FILENAME,
            ("Could not get a unique filename for the temporary proxy cert"));
        goto done;
    }
    
    unique_postfix = strrchr(unique_tmp_name, '/');
    ++unique_postfix;

    if((result = globus_i_gsi_sysconfig_create_key_string(
            unique_filename,
            & default_unique_filename,
            & status,
            "%s%s%s%s.%s.%d",
            DEFAULT_SECURE_TMP_DIR,
            FILE_SEPERATOR,
            X509_UNIQUE_PROXY_FILE,
            proc_id_string,
            unique_postfix,
            ++i)) != GLOBUS_SUCCESS)
    {
        GLOBUS_GSI_SYSCONFIG_ERROR_CHAIN_RESULT(
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_DELEG_FILENAME);
        goto done;
    }

    *unique_filename = default_unique_filename;

    if(!(*unique_filename))
    {            
        GLOBUS_GSI_SYSCONFIG_ERROR_RESULT( 
            result,
            GLOBUS_GSI_SYSCONFIG_ERROR_GETTING_DELEG_FILENAME,
            ("A file location for writing the unique proxy cert"
             " could not be found in: %s\n",
             default_unique_filename));
        
        goto done;
    }

    result = GLOBUS_SUCCESS;

 done:

    if(default_unique_filename && 
       (default_unique_filename != (*unique_filename)))
    {
        globus_libc_free(default_unique_filename);
    }

    if(proc_id_string != NULL)
    {
        free(proc_id_string);
    }
    
    GLOBUS_I_GSI_SYSCONFIG_DEBUG_EXIT;
    return result;
}

/* @} */
