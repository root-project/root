/******************************************************************************/
/* XrdFfsMisc.hh  Miscellanies functions                                      */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/* Author: Wei Yang (SLAC National Accelerator Laboratory, 2009)              */
/*         Contract DE-AC02-76-SFO0515 with the Department of Energy          */
/******************************************************************************/

#ifdef __cplusplus
  extern "C" {
#endif

#define XrdFfs_MAX_NUM_NODES 4096 /* 64*64 max number of data nodes in a cluster */

char XrdFfsMisc_get_current_url(const char *oldurl, char *newurl);
int XrdFfsMisc_get_all_urls(const char *oldurl, char **newurls, const int nnodes);
int XrdFfsMisc_get_list_of_data_servers(char* list);
void XrdFfsMisc_refresh_url_cache(const char* url);
void XrdFfsMisc_logging_url_cache(const char* url);

void XrdFfsMisc_xrd_init(const char *rdrurl, int startQueue);

void XrdFfsMisc_xrd_secsss_init();
void XrdFfsMisc_xrd_secsss_register(uid_t user_uid, gid_t user_gid);
void XrdFfsMisc_xrd_secsss_editurl(char *url, uid_t user_uid);

#ifdef __cplusplus
  }
#endif
