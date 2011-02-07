/******************************************************************************/
/* XrdFfsMisc.cc  Miscellanies functions                                      */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/* Author: Wei Yang (SLAC National Accelerator Laboratory, 2009)              */
/*         Contract DE-AC02-76-SFO0515 with the Department of Energy          */
/******************************************************************************/

#define _FILE_OFFSET_BITS 64
#include <string.h>
#include <sys/types.h>
//#include <sys/xattr.h>
#include <iostream>
#include <libgen.h>
#include <unistd.h>
#include <netdb.h>
#include <pwd.h>
#include <grp.h>
#include <time.h>
#include <pthread.h>
#include <syslog.h>

#include "XrdClient/XrdClientUrlInfo.hh"
#include "XrdClient/XrdClientEnv.hh"
#include "XrdClient/XrdClientConst.hh"
#include "XrdClient/XrdClient.hh"
#include "XrdClient/XrdClientAdmin.hh"
#include "XrdOuc/XrdOucString.hh"
#include "XrdSec/XrdSecEntity.hh"
#include "XrdSecsss/XrdSecsssID.hh"
#include "XrdFfs/XrdFfsDent.hh"
#include "XrdFfs/XrdFfsFsinfo.hh"
#include "XrdFfs/XrdFfsMisc.hh"
#include "XrdFfs/XrdFfsPosix.hh"
#include "XrdFfs/XrdFfsQueue.hh"

#ifdef __cplusplus
  extern "C" {
#endif

char XrdFfsMisc_get_current_url(const char *oldurl, char *newurl) 
{
    bool stat;
    long id, flags, modtime;
    long long size;
    struct stat stbuf;

/* if it is a directory, return oldurl */
    if (XrdFfsPosix_stat(oldurl, &stbuf) == 0 && S_ISDIR(stbuf.st_mode))
    {
        strcpy(newurl, oldurl);
        return 1;
    }

    const char* tmp = &oldurl[7];
    const char* p = index(tmp,'/');
    tmp = p+1;
    XrdOucString path = tmp;

    XrdOucString url(oldurl);
    XrdClientAdmin *adm = new XrdClientAdmin(url.c_str());
    if (adm->Connect()) 
    {
        stat = adm->Stat((char *)path.c_str(), id, size, flags, modtime);
// We might have been redirected to a destination server. Better 
// to remember it and use only this one as output.
        if (stat && adm->GetCurrentUrl().IsValid()) 
        {
            strcpy(newurl, adm->GetCurrentUrl().GetUrl().c_str());
            delete adm;
            return 1;
        }
    }
    delete adm;
    return 0;
}

uint32_t XrdFfsMisc_ip2nl(char *ip)
{
    uint32_t ipn = 0;
    char *n, tmp[16];

    strcpy(tmp, ip);
    ip = tmp;

    n = strchr(ip, '.');
    n[0] = '\0';
    ipn += atoi(ip) * 256 * 256 * 256;
    ip = n + 1;

    n = strchr(ip, '.');
    n[0] = '\0';
    ipn += atoi(ip) * 256 * 256;
    ip = n + 1;

    n = strchr(ip, '.');
    n[0] = '\0';
    ipn += atoi(ip) * 256;
    ip = n + 1;

    ipn += atoi(ip);

    return htonl(ipn);
}

char* XrdFfsMisc_getNameByAddr(char* ipaddr)
{
    char *ipname;
    struct hostent *host;
    uint32_t ip;
    ip = XrdFfsMisc_ip2nl(ipaddr);
    host = gethostbyaddr(&ip, 4, AF_INET);
    ipname = (char*)malloc(256);
    strcpy(ipname, host->h_name);
    return ipname;
}

int XrdFfsMisc_get_all_urls_real(const char *oldurl, char **newurls, const int nnodes)
{
    int rval = 0;

    const char* tmp = &oldurl[7];
    const char* p = index(tmp,'/');
    tmp = p+1;
    XrdOucString path = tmp;

    XrdOucString url = oldurl;
    XrdClientAdmin *adm = new XrdClientAdmin(url.c_str());

    XrdClientVector<XrdClientLocate_Info> allhosts;
    XrdClientLocate_Info host;

    if (adm->Connect())
    {
        adm->Locate((kXR_char *)path.c_str(), allhosts);
        if (allhosts.GetSize() > nnodes) 
        {
            rval = -1; /* array newurls doesn't have enough elements */
        }
        else 
            while (allhosts.GetSize())
            {
                host = allhosts.Pop_front();
                strcpy(newurls[rval],"root://");
                strcat(newurls[rval],(char*)host.Location);
                strcat(newurls[rval],"/");
                strcat(newurls[rval],(char*)path.c_str());
                if (host.Infotype == XrdClientLocate_Info::kXrdcLocManager ||
                    host.Infotype == XrdClientLocate_Info::kXrdcLocManagerPending)
                    rval = rval + XrdFfsMisc_get_all_urls(newurls[rval], &newurls[rval], nnodes - rval);
                else
                    rval++;
            }
    }
    delete adm;
    return rval;
}

/*
   function XrdFfsMisc_get_all_urls() has the same interface as XrdFfsMisc_get_all_urls_real(), but 
   used a cache to reduce unnecessary queries to the redirector 
*/ 

char XrdFfsMiscCururl[1024] = "";
char *XrdFfsMiscUrlcache[XrdFfs_MAX_NUM_NODES];
int XrdFfsMiscNcachedurls = 0;
time_t XrdFfsMiscUrlcachetime = 0;
pthread_mutex_t XrdFfsMiscUrlcache_mutex = PTHREAD_MUTEX_INITIALIZER;

int XrdFfsMisc_get_all_urls(const char *oldurl, char **newurls, const int nnodes)
{
    time_t currtime;
    int i, nurls;

    pthread_mutex_lock(&XrdFfsMiscUrlcache_mutex); 

    currtime = time(NULL);
/* set the cache to not expired in 10 years so that we know if a host is down */
    if (XrdFfsMiscCururl[0] == '\0' || (currtime - XrdFfsMiscUrlcachetime) > 315360000 || strcmp(XrdFfsMiscCururl, oldurl) != 0)
    {
        for (i = 0; i < XrdFfsMiscNcachedurls; i++)
            if (XrdFfsMiscUrlcache[i] != NULL) free(XrdFfsMiscUrlcache[i]);
        for (i = 0; i < XrdFfs_MAX_NUM_NODES; i++)
            XrdFfsMiscUrlcache[i] = (char*) malloc(1024);
        
        XrdFfsMiscUrlcachetime = currtime;
        strcpy(XrdFfsMiscCururl, oldurl);
        XrdFfsMiscNcachedurls = XrdFfsMisc_get_all_urls_real(oldurl, XrdFfsMiscUrlcache, nnodes);
        for (i = XrdFfsMiscNcachedurls; i < XrdFfs_MAX_NUM_NODES; i++)
            if (XrdFfsMiscUrlcache[i] != NULL) free(XrdFfsMiscUrlcache[i]);
    }

    nurls = XrdFfsMiscNcachedurls;
    for (i = 0; i < nurls; i++)
    {
        newurls[i] = (char*) malloc(1024);
        strcpy(newurls[i], XrdFfsMiscUrlcache[i]);
    }

    pthread_mutex_unlock(&XrdFfsMiscUrlcache_mutex);
    return nurls;
}

int XrdFfsMisc_get_list_of_data_servers(char* list)
{
    int i;
    char *url, *rc, *hostname, *hostip, *port, *p;
  
    rc = (char*)malloc(sizeof(char) * XrdFfs_MAX_NUM_NODES * 1024);
    rc[0] = '\0';
    pthread_mutex_lock(&XrdFfsMiscUrlcache_mutex);
    for (i = 0; i < XrdFfsMiscNcachedurls; i++)
    {
        url = strdup(XrdFfsMiscUrlcache[i]); 
        hostip = &url[7];
        p = strchr(hostip, ':');
        p[0] = '\0';
        port = ++p;
        p = strchr(port, '/');
        p[0] = '\0';

        hostname = XrdFfsMisc_getNameByAddr(hostip);
        strcat(rc, hostname); 
        strcat(rc, ":");
        strcat(rc, port);
        strcat(rc, "\n");
        free(hostname);
        free(url);
    }
    pthread_mutex_unlock(&XrdFfsMiscUrlcache_mutex);
    strcpy(list, rc);
    free(rc);
    return i;
}

void XrdFfsMisc_refresh_url_cache(const char* url)
{
    int i, nurls;
    char *surl, **turls;

    turls = (char**) malloc(sizeof(char*) * XrdFfs_MAX_NUM_NODES);

// invalid the cache first
    pthread_mutex_lock(&XrdFfsMiscUrlcache_mutex);
    XrdFfsMiscUrlcachetime = 0;
    pthread_mutex_unlock(&XrdFfsMiscUrlcache_mutex);

    if (url != NULL)
        surl = strdup(url);
    else
        surl = strdup(XrdFfsMiscCururl);

    nurls = XrdFfsMisc_get_all_urls(surl, turls, XrdFfs_MAX_NUM_NODES);

    free(surl);
    for (i = 0; i < nurls; i++) free(turls[i]);
    free(turls);
}

void XrdFfsMisc_xrd_init(const char *rdrurl, int startQueue)
{
    static int OneTimeInitDone = 0;

// Do not execute this more than once
//
   if (OneTimeInitDone) return;
   OneTimeInitDone = 1;

//    EnvPutInt(NAME_RECONNECTTIMEOUT,20);
    EnvPutInt(NAME_FIRSTCONNECTMAXCNT,2);
    EnvPutInt(NAME_DATASERVERCONN_TTL, 3600);
//    EnvPutInt(NAME_READAHEADSIZE,DFLT_READAHEADSIZE);
//    EnvPutInt(NAME_READCACHESIZE,DFLT_READCACHESIZE);
//    EnvPutInt(NAME_READAHEADSIZE,64*1024);
//    EnvPutInt(NAME_READCACHESIZE,64*64*1024);
    EnvPutInt(NAME_READAHEADSIZE,0);
    EnvPutInt(NAME_READCACHESIZE,0);
    EnvPutInt(NAME_REQUESTTIMEOUT, 30);
    EnvPutInt(NAME_FIRSTCONNECTMAXCNT, 2);

    if (getenv("XROOTDFS_SECMOD") != NULL && !strcmp(getenv("XROOTDFS_SECMOD"), "sss"))
        XrdFfsMisc_xrd_secsss_init();

    int i;
    char *hostlist, *p1, *p2;

    XrdFfsMisc_refresh_url_cache(rdrurl);

    hostlist = (char*) malloc(sizeof(char) * XrdFfs_MAX_NUM_NODES * 1024);
    i = XrdFfsMisc_get_list_of_data_servers(hostlist);

    openlog("XrootdFS", LOG_ODELAY | LOG_PID, LOG_DAEMON);
    syslog(LOG_INFO, "INFO: Starting with %d data servers :", i);
    p1 = hostlist;
    p2 = strchr(p1, '\n');
    while (p2 != NULL)
    {
        p2[0] = '\0';
        syslog(LOG_INFO, "   %s", p1);
        p1 = p2 +1;
        p2 = strchr(p1, '\n');
    }
//    closelog();

    free(hostlist);

#ifndef NOUSE_QUEUE
   if (startQueue)
   {
       if (getenv("XROOTDFS_NWORKERS") != NULL)
           XrdFfsQueue_create_workers(atoi(getenv("XROOTDFS_NWORKERS")));
       else
           XrdFfsQueue_create_workers(4);

       syslog(LOG_INFO, "INFO: Starting %d workers", XrdFfsQueue_count_workers());
   }
#endif

    XrdFfsDent_cache_init();
}


/*  SSS security module */

XrdSecEntity *XrdFfsMiscUent;
XrdSecsssID *XrdFfsMiscSssid;
bool XrdFfsMiscSecsss = false;
pthread_mutex_t XrdFfsMiscSecsss_mutex = PTHREAD_MUTEX_INITIALIZER;

void XrdFfsMisc_xrd_secsss_init()
{
    XrdFfsMiscSecsss = true;
    XrdFfsMiscUent = new XrdSecEntity("");
    XrdFfsMiscSssid = new XrdSecsssID(XrdSecsssID::idDynamic);
}

void XrdFfsMisc_xrd_secsss_register(uid_t user_uid, gid_t user_gid)
{
    struct passwd *pw;
    struct group *gr;
    char user_num[9];

    if (XrdFfsMiscSecsss)
    {
        sprintf(user_num, "%d", user_uid);
        pw = getpwuid(user_uid);
        gr = getgrgid(user_gid);

        pthread_mutex_lock(&XrdFfsMiscSecsss_mutex);
    
        XrdFfsMiscUent->name = pw->pw_name;
        XrdFfsMiscUent->grps = gr->gr_name;
        XrdFfsMiscSssid->Register(user_num, XrdFfsMiscUent, 1);

        pthread_mutex_unlock(&XrdFfsMiscSecsss_mutex);
    }
}

void XrdFfsMisc_xrd_secsss_editurl(char *url, uid_t user_uid)
{
    char user_num[9], nurl[1024];

    if (XrdFfsMiscSecsss)
    {
        sprintf(user_num, "%d", user_uid);
     
        nurl[0] = '\0';
        strcat(nurl, "root://");
        strcat(nurl, user_num);
        strcat(nurl, "@");
        strcat(nurl, &(url[7])); 
        strcpy(url, nurl);
    }
}

#ifdef __cplusplus
  }
#endif
