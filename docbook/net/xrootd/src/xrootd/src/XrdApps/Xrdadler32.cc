/************************************************************************/
/* Xrdadler32.cc                                                        */
/*                                                                      */
/* Auther: Wei Yang                                                     */
/* SLAC National Accelerator Laboratory / Stanford University, 2009     */
/*                                                                      */
/* Calculating Adler32 checksum of a local unix file (including stdin)  */
/* and file on a remote xrootd data server. Support using XROOTD_VMP.   */
/************************************************************************/

//   $Id$

const char *Xrdadler32CVSID = "$Id$";

#define _FILE_OFFSET_BITS 64

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#ifdef __linux__
  #include <sys/xattr.h>
#endif
#include <zlib.h>

#include "XrdPosix/XrdPosixExtern.hh"
#include "XrdPosix/XrdPosixXrootd.hh"
#include "XrdClient/XrdClientUrlInfo.hh"
#include "XrdClient/XrdClientConst.hh"
#include "XrdClient/XrdClient.hh"
#include "XrdClient/XrdClientEnv.hh"
#include "XrdClient/XrdClientAdmin.hh"
#include "XrdOuc/XrdOucString.hh"

void fSetXattrAdler32(int fd, const char* attr, const char *value)
{
    struct stat st;
    char mtime[12], attr_val[25];
    int rc;
    
    rc = fstat(fd, &st);
    if (rc < 0 || strlen(value) != 8) 
        return; 
    else
        sprintf(mtime, "%ld", st.st_mtime);

    strcpy(attr_val, value);
    strcat(attr_val, ":"); 
    strcat(attr_val, mtime);

#if defined(__linux__)
    rc = fsetxattr(fd, attr, attr_val, strlen(attr_val), 0x0);
#elif defined(__solaris__)
    int attrfd;
    attrfd = openat(fd, attr, O_XATTR|O_CREAT|O_TRUNC|O_WRONLY); 
    if (attrfd < 0) return;

    rc = write(attrfd, attr_val, strlen(attr_val));
/*
   Solaris extended attributes are files in orthogonal namespace.
   Their permission wont' change according to real files.
 */
    fchmod(attrfd, S_IRWXU|S_IRGRP|S_IROTH);
    close(attrfd);
#endif
    return;
}

int fGetXattrAdler32(int fd, const char* attr, char *value)
{
    struct stat st;
    char mtime[12], attr_val[25], *p;
    int rc;

    rc = fstat(fd, &st);
    if (rc < 0)
        return(0);
    else
        sprintf(mtime, "%ld", st.st_mtime);


#if defined(__linux__)
    rc = fgetxattr(fd, attr, attr_val, 25);
#elif defined(__solaris__)
    int attrfd;
    attrfd = openat(fd, attr, O_XATTR|O_RDONLY);
    if (attrfd < 0) return(0);

    rc = read(attrfd, attr_val, 25);
    close(attrfd);
#else
    return(0);
#endif

    if (rc == -1 || attr_val[8] != ':') return(0);
    attr_val[8] = '\0';
    attr_val[rc] = '\0';
    p = attr_val + 9;
     
    if (strcmp(p, mtime)) return(0);

    strcpy(value, attr_val);
    return(strlen(value));
}

/* get the actual root url pointing to the data server */
char get_current_url(const char *oldurl, char *newurl)
{
    bool stat;
    long id, flags, modtime;
    long long size;
    XrdOucString url(oldurl);

    XrdClientAdmin *adm = new XrdClientAdmin(url.c_str());
    if (adm->Connect())
    {
        XrdClientUrlInfo u(url);

        stat = adm->Stat((char *)u.File.c_str(), id, size, flags, modtime);
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

/* the rooturl should point to the data server, not redirector */
char getchksum(const char *rooturl, char *chksum) 
{
    XrdOucString url(rooturl);
    char *sum = 0, *ptb, *pte;
    long sumlen; 
    int  pte_ptb;

    XrdClientAdmin *adm = new XrdClientAdmin(url.c_str());
    if (adm->Connect()) 
    {
        XrdClientUrlInfo u(url);
        sumlen = adm->GetChecksum((kXR_char *)u.File.c_str(), (kXR_char**) &sum);
        pte = ptb = sum;
        if (sumlen != 0)
        {
            ptb = strchr(sum, ' ');
            ptb++;
            pte = strchr(ptb, ' ');
            if (pte == NULL) pte = &sum[sumlen];
        }
        pte_ptb = pte - ptb;
        strncpy(chksum, ptb, pte_ptb);
        chksum[pte_ptb] = '\0';
        free(sum);
        delete adm;
        return pte_ptb;  /* 0 means sever doesn't implement a checksum */
    }
    else
        return -1;
}

#define N 64*1024  /* reading block size */

int main(int argc, char *argv[])
{
    char path[2048], chksum[128], buf[N], adler_str[9];
    const char attr[] = "user.checksum.adler32";
    struct stat stbuf;
    int fd, len, rc;
    uLong adler;
    adler = adler32(0L, Z_NULL, 0);

    if (argc == 2 && ! strcmp(argv[1], "-h"))
    {
        printf("Usage: %s file. Calculating adler32 checksum of a given file.\n", argv[0]);
        printf("A file can be local file, stdin (if omitted), or root URL (including via XROOTD_VMP)\n");
        return 0;
    }

    path[0] = '\0';
    if (argc > 1)  /* trying to convert to root URL */
    {
        if (!strncmp(argv[1], "root://", 7))
            strcpy(path, argv[1]);
        else  
            XrdPosix_URL(argv[1], path, sizeof(path));
    }
    if (argc == 1 || path[0] == '\0')
    {                        /* this is a local file */
        if (argc > 1) 
        {
            strcpy(path, argv[1]);
            rc = stat(path, &stbuf);        
            if (rc != 0 || ! S_ISREG(stbuf.st_mode) ||
                (fd = open(path,O_RDONLY)) < 0) 
            {
                printf("Error_accessing %s\n", path);
                return 1;
            }
            else  /* see if the adler32 is saved in attribute already */
                if (fGetXattrAdler32(fd, attr, adler_str) == 8)
                {
                    printf("%s %s\n", adler_str, path);
                    return 0;
                }
        }
        else 
        {
            fd = STDIN_FILENO;
            strcpy(path, "-");
        }
        while ( (len = read(fd, buf, N)) > 0 )
            adler = adler32(adler, (const Bytef*)buf, len);

        if (fd != STDIN_FILENO) 
        {   /* try saving adler32 to attribute before close() */
            sprintf(adler_str, "%08lx", adler);
            fSetXattrAdler32(fd, attr, adler_str);
            close(fd);
        }
        printf("%08lx %s\n", adler, path);
        return 0;
    }
    else
    {                       /* this is a Xrootd file */
        EnvPutInt(NAME_DEBUG, -1);
        if (!get_current_url(path, path))
        {
            printf("Error_accessing: %s\n", argv[1]);
            return 1;
        }

        if (getchksum(path, chksum) > 0) 
        {                   /* server implements checksum */
             printf("%s %s\n", chksum, argv[1]);
             return (strcmp(chksum, "Error_accessing:") ? 0 : 1);
        }
        else
        {                   /* need to read the file and calculate */
            EnvPutInt(NAME_READAHEADSIZE, N);
            EnvPutInt(NAME_READCACHESIZE, 2*N);
            rc = XrdPosixXrootd::Stat(path, &stbuf);
            if (rc != 0 || ! S_ISREG(stbuf.st_mode) ||
                (fd = XrdPosixXrootd::Open(path, O_RDONLY, S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH)) < 0)
            {
                printf("Error_accessing: %s\n", argv[1]);
                return 1;
            }
            while ( (len = XrdPosixXrootd::Read(fd, buf, N)) > 0 )
                adler = adler32(adler, (const Bytef*)buf, len);

            XrdPosixXrootd::Close(fd);
            printf("%08lx %s\n", adler, argv[1]);
            return 0;
        }
    }
}
