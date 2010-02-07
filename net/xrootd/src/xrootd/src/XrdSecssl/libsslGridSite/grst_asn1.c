
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <string.h>

#ifndef GRST_NO_OPENSSL
#include <openssl/x509_vfy.h>
#include <openssl/err.h>
#include <openssl/pem.h>

#include <openssl/buffer.h>
#include <openssl/objects.h>
#include <openssl/asn1.h>
#endif

#ifdef SUNCC
time_t timegm (struct tm *tm) {
  time_t ret;
  char *tz;

  tz = getenv("TZ");
  setenv("TZ", "", 1);
  tzset();
  ret = mktime(tm);
  if (tz)
    setenv("TZ", tz, 1);
  else
    unsetenv("TZ");
  tzset();
  return ret;
}
#endif

#include "gridsite.h"

#ifdef R__SSL_GE_098
#define SSLARG const unsigned char**
#else
#define SSLARG unsigned char**
#endif

/// ASN1 time string (in a char *) to time_t
/**
 *  (Use ASN1_STRING_data() to convert ASN1_GENERALIZEDTIME to char * if
 *   necessary)
 */
                                                                                
time_t GRSTasn1TimeToTimeT(unsigned char *asn1time, size_t len)
{
   char   zone;
   struct tm time_tm;
   
   if (len == 0) len = strlen((char*)asn1time);
                                                                                
   if ((len != 13) && (len != 15)) return 0; /* dont understand */
                                                                                
   if ((len == 13) &&
       ((sscanf((char*)asn1time, "%02d%02d%02d%02d%02d%02d%c",
         &(time_tm.tm_year),
         &(time_tm.tm_mon),
         &(time_tm.tm_mday),
         &(time_tm.tm_hour),
         &(time_tm.tm_min),
         &(time_tm.tm_sec),
         &zone) != 7) || (zone != 'Z'))) return 0; /* dont understand */
                                                                                
   if ((len == 15) &&
       ((sscanf((char*)asn1time, "20%02d%02d%02d%02d%02d%02d%c",
         &(time_tm.tm_year),
         &(time_tm.tm_mon),
         &(time_tm.tm_mday),
         &(time_tm.tm_hour),
         &(time_tm.tm_min),
         &(time_tm.tm_sec),
         &zone) != 7) || (zone != 'Z'))) return 0; /* dont understand */
                                                                                
   /* time format fixups */
                                                                                
   if (time_tm.tm_year < 90) time_tm.tm_year += 100;
   --(time_tm.tm_mon);
                                                                                
   return timegm(&time_tm);
}

/* this function is taken from OpenSSL without modification */

static int asn1_print_info(BIO *bp, int tag, int xclass, int constructed,
	     int indent)
	{
	static const char fmt[]="%-18s";
	static const char fmt2[]="%2d %-15s";
	char str[128];
	const char *p,*p2=NULL;

	if (constructed & V_ASN1_CONSTRUCTED)
		p="cons: ";
	else
		p="prim: ";
	if (BIO_write(bp,p,6) < 6) goto err;
#if OPENSSL_VERSION_NUMBER >= 0x0090701fL
	BIO_indent(bp,indent,128);
#endif

	p=str;
	if ((xclass & V_ASN1_PRIVATE) == V_ASN1_PRIVATE)
			sprintf(str,"priv [ %d ] ",tag);
	else if ((xclass & V_ASN1_CONTEXT_SPECIFIC) == V_ASN1_CONTEXT_SPECIFIC)
		sprintf(str,"cont [ %d ]",tag);
	else if ((xclass & V_ASN1_APPLICATION) == V_ASN1_APPLICATION)
		sprintf(str,"appl [ %d ]",tag);
	else p = ASN1_tag2str(tag);

	if (p2 != NULL)
		{
		if (BIO_printf(bp,fmt2,tag,p2) <= 0) goto err;
		}
	else
		{
		if (BIO_printf(bp,fmt,p) <= 0) goto err;
		}
	return(1);
err:
	return(0);
	}

static void GRSTasn1AddToTaglist(struct GRSTasn1TagList taglist[], 
                           int maxtag, int *lasttag,
                           char *treecoords, int start, int headerlength,
                           int length, int tag)
{
   if ((strlen(treecoords) > GRST_ASN1_MAXCOORDLEN) ||
       (*lasttag + 1 > maxtag)) return;
   
   ++(*lasttag);
   
   strncpy(taglist[*lasttag].treecoords, treecoords, GRST_ASN1_MAXCOORDLEN+1);
   taglist[*lasttag].start = start;
   taglist[*lasttag].headerlength = headerlength;
   taglist[*lasttag].length = length;
   taglist[*lasttag].tag = tag;
}

int GRSTasn1SearchTaglist(struct GRSTasn1TagList taglist[], 
                                 int lasttag, char *treecoords)
{
   int i;
   
   for (i=0; i <= lasttag; ++i)
      {
        if (strcmp(treecoords, taglist[i].treecoords) == 0) return i;
      }
      
   return -1;
}

static int GRSTasn1PrintPrintable(BIO *bp, char *str, int length)
{
   int   ret = 0;
   char *dup, *p;
   
   dup = (char*)malloc(length);
   strncpy(dup,str,length);

   for (p=dup; *p != '\0'; ++p) if ((*p < ' ') || (*p > '~')) *p = '.';

   if (bp != NULL) ret = BIO_write(bp, dup, strlen(dup));

   free(dup);
   
   return ret;
}

static int GRSTasn1Parse2(BIO *bp, unsigned char **pp, long length, int offset,
	     int depth, int indent, int dump, char *treecoords,
	     struct GRSTasn1TagList taglist[], int maxtag, int *lasttag)
	{
        int sibling = 0;
        char sibtreecoords[512];

	unsigned char *p,*ep,*tot,*op,*opp;
	long len;
	int tag,xclass,ret=0;
	int nl,hl,j,r;
	ASN1_OBJECT *o=NULL;
	ASN1_OCTET_STRING *os=NULL;
	int dump_indent;


	dump_indent = 6;	/* Because we know BIO_dump_indent() */
	p= *pp;
	tot=p+length;
	op=p-1;
	while ((p < tot) && (op < p))
		{
		op=p;
		j=ASN1_get_object((SSLARG)&p,&len,&tag,&xclass,length);

		if (j & 0x80)
			{
			if ((bp != NULL) && 
			    (BIO_write(bp,"Error in encoding\n",18) <= 0))
				goto end;
			ret=0;
			goto end;
			}
		hl=(p-op);
		length-=hl;

		++sibling;
		sprintf(sibtreecoords, "%s-%d", treecoords, sibling);

                GRSTasn1AddToTaglist(taglist, maxtag, lasttag, sibtreecoords,
                               (int)offset+(int)(op - *pp),
                               (int) hl, len, tag);
                               
		if (bp != NULL)
		  {
		    BIO_printf(bp, "  %s %ld %d %ld %d ", sibtreecoords,
		           (long)offset+(long)(op - *pp), hl, len, tag);

		    GRSTasn1PrintPrintable(bp, (char*)p,
//		                   &((*pp)[(long)offset+(long)(op - *pp)+hl]),
		                           (len > 30) ? 30 : len);

		    BIO_printf(bp, "\n");
		 }


		/* if j == 0x21 it is a constructed indefinite length object */
		if ((bp != NULL) &&
		    (BIO_printf(bp,"%5ld:",(long)offset+(long)(op- *pp))
			<= 0)) goto end;

		if (j != (V_ASN1_CONSTRUCTED | 1))
			{
			if ((bp != NULL) && 
			    (BIO_printf(bp,"d=%-2d hl=%ld l=%4ld ",
				depth,(long)hl,len) <= 0))
				goto end;
			}
		else
			{
			if ((bp != NULL) &&
			    (BIO_printf(bp,"d=%-2d hl=%ld l=inf  ",
				depth,(long)hl) <= 0))
				goto end;
			}
		if ((bp != NULL) && 
		    !asn1_print_info(bp,tag,xclass,j,(indent)?depth:0))
			goto end;
		if (j & V_ASN1_CONSTRUCTED)
			{
			ep=p+len;
			if ((bp != NULL) &&
			    (BIO_write(bp,"\n",1) <= 0)) goto end;
			if (len > length)
				{
				if (bp != NULL) BIO_printf(bp,
					"length is greater than %ld\n",length);
				ret=0;
				goto end;
				}
			if ((j == 0x21) && (len == 0))
				{
				for (;;)
					{
					r=GRSTasn1Parse2(bp,&p,(long)(tot-p),
						offset+(p - *pp),depth+1,
						indent,dump,sibtreecoords,
						taglist, maxtag, lasttag);
					if (r == 0) { ret=0; goto end; }
					if ((r == 2) || (p >= tot)) break;
					}
				}
			else
				while (p < ep)
					{
					r=GRSTasn1Parse2(bp,&p,(long)len,
						offset+(p - *pp),depth+1,
						indent,dump,sibtreecoords,
						taglist, maxtag, lasttag);
					if (r == 0) { ret=0; goto end; }
					}
			}
		else if (xclass != 0)
			{
			p+=len;
			if ((bp != NULL) && 
			    (BIO_write(bp,"\n",1) <= 0)) goto end;
			}
		else
			{
			nl=0;
			if (	(tag == V_ASN1_PRINTABLESTRING) ||
				(tag == V_ASN1_T61STRING) ||
				(tag == V_ASN1_IA5STRING) ||
				(tag == V_ASN1_VISIBLESTRING) ||
				(tag == V_ASN1_UTCTIME) ||
				(tag == V_ASN1_GENERALIZEDTIME))
				{
				if ((bp != NULL) &&
				    (BIO_write(bp,":",1) <= 0)) goto end;
				if ((len > 0) && (bp != NULL) &&
					BIO_write(bp,(char *)p,(int)len)
					!= (int)len)
					goto end;
				}
			else if (tag == V_ASN1_OBJECT)
				{
				opp=op;
				if (d2i_ASN1_OBJECT(&o,(SSLARG)&opp,len+hl) != NULL)
					{
					if (bp != NULL)
					  {
					    if (BIO_write(bp,":",1) <= 0) goto end;
					    i2a_ASN1_OBJECT(bp,o);
					  }
					}
				else
					{
					if ((bp != NULL) && 
					    (BIO_write(bp,":BAD OBJECT",11) <= 0))
						goto end;
					}
				}
			else if (tag == V_ASN1_BOOLEAN)
				{
				int ii;

				opp=op;
				ii=d2i_ASN1_BOOLEAN(NULL,(SSLARG)&opp,len+hl);
				if (ii < 0)
				{
				  if ((bp != NULL) &&
				      (BIO_write(bp,"Bad boolean\n",12)))
						goto end;
				}
				if (bp != NULL) BIO_printf(bp,":%d",ii);
				}
			else if (tag == V_ASN1_BMPSTRING)
				{
				/* do the BMP thang */
				}
			else if (tag == V_ASN1_OCTET_STRING)
				{
				opp=op;
				os=d2i_ASN1_OCTET_STRING(NULL,(SSLARG) &opp,len+hl);
				if (os != NULL)
					{
					opp=os->data;

					if (os->length > 0)
					  {
					    if ((bp != NULL) &&
						    (BIO_write(bp,":",1) <= 0))
							goto end;
					    if ((bp != NULL) &&
					        (GRSTasn1PrintPrintable(bp,
					                (char*)opp,
							os->length) <= 0))
							goto end;
					  }

					M_ASN1_OCTET_STRING_free(os);
					os=NULL;
					}
				}
			else if (tag == V_ASN1_INTEGER)
				{
				ASN1_INTEGER *bs;
				int i;

				opp=op;
				bs=d2i_ASN1_INTEGER(NULL,(SSLARG)&opp,len+hl);
				if (bs != NULL)
					{
					if ((bp != NULL) &&
					    (BIO_write(bp,":",1) <= 0)) goto end;
					if (bs->type == V_ASN1_NEG_INTEGER)
						if ((bp != NULL) &&
						    (BIO_write(bp,"-",1) <= 0))
							goto end;
					for (i=0; i<bs->length; i++)
						{
						if ((bp != NULL) &&
						    (BIO_printf(bp,"%02X",
							bs->data[i]) <= 0))
							goto end;
						}
					if (bs->length == 0)
						{
						if ((bp != NULL) && 
						    (BIO_write(bp,"00",2) <= 0))
							goto end;
						}
					}
				else
					{
					if ((bp != NULL) && 
					    (BIO_write(bp,"BAD INTEGER",11) <= 0))
						goto end;
					}
				M_ASN1_INTEGER_free(bs);
				}
			else if (tag == V_ASN1_ENUMERATED)
				{
				ASN1_ENUMERATED *bs;
				int i;

				opp=op;
				bs=d2i_ASN1_ENUMERATED(NULL,(SSLARG) &opp,len+hl);
				if (bs != NULL)
					{
					if ((bp != NULL) &&
					    (BIO_write(bp,":",1) <= 0)) goto end;
					if (bs->type == V_ASN1_NEG_ENUMERATED)
						if ((bp != NULL) &&
						    (BIO_write(bp,"-",1) <= 0))
							goto end;
					for (i=0; i<bs->length; i++)
						{
						if ((bp != NULL) &&
						    (BIO_printf(bp,"%02X",
							bs->data[i]) <= 0))
							goto end;
						}
					if (bs->length == 0)
						{
						if ((bp != NULL) &&
						    (BIO_write(bp,"00",2) <= 0))
							goto end;
						}
					}
				else
					{
					if ((bp != NULL) &&
					    (BIO_write(bp,"BAD ENUMERATED",11) <= 0))
						goto end;
					}
				M_ASN1_ENUMERATED_free(bs);
				}
			else if (len > 0 && dump)
				{
				if (!nl) 
					{
					if ((bp != NULL) &&
					    (BIO_write(bp,"\n",1) <= 0))
						goto end;
					}
				if ((bp != NULL) &&
				    (BIO_dump_indent(bp,(char *)p,
					((dump == -1 || dump > len)?len:dump),
					dump_indent) <= 0))
					goto end;
				nl=1;
				}

			if (!nl) 
				{
				if ((bp != NULL) &&
				    (BIO_write(bp,"\n",1) <= 0)) goto end;
				}
			p+=len;
			if ((tag == V_ASN1_EOC) && (xclass == 0))
				{
				ret=2; /* End of sequence */
				goto end;
				}
			}

		length-=len;
		}
	ret=1;
end:
	if (o != NULL) ASN1_OBJECT_free(o);
	if (os != NULL) M_ASN1_OCTET_STRING_free(os);
	*pp=p;
	return(ret);
	}

int GRSTasn1ParseDump(BIO *bp, unsigned char *pp, long len,
                      struct GRSTasn1TagList taglist[], 
                      int maxtag, int *lasttag)
        {
           return(GRSTasn1Parse2(bp,&pp,len,0,0,0,0,"",
                                 taglist, maxtag, lasttag));
        }                        

int GRSTasn1GetX509Name(char *x509name, int maxlength, char *coords,
                        char *asn1string,
                        struct GRSTasn1TagList taglist[], int lasttag)                        
{
   int i, iobj, istr, n, len = 0;
   ASN1_OBJECT *obj = NULL;
   unsigned char coordstmp[81], *q;
   const unsigned char *shortname;

   for (i=1; ; ++i)
      {
        snprintf((char*)coordstmp, sizeof(coordstmp), coords, i, 1);
        iobj = GRSTasn1SearchTaglist(taglist, lasttag, (char*)coordstmp);
        if (iobj < 0) break;
        
        snprintf((char*)coordstmp, sizeof(coordstmp), coords, i, 2);
        istr = GRSTasn1SearchTaglist(taglist, lasttag, (char*)coordstmp);
        if (istr < 0) break;
        
        q = (unsigned char*) &asn1string[taglist[iobj].start];
        d2i_ASN1_OBJECT(&obj, (SSLARG) &q, taglist[iobj].length +
                                  taglist[iobj].headerlength);

        n = OBJ_obj2nid(obj);
// free obj now?
//	if (obj) free (obj);
        shortname = (const unsigned char*) OBJ_nid2sn(n);
        
        if (len + 2 + strlen((char*)shortname) + taglist[istr].length >= maxlength)
          {
            x509name[0] = '\0';
            return GRST_RET_FAILED;          
          }
        
        sprintf(&x509name[len], "/%s=%.*s", shortname, 
                                taglist[istr].length, 
               &asn1string[taglist[istr].start+taglist[istr].headerlength]);
        len += 2 + strlen((char*)shortname) + taglist[istr].length;
      }
      
   x509name[len] = '\0';
   
   return (x509name[0] != '\0') ? GRST_RET_OK : GRST_RET_FAILED;
}
