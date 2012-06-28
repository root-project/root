/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/*********************************************************************
* strstream.h
*
*********************************************************************/

#pragma if !defined(G__STRSTREAM_H) && !defined(G__SSTREAM_H)

#include <iostream.h>

#ifndef G__STRSTREAM_H
#define G__STRSTREAM_H


/*********************************************************************
* ostrstream, istrstream
*
*********************************************************************/

class strstream;

class ostrstream {
	char *buf;
	int size;
	int point;
      public:
	ostrstream(char *p,int bufsize);

	ostrstream& operator <<(char c);
	ostrstream& operator <<(char *s);
	ostrstream& operator <<(long i);
	ostrstream& operator <<(unsigned long i);
	ostrstream& operator <<(double d);
	ostrstream& operator <<(void *p);
};

ostrstream::ostrstream(char *p,int bufsize)
{
	if(p==NULL) {
		fprintf(stderr,"NULL pointer given to istrstream\n");
		return;
	}
	buf=p;
	size=bufsize;
	point=0;
}

ostrstream& ostrstream::operator <<(char c)
{
	if(point+1<size-1) sprintf(buf+point,"%c",c);
	return(*this);
}

ostrstream& ostrstream::operator <<(char *s)
{
	int len;
	if(point+strlen(s)<size-1) sprintf(buf+point,"%s",s);
	return(*this);
}

ostrstream& ostrstream::operator <<(long i)
{
	char temp[50];
	sprintf(temp,"%d",i);
	if(point+strlen(temp)<size-1) sprintf(buf+point,"%s",temp);
	return(*this);
}

ostrstream& ostrstream::operator <<(unsigned long i)
{
	char temp[50];
	sprintf(temp,"%u",i);
	if(point+strlen(temp)<size-1) sprintf(buf+point,"%s",temp);
	return(*this);
}

ostrstream& ostrstream::operator <<(double d)
{
	char temp[50];
	sprintf(temp,"%g",d);
	if(point+strlen(temp)<size-1) sprintf(buf+point,"%s",temp);
	return(*this);
}

ostrstream& ostrstream::operator <<(void *p)
{
	char temp[50];
	sprintf(temp,"0x%x",p);
	if(point+strlen(temp)<size-1) sprintf(buf+point,"%s",temp);
	return(*this);
}



/*********************************************************************
* istrstream
*
* NOT COMPLETE
*********************************************************************/

class istrstream {
	char *buf;
	int size;
	int point;
      public:
	istrstream(char *p,int bufsize);

	istrstream& operator >>(char& c);
	istrstream& operator >>(char *s);
	istrstream& operator >>(short& s);
	istrstream& operator >>(int& i);
	istrstream& operator >>(long i);
	istrstream& operator >>(unsigned char& c);
	istrstream& operator >>(unsigned short& s);
	istrstream& operator >>(unsigned int i);
	istrstream& operator >>(unsigned long i);
	istrstream& operator >>(double d);
	istrstream& operator >>(float d);
};

istrstream::istrstream(char *p,int bufsize)
{
	if(p==NULL) {
		fprintf(stderr,"NULL pointer given to istrstream\n");
	}
	buf=p;
	size=bufsize;
	point=0;
}


istrstream& istrstream::operator >>(char& c)
{
	sscanf(buf+point,"%c",&c);
	++point;
	return(*this);
}

istrstream& istrstream::operator >>(char *s)
{
	sscanf(buf+point,"%s",s);
	point += strlen(s);
	return(*this);
}

istrstream& istrstream::operator >>(short& s)
{
	sscanf(buf+point,"%hd",&s);
	return(*this);
}

istrstream& istrstream::operator >>(int& i)
{
	sscanf(buf+point,"%d",&i);
	return(*this);
}

istrstream& istrstream::operator >>(long& i)
{
	sscanf(buf+point,"%ld",&i);
	return(*this);
}

istrstream& istrstream::operator >>(unsigned char& c)
{
	int i;
	sscanf(buf+point,"%u",&i);
	c = i;
	return(*this);
}
istrstream& istrstream::operator >>(unsigned short& s)
{
	sscanf(buf+point,"%hu",&s);
	return(*this);
}
istrstream& istrstream::operator >>(unsigned int& i)
{
	sscanf(buf+point,"%u",&i);
	return(*this);
}
istrstream& istrstream::operator >>(unsigned long& i)
{
	sscanf(buf+point,"%lu",&i);
	return(*this);
}

istrstream& istrstream::operator >>(float& f)
{
	sscanf(buf+point,"%g",&f);
	return(*this);
}

istrstream& istrstream::operator >>(double& d)
{
	sscanf(buf+point,"%lg",&d);
	return(*this);
}




/*********************************************************************
* iostrstream
*
* NOT COMPLETE
*********************************************************************/
class iostrstream : public istrstream , public ostrstream {
	iostrstream(char *p,int bufsize)
		: istrstream(p,bufsize) , ostrstream(p,bufsize) { }
};


#else

istrstream::istrstream() {
}

typedef ostrstream ostringstream;
typedef istrstream istringstream;
//typedef strstream stringstream;

#endif

#pragma endif
