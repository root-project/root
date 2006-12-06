/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/***************************************************************************
* graphbuf.h
*
*  array and carray type graph data buffer
*
*	buf  << x << y << "\n";
*
*	buf  << "title of graph"
*            << x  << "xunit"           << xmin | xmax << LOG
*     	     << y1 << "name of y1 data" << ymin | ymax << LOG
*     	     << y2 << "name of y2 data"  // multiple y axis data(optional)
*     	     << y3 << "name of y3 data"  //
*     	     << "yunit\n" ;              // must end with \n
*
*       plot << buf ;
*       arrayostream("fname") << buf ;
*       arrayostream("fname") >> buf ;
*
****************************************************************************/


#ifndef G__GRAPHBUF_H
#define G__GRAPHBUF_H

#ifndef G__ARRAY_H
class array;
#endif

#ifndef G__CARRAY_H
class carray;
#endif

#define G__ARRAYMAX 100

typedef unsigned char ISLOG;
ISLOG LIN=0;  // log flag linear
ISLOG LOG=1;  // log flag log

/**********************************************************
* graphbuf
**********************************************************/
class graphbuf {
public:
  enum DATATYPE { G__ARRAYTYPE, G__CARRAYTYPE, G__CARRAYTYPE_RE, G__CARRAYTYPE_IM };
  enum GRSTATUS { G__GRAPHBUF_PUSHING , G__GRAPHBUF_FIXED };

private:
  // data member
  int minn;                     // minimum of n[]
  int pout;                     // output pointer
  int status;                   // input status
  int iden[G__ARRAYMAX];        // identity, array, carray re,im
  int nplot;                    // number of data, input pointer
  int xlog,ylog;                // log flag
  double xmin,xmax,ymin,ymax;   // scale
  int n[G__ARRAYMAX];           // array of number of data
  
  char *title;                  // title of graph
  char *yunit;                  // y axis unit
  
  char *dataname[G__ARRAYMAX];  // name of data
public:
  double *pdat[G__ARRAYMAX];    // pointer to data array
  
public:
  // constructor, destructor, initialization
  graphbuf(void);
  ~graphbuf(void);
  void setnull(void);
  void freebuf(void);
  
  // assignment operator
  graphbuf& operator =(graphbuf& X);
  
  // ostream pipeline operator
  graphbuf& operator <<(array& a);
  graphbuf& operator <<(carray& a);
  graphbuf& operator <<(char *s);    // give name or title+do plot
  graphbuf& operator <<(double min); // specify min scale
  graphbuf& operator  |(double max); // specify max scale
  graphbuf& operator <<(ISLOG log);  // specify log scale
  virtual graphbuf& operator <<(char c);  // do plot
  
  // istream pipeline operator
  graphbuf& operator >>(array& a);
  graphbuf& operator >>(carray& a);
  graphbuf& operator >>(char *s);    // give name or title+do plot
  graphbuf& operator >>(double min); // dummy
  graphbuf& operator >>(ISLOG log);  // dummy
  graphbuf& operator >>(char c);     // 
  
  int isterminate(void) {
    if(G__GRAPHBUF_FIXED==status) return(1);
    else                          return(0);
  }
  
  int Nplot(void)   { return(nplot); }
  int Npoint(int i) { return(n[i]); }
  int Minn(void)    { return(minn); }
  char *Name(int i) { return(dataname[i]); }
  char *Title(void) { return(title); }
  char *Yunit(void) { return(yunit); }
  double *Pdat(int i) { return(pdat[i]); }
  double Dat(int i,int n) { return(pdat[i][n]); }
  int Xlog(void)    { return(xlog); }
  int Ylog(void)    { return(ylog); }
  double Xmin(void) { return(xmin); }
  double Xmax(void) { return(xmax); }
  double Ymin(void) { return(ymin); }
  double Ymax(void) { return(ymax); }
  int Pout(void) { return(pout); }
  int Status(void) { return(status); }
  int Iden(int i)  { return(iden[i]); }
  void setStatus(int stat) { status = stat; }
  
  void dumpdata(FILE *fp);
  void loaddata(FILE *fp);
} ;


// constructor
graphbuf::graphbuf(void)
{
  setnull();
  pout=0;
}

// initialization
void graphbuf::setnull(void)
{
	int i;
	for(i=0;i<G__ARRAYMAX;i++) {
		pdat[i] = NULL;
		n[i] = 0;
		dataname[i] = NULL;
	}
	title=NULL;
	yunit=NULL;
	nplot=0;
	status=G__GRAPHBUF_PUSHING;
	minn=2000000;
	xmin=xmax=ymin=ymax=0;
	xlog=ylog=0;
}

void graphbuf::freebuf(void)
{
	int i;
	for(i=nplot-1;i>=0;i--) {
		if(dataname[i]) free(dataname[i]);
		if(pdat[i])     free(pdat[i]);
	}
	if(yunit) free(yunit);
	if(title) free(title);
	setnull();
}


// destructor
graphbuf::~graphbuf(void)
{
	freebuf();
}


/**************************************************************************
* operator overloading
**************************************************************************/
graphbuf& graphbuf::operator =(graphbuf& a)
{
	int i;
	if(G__GRAPHBUF_FIXED==status) {
		freebuf();
	}
	if(title) free(title);
	if(a.Title()) {
		title=malloc(strlen(a.Title())+1);
		strcpy(title,a.Title());
	}
	else title=NULL;

	if(yunit) free(yunit);
	if(a.Yunit()) {
		yunit=malloc(strlen(a.Yunit())+1);
		strcpy(yunit,a.Yunit());
	}
	else yunit=NULL;
	
	minn=a.Minn();
	xlog=a.Xlog();
	ylog=a.Ylog();
	xmin=a.Xmin();
	xmax=a.Xmax();
	ymin=a.Ymin();
	ymax=a.Ymax();
	nplot=a.Nplot();
	pout=a.Pout();
	status=a.Status();

	for(i=0;i<nplot;i++) {
		n[i]=a.Npoint(i);
		iden[i]=a.Iden(i);

		if(pdat[i]) free(pdat[i]);
		pdat[i] = malloc(n[i]*sizeof(double));
		memcpy((char*)pdat[i],(char*)a.pdat[i],n[i]*sizeof(double));

		if(dataname[i]) free(dataname);
		if(a.Name(i)) {
			dataname[i]=malloc(strlen(a.Name(i))+1);
			strcpy(dataname[i],a.Name(i));
		}
		else dataname[i]=NULL;
	}
	return(*this);
}


// add array or carray to plot
graphbuf& graphbuf::operator <<(array& a)
{
	if(G__GRAPHBUF_FIXED==status) {
		freebuf();
	}
	if(pdat[nplot]) free(pdat[nplot]);
	pdat[nplot] = malloc(a.n*sizeof(double));
	memcpy((char*)pdat[nplot],(char*)a.dat,a.n*sizeof(double));
	iden[nplot]=G__ARRAYTYPE;
	n[nplot]=a.n;
	if(minn>a.n) minn=a.n;
	++nplot;
	return(*this);
}

graphbuf& graphbuf::operator <<(carray& a)
{
	if(G__GRAPHBUF_FIXED==status) {
		freebuf();
	}
	// real part
	if(pdat[nplot]) free(pdat[nplot]);
	pdat[nplot] = malloc(a.n*sizeof(double));
	memcpy((char*)pdat[nplot],(char*)a.re,a.n*sizeof(double));
	iden[nplot]=G__CARRAYTYPE_RE;
	n[nplot]=a.n;
	if(minn>a.n) minn=a.n;
	++nplot;
	// imaginary part
	if(pdat[nplot]) free(pdat[nplot]);
	pdat[nplot] = malloc(a.n*sizeof(double));
	memcpy((char*)pdat[nplot],(char*)a.im,a.n*sizeof(double));
	iden[nplot]=G__CARRAYTYPE_IM;
	n[nplot]=a.n;
	++nplot;
	return(*this);
}


// add min scale information
graphbuf& graphbuf::operator <<(double min)
{
	if(1==nplot) { // on x data
		xmin=min;
	}
	else { // on y data
		ymin=min;
	}
	return(*this);
}

// add max scale information
graphbuf& graphbuf::operator |(double max)
{
	if(G__GRAPHBUF_PUSHING==status) {
		if(1==nplot) { // on x data
			xmax=max;
		}
		else { // on y data
			ymax=max;
		}
	}
	return(*this);
}

// add log information
graphbuf& graphbuf::operator <<(ISLOG log)
{
	if(1==nplot) xlog = (int)log;
	else         ylog = (int)log;
	return(*this);
}

// add title of plot
graphbuf& graphbuf::operator <<(char *s)
{
	if(G__GRAPHBUF_FIXED==status) {
		freebuf();
	}
	if(strcmp(s,"\n")==0) {
		*this << '\n';
	}
	else if(strlen(s)>0 && s[strlen(s)-1]=='\n') {
		if(yunit) free(yunit);
		yunit=malloc(strlen(s)+1);
		sprintf(yunit,"%s",s);
		yunit[strlen(s)-1]='\0';
		*this << '\n';
	}
	else {
		if(nplot>0) {
			if(G__ARRAYTYPE==iden[nplot-1]) {
				if(dataname[nplot-1]) free(dataname[nplot-1]);
				dataname[nplot-1]=malloc(strlen(s)+1);
				strcpy(dataname[nplot-1],s);
			}
			else {
				if(dataname[nplot-2]) free(dataname[nplot-2]);
				dataname[nplot-2]=malloc(strlen(s)+5);
				sprintf(dataname[nplot-2],"%s(re)",s);
				if(dataname[nplot-1]) free(dataname[nplot-1]);
				dataname[nplot-1]=malloc(strlen(s)+5);
				sprintf(dataname[nplot-1],"%s(im)",s);
			}
		}
		else  {
			if(title) free(title);
			title=malloc(strlen(s)+1);
			strcpy(title,s);
		}
	}
	return(*this);
}

// do plot or print
graphbuf& graphbuf::operator <<(char c)
{
	status=G__GRAPHBUF_FIXED;
	return(*this);
}


/**********************************************************
* output
**********************************************************/

// add array or carray 
graphbuf& graphbuf::operator >>(array& a)
{
	if(pout>nplot) {
		cerr << "Error: no more data in graphbuf\n";
		return(*this);
	}
	a = array(pdat[pout],n[pout]);
	++pout;
	return(*this);
}

graphbuf& graphbuf::operator >>(carray& a)
{
	if(pout>nplot-1) {
		cerr << "Error: no more data in graphbuf\n";
		return(*this);
	}
	a = carray(pdat[pout],pdat[pout+1],n[pout]);
	pout+=2;
	return(*this);
}

// dummy 
graphbuf& graphbuf::operator >>(double d)
{
	return(*this);
}

#ifdef NEVER
graphbuf& graphbuf::operator |(double d)
{
	return(*this);
}
#endif

graphbuf& graphbuf::operator >>(ISLOG log)
{
	return(*this);
}

// reset ouput pointer
graphbuf& graphbuf::operator >>(char *s)
{
	pout=0;
	return(*this);
}

graphbuf& graphbuf::operator >>(char c)
{
	pout=0;
	return(*this);
}


/********************************************************************
*  binary dump
********************************************************************/
void graphbuf::dumpdata(FILE *fp)
{
	int i;
	int len;

	fwrite(&nplot,sizeof(nplot),1,fp);
	fwrite(&xlog,sizeof(xlog),1,fp);
	fwrite(&ylog,sizeof(ylog),1,fp);
	fwrite(&xmin,sizeof(xmin),1,fp);
	fwrite(&xmax,sizeof(xmax),1,fp);
	fwrite(&ymin,sizeof(ymin),1,fp);
	fwrite(&ymax,sizeof(ymax),1,fp);
	fwrite(&minn,sizeof(minn),1,fp);
	fwrite(&pout,sizeof(pout),1,fp);
	fwrite(&status,sizeof(status),1,fp);
	fwrite(&n,sizeof(n),1,fp);
	fwrite(&iden,sizeof(iden),1,fp);

	if(title) len=strlen(title);
	else      len=0;
	fwrite(&len,sizeof(len),1,fp);
	if(len) fwrite(title,len+1,1,fp);
	

	if(yunit) len=strlen(yunit);
	else      len=0;
	fwrite(&len,sizeof(len),1,fp);
	if(len) fwrite(yunit,len+1,1,fp);

	for(i=0;i<nplot;i++) {
		fwrite(pdat[i],n[i]*sizeof(double),1,fp);

		if(dataname[i]) len=strlen(dataname[i]);
		else            len=0;
		fwrite(&len,sizeof(len),1,fp);
		if(len) fwrite(dataname[i],len+1,1,fp);
	}
}


/********************************************************************
*  binary load
********************************************************************/
void graphbuf::loaddata(FILE *fp)
{
	int i;
	int len;

	fread(&nplot,sizeof(nplot),1,fp);
	fread(&xlog,sizeof(xlog),1,fp);
	fread(&ylog,sizeof(ylog),1,fp);
	fread(&xmin,sizeof(xmin),1,fp);
	fread(&xmax,sizeof(xmax),1,fp);
	fread(&ymin,sizeof(ymin),1,fp);
	fread(&ymax,sizeof(ymax),1,fp);
	fread(&minn,sizeof(minn),1,fp);
	fread(&pout,sizeof(pout),1,fp);
	fread(&status,sizeof(status),1,fp);
	fread(&n,sizeof(n),1,fp);
	fread(&iden,sizeof(iden),1,fp);

	fread(&len,sizeof(len),1,fp);
	if(title) {
		free(title);
		title=NULL;
	}
	if(len) {
		title=malloc(len+1);
		fread(title,len+1,1,fp);
	}

	fread(&len,sizeof(len),1,fp);
	if(yunit) {
		free(yunit);
		yunit=NULL;
	}
	if(len) {
		yunit=malloc(len+1);
		fread(yunit,len+1,1,fp);
	}

	for(i=0;i<nplot;i++) {
		if(pdat[i]) free(pdat[i]);
		pdat[i]=malloc(n[i]*sizeof(double));
		fread(pdat[i],n[i]*sizeof(double),1,fp);

		fread(&len,sizeof(len),1,fp);
		if(dataname[i]) {
			free(dataname[i]);
			dataname[i]=NULL;
		}
		if(len) {
			dataname[i]=malloc(len+1);
			fread(dataname[i],len+1,1,fp);
		}
	}
}

#endif
