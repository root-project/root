/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/***************************************************************************
* arrayiostream.h
*
*  array and carray type graph plot and file I/O class
*
*	plot << x << y << "\n";
*
*	plot << "title of graph"
*            << x  << "xunit"           << xmin >> xmax << LOG
*     	     << y1 << "name of y1 data" << ymin >> ymax << LOG
*     	     << y2 << "name of y2 data"  // multiple y axis data(optional)
*     	     << y3 << "name of y3 data"  //
*     	     << "yunit\n" ;              // must end with \n
*
*       fout << x << y1 << y2 << y3 << "\n";
*       fin  >> x >> y1 >> y2 >> y3 >> "\n";
*
****************************************************************************/
#ifndef G__ARRAYIOSTREAM_H
#define G__ARRAYIOSTREAM_H

#ifndef G__XGRAPHSL

#ifdef G__SHAREDLIB
#pragma include_noerr <xgraph.dl>
# ifndef G__XGRAPHSL
#include <xgraph.c>
# endif
#else
#include <xgraph.c>
#endif

#endif

typedef unsigned short UNITCIRCLE_t;
UNITCIRCLE_t UNITCIRCLE;

/**********************************************************
* arrayostream
**********************************************************/
class arrayostream {
  // Due to cint limitation, base class inheritance is emulated by
  // having pointer of graphbuf and construct it by new operator.
  graphbuf *buf;  
  
  // data member
  FILE *fp;
  int filen;
  int importedfp;
  int unitcircle;               // unit circle mode
  int iscsv;
  
  // private member function
  void plotdata(void);
  void printdata(void);
  void csvdata(void);
  
 public:
  // constructor
  arrayostream(FILE *fp_init,int iscsv=0);
  arrayostream(char *filename,int iscsv=0);
  ~arrayostream();
  
  void init(FILE *fp_init)
    {if(fp&&fp!=stdout&&fp!=stderr) fclose(fp); fp=fp_init;}
  void init(char fname)
    {if(fp&&fp!=stdout&&fp!=stderr) fclose(fp);fp=fopen(fname,"w");}
  void close() 
    {if(fp&&fp!=stdout&&fp!=stderr) fclose(fp); fp=NULL;}
  
  // ostream pipeline operator
  arrayostream& operator <<(array& a);
  arrayostream& operator <<(carray& a);
  arrayostream& operator <<(char *s);    // give name or title+do plot
  arrayostream& operator <<(double min); // specify min scale
  arrayostream& operator  |(double max); // specify max scale
  arrayostream& operator <<(ISLOG log);  // specify log scale
  arrayostream& operator <<(UNITCIRCLE_t uc);  // specify log scale
  arrayostream& operator <<(char c);     // do plot
} ;


// constructor

arrayostream::arrayostream(FILE *fp_init,int iscsvin)
{
  buf = new graphbuf; // base class emulation
  filen=0;
  fp=fp_init;
  importedfp=1;
  unitcircle=0;
  iscsv = iscsvin;
}

arrayostream::arrayostream(char *filename,int iscsvin)
{
  buf = new graphbuf; // base class emulation
  filen=0;
  fp=fopen(filename,"wb");
  if(NULL==fp) cerr << filename << " could not open\n" ;
  importedfp=0;
  unitcircle=0;
  iscsv = iscsvin;
}

arrayostream::~arrayostream()
{
  if(0==importedfp && fp && stdout!=fp && stderr!=fp) fclose(fp);
  delete buf; // base class emulation
}


//
arrayostream& arrayostream::operator <<(graphbuf& a)
{
  *buf = a;
  if(buf->isterminate()) {
    terminated();
  }
  return(*this);
}

// add array or carray to plot
arrayostream& arrayostream::operator <<(array& a)
{
  *buf << a;
  return(*this);
}

arrayostream& arrayostream::operator <<(carray& a)
{
  *buf << a;
  return(*this);
}


// add min scale information
arrayostream& arrayostream::operator <<(double min)
{
  *buf << min;
  return(*this);
}

// add max scale information
arrayostream& arrayostream::operator |(double max)
{
  *buf | max;
  return(*this);
}
arrayostream& arrayostream::operator >>(double max)
{
  *buf | max;
  return(*this);
}

// add log information
arrayostream& arrayostream::operator <<(ISLOG log)
{
  *buf << log;
  return(*this);
}

// add unit circle information
arrayostream& arrayostream::operator <<(UNITCIRCLE_t uc)
{
  unitcircle=1;
  return(*this);
}

// add title of plot
arrayostream& arrayostream::operator <<(char *s)
{
  *buf << s;
  if(buf->isterminate()) {
    terminated();
  }
  return(*this);
}

// do plot or print
arrayostream& arrayostream::operator <<(char c)
{
  *buf << c;
  terminated();
  return(*this);
}
arrayostream& arrayostream::operator <<(G__CINT_ENDL c)
{
  *buf << '\n';
  terminated();
  return(*this);
}

arrayostream::terminated(void)
{
  int i;
  if(iscsv) {
    csvdata();
  }
  else if(NULL==fp) {
    plotdata();
  }
  else if(importedfp) {
    printdata();
  }
  else {
    FILE *FP;
    FP=fp;             // Assigning tp FP
    buf->dumpdata(FP); // cint bug work around
  }
  buf->freebuf();
}

// plot data using xgraph
void arrayostream::plotdata(void)
{
  int i,nplot,n,n;
  char fname[100];
  
  nplot = buf->Nplot();

  switch(nplot) {
  case 0:
    cerr << "No data to plot\n";
    break;
  case 1:
    n=buf->Npoint(0);
    array tmpx=array(0,n-1,n);
    sprintf(fname,"G__graph%d",filen++);
    xgraph_open(fname,buf->Title());
    xgraph_data(fname
		,tmpx.dat
		,buf->Pdat(0)
		,n
		,buf->Name(0)
		);
    xgraph_invoke(fname
		  ,buf->Xmin(),buf->Xmax()
		  ,buf->Ymin(),buf->Ymax()
		  ,buf->Xlog(),buf->Ylog()
		  ,(char*)NULL,buf->Yunit());
    break;
  default:
    sprintf(fname,"G__graph%d",filen++);
    xgraph_open(fname,buf->Title());
    for(i=1;i<nplot;i++) {
      if(buf->Npoint(0)>buf->Npoint(i))  n = buf->Npoint(i);
      else                               n = buf->Npoint(0);
      xgraph_data(fname
		  ,buf->Pdat(0)
		  ,buf->Pdat(i)
		  ,n
		  ,buf->Name(i)
		  );
    }
    if(unitcircle) {
      int nuc = 100;
      double dpi = 3.141592*2/100;
      double x[100],y[100];
      for(i=0;i<nuc;i++) {
	x[i] = cos(dpi*i);
	y[i] = sin(dpi*i);
      }
      xgraph_data(fname
		  ,x
		  ,y
		  ,nuc
		  ,"unit_circle"
		  );
    }
    xgraph_invoke(fname
		  ,buf->Xmin(),buf->Xmax()
		  ,buf->Ymin(),buf->Ymax()
		  ,buf->Xlog(),buf->Ylog()
		  ,buf->Name(0),buf->Yunit());
    break;
  }
}

// print data to file
void arrayostream::printdata(void)
{
  int i,n;
  int Ndata,Nplot;
  
  Ndata = buf->Minn();
  Nplot = buf->Nplot();
  for(i=0;i<Ndata;i++) {
    for(n=0;n<Nplot;n++) {
      fprintf(fp,"%g ",buf->pdat[n][i]);
      // Doing above due to cint limitation,
      // supposed to be like below.
      // fprintf(fp,"%g ",buf->Dat(n,i)); 
    }
    fprintf(fp,"\n");
  }
}

// output data in CSV (Comma Separated Value)
void arrayostream::csvdata(void)
{
  int i,nplot,n,n;
  int nplot = buf->Nplot();
  int n=buf->Npoint(0);
  double *p;
  if(!fp) fp=stdout;
  int flag=0;
  for(int j=0;j<nplot;j++) {
    if(buf->Name(j) && strlen(buf->Name(j))) flag=1;
  }
  if(flag) {
    for(int j=0;j<nplot;j++) {
      if(buf->Name(j)) fprintf(fp,"%s,",buf->Name(j));
      else fprintf(fp,",");
    }
  }
  fprintf(fp,"\n");
  for(int i=0;i<n;i++) {
    for(int j=0;j<nplot;j++) {
      p = buf->Pdat(j);
      fprintf(fp,"%g,",p[i]);
    }
    fprintf(fp,"\n");
  }
}


/**********************************************************
* istream
**********************************************************/
class arrayistream {
  // data member
  double *dat[G__ARRAYMAX];
  int ndat;
  int nplot;
  FILE *fp;
  int importedfp;
  
 public:
  // constructor
  arrayistream(FILE *fp_init);
  arrayistream(char *filename);
  ~arrayistream();
  
  // operator overloading
  arrayistream& operator >>(array& a);
  arrayistream& operator >>(carray& a);
  arrayistream& operator >>(char *s);
  arrayistream& operator >>(char c);
  arrayistream& operator >>(double d);  // dummy
  arrayistream& operator >>(ISLOG log); // dummy
  arrayistream& operator >>(graphbuf& a);
  
  void close(void);
} ;


arrayistream::arrayistream(FILE *fp_init) 
{ 
  fp=fp_init;
  nplot=0; 
  ndat=2000000; 
  importedfp=1;
}

arrayistream::arrayistream(char *filename)
{ 
  fp=fopen(filename,"rb");
  if(NULL==fp) cerr << filename << " could not open\n" ;
  nplot=0; 
  ndat=2000000; 
  importedfp=0;
}

arrayistream::~arrayistream()
{
  close();
}

void arrayistream::close(void)
{
  if(0==importedfp) fclose(fp);
}


arrayistream& arrayistream::operator >>(graphbuf& a)
{
  if(fp) {
    FILE *FP;       
    FP=fp;          // Assigning to FP
    a.loaddata(FP); // cint bug work around
    nplot=0;
		ndat=2000000;
  }
  else {
    cerr << "Can not read data\n";
    a.freebuf();
  }
  return(*this);
}

// add array or carray 
arrayistream& arrayistream::operator >>(array& a)
{
  dat[nplot++] = a.dat;
  if(ndat>a.n) ndat=a.n;
  return(*this);
}

arrayistream& arrayistream::operator >>(carray& a)
{
  dat[nplot++] = a.re;
  dat[nplot++] = a.im;
  if(ndat>a.n) ndat=a.n;
  return(*this);
}

// dummy 
arrayistream& arrayistream::operator >>(double d)
{
  return(*this);
}

arrayistream& arrayistream::operator |(double d)
{
  return(*this);
}

arrayistream& arrayistream::operator >>(ISLOG log)
{
  return(*this);
}

// do input
arrayistream& arrayistream::operator >>(char *s)
{
  if(s[strlen(s)-1]=='\n') 
    *this >> '\n';
  return(*this);
}

arrayistream& arrayistream::operator >>(char c)
{
  int i,n;
  if(NULL==fp) {
    cerr << "arrayistream: File pointer null\n" ;
    return;
  }
  
  if(nplot) {
    for(i=0;i<ndat;i++) {
      $read(fp);
      for(n=0;n<nplot;n++) {
	dat[n][i]=atof($(n+1));
      }
    }
  }
  
  nplot=0;
  ndat=2000000;
  return(*this);
}

// global object
arrayostream plot=arrayostream((FILE*)NULL);
arrayostream arraycout=arrayostream(stdout);
arrayostream arraycerr=arrayostream(stderr);
arrayistream arraycin=arrayistream(stdin);
arrayostream csv=arrayostream(stdout,1);


#endif
