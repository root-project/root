/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/***************************************************************************
* xygraphbuf.h
*
*  array and carray type graph plot and file I/O class
*
*	plot << x << y << "\n";
*
*	plot << "title of graph"
*            << x  << "xunit"           << xmin | xmax << LOG
*     	     << y1 << "name of y1 data" << ymin | ymax << LOG
*     	     << y2 << "name of y2 data"  // multiple y axis data(optional)
*     	     << y3 << "name of y3 data"  //
*     	     << "yunit\n" ;              // must end with \n
*
*       fout << x << y1 << y2 << y3 << "\n";
*       fin  >> x >> y1 >> y2 >> y3 >> "\n";
*
****************************************************************************/
#ifndef G__XYGRAPHBUF_H
#define G__XYGRAPHBUF_H

/**********************************************************
* arrayostream
**********************************************************/
class graphbuf {
	// data member
	double *pdat[100];
	union {
		array  *pa;
		carray *pc;
	} pary[100];
	int iden[100];        // to distinguish array or carray 
	char *dataname[100];  // name of data
	int nplot;            // number of data
	FILE *fp;             // File pointer. NULL for plot
	char title[100];      // title of graph
	char yunit[50];       // y axis unit
	char fname[100];      // temp file name
	int filen;            // number of temp file in this process
	double xmin,xmax,ymin,ymax;  // scale
	int xlog,ylog;        // log flag

	// constructor
	arrayostream(FILE *fp_init);
	arrayostream(char *filename);
	~arrayostream();

	// private member function
	void plotdata(void);
	void printdata(void);
	
	// ostream pipeline operator
	arrayostream& operator <<(array& a);
	arrayostream& operator <<(carray& a);
	arrayostream& operator <<(char *s);    // give name or title+do plot
	arrayostream& operator <<(double min); // specify min scale
	arrayostream& operator  |(double max); // specify max scale
	arrayostream& operator <<(ISLOG log);  // specify log scale
	arrayostream& operator <<(char c);     // do plot
} ;


// constructor

arrayostream::arrayostream(FILE *fp_init)
{
	int i;
	nplot=0;
	filen=0;
	xmin=xmax=ymin=ymax=0;
	xlog=ylog=0;
	title[0]='\0';
	yunit[0]='\0';
	for(i=0;i<100;i++) dataname[i]=NULL;
	fp=fp_init;
}

arrayostream::arrayostream(char *filename)
{
	int i;
	nplot=0;
	filen=0;
	xmin=xmax=ymin=ymax=0;
	xlog=ylog=0;
	title[0]='\0';
	yunit[0]='\0';
	for(i=0;i<100;i++) dataname[i]=NULL;
	fp=fopen(filename,"w");
}

arrayostream::~arrayostream()
{
	switch(fp) {
	case stdout:
	case stderr:
	case NULL:
		break;
	default:
		fclose(fp);
		break;
	}
}



// add array or carray to plot
arrayostream& arrayostream::operator <<(array& a)
{
	pary[nplot].pa = &a;
	iden[nplot] = G__ARRAYTYPE;
	dataname[nplot] = NULL;
	++nplot;
	return(*this);
}

arrayostream& arrayostream::operator <<(carray& a)
{
	pary[nplot].pc = &a;
	iden[nplot] = G__CARRAYTYPE;
	dataname[nplot] = NULL;
	++nplot;
	return(*this);
}


// add min scale information
arrayostream& arrayostream::operator <<(double min)
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
arrayostream& arrayostream::operator |(double max)
{
	if(1==nplot) { // on x data
		xmax=max;
	}
	else { // on y data
		ymax=max;
	}
	return(*this);
}

// add log information
arrayostream& arrayostream::operator <<(ISLOG log)
{
	if(1==nplot) xlog = (int)log;
	else         ylog = (int)log;
	return(*this);
}

// add title of plot
arrayostream& arrayostream::operator <<(char *s)
{
	if(strcmp(s,"\n")==0) {
		*this << '\n';
	}
	else if(s[strlen(s)-1]=='\n') {
		sprintf(yunit,"%s",s);
		yunit[strlen(s)-1]='\0';
		*this << '\n';
	}
	else {
		if(nplot>0) {
			if(NULL==dataname[nplot-1]) {
				dataname[nplot-1]=malloc(strlen(s)+1);
				strcpy(dataname[nplot-1],s);
			}
		}
		else        strcpy(title,s);
	}
	return(*this);
}

// do plot or print
arrayostream& arrayostream::operator <<(char c)
{
	int i;
	if((FILE*)NULL!=fp) {
		printdata();
	}
	else {
		plotdata();
	}
	for(i=0;i<nplot;i++) {
		if(dataname[i]) {
			free(dataname[i]);
		}
		dataname[i]=NULL;
	}
	nplot=0;
	return(*this);
}

// plot data using xgraph
void arrayostream::plotdata(void)
{
	int i;
	char fname[100];
	char temp[100];
	switch(nplot) {
	case 0:
	case 1:
		cerr << "Too few data to plot\n";
		break;
	default:
		sprintf(fname,"G__graph%d",filen++);
		xgraph_open(fname,title);
		for(i=1;i<nplot;i++) {
			switch(iden[i]) {
			case G__ARRAYTYPE:
				xgraph_data(fname
					    ,pary[0].pa->dat
					    ,pary[i].pa->dat
					    ,pary[i].pa->n
					    ,dataname[i]);
				break;
			case G__CARRAYTYPE:
				sprintf(temp,"%s(re)",dataname[i]);
				xgraph_data(fname
					    ,pary[0].pa->dat
					    ,pary[i].pc->re
					    ,pary[i].pc->n
					    ,temp);
				sprintf(temp,"%s(im)",dataname[i]);
				xgraph_data(fname
					    ,pary[0].pa->dat
					    ,pary[i].pc->im
					    ,pary[i].pc->n
					    ,temp);
				break;
			}
		}
		xgraph_invoke(fname
			      ,xmin,xmax,ymin,ymax
			      ,xlog,ylog
			      ,dataname[0],yunit);
		// xgraph_remove(fname,0);
		break;
	}
}

// print data to file
void arrayostream::printdata(void)
{
	double *dat[100];
	int Ndat=2000000,Nplot=0;
	int i,n;
	for(n=0;n<nplot;n++) {
		switch(iden[n]) {
		case G__ARRAYTYPE:
			dat[Nplot++] = pary[n].pa->dat ;
			if(Ndat>pary[n].pa->n) Ndat=pary[n].pa->n;
			break;
		case G__CARRAYTYPE:
			dat[Nplot++] = pary[n].pc->re ;
			dat[Nplot++] = pary[n].pc->im ;
			if(Ndat>pary[n].pc->n) Ndat=pary[n].pc->n;
			break;
		}
	}

	for(i=0;i<Ndat;i++) {
		for(n=0;n<Nplot;n++) {
			fprintf(fp,"%g ",dat[n][i]);
		}
		fprintf(fp,"\n");
	}
}


/**********************************************************
* istream
**********************************************************/
class arrayistream {
	double *dat[100];
	int ndat;
	int nplot;
	FILE *fp;

	arrayistream(FILE *fp_init);
	arrayistream(char *filename);
	~arrayistream();
	arrayistream& operator >>(array& a);
	arrayistream& operator >>(carray& a);
	arrayistream& operator >>(char *s);
	arrayistream& operator >>(char c);
	arrayistream& operator >>(double d);  // dummy
	arrayistream& operator >>(ISLOG log); // dummy
	void close(void);
} ;


arrayistream::arrayistream(FILE *fp_init) 
{ 
	fp=fp_init;
	nplot=0; 
	ndat=2000000; 
}

arrayistream::arrayistream(char *filename)
{ 
	fp=fopen(filename,"r");
	nplot=0; 
	ndat=2000000; 
}

arrayistream::~arrayistream()
{
	close();
}

void arrayistream::close(void)
{
	switch(fp) {
	case stdin:
		break;
	default:
		fclose(fp);
		break;
	}
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

	for(i=0;i<ndat;i++) {
		$read(fp);
		for(n=0;n<nplot;n++) {
			dat[n][i]=atof($(n+1));
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

#endif
