/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
//**************************************************************************
// cint text2tex.c [textfile] <-simple>
//
//  Plain text to Latex source converter implemented in cint script.
//
//  Masaharu Goto
// 
//**************************************************************************
#include <stdio.h>
#include <stdlib.h>

#if getenv("CINTSYSDIR")
   // If CINTSYSDIR is set $CINTSYSDIR/include/iostream.h is loaded.
   #include <iostream.h>
#else
   // If CINTSYSDIR is not set ../include/iostream.h is loaded.
   #include "../include/iostream.h"
#endif

// remove this line to compile speed critical function
#pragma disablecompile

int main(int argc,char **argv)
{
	FILE *fp;
	char charsize[20] = "Large";
	int simple=0;

	cerr << "=========================================================\n";
	cerr << "= text2tex \n";
	cerr << "=   Plain text to Latex source converter\n";
	cerr << "=   Author: Masaharu Goto, HSTD R&D\n";
	cerr << "=   Date  : 28 July 1994 \n";
	cerr << "=   Implementation: C++ cint script\n";
	cerr << "=========================================================\n";

	if(argc<2) { // Usage message
		cerr << "Usage: " << argv[0] << " [text_file] <-simple>\n";
		exit(0);
	}
	if(argc>2) { // optional character size
		simple=1;
	}

	// Open input file
	fp = fopen(argv[1],"r");
	if(NULL==fp) {
		cerr << argv[1] << " cannot open\n" ;
		exit(1);
	}

	// output Latex source
	cerr << "Translating plain text to Latex source\n";
	head(charsize);
	if(simple) {
		while($read(fp)) {
			if(strcmp($1,"")==0) vspace();
			else                 linesimple($0);
		}
	}
	else {
		while($read(fp)) {
			if(strcmp($1,"")==0) vspace();
			else                 line($0);
		}
	}
	tail();
	cerr << "Translation finished\n";

	fclose(fp);

	exit(0);
}

void head(char *charsize)
{
	cout << "\\documentstyle[]{jarticle}\n" ; 
	cout << "\\setcounter{secnumdepth}{6}\n" ;
	cout << "\\setcounter{tocdepth}{6}\n" ;
	cout << "\\topsep=0.1cm\n" ;
	cout << "\\parsep=0.1cm\n" ;
	cout << "\\itemsep=0.0cm\n" ;
	cout << "\\renewcommand{\\bf}{\\protect\\pbf\\protect\\pdg}\n" ;
	cout << "\\begin{document}\n" ;
	cout << "\\" << charsize << "\n" ;
	cout << "\\medskip\n" ;
	cout << "\\vspace{3.0cm}\n" ;
}

void tail(void)
{
	cout << "\\end{document}\n" ;
}

void vspace(void)
{
	cout << "\\medskip\n";
	cout << "\\vspace{0.5cm}\n"; 
}

void indent(int& indent,int i)
{
	if(indent) {
		if(1==indent && 1!=i)
			cout << ' ';
		else
			cout << "\\hspace{" << indent*0.2 << "cm}\n" ;
	}
	indent=0;
}


#pragma compile
void line(char *line)
{
	int indent=0,i=0;
	char c;
	cout << "\\medskip\n" ;
	cout << "\\par\n" ;

	while(c=line[i]) {
		switch(c) {
		      case ' ':
			++indent;
			break;
		      case '\t':
			indent+=8;
			break;
		      case '<':
		      case '>':
		      case '!':
			indent(indent,i);
			cout << "$" << c << "$";
			break;
		      case '#':
		      case '$':
		      case '%':
		      case '&':
		      case '{':
		      case '}':
		      case '[':
		      case ']':
			indent(indent,i);
			cout << "\\" << c ;
			break;
		      case '~':
			indent(indent,i);
			cout << "$\\sim$";
			break;
		      case '|':
			indent(indent,i);
			cout << "$\\mid$";
			break;
		      case '^':
			indent(indent,i);
			cout << "$\\hat{}$";
			break;
		      case '\\':
			indent(indent,i);
			cout << line[++i];
			break;
		      default:
			indent(indent,i);
			cout << c;
			break;
		}
		i++;
	}
	cout << '\n';
}
#pragma endcompile


void linesimple(char *line)
{
	int indent=0,i=0;
	cout << "\\medskip\n" ;
	cout << "\\par\n" ;

	while(isspace(line[i])) {
		switch(line[i]) {
		case '\t':
			indent+=8;
			break;
		default:
			++indent;
			break;
		}
		i++;
	}
	if(indent) cout << "\\hspace{" << indent*0.2 << "cm}\n" ;
	cout << line << '\n' ;
}
