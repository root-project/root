/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

//////////////////////////////////////////////////////////////////////////////
//
// RooStreamParser is a utility class to parse istreams into tokens and optionally
// convert them into basic types (double,int,string)
// 
// The general tokenizing philosophy is that there are two kinds of tokens: value
// and punctuation. The former are variable length, the latter always
// one character. A token is terminated if one of the following conditions
// occur
//         - space character found (' ',tab,newline)
//         - change of token type (value -> punctuation or vv)
//         - end of fixed-length token (punctuation only)
//         - start or end of quoted string
//
// The parser is aware of floating point notation and will assign leading
// minus signs, decimal points etc to a value token when this is obvious
// from the context. The definition of what is punctuation can be redefined.
//


#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <stdlib.h>
#include <ctype.h>

#ifndef _WIN32
#include <strings.h>
#endif

#include "RooStreamParser.h"
#include "RooMsgService.h"
#include "RooNumber.h"


using namespace std;

ClassImp(RooStreamParser)


//_____________________________________________________________________________
RooStreamParser::RooStreamParser(istream& is) : 
  _is(&is), _atEOL(kFALSE), _atEOF(kFALSE), _prefix(""), _punct("()[]<>|/\\:?.,=+-&^%$#@!`~")
{
  // Construct parser on given input stream
}


//_____________________________________________________________________________
RooStreamParser::RooStreamParser(istream& is, const TString& errorPrefix) : 
  _is(&is), _atEOL(kFALSE), _atEOF(kFALSE), _prefix(errorPrefix), _punct("()[]<>|/\\:?.,=+-&^%$#@!`~")
{
  // Construct parser on given input stream. Use given errorPrefix to 
  // prefix any parsing error messages
}



//_____________________________________________________________________________
RooStreamParser::~RooStreamParser()
{
  // Destructor
}



//_____________________________________________________________________________
Bool_t RooStreamParser::atEOL() 
{ 
  // If true, parser is at end of line in stream

  Int_t nc(_is->peek()) ; 
  return (nc=='\n'||nc==-1) ; 
}



//_____________________________________________________________________________
void RooStreamParser::setPunctuation(const TString& punct) 
{
  // Change list of characters interpreted as punctuation

  _punct = punct ;
}



//_____________________________________________________________________________
Bool_t RooStreamParser::isPunctChar(char c) const 
{
  // Check if given char is considered punctuation

  const char* punct = _punct.Data() ;
  for (int i=0 ; i<_punct.Length() ; i++)
    if (punct[i] == c) {
      return kTRUE ;
    }
  return kFALSE ;
}



//_____________________________________________________________________________
TString RooStreamParser::readToken() 
{
  // Read one token separated by any of the know punctuation characters
  // This function recognizes and handles comment lines in the istream (those
  // starting with '#', quoted strings ("") the content of which is not tokenized
  // and '+-.' characters that are part of a floating point numbers and are exempt
  // from being interpreted as a token separator in case '+-.' are defined as
  // token separators.

  // Smart tokenizer. Absorb white space and token must be either punctuation or alphanum
  Bool_t first(kTRUE), quotedString(kFALSE), lineCont(kFALSE) ;
  char buffer[10240], c(0), cnext='\0', cprev=' ' ;
  Bool_t haveINF(kFALSE) ;
  Int_t bufptr(0) ;

  // Check for end of file 
   if (_is->eof() || _is->fail()) {
     _atEOF = kTRUE ;
     return TString("") ;
   }

  //Ignore leading newline
  if (_is->peek()=='\n') {
    _is->get(c) ;

    // If new line starts with #, zap it    
    while (_is->peek()=='#') {
      zapToEnd(kFALSE) ;
      _is->get(c) ; // absorb newline
    }
  }

  while(1) {
    // Buffer overflow protection
    if (bufptr>=10239) {
      oocoutW((TObject*)0,InputArguments) << "RooStreamParser::readToken: token length exceeds buffer capacity, terminating token early" << endl ;
      break ;
    }

    // Read next char
    _is->get(c) ;



    // Terminate at EOF, EOL or trouble
    if (_is->eof() || _is->fail() || c=='\n') break ;

    // Terminate as SPACE, unless we haven't seen any non-SPACE yet
    if (isspace(c)) {
      if (first) 
	continue ; 
      else 
	if (!quotedString) {
	  break ;
	}
    }

    // If '-' or '/' see what the next character is
    if (c == '.' || c=='-' || c=='+' || c=='/' || c=='\\') {
      _is->get(cnext) ;


      if (cnext=='I' || cnext=='i') {
	char tmp1,tmp2 ;
	_is->get(tmp1) ;
	_is->get(tmp2) ;
	_is->putback(tmp2) ;
	_is->putback(tmp1) ;
	haveINF = ((cnext=='I' && tmp1 == 'N' && tmp2 == 'F') || (cnext=='i' && tmp1 == 'n' && tmp2 == 'f')) ;
      } else {
	haveINF = kFALSE ;
      }

      _is->putback(cnext) ;
    }


    // Check for line continuation marker
    if (c=='\\' && cnext=='\\') {
      // Kill rest of line including endline marker
      zapToEnd(kFALSE) ;
      _is->get(c) ;
      lineCont=kTRUE ;
      break ;
    }

    // Stop if begin of comments is encountered
    if (c=='/' && cnext=='/') {
      zapToEnd(kFALSE) ;
      break ;
    }

    // Special handling of quoted strings
    if (c=='"') {
      if (first) {
	quotedString=kTRUE ;		
      } else if (!quotedString) {
	// Terminate current token. Next token will be quoted string
	_is->putback('"') ;
	break ;
      }
    }

    if (!quotedString) {
      // Decide if next char is punctuation (exempt - and . that are part of floating point numbers, or +/- preceding INF)
      if (isPunctChar(c) && !(c=='.' && (isdigit(cnext)||isdigit(cprev))) 
	  && (!first || !((c=='-'||c=='+') && (isdigit(cnext)||cnext=='.'||haveINF)))) {

	if (first) {
	  // Make this a one-char punctuation token
	  buffer[bufptr++]=c ;
	  break ;
	} else {
	  // Put back punct. char and terminate current alphanum token
	  _is->putback(c) ;
	  break ;
	} 
      }       
    } else {
      // Inside quoted string conventional tokenizing rules do not apply

      // Terminate token on closing quote
      if (c=='"' && !first) {
	buffer[bufptr++]=c ;	
	quotedString=kFALSE ;
	break ;
      }
    }

    // Store in buffer
    buffer[bufptr++]=c ;
    first=kFALSE ;
    cprev=c ;
  }

  if (_is->eof() || _is->bad()) {
    _atEOF = kTRUE ;
  }

  // Check if closing quote was encountered
  if (quotedString) {
    oocoutW((TObject*)0,InputArguments) << "RooStreamParser::readToken: closing quote (\") missing" << endl ;
  }

  // Absorb trailing white space or absorb rest of line if // is encountered
  if (c=='\n') {
    if (!lineCont) {
      _is->putback(c) ;
    }
  } else {
    c = _is->peek() ;

    while ((isspace(c) || c=='/') && c != '\n') {
      if (c=='/') {
	_is->get(c) ;
	if (_is->peek()=='/') {
	  zapToEnd(kFALSE) ;	
	} else {
	  _is->putback('/') ;
	}
	break ;
      } else {
	_is->get(c) ;
	c = _is->peek() ;
      }
    }
  }

  // If no token was read line is continued, return first token on next line
  if (bufptr==0 && lineCont) {
    return readToken() ;
  }
  
  // Zero terminate buffer and convert to TString
  buffer[bufptr]=0 ;
  return TString(buffer) ;
}



//_____________________________________________________________________________
TString RooStreamParser::readLine() 
{
  // Read an entire line from the stream and return as TString
  // This method recognizes the use of '\\' in the istream 
  // as line continuation token.

  char c,buffer[10240] ;
  Int_t nfree(10239) ; 
  
  if (_is->peek()=='\n') _is->get(c) ;

  // Read till end of line
  _is->getline(buffer,nfree,'\n') ;

  // Look for eventual continuation line sequence  
  char *pcontseq = strstr(buffer,"\\\\") ;
  if (pcontseq) nfree -= (pcontseq-buffer) ;
  while(pcontseq) {
    _is->getline(pcontseq,nfree,'\n') ;

    char* nextpcontseq = strstr(pcontseq,"\\\\") ;
    if (nextpcontseq) nfree -= (nextpcontseq-pcontseq) ;
    pcontseq = nextpcontseq ;
  }    

  // Chop eventual comments
  char *pcomment = strstr(buffer,"//") ;
  if (pcomment) *pcomment=0 ;

  // Chop leading and trailing space
  char *pstart=buffer ;
  while (isspace(*pstart)) {
    pstart++ ;
  }
  char *pend=buffer+strlen(buffer)-1 ;
  if (pend>pstart)
    while (isspace(*pend)) { *pend--=0 ; }

  if (_is->eof() || _is->fail()) {
    _atEOF = kTRUE ;
  }

  // Convert to TString
  return TString(pstart) ;
}



//_____________________________________________________________________________
void RooStreamParser::zapToEnd(Bool_t inclContLines) 
{
  // Eat all characters up to and including then end of the
  // current line. If inclContLines is kTRUE, all continuation lines
  // marked by the '\\' token are zapped as well

  // Skip over everything until the end of the current line
  if (_is->peek()!='\n') {

    char buffer[10240] ;
    Int_t nfree(10239) ; 

    // Read till end of line
    _is->getline(buffer,nfree,'\n') ;

    if (inclContLines) {
      // Look for eventual continuation line sequence  
      char *pcontseq = strstr(buffer,"\\\\") ;
      if (pcontseq) nfree -= (pcontseq-buffer) ;
      while(pcontseq) {
	_is->getline(pcontseq,nfree,'\n') ;
	
	char* nextpcontseq = strstr(pcontseq,"\\\\") ;
	if (nextpcontseq) nfree -= (nextpcontseq-pcontseq) ;
	pcontseq = nextpcontseq ;
      }    
    }

    // Put back newline character in stream buffer
   _is->putback('\n') ;
  }
}



//_____________________________________________________________________________
Bool_t RooStreamParser::expectToken(const TString& expected, Bool_t zapOnError) 
{
  // Read the next token and return kTRUE if it is identical to the given 'expected' token.

  TString token(readToken()) ;

  Bool_t error=token.CompareTo(expected) ;
  if (error && !_prefix.IsNull()) {
    oocoutW((TObject*)0,InputArguments) << _prefix << ": parse error, expected '" 
					<< expected << "'" << ", got '" << token << "'" << endl ;
    if (zapOnError) zapToEnd(kTRUE) ;
  }
  return error ;
}



//_____________________________________________________________________________
Bool_t RooStreamParser::readDouble(Double_t& value, Bool_t /*zapOnError*/) 
{
  // Read the next token and convert it to a Double_t. Returns true
  // if an error occurred in reading or conversion

  TString token(readToken()) ;
  if (token.IsNull()) return kTRUE ;
  return convertToDouble(token,value) ;
  
}



//_____________________________________________________________________________
Bool_t RooStreamParser::convertToDouble(const TString& token, Double_t& value) 
{
  // Convert given string to a double. Return true if the conversion fails.

  char* endptr = 0;
  const char* data=token.Data() ;

  // Handle +/- infinity cases, (token is guaranteed to be >1 char long)
  if (!strcasecmp(data,"inf") || !strcasecmp(data+1,"inf")) {
    value = (data[0]=='-') ? -RooNumber::infinity() : RooNumber::infinity() ;
    return kFALSE ;
  }

  value = strtod(data,&endptr) ;
  Bool_t error = (endptr-data!=token.Length()) ;

  if (error && !_prefix.IsNull()) {
    oocoutE((TObject*)0,InputArguments) << _prefix << ": parse error, cannot convert '" 
					<< token << "'" << " to double precision" <<  endl ;
  }
  return error ;
}



//_____________________________________________________________________________
Bool_t RooStreamParser::readInteger(Int_t& value, Bool_t /*zapOnError*/) 
{
  // Read a token and convert it to an Int_t. Returns true
  // if an error occurred in reading or conversion

  TString token(readToken()) ;
  if (token.IsNull()) return kTRUE ;
  return convertToInteger(token,value) ;
}



//_____________________________________________________________________________
Bool_t RooStreamParser::convertToInteger(const TString& token, Int_t& value) 
{
  // Convert given string to an Int_t. Returns true if an error
  // occurred in conversion

  char* endptr = 0;
  const char* data=token.Data() ;
  value = strtol(data,&endptr,10) ;
  Bool_t error = (endptr-data!=token.Length()) ;

  if (error && !_prefix.IsNull()) {
    oocoutE((TObject*)0,InputArguments)<< _prefix << ": parse error, cannot convert '" 
				       << token << "'" << " to integer" <<  endl ;
  }
  return error ;
}



//_____________________________________________________________________________
Bool_t RooStreamParser::readString(TString& value, Bool_t /*zapOnError*/) 
{
  // Read a string token. Returns true if an error occurred in reading
  // or conversion.  If a the read token is enclosed in quotation
  // marks those are stripped in the returned value

  TString token(readToken()) ;
  if (token.IsNull()) return kTRUE ;
  return convertToString(token,value) ;
}



//_____________________________________________________________________________
Bool_t RooStreamParser::convertToString(const TString& token, TString& string) 
{
  // Convert given token to a string (i.e. remove eventual quotation marks)

  // Transport to buffer 
  char buffer[10240],*ptr ;
  strncpy(buffer,token.Data(),10239) ;
  if (token.Length()>=10239) {
    oocoutW((TObject*)0,InputArguments) << "RooStreamParser::convertToString: token length exceeds 1023, truncated" << endl ;
    buffer[10239]=0 ;
  }
  int len = strlen(buffer) ;

  // Remove trailing quote if any
  if ((len) && (buffer[len-1]=='"'))
    buffer[len-1]=0 ;

  // Skip leading quote, if present
  ptr=(buffer[0]=='"') ? buffer+1 : buffer ;

  string = ptr ;
  return kFALSE ;
}
