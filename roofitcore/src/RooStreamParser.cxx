/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooStreamParser.cc,v 1.5 2001/05/17 00:43:16 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --
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


#include <iostream.h>
#include <stdlib.h>
#include <ctype.h>
#include "RooFitCore/RooStreamParser.hh"


ClassImp(RooStreamParser)


RooStreamParser::RooStreamParser(istream& is) : 
  _is(is), _prefix(""), _punct("()[]<>|/\\:?.,=+-&^%$#@!`~")
{
  // Constructor
}

RooStreamParser::RooStreamParser(istream& is, TString errorPrefix) : 
  _is(is), _prefix(errorPrefix), _punct("()[]<>|/\\:?.,=+-&^%$#@!`~")
{
  // Constructor with error message prefix
}


RooStreamParser::~RooStreamParser()
{
  // Destructor
}


void RooStreamParser::setPunctuation(TString punct) 
{
  // Change list of punctuation characters
  _punct = punct ;
}


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


TString RooStreamParser::readToken() 
{
  // Read one token

  // Smart tokenizer. Absorb white space and token must be either punctuation or alphanum
  Bool_t first(kTRUE), quotedString(kFALSE) ;
  char buffer[1024], c, cnext, cprev=' ' ;
  Int_t bufptr=0 ;

  //Ignore leading newline
  if (_is.peek()=='\n') _is.get(c) ;

  while(1) {
    _is.get(c) ;

    // Terminate at EOF, EOL or trouble
    if (!_is.good() || c=='\n') break ;

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
    if (c == '.' || c=='-' || c=='/') {
      _is.get(cnext) ;
      _is.putback(cnext) ;
    }
    
    // Stop if begin of comments is encountered
    if (c=='/' && cnext=='/') {
      zapToEnd() ;
      break ;
    }

    // Special handling of quoted strings
    if (c=='"') {
      if (first) {
	quotedString=kTRUE ;		
      } else if (!quotedString) {
	// Terminate current token. Next token will be quoted string
	_is.putback('"') ;
	break ;
      }
    }

    if (!quotedString) {
      // Decide if next char is punctuation (exempt - and . that are part of floating point numbers)
      if (isPunctChar(c) && !(c=='.' && (isdigit(cnext)||isdigit(cprev))) 
	  && !(c=='-' && (isdigit(cnext)||cnext=='.'))) {
	if (first) {
	  // Make this a one-char punctuation token
	  buffer[bufptr++]=c ;
	  break ;
	} else {
	  // Put back punct. char and terminate current alphanum token
	  _is.putback(c) ;
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

  // Check if closing quote was encountered
  if (quotedString) {
    cout << "RooStreamParser::readToken: closing quote (\") missing" << endl ;
  }

  // Absorb trailing white space or absorb rest of line if // is encountered
  if (c=='\n') {
    _is.putback(c) ;
  } else {
    c = _is.peek() ;

    while ((isspace(c) || c=='/') && c != '\n') {
      if (c=='/') {
	_is.get(c) ;
	if (_is.peek()=='/') {
	  zapToEnd() ;	
	} else {
	  _is.putback('/') ;
	}
	break ;
      } else {
	_is.get(c) ;
	c = _is.peek() ;
      }
    }
  }
  
  // Zero terminate buffer and convert to TString
  buffer[bufptr]=0 ;
  return TString(buffer) ;
}


TString RooStreamParser::readLine() 
{
  // Read an entire line

  char c,buffer[1024] ;
  
  if (_is.peek()=='\n') _is.get(c) ;

  // Read till end of line
  _is.getline(buffer,1024,'\n') ;

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
  
  // Convert to TString
  return TString(pstart) ;
}


void RooStreamParser::zapToEnd() 
{
  // Skip over everything until the end of the current line
  if (!_is.peek()!='\n') {
    _is.ignore(1000,'\n') ;
    _is.putback('\n') ;
  }
}


void RooStreamParser::putBackToken(TString& token) 
{
  // Put back given token in stream buffer

  const char* buf=token.Data() ;
  int len=token.Length() ;
  for (int i=len-1 ; i>=0 ; i--)
    _is.putback(buf[i]) ;
  
  // Add a space to keep the token separate
  if (buf[0]=='\n') {
  } else {
  _is.putback(' ') ;  
  }
}


Bool_t RooStreamParser::expectToken(TString expected, Bool_t zapOnError) 
{
  // Read a token and check if it matches the given expected value
  TString token=readToken() ;

  Bool_t error=token.CompareTo(expected) ;
  if (error && !_prefix.IsNull()) {
    cout << _prefix << ": parse error, expected '" 
	 << expected << "'" << ", got '" << token << "'" << endl ;
    if (zapOnError) zapToEnd() ;
  }
  return error ;
}


Bool_t RooStreamParser::readDouble(Double_t& value, Bool_t zapOnError) 
{
  // Read a token and convert it to a Double_t
  TString token=readToken() ;
  if (token.IsNull()) return kTRUE ;
  return convertToDouble(token,value) ;
  
}


Bool_t RooStreamParser::convertToDouble(TString token, Double_t& value) 
{
  // Convert given string to a double
  char* endptr(0) ;
  const char* data=token ;
  value = strtod(data,&endptr) ;
  Bool_t error = (endptr-data!=token.Length()) ;

  if (error && !_prefix.IsNull()) {
    cout << _prefix << ": parse error, cannot convert '" 
	 << token << "'" << " to double precision" <<  endl ;
  }
  return error ;
}


Bool_t RooStreamParser::readInteger(Int_t value, Bool_t zapOnError) 
{
  // Read a token and convert it to an Int_t
  TString token=readToken() ;
  if (token.IsNull()) return kTRUE ;
  return convertToInteger(token,value) ;
}


Bool_t RooStreamParser::convertToInteger(TString token, Int_t& value) 
{
  // Convert given string to an Int_t
  char* endptr(0) ;
  const char* data=token ;
  value = strtol(data,&endptr,10) ;
  Bool_t error = (endptr-data!=token.Length()) ;

  if (error && !_prefix.IsNull()) {
    cout << _prefix << ": parse error, cannot convert '" 
	 << token << "'" << " to integer" <<  endl ;
  }
  return error ;
}


Bool_t RooStreamParser::readString(TString& value, Bool_t zapOnError) 
{
  // Read a string token
  TString token=readToken() ;
  if (token.IsNull()) return kTRUE ;
  return convertToString(token,value) ;
}


Bool_t RooStreamParser::convertToString(TString token, TString& string) 
{
  // Convert given token to a string (i.e. remove eventual quotation marks)

  // Transport to buffer 
  char buffer[1024],*ptr ;
  strcpy(buffer,token) ;
  int len = strlen(buffer) ;

  // Remove trailing quote if any
  if ((len) && (buffer[len-1]=='"'))
    buffer[len-1]=0 ;

  // Skip leading quote, if present
  ptr=(buffer[0]=='"') ? buffer+1 : buffer ;

  string = ptr ;
  return kFALSE ;
}





