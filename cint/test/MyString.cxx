/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// MyString.cxx

#include <string.h>
#include <stdio.h>
#include "MyString.h"

// デフォルト・コンストラクタ ///////////////////////////////////////
MyString::MyString() 
{
  len=0;
  pstr=0;
}

// コピー・コンストラクタ ///////////////////////////////////////////
MyString::MyString(const MyString& strin) 
{
  len=0;
  pstr=0;
  if(0!=strin.len) {
    len = strin.len;
    pstr = new char[len+1];
    strcpy(pstr,strin.pstr);
  }
}

// char* を MyString に変換するコンストラクタ ///////////////////////
MyString::MyString(const char* charin)
{
  len=0;
  pstr=0;
  if(0!=charin && 0!=charin[0]) {
    len = strlen(charin);
    pstr = new char[len+1];
    strcpy(pstr,charin);
  }
}

// 数値を MyString に変換するコンストラクタ /////////////////////////
MyString::MyString(const double din)
{
  char buf[50];
  len=0;
  pstr=0;
  sprintf(buf,"%g",din);
#if 1
  *(this) = buf; // TODO,  segv
#else
  len = strlen(buf);
  pstr=new char[len+1];
  strcpy(pstr,buf);
#endif
}

// デストラクタ /////////////////////////////////////////////////////
MyString::~MyString()
{
  if(pstr) delete[] pstr;
}

// 代入演算子 //////////////////////////////////////////////////////
MyString& MyString::operator=(const MyString& strin) 
{
  // 以前の文字列内容を消去
  if(pstr) delete[] pstr;
  pstr=0;
  len=0;

  // 右辺の文字列を代入
  if(strin.len) {
    len = strin.len;
    pstr = new char[len+1];
    //fprintf(stderr,"a %s %p\n",pstr,&strin);
    strcpy(pstr,strin.pstr);
  }
  return(*this);
}

// メンバ関数による文字列の連接 ///////////////////////////////////
MyString& MyString::operator+=(const MyString& strin) 
{
  if(0==strin.pstr) return(*this);
  if(0==pstr) {
    len = strin.len;
    delete[] pstr;
    pstr = new char[len+1];
    strcpy(pstr,strin.pstr);
  }
  else {
    len += strin.len;
    char *p = new char[len+1];
    strcpy(p,pstr);
    strcat(p,strin.pstr);
    delete[] pstr;
    pstr = p;
  }
  return(*this);
}

// フレンド関数による連接 /////////////////////////////////////////
MyString operator+(const MyString& str1,const MyString& str2)
{
  if(0==str1.pstr) return(str2);
  if(0==str2.pstr) return(str1);

  MyString str3(str1);
  str3 += str2;
  return(str3);
}

// 等値判定演算子 /////////////////////////////////////////////////
bool operator==(const MyString& str1,const MyString& str2)
{
  if( (0==str1.pstr && 0==str2.pstr) ||
      (str1.len==str2.len && 0==strcmp(str1.pstr,str2.pstr) ) ) {
    return(true);
  }
  else {
    return(false);
  }
}

// 出力ストリーム演算子 //////////////////////////////////////////
ostream& operator<<(ostream& ost,const MyString& str)
{
  if(str.pstr) ost << str.pstr ;
  return(ost);
}

// 文字要素取得 /////////////////////////////////////////////////
char MyString::operator[](int index) const
{
  if(index<0||len<index) return('\0');
  else                   return(pstr[index]);
}

