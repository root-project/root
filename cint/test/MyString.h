/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// MyString.h

#ifndef MYSTRING_H
#define MYSTRING_H

#ifdef __hpux
#include <iostream.h>
#else
#include <iostream>
using namespace std;
#endif

// 文字列クラス ///////////////////////////////////////////////////////
class MyString {
 public:
  // 初期化
  MyString();                      // デフォルト・コンストラクタ
  MyString(const MyString& strin); // コピー・コンストラクタ
  MyString(const char* charin);    // char* を MyString に変換
  MyString(const double din);      // 数値を MyString に変換

  // 終了処理
  ~MyString(); // デストラクタ

  // メンバ関数による演算子多重定義
  MyString& operator=(const MyString& strin); //代入演算子
  MyString& operator+=(const MyString& str1); //文字列の連接

  // フレンド関数による演算子多重定義
  friend bool operator==(const MyString& str1,const MyString& str2);
  friend bool operator!=(const MyString& str1,const MyString& str2)
    { return(!(str1==str2)); }
  friend MyString operator+(const MyString& str1,const MyString& str2);
  friend ostream& operator<<(ostream& ost,const MyString& str);

  // 情報取得
  char operator[](int index) const;     //文字要素取得
  char* c_str() const { return(pstr); } // char* として文字列を返す
  int length() const { return(len); }   // 文字列長

 private:
  char *pstr;
  int len;
};

#endif
