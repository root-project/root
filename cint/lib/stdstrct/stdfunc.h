/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * header file stdfunc.h
 ************************************************************************
 * Description:
 *  Stub file for making ANSI C standard structs
 ************************************************************************
 * Copyright(c) 1991~1999,   Masaharu Goto (MXJ02154@niftyserve.or.jp)
 *
 ************************************************************************/

#ifndef G__STDFUNC
#define G__STDFUNC


#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include <time.h>
#include <locale.h>

#ifdef __MAKECINT__

void abort(void);
int abs(int n);
double acos(double arg);
double asin(double arg);
char* asctime(struct tm* timestruct);
double atan(double arg);
double atan2(double num,double den);
double atof(char *string);
int atoi(char *string);
long atol(char *string);
void *calloc(size_t count,size_t size);
double ceil(double z);
void clearerr(FILE *fp);
clock_t clock(void);
double cos(double radian);
double cosh(double value);
char *ctime(time_t *timeptr);
double difftime(time_t newtime,time_t oldtime);
div_t div(int numerator,int denominator);
double exp(double z);
double fabs(double z);
int fclose(FILE *fp);
int feof(FILE *fp);
int ferror(FILE *fp);
int fflush(FILE *fp);
int fgetc(FILE *fp);
int fgetpos(FILE *fp,fpos_t *position);
char *fgets(char *string,int n,FILE *fp);
double floor(double z);
double fmod(double number,double divisor);
FILE *fopen(char *file,char *mode);
int fputc(int character,FILE *fp);
int fputs(char *string,FILE *fp);
size_t fread(void *buffer,size_t size,size_t n,FILE *fp);
void free(void *ptr);
FILE *freopen(char *file,char *mode,FILE *fp);
double frexp(double real,int *exp1);
int fseek(FILE *fp,long offset,int whence);
int fsetpos(FILE *fp,fpos_t *position);
long ftell(FILE *fp);
size_t fwrite(void *buffer,size_t size,size_t n,FILE *fp);
int getc(FILE *fp);
int getchar(void);
char *getenv(char *variable);
char *gets(char *buffer);
struct tm* gmtime(time_t *caltime);
int isalnum(int c);
int isalpha(int c);
int iscntrl(int c);
int isdigit(int c);
int isgraph(int c);
int islower(int c);
int isprint(int c);
int ispunct(int c);
int isspace(int c);
int isupper(int c);
int isxdigit(int c);
long labs(long n);
double ldexp(double number,int n);
ldiv_t ldiv(long numerator,long denominator);
struct lconv* localeconv(void);
struct tm* localtime(time_t* timeptr);
double log(double z);
double log10(double z);
void *malloc(size_t size);
int mblen(char *address,size_t number);
size_t mbstowcs(wchar_t *widechar,char *multibyte,size_t number);
int mbtowc(wchar_t *charptr,char *address,size_t number);
void *memchr(void *region,int character,size_t n);
int memcmp(void *region1,void *region2,size_t count);
void *memcpy(void *region1,void *region2,size_t n);
void *memmove(void *region1,void *region2,size_t count);
void *memset(void *buffer,int character,size_t n);
time_t mktime(struct tm *timeptr);
double modf(double real,double *ip);
void perror(char *string);
double pow(double z,double x);
int putc(int character,FILE *fp);
int putchar(int character);
int puts(char *string);
int raise(int signal);
int rand(void);
void *realloc(void *ptr,size_t size);
int remove(char *filename);
int rename(char *old,char *new);
void rewind(FILE *fp);
void setbuf(FILE *fp,char *buffer);
char* setlocale(int position,char *locale);
int setvbuf(FILE *fp,char *buffer,int mode,size_t size);
double sin(double radian);
double sinh(double value);
double sqrt(double z);
void srand(unsigned int seed);
char *strcat(char *string1,char *string2);
char *strchr(char *string,int character);
int strcmp(char *string1,char *string2);
int strcoll(char *string1,char *string2);
char *strcpy(char *string1,char *string2);
size_t strcspn(char *string1,char *string2);
char *strerror(int error);
size_t strftime(char *string,size_t maximum,char *format,struct tm*brokentime);
size_t strlen(char *string);
char *strncat(char *string1,char *string2,size_t n);
int strncmp(char *string1,char *string2,size_t n);
char *strncpy(char *string1,char *string2,size_t n);
char *strpbrk(char *string1,char *string2);
char *strrchr(char *string,int character);
size_t strspn(char *string1,char *string2);
char *strstr(char *string1,char *string2);
double strtod(char *string,char **tailptr);
char *strtok(char *string1,char *string2);
long strtol(char *sprt,char **tailptr,int base);
unsigned long strtoul(char *sprt,char **tailptr,int base);
size_t strxfrm(char *string1,char *string2,size_t n);
int system(char *program);
double tan(double radian);
double tanh(double value);
time_t time(time_t *tp);
FILE *tmpfile(void);
char *tmpnam(char *name);
int tolower(int c);
int toupper(int c);
int ungetc(int character,FILE *fp);
size_t wcstombs(char *multibyte,wchar_t *widechar,size_t number);
int wctomb(char *string,wchar_t widecharacter);
//int fprintf(FILE *fp,char *format,arglist,...);
//int printf(char *format,arglist,...);
//int sprintf(char *string,char *format,arglist,...);
//int fscanf(FILE *fp,char *format,arglist,...);
//int scanf(char *format,arglist,...);
int sscanf(char *string,char *format,arglist,...);
void exit(int status);
//int atexit(void(*function)(void));

#endif

#endif
