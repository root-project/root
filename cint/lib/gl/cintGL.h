#ifdef __MAKECINT__
#ifndef G__WIN32
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#endif
/* Following pragmas delete symbols in abobe headers only.
 * Symbols in following headers will not be affected. */
#pragma link off all functions;
#pragma link off all typedefs;
#pragma link off all classes;
#endif

#define WINGDIAPI 
#define APIENTRY
#define CALLBACK
#ifdef __MAKECINT__
#define HGLRC
#endif

#include <stddef.h>
#include <GL/gl.h>
#include <GL/glu.h>
//#include <GL/glut.h>
#if defined(G__WIN32) || defined(_WIN32)
#include <GL/glaux.h>
#else
#include <GL/glx.h>
#endif
//#include <GL/xmesa.h>
