/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

#define NAME "A Simple Window"
#define ICON_NAME "Simple"
char *STRING= "Text inside the simple window.";
#ifdef __hpux
#define FONT "vbee-36"
#else
#define FONT "r14"
#endif


XWMHints xwmh ;

setXWMHints(XWMHints *p)
{
  p->flags = InputHint|StateHint;
  p->input = False;
  p->initial_state = NormalState;
  p->icon_pixmap=0;
  p->icon_window=0;
  p->icon_x=0;
  p->icon_y=0;
  p->icon_mask=0;
  p->window_group=0;
}

main(int argc,char **argv)
{
  unsigned int fontheight,pad,fg,bg,bd,bw;
  Display *dpy;
  Window win;
  GC gc;
  XFontStruct *fontstruct;
  XEvent event;
  XSizeHints xsh;
  XWindowAttributes xwa;
  XSetWindowAttributes xswa;
  
  setXWMHints(&xwmh);
  
  if((dpy=XOpenDisplay(NULL))==NULL) {
    fprintf(stderr,"%s: can't open %s.\n"
	    ,argv[0],XDisplayName(NULL));
    exit(1);
  }
  
  if((fontstruct = XLoadQueryFont(dpy,FONT))==NULL) {
    fprintf(stderr,"%s: display %s doesn't know font %s\n"
	    ,argv[0],DisplayString(dpy),FONT);
    exit(1);
  }
  
  fontheight = fontstruct->max_bounds.ascent+fontstruct->max_bounds.descent;
  
  bd = WhitePixel(dpy,DefaultScreen(dpy));
  bg = BlackPixel(dpy,DefaultScreen(dpy));
  fg = WhitePixel(dpy,DefaultScreen(dpy));
  
  bw=1;
  pad = 1;
  
  xsh.flags = (PPosition|PSize);
  xsh.height = fontheight + 2 * pad;
  xsh.width = XTextWidth(fontstruct,STRING,strlen(STRING))+2*pad;
  xsh.x = (DisplayWidth(dpy,DefaultScreen(dpy))-xsh.width)/2;
  xsh.y = (DisplayHeight(dpy,DefaultScreen(dpy))-xsh.height)/2;
  xsh.height *= 10;
  xsh.width *= 2;
#if 0
  printf("height=%d width=%d x=%d y=%d\n",xsh.height,xsh.width,xsh.x,xsh.y);
#endif
  
  win = XCreateSimpleWindow(dpy,DefaultRootWindow(dpy),
			    xsh.x,xsh.y,xsh.width,xsh.height,bw,bd,bg);
  
  XSetStandardProperties(dpy,win,NAME,ICON_NAME,None,argv,argc,&xsh);
  XSetWMHints(dpy,win,&xwmh);
  
  xswa.colormap = DefaultColormap(dpy,DefaultScreen(dpy));
  xswa.bit_gravity = CenterGravity;
  XChangeWindowAttributes(dpy,win,(CWColormap|CWBitGravity),&xswa);
  
  gc = DefaultGC(dpy,DefaultScreen(dpy));
  XSetFont(dpy,gc,fontstruct->fid);
  XSetForeground(dpy,gc,fg);
  XSetBackground(dpy,gc,bg);
  
  XSelectInput(dpy,win ,ExposureMask|StructureNotifyMask);
  
  XMapWindow(dpy,win);

  printf("!!!Hit ctl-C twice and type 'q' command to quit\n");
  
  while(1) {
    XNextEvent(dpy,&event);
    
    if((event.type==ConfigureNotify) ||
       (event.type==Expose)) {
      int i;
      int x,y;
      if(XGetWindowAttributes(dpy,win,&xwa) ==0) 
	break;
      x = (xwa.width-XTextWidth(fontstruct,STRING,strlen(STRING)))/2;
      y = (xwa.height+fontstruct->max_bounds.ascent-
	   fontstruct->max_bounds.descent)/2;
      XClearWindow(dpy,win);

      /* Draw graphic objects */
      XDrawString(dpy,win,gc,x,y,STRING,strlen(STRING));
      XDrawString(dpy,win,gc,x+50,y+50
		  ,"Hit ctl-C twice & type 'q' to quit",34);
      for(i=10;i<150;i+=10) XDrawPoint(dpy,win,gc,i,i);
      XDrawLine(dpy,win,gc,80,40,300,10);
      XDrawRectangle(dpy,win,gc,95,50,240,60);
      XDrawArc(dpy,win,gc,320,20,50,40,0,180*64);
      XDrawArc(dpy,win,gc,350,70,40,40,0,360*64);

      while(XCheckTypedEvent(dpy,Expose,&event)) ;
    }
  }
  
  fprintf(stderr,"Can't getwindow attributes\n");
  exit(1);
}




