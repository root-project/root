#!/usr/bin/env python

import sys

if __name__ == '__main__':

    if len(sys.argv)<2:
        sys.exit(0)

    filename = sys.argv[1]

    f = open(filename,"r")
    lines = f.readlines()
    f.close()

    fout = open(filename+".html","w")

    msglevels = ["FATAL", "ERROR", "WARNING", "DEBUG", "VERBOSE"]
        
    colors = { "FATAL":   "#990000",
               "ERROR":   "#FF0033",
               "WARNING": "#FF9933",
               "DEBUG":   "#CCFFCC",
               "VERBOSE": "#FFCCCC"
               }

    print >>fout, '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">'
    print >>fout, '<style>'
    print >>fout, """body {
  color: black;
  link: navy;
  vlink: maroon;
  alink: tomato;
  background: floralwhite;
  font-family: 'Lucida Console', 'Courier New', Courier, monospace;
  font-size: 10pt;
}"""
    for lvl in msglevels:
        print >>fout,"#%s {background-color: %s;}" % (lvl,colors[lvl])
    print >>fout, "</style>"
    print >>fout, """<html>
  <head>
    <title>%s</title>
  </head>

  <body><pre>""" % filename

    for l in lines:
        newline = l.rstrip('\n').replace("<","&lt;").replace(">","&gt;")
        colored=False
        for lvl in msglevels:
            if lvl in newline:
                newline = '<div id="%s">%s</div>' % (lvl,newline)
                colored=True
                break
        #if not colored:
        #    newline = newline+"<BR>"
        print >>fout, newline

    print >>fout, """  </pre></body>
</html>
"""
    fout.close()
