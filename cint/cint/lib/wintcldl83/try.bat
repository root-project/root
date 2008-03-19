
cint -BG__cinttk_init -w0 -zwildc -Wwildc -Nwildc -nwildcIF.c -c-2 -I"\Program Files\tcl\include" -I"\Program Files\tcl\include\X11" -D__MAKECINT__ -DSTATIC_BUILD TOP.h
del WILDC.DEF
copy wildc.dbk WILDC.DEF
cd wildc
nmake /F wildc.mak CFG="wildc - Win32 Release"
cd ..
