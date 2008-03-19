############################################################
# Automatically created makefile for wildc
############################################################

# Copying $CINTSYSDIR/MAKEINFO #############################

############################################################
# platform/linux_RH6.2
#  Platform dependent information for LINUX 2.0 RedHatH6.2 or later
#  Redhat-6.2
############################################################

# Tools
RM	    = rm -f
CP	    = cp
AR	    = ar
AROPT       = clq
KRCC        = gcc -traditional
CC          = gcc -fexceptions -Wall
CPP         = g++ -Wall -fguiding-decls
LD          = g++
CPREP       = gcc -E -C
CPPPREP     = g++ -E -C

# Compiler and linker option
CCDLLOPT    = -fPIC
LDDLLOPT    = -shared
OPTIMIZE    = -O2
LDOPT       = -lm -L/usr/lib/termcap -ltermcap -lbsd -ldl -rdynamic
SYSMACRO    = -DG__REGEXP -DG__SHAREDLIB -DG__OSFDLL -DG__ANSI -DG__ERRORCALLBACK
OTHMACRO    = -DG__P2FCAST -DG__REDIRECTIO -DG__DETECT_NEWDEL -DG__POSIX -DG__STD_EXCEPTION 
SYSIPATH    =

# libraries
MAINO	    = $(CINTSYSDIR)/main/G__cppmain.o
CINTLIB     = $(CINTSYSDIR)/src/G__ci.a
READLINEA   = /usr/lib/libreadline.a
APIO	    = Api.o Class.o BaseCls.o Type.o DataMbr.o Method.o MethodAr.o CallFunc.o Typedf.o Apiif.o Token.o
RANLIB	    = /usr/bin/ranlib
STDLIBS     = libstrm.o stdstrct.o
PLATFORM    =

# source code postfix
CSRCPOST    = .c
CHDRPOST    = .h
CPPSRCPOST  = .C
CPPHDRPOST  = .h
OBJPOST     = .o
DLLPOST     = .dl




# End of $CINTSYSDIR/MAKEINFO ##############################

# Set variables ############################################
IPATH      = $(SYSIPATH) \
		-I/home/gotom/src/cint \
		-I/usr/X11R6/include \
		-I/usr/local/include \
		-I/usr/local/X11R6/include 
MACRO      = $(SYSMACRO)
CINTSYSDIR = /home/gotom/src/cint
CINTIPATH  = -I$(CINTSYSDIR)
OBJECT     = wildc
OPTION     =
DLLSPEC    =
LINKSPEC   = -DG__CLINK_ON

# Set File names ###########################################
CIFC       = G__c_wildc.c
CIFH       = G__c_wildc.h
CIFO       = G__c_wildc.o
CPPIFC     =
CPPIFH     =
CPPIFO     =

LIBS       = \
		-L/usr/X11R6/lib \
		-L/usr/local/lib \
		-lX11 \
		-ltk \
		-ltcl 

CCOPT      = 

CINTOPT      = 

COFILES    = \
		AppInit.o \
		TkInit.o \
		WildCard.o 

RMCOFILES  = \
		AppInit.o \
		TkInit.o \
		WildCard.o 

CHEADER    = \
		TOP.h \
		TCLTK.h \
		TKMACRO.h \
		TCLMACRO.h 
CHEADERCINT = \
		TOP.h \
		TCLTK.h \
		TKMACRO.h \
		TCLMACRO.h 

CSTUB      = 
CSTUBCINT  = 

CPPOFILES  = \
		Main.o 

RMCPPOFILES = \
		Main.o 

CPPHEADER  = 
CPPHEADERCINT  = 

CPPSTUB    = 
CPPSTUBCINT = 

# Link Object #############################################
$(OBJECT) : $(CINTLIB) $(READLINEA) G__setup.o $(COFILES) $(CPPOFILES) $(CIFO) $(CPPIFO) 
	$(LD) $(OPTIMIZE) $(IPATH) $(MACRO) $(CCOPT) -o $(OBJECT) $(CIFO) $(CPPIFO) $(COFILES) $(CPPOFILES) $(CINTLIB) G__setup.o $(READLINEA) $(LIBS) $(LDOPT)

# Compile User C source files ##############################
AppInit.o : AppInit.c $(CHEADER)
	$(CC) $(IPATH) $(MACRO) $(OPTIMIZE) $(OPTION) $(CCOPT) -o AppInit.o -c AppInit.c

TkInit.o : TkInit.c $(CHEADER)
	$(CC) $(IPATH) $(MACRO) $(OPTIMIZE) $(OPTION) $(CCOPT) -o TkInit.o -c TkInit.c

WildCard.o : WildCard.c $(CHEADER)
	$(CC) $(IPATH) $(MACRO) $(OPTIMIZE) $(OPTION) $(CCOPT) -o WildCard.o -c WildCard.c


# Compile User C++ source files ############################
Main.o : Main.cxx $(CPPHEADER)
	$(CPP) $(IPATH) $(MACRO) $(OPTIMIZE) $(OPTION) $(CCOPT) -o Main.o -c Main.cxx


# Compile dictionary setup routine #######################
G__setup.o : $(CINTSYSDIR)/main/G__setup.c $(CINTSYSDIR)/G__ci.h
	$(CC) $(LINKSPEC) $(CINTIPATH) $(OPTIMIZE) $(OPTION) -o G__setup.o -c $(CINTSYSDIR)/main/G__setup.c

# Compile C Interface routine ############################
$(CIFO) : $(CIFC)
	$(CC) $(CINTIPATH) $(IPATH) $(MACRO) $(OPTIMIZE) $(OPTION) $(CCOPT) -c $(CIFC)

# Create C Interface routine #############################
$(CIFC) : $(CHEADER) $(CSTUB) $(CINTSYSDIR)/cint
	$(CINTSYSDIR)/cint -BG__cinttk_init -K -w0 -zwildc -n$(CIFC) $(DLLSPEC) -D__MAKECINT__ -DG__MAKECINT  -c-2 $(KRMODE) $(IPATH) $(MACRO) $(CINTOPT) $(CHEADERCINT)


# Clean up #################################################
clean :
	$(RM) $(OBJECT) core $(CIFO) $(CIFC) $(CIFH) $(CPPIFO) $(CPPIFC) $(CPPIFH) $(RMCOFILES) $(RMCPPOFILES) G__setup.o

# re-makecint ##############################################
makecint :
	makecint -B G__cinttk_init -mk Makefile -o wildc -I/home/gotom/src/cint -I/usr/X11R6/include -I/usr/local/include -I/usr/local/X11R6/include -m -C AppInit.c TkInit.c WildCard.c -C++ Main.cxx -h TOP.h TCLTK.h TKMACRO.h TCLMACRO.h -l -L/usr/X11R6/lib -L/usr/local/lib -lX11 -ltk -ltcl 

