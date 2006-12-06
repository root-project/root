############################################################
# Automatically created makefile for ehdemo.dll
############################################################

# Copying $CINTSYSDIR/MAKEINFO #############################

############################################################
# platform/linux_RH7.3
#  Platform dependent information for LINUX 2.4 RedHatH7.3 or later
#  Redhat-7.3
############################################################

# Tools
RM	    = rm -f
CP	    = cp
AR	    = ar
AROPT       = clq
KRCC        = gcc -traditional
CC          = gcc -fexceptions -Wall
CPP         = g++ -Wall
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
IPATH      = $(SYSIPATH) 
MACRO      = $(SYSMACRO)
CINTSYSDIR = /home/gotom/src/cint
CINTIPATH  = -I$(CINTSYSDIR)
OBJECT     = ehdemo.dll
OPTION     = $(CCDLLOPT)
DLLSPEC    =
LINKSPEC   = -DG__CPPLINK_ON

# Set File names ###########################################
CIFC       =
CIFH       =
CIFO       =
CPPIFC     = G__cpp_ehdemo.C
CPPIFH     = G__cpp_ehdemo.h
CPPIFO     = G__cpp_ehdemo.o

LIBS       = 

CCOPT      = 

CINTOPT      = 

COFILES    = 

RMCOFILES  = 

CHEADER    = 
CHEADERCINT = 

CSTUB      = 
CSTUBCINT  = 

CPPOFILES  = 

RMCPPOFILES = 

CPPHEADER  = \
		ehdemo.h 
CPPHEADERCINT  = \
		ehdemo.h 

CPPSTUB    = 
CPPSTUBCINT = 

# Link Object #############################################
$(OBJECT) : $(COFILES) $(CPPOFILES) $(CIFO) $(CPPIFO)
	$(LD) $(LDDLLOPT) $(OPTIMIZE) $(IPATH) $(MACRO) $(CCOPT) -o $(OBJECT) $(COFILES) $(CIFO) $(CPPIFO) $(CPPOFILES) $(LIBS)

# Compile User C source files ##############################

# Compile User C++ source files ############################


# Compile C++ Interface routine ##########################
$(CPPIFO) : $(CPPIFC)
	$(CPP) $(CINTIPATH) $(IPATH) $(MACRO) $(OPTIMIZE) $(OPTION) $(CCOPT) -c $(CPPIFC)

# Create C++ Interface routine ###########################
$(CPPIFC) : $(CPPHEADER) $(CPPSTUB) $(CINTSYSDIR)/cint
	$(CINTSYSDIR)/cint  -w1 -zehdemo -n$(CPPIFC) $(DLLSPEC) -D__MAKECINT__ -DG__MAKECINT  -c-1 -A $(IPATH) $(MACRO) $(CINTOPT) $(CPPHEADERCINT)


# Clean up #################################################
clean :
	$(RM) $(OBJECT) core $(CIFO) $(CIFC) $(CIFH) $(CPPIFO) $(CPPIFC) $(CPPIFH) $(RMCOFILES) $(RMCPPOFILES)

# re-makecint ##############################################
makecint :
	makecint -mk Makefile -dl ehdemo.dll -H ehdemo.h 

