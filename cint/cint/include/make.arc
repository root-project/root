############################################################
# Automatically created makefile for ../cint
############################################################

# Copying $CINTSYSDIR/MAKEINFO #############################

############################################################
# platform/linux2.0RH6.2
#  Platform dependent information for LINUX 2.0 RedHatH6.2 or later
#  Redhat-5.2
############################################################

# Tools
RM	    = rm -f
CP	    = cp
AR	    = ar
AROPT       = clq
KRCC        = gcc -traditional
CC          = gcc -Wall
CPP         = g++ -Wall -fguiding-decls
LD          = g++
CPREP       = gcc -E -C
CPPPREP     = g++ -E -C

# Compiler and linker option
CCDLLOPT    = -fPIC
LDDLLOPT    = -shared
OPTIMIZE    = -O2
LDOPT       = -lm -L/usr/lib/termcap -ltermcap -lbsd -ldl -rdynamic
SYSMACRO    = -DG__REGEXP -DG__SHAREDLIB -DG__OSFDLL -DG__ANSI
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
OBJECT     = ../cint
OPTION     =
DLLSPEC    =
LINKSPEC   = -DG__CLINK_ON -DG__CPPLINK_ON

# Set File names ###########################################
CIFC       = G__c_cint.c
CIFH       = G__c_cint.h
CIFO       = G__c_cint.o
CPPIFC     = G__cpp_cint.C
CPPIFH     = G__cpp_cint.h
CPPIFO     = G__cpp_cint.o

LIBS       = 

CCOPT      = 

CINTOPT      = 

COFILES    = 

RMCOFILES  = 

CHEADER    = \
		statistics.c \
		array.c \
		carray.c \
		fft.c \
		lsm.c \
		xgraph.c 
CHEADERCINT = \
		statistics.c \
		array.c \
		carray.c \
		fft.c \
		lsm.c \
		xgraph.c 

CSTUB      = 
CSTUBCINT  = 

CPPOFILES  = \
		ReadF.o \
		RegE.o 

RMCPPOFILES = \
		ReadF.o \
		RegE.o 

CPPHEADER  = \
		ReadF.h \
		RegE.h 
CPPHEADERCINT  = \
		ReadF.h \
		RegE.h 

CPPSTUB    = 
CPPSTUBCINT = 

# Link Object #############################################
$(OBJECT) : $(MAINO) $(CINTLIB) $(READLINEA) G__setup.o $(COFILES) $(CPPOFILES) $(CIFO) $(CPPIFO) 
	$(LD) $(OPTIMIZE) $(IPATH) $(MACRO) $(CCOPT) -o $(OBJECT) $(MAINO) $(CIFO) $(CPPIFO) $(COFILES) $(CPPOFILES) $(CINTLIB) G__setup.o $(READLINEA) $(LIBS) $(LDOPT)

# Compile User C source files ##############################

# Compile User C++ source files ############################
ReadF.o : ReadF.C $(CPPHEADER)
	$(CPP) $(IPATH) $(MACRO) $(OPTIMIZE) $(OPTION) $(CCOPT) -o ReadF.o -c ReadF.C

RegE.o : RegE.C $(CPPHEADER)
	$(CPP) $(IPATH) $(MACRO) $(OPTIMIZE) $(OPTION) $(CCOPT) -o RegE.o -c RegE.C


# Compile dictionary setup routine #######################
G__setup.o : $(CINTSYSDIR)/main/G__setup.c $(CINTSYSDIR)/G__ci.h
	$(CC) $(LINKSPEC) $(CINTIPATH) $(OPTIMIZE) $(OPTION) -o G__setup.o -c $(CINTSYSDIR)/main/G__setup.c

# Compile C Interface routine ############################
$(CIFO) : $(CIFC)
	$(CC) $(CINTIPATH) $(IPATH) $(MACRO) $(OPTIMIZE) $(OPTION) $(CCOPT) -c $(CIFC)

# Create C Interface routine #############################
$(CIFC) : $(CHEADER) $(CSTUB) $(CINTSYSDIR)/cint
	cint  -K -w0 -zcint -n$(CIFC) $(DLLSPEC) -D__MAKECINT__ -DG__MAKECINT  -c-2 $(KRMODE) $(IPATH) $(MACRO) $(CINTOPT) $(CHEADERCINT)

# Compile C++ Interface routine ##########################
$(CPPIFO) : $(CPPIFC)
	$(CPP) $(CINTIPATH) $(IPATH) $(MACRO) $(OPTIMIZE) $(OPTION) $(CCOPT) -c $(CPPIFC)

# Create C++ Interface routine ###########################
$(CPPIFC) : $(CPPHEADER) $(CPPSTUB) $(CINTSYSDIR)/cint
	cint  -w0 -zcint -n$(CPPIFC) $(DLLSPEC) -D__MAKECINT__ -DG__MAKECINT  -c-1 -A $(IPATH) $(MACRO) $(CINTOPT) $(CPPHEADERCINT)


# Clean up #################################################
clean :
	$(RM) $(OBJECT) core $(CIFO) $(CIFC) $(CIFH) $(CPPIFO) $(CPPIFC) $(CPPIFH) $(RMCOFILES) $(RMCPPOFILES) G__setup.o

# re-makecint ##############################################
makecint :
	makecint -mk make.arc -o ../cint -c statistics.c array.c carray.c fft.c lsm.c xgraph.c -H ReadF.h RegE.h -C++ ReadF.C RegE.C 

