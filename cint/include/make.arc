############################################################
# Automatically created makefile for ../cint
############################################################

# Copying $CINTSYSDIR/MAKEINFO #############################

############################################################
# platform/linux1.1
#  Platform dependent information for LINUX
############################################################

# Tools
RM	    = rm -f
CP	    = cp
AR	    = ar
AROPT       = clq
KRCC        = gcc -traditional
CC          = gcc
CPP         = g++
LD          = g++
CPREP       = gcc -E
CPPPREP     = g++ -E

# Compiler and linker option
CCDLLOPT    = -shared
LDDLLOPT    = 
OPTIMIZE    = -O
LDOPT       = -lm -ltermcap -lbsd
SYSMACRO    = -DG__REGEXP 
OTHMACRO    = -DG__P2FCAST -DG__REDIRECTIO
SYSIPATH    =

# libraries
MAINO	    = $(CINTSYSDIR)/main/G__cppmain.o
CINTLIB     = $(CINTSYSDIR)/src/G__ci.a
READLINEA   = $(CINTSYSDIR)/readline/libreadline.a
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
DLLPOST     = 




# End of $CINTSYSDIR/MAKEINFO ##############################

# Set variables ############################################
IPATH      = $(SYSIPATH) 
MACRO      = $(SYSMACRO)
CINTSYSDIR = /home/gotom/src/cintlinux1.1
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
	$(LD) $(OPTIMIZE) $(IPATH) $(MACRO) -o $(OBJECT) $(MAINO) $(CIFO) $(CPPIFO) $(COFILES) $(CPPOFILES) $(CINTLIB) G__setup.o $(READLINEA) $(LIBS) $(LDOPT)

# Compile User C source files ##############################

# Compile User C++ source files ############################
ReadF.o : ReadF.C $(CPPHEADER)
	$(CPP) $(IPATH) $(MACRO) $(OPTIMIZE) $(OPTION) -o ReadF.o -c ReadF.C

RegE.o : RegE.C $(CPPHEADER)
	$(CPP) $(IPATH) $(MACRO) $(OPTIMIZE) $(OPTION) -o RegE.o -c RegE.C


# Compile dictionary setup routine #######################
G__setup.o : $(CINTSYSDIR)/main/G__setup.c $(CINTSYSDIR)/G__ci.h
	$(CC) $(LINKSPEC) $(CINTIPATH) $(OPTIMIZE) $(OPTION) -o G__setup.o -c $(CINTSYSDIR)/main/G__setup.c

# Compile C Interface routine ############################
$(CIFO) : $(CIFC)
	$(CC) $(CINTIPATH) $(IPATH) $(MACRO) $(OPTIMIZE) $(OPTION) -c $(CIFC)

# Create C Interface routine #############################
$(CIFC) : $(CHEADER) $(CSTUB) $(CINTSYSDIR)/cint
	cint  -K -w0 -zcint -n$(CIFC) $(DLLSPEC) -D__MAKECINT__ -DG__MAKECINT  -c-2 $(KRMODE) $(IPATH) $(MACRO) $(CHEADERCINT)

# Compile C++ Interface routine ##########################
$(CPPIFO) : $(CPPIFC)
	$(CPP) $(CINTIPATH) $(IPATH) $(MACRO) $(OPTIMIZE) $(OPTION) -c $(CPPIFC)

# Create C++ Interface routine ###########################
$(CPPIFC) : $(CPPHEADER) $(CPPSTUB) $(CINTSYSDIR)/cint
	cint  -w0 -zcint -n$(CPPIFC) $(DLLSPEC) -D__MAKECINT__ -DG__MAKECINT  -c-1 -A $(IPATH) $(MACRO) $(CPPHEADERCINT)


# Clean up #################################################
clean :
	$(RM) $(OBJECT) core $(CIFO) $(CIFC) $(CIFH) $(CPPIFO) $(CPPIFC) $(CPPIFH) $(RMCOFILES) $(RMCPPOFILES) G__setup.o

# re-makecint ##############################################
makecint :
	makecint -mk make.arc -o ../cint -c statistics.c array.c carray.c fft.c lsm.c xgraph.c -H ReadF.h RegE.h -C++ ReadF.C RegE.C 

