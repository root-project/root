############################################################
# Automatically created makefile for Array.dl
############################################################

# Copying $CINTSYSDIR/MAKEINFO #############################

############################################################
# platform/cygwin51_gcc3_so
#  Platform dependent information for Cygwin_NT-5.1
#  using gcc-3.00
#  Build libcint.dll instead of G__ci.a
############################################################

# Tools
AS	    = as
DLLTOOL	    = dlltool
LDDLL 	    = ld
#LIBS	    = -lcygwin -lkernel32 -luser32 -lgdi32 -lcomdlg32

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
CCDLLOPT    = -DBUILDING_DLL=1
LDDLLOPT    = -shared
# gcc-3.00 has a bug in -O2. Use -O instead. -O2 may be OK for later version
OPTIMIZE    = -O
LDOPT       = -lm -ltermcap -lreadline
SYSMACRO    = -DG__REGEXP -DG__SHAREDLIB -DG__OSFDLL -DG__ANSI -DG__ERRORCALLBACK -DG__CYGWIN=51 -DG__NEWSTDHEADER
OTHMACRO    = -DG__P2FCAST -DG__REDIRECTIO -DG__DETECT_NEWDEL -DG__POSIX -DG__STD_EXCEPTION $(CCDLLPOST) -DG__GNUREADLINE
SYSIPATH    =

# libraries
MAINO	    = $(CINTSYSDIR)/main/G__cppmain.o
CINTLIB     = $(CINTSYSDIR)/libcint.dll
READLINEA   = 
APIO	    = Api.o Class.o BaseCls.o Type.o DataMbr.o Method.o MethodAr.o CallFunc.o Typedf.o Apiif.o Token.o
RANLIB	    = /usr/bin/ranlib
STDLIBS     = gcc3strm.o stdstrct.o
PLATFORM    = $(CINTSYSDIR)/main/G__setup.o

# source code postfix
CSRCPOST    = .c
CHDRPOST    = .h
CPPSRCPOST  = .C
CPPHDRPOST  = .h
OBJPOST     = .o
DLLPOST     = .dll





##########################################################
# Control default behavior of cint command line input
##########################################################
#INPUTMODE = C++
#INPUTMODE = root
INPUTMODE = cint
#INPUTMODELOCK = on
INPUTMODELOCK = off

# End of $CINTSYSDIR/MAKEINFO ##############################

# Set variables ############################################
IPATH      = $(SYSIPATH) \
		-I../Complex 
MACRO      = $(SYSMACRO)
CINTSYSDIR = /home/gotom/src/cint
CINTIPATH  = -I$(CINTSYSDIR)
OBJECT     = Array.dl
OPTION     = $(CCDLLOPT)
DLLSPEC    =
LINKSPEC   = -DG__CPPLINK_ON

# Set File names ###########################################
CIFC       =
CIFH       =
CIFO       =
CPPIFC     = G__cpp_Array.C
CPPIFH     = G__cpp_Array.h
CPPIFO     = G__cpp_Array.o

LIBS       = 

CCOPT      = 

CINTOPT      = 

COFILES    = 

RMCOFILES  = 

CHEADER    = 
CHEADERCINT = 

CSTUB      = 
CSTUBCINT  = 

CPPOFILES  = \
		Fundament.o \
		Array.o \
		../Complex/Complex.o 

RMCPPOFILES = \
		Fundament.o \
		Array.o \
		../Complex/Complex.o 

CPPHEADER  = \
		Fundament.h \
		Array.h 
CPPHEADERCINT  = \
		Fundament.h \
		Array.h 

CPPSTUB    = 
CPPSTUBCINT = 

# Link Object #############################################
$(OBJECT) : $(CINTLIB) $(COFILES) $(CPPOFILES) $(CIFO) $(CPPIFO)
	$(LD) $(LDDLLOPT) $(OPTIMIZE) $(IPATH) $(MACRO) $(CCOPT) -o $(OBJECT) $(COFILES) $(CIFO) $(CPPIFO) $(CPPOFILES) $(LIBS) -L$(CINTSYSDIR) -lcint

# Compile User C source files ##############################

# Compile User C++ source files ############################
Fundament.o : Fundament.C $(CPPHEADER)
	$(CPP) $(IPATH) $(MACRO) $(OPTIMIZE) $(OPTION) $(CCOPT) -o Fundament.o -c Fundament.C

Array.o : Array.C $(CPPHEADER)
	$(CPP) $(IPATH) $(MACRO) $(OPTIMIZE) $(OPTION) $(CCOPT) -o Array.o -c Array.C

../Complex/Complex.o : ../Complex/Complex.C $(CPPHEADER)
	$(CPP) $(IPATH) $(MACRO) $(OPTIMIZE) $(OPTION) $(CCOPT) -o ../Complex/Complex.o -c ../Complex/Complex.C



# Compile C++ Interface routine ##########################
$(CPPIFO) : $(CPPIFC)
	$(CPP) $(CINTIPATH) $(IPATH) $(MACRO) $(OPTIMIZE) $(OPTION) $(CCOPT) -c $(CPPIFC)

# Create C++ Interface routine ###########################
$(CPPIFC) : $(CPPHEADER) $(CPPSTUB) $(CINTSYSDIR)/cint
	$(CINTSYSDIR)/cint  -w1 -zArray -n$(CPPIFC) $(DLLSPEC) -D__MAKECINT__ -DG__MAKECINT  -c-1 -A $(IPATH) $(MACRO) $(CINTOPT) $(CPPHEADERCINT)


# Clean up #################################################
clean :
	$(RM) $(OBJECT) core $(CIFO) $(CIFC) $(CIFH) $(CPPIFO) $(CPPIFC) $(CPPIFH) $(RMCOFILES) $(RMCPPOFILES)

# re-makecint ##############################################
makecint :
	makecint -mk Makefile -dl Array.dl -I../Complex -H Fundament.h Array.h -C++ Fundament.C Array.C ../Complex/Complex.C 

