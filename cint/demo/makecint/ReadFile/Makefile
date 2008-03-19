############################################################
# Automatically created makefile for ReadFile.dl
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
IPATH      = $(SYSIPATH) 
MACRO      = $(SYSMACRO)
CINTSYSDIR = /home/gotom/src/cint
CINTIPATH  = -I$(CINTSYSDIR)
OBJECT     = ReadFile.dl
OPTION     = $(CCDLLOPT)
DLLSPEC    =
LINKSPEC   = -DG__CPPLINK_ON

# Set File names ###########################################
CIFC       =
CIFH       =
CIFO       =
CPPIFC     = G__cpp_ReadFile.C
CPPIFH     = G__cpp_ReadFile.h
CPPIFO     = G__cpp_ReadFile.o

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
		ReadFile.o 

RMCPPOFILES = \
		ReadFile.o 

CPPHEADER  = \
		ReadFile.h 
CPPHEADERCINT  = \
		ReadFile.h 

CPPSTUB    = 
CPPSTUBCINT = 

# Link Object #############################################
$(OBJECT) : $(CINTLIB) $(COFILES) $(CPPOFILES) $(CIFO) $(CPPIFO)
	$(LD) $(LDDLLOPT) $(OPTIMIZE) $(IPATH) $(MACRO) $(CCOPT) -o $(OBJECT) $(COFILES) $(CIFO) $(CPPIFO) $(CPPOFILES) $(LIBS) -L$(CINTSYSDIR) -lcint

# Compile User C source files ##############################

# Compile User C++ source files ############################
ReadFile.o : ReadFile.C $(CPPHEADER)
	$(CPP) $(IPATH) $(MACRO) $(OPTIMIZE) $(OPTION) $(CCOPT) -o ReadFile.o -c ReadFile.C



# Compile C++ Interface routine ##########################
$(CPPIFO) : $(CPPIFC)
	$(CPP) $(CINTIPATH) $(IPATH) $(MACRO) $(OPTIMIZE) $(OPTION) $(CCOPT) -c $(CPPIFC)

# Create C++ Interface routine ###########################
$(CPPIFC) : $(CPPHEADER) $(CPPSTUB) $(CINTSYSDIR)/cint
	$(CINTSYSDIR)/cint  -w1 -zReadFile -n$(CPPIFC) $(DLLSPEC) -D__MAKECINT__ -DG__MAKECINT  -c-1 -A $(IPATH) $(MACRO) $(CINTOPT) $(CPPHEADERCINT)


# Clean up #################################################
clean :
	$(RM) $(OBJECT) core $(CIFO) $(CIFC) $(CIFH) $(CPPIFO) $(CPPIFC) $(CPPIFH) $(RMCOFILES) $(RMCPPOFILES)

# re-makecint ##############################################
makecint :
	makecint -mk Makefile -dl ReadFile.dl -H ReadFile.h -C++ ReadFile.C 

