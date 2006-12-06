############################################################
# Automatically created makefile for pthread.dll
############################################################

# Copying $CINTSYSDIR/MAKEINFO #############################

############################################################
# platform/macos_gcc3_v3_v6
#  Platform dependent information for g++ with mininum config
#  By Alessandro Pistocchi
#  It seems like this setup is the absolute minimum. I believe
#  we can explore much more with MacOSX. Probably, you can get
#  as much as Linux_xxx. Contribution will be greatly appreciated.
#  Cint ver6 features turned on
############################################################

# Tools
RM	    = rm -f
CP	    = cp
AR	    = ar
AROPT       = clq
KRCC        = cc
CC          = gcc
CPP         = g++
LD          = g++
CPREP       = gcc -E -C
CPPPREP     = g++ -E -C

# Compiler and linker option
CCDLLOPT    = -fno-common
LDDLLOPT    = -bundle -flat_namespace -undefined suppress
OPTIMIZE    = -O
LDOPT       = -lm -ldl -lcurses 
SYSMACRO    = -DG__SHAREDLIB -DG__OSFDLL -DG__ANSI -DG__ERRORCALLBACK -DG__MACOSX__ -DG__SIGNEDCHAR -DG__LONGBUF -DG__NEWSTDHEADER -DG__CINT_VER6 -DG__NATIVELONGLONG
OTHMACRO   = -DG__APPLE -DG__P2FCAST -DG__NOPOWEROPR -DG__EXCEPTIONWRAPPER -DG__STD_EXCEPTION 
SYSIPATH    =

# libraries
MAINO	    = $(CINTSYSDIR)/main/G__cppmain.o
CINTLIB     = $(CINTSYSDIR)/src/G__ci.a
READLINEA   = $(CINTSYSDIR)/readline/libreadline.a
APIO	    = Api.o Class.o BaseCls.o Type.o DataMbr.o Method.o MethodAr.o CallFunc.o Typedf.o Apiif.o Token.o
RANLIB	    = /usr/bin/ranlib
STDLIBS     = gcc3strm.o stdstrct.o
PLATFORM    =  
CINT_V6     = bc_autoobj.o bc_cfunc.o bc_inst.o bc_parse.o bc_reader.o bc_type.o bc_exec.o bc_vtbl.o bc_eh.o bc_debug.o bc_assign.o

# source code postfix
CSRCPOST    = .c
CHDRPOST    = .h
CPPSRCPOST  = .C
CPPHDRPOST  = .h
OBJPOST     = .o
DLLPOST     = .so
 



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
MACRO      = $(SYSMACRO)\
		-D_POSIX_C_SOURCE=199506L 
CINTSYSDIR = /Users/gotom/src/cint
CINTIPATH  = -I$(CINTSYSDIR)
OBJECT     = pthread.dll
OPTION     = $(CCDLLOPT)
DLLSPEC    =
LINKSPEC   = -DG__CPPLINK_ON

# Set File names ###########################################
CIFC       =
CIFH       =
CIFO       =
CPPIFC     = G__cpp_pthread.C
CPPIFH     = G__cpp_pthread.h
CPPIFO     = G__cpp_pthread.o

LIBS       = \
		-lpthread 

CCOPT      = 

CINTOPT      = \
		-Z0 

COFILES    = 

RMCOFILES  = 

CHEADER    = 
CHEADERCINT = 

CSTUB      = 
CSTUBCINT  = 

CPPOFILES  = 

RMCPPOFILES = 

CPPHEADER  = \
		pthd.h 
CPPHEADERCINT  = \
		pthd.h 

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
	$(CINTSYSDIR)/cint  -w1 -zpthread -n$(CPPIFC) $(DLLSPEC) -D__MAKECINT__ -DG__MAKECINT  -c-1 -A $(IPATH) $(MACRO) $(CINTOPT) $(CPPHEADERCINT)


# Clean up #################################################
clean :
	$(RM) $(OBJECT) core $(CIFO) $(CIFC) $(CIFH) $(CPPIFO) $(CPPIFC) $(CPPIFH) $(RMCOFILES) $(RMCPPOFILES)

# re-makecint ##############################################
makecint :
	makecint -mk Makefile -dl pthread.dll -H pthd.h -D_POSIX_C_SOURCE=199506L -l -lpthread -cint -Z0 

