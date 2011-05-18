#!/bin/sh 
#

need_krb=0
need_qt=0
if test "x$1" = "xrpm" ; then 
    shift
    for i in $* ; do 
	case $i in 
	    *-dev)							;;
	    libroot-bindings-python)         
		echo "BuildRequires: python-devel >= 2.1"		;;
	    libroot-bindings-ruby)	   
		echo "BuildRequires: ruby-devel >= 1.8"			
		echo "BuildRequires: ruby >= 1.8"			;;
	    libroot-core)						;;
	    libroot-geom) 						;;
	    libroot-graf2d-gpad)					;;
	    libroot-graf2d-graf)					;;
	    libroot-graf2d-postscript)					;;
	    libroot-graf3d-eve)						;;
	    libroot-graf3d-g3d)						;;
	    libroot-graf3d-gl)
		echo "BuildRequires: mesa-libGLU-devel"			;;
	    libroot-gui)  						;;
	    libroot-gui-ged)        					;;
	    libroot-hist) 						;;
	    libroot-hist-spectrum)					;;
	    libroot-io)   						;;
	    libroot-io-xmlparser)
	   	echo "BuildRequires: libxml2-devel"			;;
	    libroot-math-physics)					;;
	    libroot-math-foam) 						;;
	    libroot-math-genvector)					;;
	    libroot-math-mathcore)                                      ;;
	    libroot-math-mathmore)
		echo "BuildRequires: gsl-devel"				;;
	    libroot-math-matrix)  					;;
	    libroot-math-minuit)					;;
	    libroot-math-mlp)						;;
	    libroot-math-quadp)						;;
	    libroot-math-smatrix) 					;;
	    libroot-math-splot) 					;;
	    libroot-math-unuran)					;;
	    libroot-misc-table) 					;;
	    libroot-misc-minicern) 				        
		echo "BuildRequires: gcc-gfortran"		 	;;
	    libroot-montecarlo-eg) 					;;
	    libroot-montecarlo-vmc) 					;;
	    libroot-net-ldap)		
		echo "BuildRequires: openldap-devel"			;;
	    libroot-proof)						;;
	    libroot-proof-clarens)  
		echo "BuildRequires: xmlrpc-c-devel"			;;
	    libroot-roofit)						;;
	    libroot-static)						;;
            libroot-tmva)						;;
	    libroot-tree)  						;;
	    libroot-tree-treeplayer)   					;;
	    libroot-net)   						;;
	    libroot-net-auth)   					;;
	    root-plugin-geom-geompainter)				;;
	    root-plugin-geom-geombuilder)				;;
	    root-plugin-geom-gdml)					;;
	    root-plugin-graf2d-x11)        				;;
# Build dependency on AfterStep-devel temporarily commented out 
# until such a time when ROOT can use the normal libAfterImage.
# Input the build dependencies of the libafterimage-dev package
#	    *asimage)	echo -n ", AfterStep-devel"			;;
	    root-plugin-graf2d-asimage)  				
		echo "BuildRequires:  freetype-devel"		
		echo "BuildRequires:  zlib-devel"		
		echo "BuildRequires:  libtiff-devel"		
		echo "BuildRequires:  libpng-devel"		
		echo "BuildRequires:  libungif-devel"		
		echo "BuildRequires:  libjpeg-devel"		
		echo "BuildRequires:  libICE-devel"		
		echo "BuildRequires:  libSM-devel"		
		echo "BuildRequires:  gawk"				;;
	    root-plugin-graf2d-qt)	  	need_qt=1 		;;
	    root-plugin-graf3d-x3d)        				;;
	    root-plugin-gui-fitpanel)   				;;
	    root-plugin-gui-guibuilder) 				;;
	    root-plugin-gui-qt)	   		need_qt=1		;;
	    root-plugin-gui-sessionviewer)				;;
	    root-plugin-hist-hbook)	   				;;
	    root-plugin-hist-histpainter)  				;;
	    root-plugin-hist-spectrumpainter)				;;
	    root-plugin-io-castor)   	
		echo "BuildRequires: castor-devel"			;;
	    root-plugin-io-chirp)					;;
	    root-plugin-io-dcache)   					
		echo "BuildRequires: d-cache-client"			;;
	    root-plugin-io-sql)						;;
	    root-plugin-io-xml)						;;
	    root-plugin-math-fftw3)	   				
		echo "BuildRequires: fftw3-devel"			;;
	    root-plugin-math-fumili)   					;;
	    root-plugin-math-mathmore)					;;
	    root-plugin-math-minuit2)					;;
	    root-plugin-math-mlp)					;;
	    root-plugin-montecarlo-pythia6)
		echo "BuildRequires: pythia6-devel"			;;
	    root-plugin-montecarlo-pythia8)
		echo "BuildRequires: pythia8-devel"			;;
	    root-plugin-net-alien)		
	   	echo "BuildRequires: AliEn-Client"			;;
	    root-plugin-net-globus)   	
		echo "BuildRequires: globus"				;;
	    root-plugin-net-krb5)	
		echo "BuildRequires: krb5-devel"			;;
	    root-plugin-net-netx)					;;
	    root-plugin-net-srp)	   				
		echo "BuildRequires: srp-devel"				;;
	    root-plugin-net-xrootd)					;;
	    root-plugin-proof-peac)					;;
	    root-plugin-proof-proofplayer)  				;;
	    root-plugin-proof-xproof)	  				;;
	    root-plugin-sql-oracle)   
		"BuildRequires: oracle-instantclient-devel"		;;
	    root-plugin-sql-mysql)	   
		cat <<EOF
%if %{?_vendor} 
 %if %{_vendor} == "MandrakeSoft"
BuildRequires: MySQL-devel >= 4.1.0
 %else
BuildRequires: mysql-devel >= 4.1.0
 %endif
%else
BuildRequires: mysql-devel >= 4.1.0
%endif
EOF
		;;
	    root-plugin-sql-odbc)	   	
		echo "BuildRequires: unixODBC-devel >= 2.2.11"		;;
	    root-plugin-sql-pgsql)	        
		echo "BuildRequires: postgresql-devel"			;;
	    root-plugin-sql-maxdb)	   
		echo "BuildRequires: libsqlod75-dev"			;;
	    root-plugin-tree-treeviewer)				;;
	    root-system-bin)						;;
	    root-system-common)						;;
	    root-system-doc)  						;;
	    root-system-proofd)						;;
	    root-system-rootd)						;;
	    root-system-xrootd)		need_krb=1			;;
	    ttf-root*)							;;
	    *) 
		echo "*** Warning *** Unknown package $i - please update $0" \
		    > /dev/stderr ;;
	esac
    done
    if test $need_qt -gt 0 ; then 
		    cat <<EOF
%if %{?_vendor} 
 %if %{_vendor} == "MandrakeSoft"
BuildRequires: libqt4-devel
 %else
   %if %{_vendor} == "suse"
BuildRequires: qt4-devel >= 4.3.0
   %else 
BuildRequires: qt4-devel >= 4.3.0
   %endif
 %endif
%else
BuildRequires: qt4-devel >= 4.3.0
%endif
EOF
    fi
    if test $need_krb -gt 0 ; then 
	echo "BuildRequires: krb5-devel"
    fi
    exit 0
fi    

### echo %%% Making build dependencies
bd=
have_krb=0
have_qt=0
for i in $* ; do 
    case $i in 
	*-dev)							        ;;
	libroot-bindings-python)         
	    echo -n ", python-support (>= 0.3)"				;;
	libroot-bindings-ruby)	   
	    echo -n ", ruby (>= 1.8), ruby1.8-dev | ruby-dev (>= 1.8)"	;;
	libroot-core)							;;
	libroot-geom) 							;;
	libroot-graf2d-gpad)						;;
	libroot-graf2d-graf)						;;
	libroot-graf2d-postscript)					;;
	libroot-graf3d-eve)						;;
	libroot-graf3d-g3d)						;;
	libroot-graf3d-gl)					        
	    echo -n ", libglu1-mesa-dev | libglu1-xorg-dev "
	    echo -n "| xlibmesa-glu-dev |  libglu-dev"			
	    echo -n ", libftgl-dev | ftgl-dev"				;;
	libroot-gui)  							;;
	libroot-gui-ged)        					;;
	libroot-hist) 							;;
	libroot-hist-spectrum)						;;
	libroot-io)   							;;
	libroot-io-xmlparser)	   	echo -n ", libxml2-dev"		;;
	libroot-math-physics)						;;
	libroot-math-foam) 						;;
	libroot-math-genvector)						;;
	libroot-math-mathcore)						;;
	libroot-math-mathmore)		echo -n ", libgsl0-dev"		;;
	libroot-math-matrix)  						;;
	libroot-math-minuit)						;;
	libroot-math-mlp)						;;
	libroot-math-quadp)						;;
	libroot-math-smatrix) 						;;
	libroot-math-splot) 						;;
	libroot-math-unuran)						;;
	libroot-misc-table) 						;;
	libroot-misc-minicern) 						
	    echo -n ", gfortran|fortran-compiler" 			;;
	libroot-montecarlo-eg) 						;;
	libroot-montecarlo-vmc) 					;;
	libroot-net-ldap)		
	    echo -n ", libldap2-dev | libldap-dev"			;;
	libroot-proof)							;;
	libroot-proof-clarens)  
	    echo -n ", libxmlrpc-c3-dev | libxmlrpc-c-dev"
	    echo -n ", libcurl4-gnutls-dev | libcurl4-openssl-dev | libcurl-dev"	
	    								;;
	libroot-roofit)							;;
        libroot-tmva)							;;
	libroot-tree)  							;;
	libroot-tree-treeplayer)   					;;
	libroot-net)   							;;
	libroot-net-auth)   						;;
	root-plugin-geom-geompainter)					;;
	root-plugin-geom-geombuilder)					;;
	root-plugin-geom-gdml)						;;
	root-plugin-graf2d-x11)        					;;
# Build dependency libafterimage-dev temporarily commented out 
# until such a time that ROOT can use the normal libAfterImage.
#	root-plugin-graf3d-asimage)  	echo -n ", libafterimage-dev"	;;
# Input the build dependencies of the libafterimage-dev package
	root-plugin-graf2d-asimage)  
	    echo -n ", libjpeg62-dev, libpng12-dev, libtiff4-dev"
            echo -n ", libgif-dev | libungif4-dev, libxinerama-dev";;
	root-plugin-graf2d-qt)	      need_qt=1		  	        ;;
	root-plugin-graf3d-x3d)        					;;
	root-plugin-gui-fitpanel)   					;;
	root-plugin-gui-guibuilder) 					;;
	root-plugin-gui-qt)	   	need_qt=1			;;
	root-plugin-gui-sessionviewer)					;;
	root-plugin-hist-hbook)	   					;;
	root-plugin-hist-histpainter)  					;;
	root-plugin-hist-spectrumpainter)				;;
	root-plugin-io-castor)   	echo -n ", libshift-dev"	;;
	root-plugin-io-chirp)	 	echo -n ", libchirp-dev"	;;
	root-plugin-io-dcache)   	echo -n ", libdcap-dev"		;;
	root-plugin-io-sql)						;;
	root-plugin-io-xml)						;;
	root-plugin-math-fftw3)	   	
	    echo -n ", libfftw3-dev | fftw3-dev"			;;
	root-plugin-math-fumili)   					;;
	root-plugin-math-minuit2)					;;
	root-plugin-math-mlp)						;;
	root-plugin-montecarlo-pythia5) echo -n ", pythia5-dev"		;;
	root-plugin-montecarlo-pythia6)	echo -n ", pythia6-dev"		;;
	root-plugin-montecarlo-pythia8)	echo -n ", pythia8-dev"		;;
	root-plugin-net-alien)	   	echo -n ", libgapiui-dev" 	;;
	root-plugin-net-globus)   	
	    echo -n ", libglobus-gss-assist-dev"
            echo -n ", libglobus-gssapi-gsi-dev"
            echo -n ", libglobus-gsi-credential-dev"
            echo -n ", libglobus-common-dev"
            echo -n ", libglobus-gsi-callback-dev"
            echo -n ", libglobus-proxy-ssl-dev"
            echo -n ", libglobus-gsi-sysconfig-dev"
            echo -n ", libglobus-openssl-error-dev"
            echo -n ", libglobus-gssapi-gsi-dev"
            echo -n ", libglobus-gsi-callback-dev"
            echo -n ", libglobus-oldgaa-dev"
            echo -n ", libglobus-gsi-cert-utils-dev"
            echo -n ", libglobus-openssl-dev"
            echo -n ", libglobus-gsi-proxy-core-dev"
	    echo -n ", libglobus-callout-dev"
 	    ;;
	root-plugin-net-krb5)		 need_krb5=1			
	    echo -n ",krb5-user|heimdal-clients"		        ;;
	root-plugin-net-netx)						;;
	root-plugin-net-srp)	   	echo -n ", libsrputil-dev"	;;
	root-plugin-net-xrootd)						;;
	root-plugin-proof-peac)						;;
	root-plugin-proof-proofplayer)  				;;
	root-plugin-proof-xproof) 	 				;;
	root-plugin-sql-oracle)   
	    echo -n ", oracle-instantclient-devel"			;;
	root-plugin-sql-mysql)	   
	    echo -n ", libmysqlclient15-dev | libmysqlclient14-dev"
	    echo -n "| libmysqlclient12-dev| libmysqlclient-dev"	;;
	root-plugin-sql-odbc)	   	
	    echo -n ", libiodbc2-dev | unixodbc-dev"			;;
	root-plugin-sql-pgsql)	        
	    echo -n ",  libpq-dev | postgresql-dev"			;;
	root-plugin-sql-maxdb)	   
	    echo -n ", libsqlod75-dev [i386 ia64 amd64]"		;;
	root-plugin-tree-treeviewer)					;;
	root-system-bin)						;;
	root-system-common)						;;
	root-system-doc)  						;;
	root-system-proofd)						;;
	root-system-rootd)						;;
	root-system-xrootd)	need_krb5=1				;;
	ttf-root*)							;;
	*) 
	    echo "*** Warning *** Unknown package $i - please update $0" \
		> /dev/stderr ;;
    esac
done
if test $need_qt -gt 0 ; then 
    echo -n ", libqt4-dev (>= 4.3.0) | libqt3-mt-dev (>= 3.3.0)"
    echo -n ", qt4-dev-tools (>= 4.3.0) | qt3-dev-tools (>= 3.3.0)"
    echo -n ", libqt4-opengl-dev"
fi
if test $need_krb5 -gt 0 ; then
    echo -n ", libkrb5-dev|heimdal-dev"
fi 

#
# EOF
#
