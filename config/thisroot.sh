# Source this script to set up the ROOT build that this script is part of.
#
# Conveniently an alias like this can be defined in .bashrc or .zshrc:
#   alias thisroot=". bin/thisroot.sh"
#
# This script is for the bash like shells, see thisroot.csh for csh like shells.
#
# Author: Fons Rademakers, 18/8/2006

# shellcheck disable=SC2123,SC3043
# Whole-file directives
# Disable SC2123 to not warn PATH modification
# Disable SC3043 to not warn for the local keyword

drop_from_path()
{
   # Assert that we got enough arguments
   if test $# -ne 2 ; then
      echo "drop_from_path: needs 2 arguments" >&2
      return 1
   fi

   local p="$1" 2>/dev/null
   local drop="$2" 2>/dev/null

   newpath="$(echo "$p" | sed -e "s;:${drop}:;:;g" \
                          -e "s;:${drop}\$;;g"   \
                          -e "s;^${drop}:;;g"   \
                          -e "s;^${drop}\$;;g")"
}

clean_environment()
{

   if [ -n "${old_rootsys-}" ] ; then
      if [ -n "${PATH-}" ]; then
         drop_from_path "$PATH" "${old_rootsys}/bin"
         PATH=$newpath
      fi
      if [ -n "${LD_LIBRARY_PATH-}" ]; then
         drop_from_path "$LD_LIBRARY_PATH" "${old_rootsys}/lib"
         drop_from_path "$LD_LIBRARY_PATH" "${old_rootsys}/lib/root"
         LD_LIBRARY_PATH=$newpath
      fi
      if [ -n "${DYLD_LIBRARY_PATH-}" ]; then
         drop_from_path "$DYLD_LIBRARY_PATH" "${old_rootsys}/lib"
         drop_from_path "$DYLD_LIBRARY_PATH" "${old_rootsys}/lib/root"
         DYLD_LIBRARY_PATH=$newpath
      fi
      if [ -n "${SHLIB_PATH-}" ]; then
         drop_from_path "$SHLIB_PATH" "${old_rootsys}/lib"
         drop_from_path "$SHLIB_PATH" "${old_rootsys}/lib/root"
         SHLIB_PATH=$newpath
      fi
      if [ -n "${LIBPATH-}" ]; then
         drop_from_path "$LIBPATH" "${old_rootsys}/lib"
         drop_from_path "$LIBPATH" "${old_rootsys}/lib/root"
         LIBPATH=$newpath
      fi
      if [ -n "${PYTHONPATH-}" ]; then
         drop_from_path "$PYTHONPATH" "${old_rootsys}/lib"
         drop_from_path "$PYTHONPATH" "${old_rootsys}/lib/root"
         PYTHONPATH=$newpath
      fi
      if [ -n "${MANPATH-}" ]; then
         drop_from_path "$MANPATH" "${old_rootsys}/man"
         MANPATH=$newpath
      fi
      if [ -n "${CMAKE_PREFIX_PATH-}" ]; then
         drop_from_path "$CMAKE_PREFIX_PATH" "${old_rootsys}"
         CMAKE_PREFIX_PATH=$newpath
      fi
      if [ -n "${JUPYTER_PATH-}" ]; then
         drop_from_path "$JUPYTER_PATH" "${old_rootsys}/etc/notebook"
         JUPYTER_PATH=$newpath
      fi
      if [ -n "${JUPYTER_CONFIG_PATH-}" ]; then
         drop_from_path "$JUPYTER_CONFIG_PATH" "${old_rootsys}/etc/notebook"
         JUPYTER_CONFIG_PATH=$newpath
      fi
   fi
   if [ -z "${MANPATH-}" ]; then
      # Grab the default man path before setting the path to avoid duplicates
      if command -v manpath >/dev/null; then
         default_manpath="$(manpath)"
      elif command -v man >/dev/null; then
         default_manpath="$(man -w 2> /dev/null)"
      else
         default_manpath=""
      fi
   fi
}

set_environment()
{
   if [ -z "${PATH-}" ]; then
      PATH=@bindir@; export PATH
   else
      PATH=@bindir@:$PATH; export PATH
   fi

   if [ -z "${LD_LIBRARY_PATH-}" ]; then
      LD_LIBRARY_PATH=@libdir@
      export LD_LIBRARY_PATH       # Linux, ELF HP-UX
   else
      LD_LIBRARY_PATH=@libdir@:$LD_LIBRARY_PATH
      export LD_LIBRARY_PATH
   fi

   if [ -z "${DYLD_LIBRARY_PATH-}" ]; then
      DYLD_LIBRARY_PATH=@libdir@
      export DYLD_LIBRARY_PATH       # Linux, ELF HP-UX
   else
      DYLD_LIBRARY_PATH=@libdir@:$DYLD_LIBRARY_PATH
      export DYLD_LIBRARY_PATH
   fi

   if [ -z "${SHLIB_PATH-}" ]; then
      SHLIB_PATH=@libdir@
      export SHLIB_PATH       # Linux, ELF HP-UX
   else
      SHLIB_PATH=@libdir@:$SHLIB_PATH
      export SHLIB_PATH
   fi

   if [ -z "${LIBPATH-}" ]; then
      LIBPATH=@libdir@
      export LIBPATH       # Linux, ELF HP-UX
   else
      LIBPATH=@libdir@:$LIBPATH
      export LIBPATH
   fi

   if [ -z "${PYTHONPATH-}" ]; then
      PYTHONPATH=@libdir@
      export PYTHONPATH       # Linux, ELF HP-UX
   else
      PYTHONPATH=@libdir@:$PYTHONPATH
      export PYTHONPATH
   fi

   if [ -z "${MANPATH-}" ]; then
      MANPATH=@mandir@:${default_manpath}; export MANPATH
   else
      MANPATH=@mandir@:$MANPATH; export MANPATH
   fi

   if [ -z "${CMAKE_PREFIX_PATH-}" ]; then
      CMAKE_PREFIX_PATH=$ROOTSYS; export CMAKE_PREFIX_PATH       # Linux, ELF HP-UX
   else
      CMAKE_PREFIX_PATH=$ROOTSYS:$CMAKE_PREFIX_PATH; export CMAKE_PREFIX_PATH
   fi

   if [ -z "${JUPYTER_PATH-}" ]; then
      JUPYTER_PATH=$ROOTSYS/etc/notebook; export JUPYTER_PATH       # Linux, ELF HP-UX
   else
      JUPYTER_PATH=$ROOTSYS/etc/notebook:$JUPYTER_PATH; export JUPYTER_PATH
   fi

   if [ -z "${JUPYTER_CONFIG_PATH-}" ]; then
      JUPYTER_CONFIG_PATH=$ROOTSYS/etc/notebook; export JUPYTER_CONFIG_PATH # Linux, ELF HP-UX
   else
      JUPYTER_CONFIG_PATH=$ROOTSYS/etc/notebook:$JUPYTER_CONFIG_PATH; export JUPYTER_CONFIG_PATH
   fi
}

getTrueShellExeName() { # mklement0 https://stackoverflow.com/a/23011530/7471760
    # shellcheck disable=SC3043 # assume that local is available
   local trueExe nextTarget 2>/dev/null # ignore error in shells without `local`
   # Determine the shell executable filename.
   if [ -r "/proc/$$/cmdline" ]; then
      trueExe=$(cut -d '' -f1 /proc/$$/cmdline 2>/dev/null) || trueExe=$(xargs -0 -n 1 < /proc/$$/cmdline | head -n 1) || return 1
      # Qemu emulation has cmdline start with the emulator
      if [ "${trueExe##*qemu*}" != "${trueExe}" ]; then
         # but qemu sets comm to the emulated command
         trueExe=$(cat /proc/$$/comm) || return 1
      fi
   else
      trueExe=$(ps -p $$ -o comm=) || return 1
   fi
   # Strip a leading "-", as added e.g. by macOS for login shells.
   if [ "${trueExe#-}" != "$trueExe" ]; then
      trueExe=${trueExe#-}
   fi
   # Determine full executable path.
   if [ "${trueExe#/}" = "$trueExe" ]; then
      trueExe=$(if [ -n "${ZSH_VERSION-}" ]; then builtin which -p "$trueExe"; else which "$trueExe"; fi)
   fi
   # If the executable is a symlink, resolve it to its *ultimate*
   # target.
   while nextTarget=$(readlink "$trueExe"); do trueExe=$nextTarget; done
   # Output the executable name only.
   printf '%s' "$(basename "$trueExe")"
}

### main ###


if [ -n "${ROOTSYS-}" ] ; then
   old_rootsys=${ROOTSYS}
fi


SHELLNAME=$(getTrueShellExeName)
if [ "$SHELLNAME" = "bash" ]; then
   # shellcheck disable=SC3028,SC3054 # Bash-only syntax
   SOURCE=${BASH_ARGV[0]}
elif [ -z "${SHELLNAME}" ]; then # in case getTrueShellExeName does not work, fall back to default
   echo "WARNING: shell name was not found. Assuming 'bash'."
   # shellcheck disable=SC3028,SC3054 # Bash-only syntax
   SOURCE=${BASH_ARGV[0]}
elif [ "$SHELLNAME" = "zsh" ]; then
   # shellcheck disable=all
   SOURCE=${(%):-%N}
else # dash or ksh
   x=$(lsof -p $$ -Fn0 2>/dev/null | tail -1); # Paul Brannan https://stackoverflow.com/a/42815001/7471760
   SOURCE=${x#*n}
fi


if [ -z "${SOURCE}" ]; then
   if [ -f bin/thisroot.sh ]; then
      ROOTSYS="$PWD"; export ROOTSYS
   elif [ -f ./thisroot.sh ]; then
      ROOTSYS=$(cd .. > /dev/null && pwd); export ROOTSYS
      if [ -z "$ROOTSYS" ]; then
         echo "ERROR: \"cd ..\" or \"pwd\" failed" >&2
         return 1
      fi
   else
      if [ "$SHELLNAME" = "bash" ] ; then
         echo "ERROR: please turn on extdebug using \"shopt -s extdebug\"" >&2
         echo "or \"cd where/root/is\" before calling \". bin/thisroot.sh\"" >&2
      else
         echo "ERROR: must \"cd where/root/is\" before calling \". bin/thisroot.sh\" for this version of \"$SHELLNAME\"!" >&2
      fi
      ROOTSYS=; export ROOTSYS
      return 1
   fi
else
   # get param to "."
   thisroot="$(dirname "${SOURCE}")"
   ROOTSYS=$(cd "${thisroot}/.." > /dev/null && pwd); export ROOTSYS
   if [ -z "$ROOTSYS" ]; then
      echo "ERROR: \"cd ${thisroot}/..\" or \"pwd\" failed" >&2
      return 1
   fi
fi


clean_environment
set_environment


if (root-config --arch | grep -v win32gcc | grep -q -i win32); then
   ROOTSYS="$(cygpath -w "$ROOTSYS")"
fi

unset old_rootsys
unset thisroot
unset -f drop_from_path
unset -f clean_environment
unset -f set_environment
