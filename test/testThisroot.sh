#!/bin/sh

set -eu

shell=$1
root_dir=$2
shell_name=${shell##*/}
unset ROOTSYS

# Variables are expanded by the child shell.
# shellcheck disable=SC2016
shell_command='
if [ -n "${BASH_VERSION-}" ]; then shopt -s extdebug; fi;
. "$1/bin/thisroot.sh" || exit 1;
test "$ROOTSYS" = "$1" || exit 1;
test "$SHELLNAME" = "$2" || exit 1;
'

single_line_command=$(printf '%s' "$shell_command" | tr '\n' ' ')

for layout in single multiline long; do
   printf 'Checking %s with %s commands\n' "$shell_name" "$layout"
   case "$layout" in
      single) command_text=$single_line_command ;;
      multiline) command_text=$shell_command ;;
      long)
         padding=$(printf '%65536s' '')
         command_text="set -o pipefail; $single_line_command #$padding"
         ;;
   esac
   "$shell" -c "$command_text" thisroot-test "$root_dir" "$shell_name"
done
