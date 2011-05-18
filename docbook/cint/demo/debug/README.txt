File: $CINTSYSDIR/demo/debug/README

Tutorial programs to teach you how to use cint debugger interface.
Just execute the source code and follow the instruction.

	$ cint debug.c


For another simple tutorial(debug2.cxx/debug2.com), there are 3 ways to run it.
    
     1.
        $ cint -X debug2.com               : run a command dump file debug.com
        cint>  q                           : quit cint
    
     2.
        $ cint
        cint>  < debug2.com                : run a command dump file debug.com
        cint>  q                           : quit cint
    
     3.
        $ cint
        cint>  L debug2.cxx                : load a source file debug2.cxx
        cint>  p test()                    : run through test() 
        cint>  s test()                    : run and step into test() 
        cint>  S                           : step over a line
        cint>  S                           : step over a line
        cint>  s                           : step into subroutine
        cint>  p x                         : verify value of x
        cint>  S                           : step over a line
        cint>  p *this                     : verify value of *this
        cint>  S                           : step over a line
        cint>  S                           : step over a line
        cint>  S                           : step over a line
        cint>  U debug2.cxx                : unload the source file
        cint>  q                           : quit cint
