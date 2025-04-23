import os, sys, re

if len(sys.argv) < 2 :
  print 'Usage: ', __file__, ' <logfile>'
  exit()

logfile = sys.argv[1]

currdir = '.'
for line in file(logfile).readlines() :
  matchobj = re.match( r'Running test in (.*)$', line, re.M|re.I)
  if matchobj :
     currdir = matchobj.group(1)
     continue

  matchobj = re.match( r'root.exe (.*)', line, re.M|re.I)
  if matchobj :
     command = matchobj.group(1)
     quote = False
     for arg in command.split():
        if quote and (arg[-1] == '"' or arg[-1] == "'"):
           quote = False
           continue
        elif quote:
           continue
        elif arg[0] == '"' or arg[0] == "'":
           quote = True
           continue
        elif arg == '-l' or arg == '-q' or arg == '-b':
           continue
        elif arg == '-e' :
           skipnext = True
           continue
        else :
           macro = arg
           if macro == '-' : print command
           break
     # Distinguish the the different commands
     matchobj = re.match( r'.*scripts/build[.]C(.*)', macro, re.M|re.I)
     if matchobj:
        print 'BUILD %s/%s' %(currdir, matchobj.group(1)[4:-4])
        continue
     matchobj = re.match( r'([^+]*)', macro, re.M|re.I)
     if matchobj :
        print 'RUN   %s/%s' %(currdir, matchobj.group(1))
     else :
        print 'ERROR   %s/%s' %(currdir, macro)





