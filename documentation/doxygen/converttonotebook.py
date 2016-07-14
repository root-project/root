import json
import sys
import textwrap
import time
import subprocess
import os
from nbformat import v3, v4
from datetime import datetime, date

starttime= time.time()

#-------------------------------------
#-------- Fuction definitions---------
#-------------------------------------

def unindenter(string):
    """
    Returns string with each line unindented by 3 spaces. If line isn't indented, it stays the same.
    """
    newstring=''
    lines = string.splitlines()
    for line in lines:
        if line.startswith("   "):
            print "yes"
            newstring+= (line[3:] + "\n")
        else:
            print "no"
            newstring+= (line + "\n")
        print line
    return newstring

def pythonheader(text):
    """
    Extract author and descrition from header, eliminate header from text
    """

    lines = text.splitlines()

    description=''
    author=''
    notebook=False
    for i, line in enumerate(lines):
        if line.startswith("## \\aut"):
            author = line[11:]
        if line.startswith("## \\note"):
            notebook=True
        if line.startswith("##"):
            if not line.startswith("## \\") and not line == "##":
                description+=(line[3:]+ '\n')
        else:
            break
    newtext=''
    for j in lines[i:]:
        newtext += (j+"\n")

    return newtext, description, author, notebook

def pythoncomments (text):
    """
    Converts comments delimited by # or ## and on a new line into a markdown cell. For python files only
    """
    text=text.splitlines()
    newtext=''
    incomment=False
    for i, line in enumerate(text):
        if line.startswith("#") and not incomment: ## True if first line of comment
            incomment=True
            newtext+="# <markdowncell>\n"
            newtext+=(line + "\n")
        elif incomment and not line.startswith("#"): ## True if first line after comment
            incomment=False
            newtext+="# <codecell>\n"
            newtext+=(line+"\n")
        else:
            newtext+=(line+"\n")
    return newtext


def cppheader(text):
    """
    Extract author and descrition from header, eliminate header from text
    """
    lines = text.splitlines()

    description=''
    author=''
    notebook=False
    for i, line in enumerate(lines):
        if line.startswith("/// \\aut"):
            author = line[12:]
        if line.startswith("/// \\note"):
            notebook=True
        if line.startswith("///"):
            if not line.startswith("/// \\") and not line == "///":
                description+=(line[4:]+ '\n')
        else:
            break
    newtext=''
    for j in lines[i:]:
        newtext += (j+"\n")

    return newtext, description, author, notebook

def cppfunction(text):
    """
    Extracts main function for the function enclosure. Creates a cell break above it.
    Creates new cells for all help functions, and adds the %%cpp -d magic on top of all of them
    """
    lines=text.splitlines()
    functionline=0
    mainbrace=0

    for i,line in enumerate(lines):
        if "void" in line and name in line:
            functionline=i
        if functionline and line == "}" and not mainbrace:
            mainbrace=i
   
    newtext=''

    if functionline:
        for line in lines[:functionline]:
            newtext += (line+"\n")
        newtext+="# <codecell>\n"

        unindent=''
        for line in lines[functionline+2:mainbrace]:
            unindent += (line+"\n")
        newtext+=unindenter(unindent)
        for line in lines[mainbrace+1:]:
            newtext += (line+"\n")
    else:
        for line in lines:
            newtext += (line+"\n")

    ## print text
    ## print newtext
    ## print functionline

    ## for i, line in enumerate(lines):
    ##      print str(i) + "       " +  line

    ## print "Functionline, mainbrace: ", functionline, mainbrace

    ## for i, line in enumerate(newtext.splitlines()):
    ##      print str(i) + "       " +  line


    return newtext, functionline, mainbrace-2

def cpphelpers(text, functionline, mainbrace):

    lines=text.splitlines()
    helperlines=[]
    helperbraces=[]
    inhelper=False
    for i,line in enumerate(lines):
        if "void" in line and lines[i+1] == "{":
            helperlines.append(i)
            inhelper=True
        if inhelper and line == "}":
            helperbraces.append(i)
            inhelper=False

    helpers = []
    for i , helper in enumerate(helperlines):
        for j, brace in enumerate(helperbraces):
            try:
                if helperlines[i+1] < brace:
                    break
                else:
                    helpers.append([helper,brace])
            except:
                 if helperlines[-1] < helperbraces [-1]:
                     helpers.append([helperlines[-1] , helperbraces [-1]])
                     break

    print helperlines, helperbraces
    print helpers
    for i, line in enumerate(lines):
        print str(i) + "       " +  line
    print helpers

    
    newtext=''
    counter=0
    for i , line in enumerate(lines):
        inhelper=False
        for pair in helpers:
            if pair[0] <= i <= pair[1]:
                inhelper = True
        if not inhelper and (i < functionline or i > mainbrace):
            print line
            newtext += (line+"\n")
            counter+=1  

    print "*************"
    print "^^Not helper, not main ^^"
    print "*************"
    
    for pair in helpers:
        newtext+= "# <markdowncell>\n A helper function is created: \n# <codecell>\n%%cpp -d\n"
        counter+=4
        for line in lines[pair[0]:pair[1]+1]:
            print line
            newtext += (line+"\n")
            counter+=1
        print "*************"
        print "^^helper ^^"

    print "*************"
    print " Main below"
    print "*************"
    for i , line in enumerate(lines):
        inhelper=False
        if functionline <= i <= mainbrace:
            print line
            newtext += (line+"\n")
    
    return newtext, counter

def cppcomments (text, counter):
    """
    Converts comments delimited by // and on a new line into a markdown cell. For C++ files only.
    """
    text=text.splitlines()
    newtext=''
    incomment=False
    
    for i, line in enumerate(text):
        if i <= counter:
            newtext+=(line+"\n")
        elif line.startswith("//") and not incomment: ## True if first line of comment
            incomment=True
            newtext+="# <markdowncell>\n"
            newtext+=("# " + line[2:]+"\n")
        elif incomment and not line.startswith("//"):## True if first line after comment
            incomment=False
            newtext+="# <codecell>\n"
            newtext+=(line+"\n")
        elif incomment and line.startswith("//"): ## True if in the middle of a comment block
            newtext+=("# " + line[2:]+"\n")
        else:
            newtext+=(line+"\n")

        print str(i)  + "    "+ str(i <= counter) + "    " + line

    return newtext

#-------------------------------------
#----- Preliminary definitions--------
#-------------------------------------


## Extract and define the name of the file as well as its derived names
pathname = str(sys.argv[1])
pathnoext = os.path.dirname(pathname)
filename = os.path.basename(pathname)
name,extension = filename.split(".")
outname= filename + ".ipynb"

## print pathname, filename, name, extension, outname
## Extract output directory
try:
    outdir = str(sys.argv[2])
except:
    outdir = pathnoext + "/"

## Find and define the time and date this script is run
date = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")


#-------------------------------------
#------------ Main Program------------
#-------------------------------------

with open(pathname) as fpin:
    text = fpin.read()

if extension == "py":
    ## print "Python"
    text, description, author, notebook = pythonheader(text)

elif extension ==  "c" or extension == "C":
    ## print "C++"
    ## print text

    text, description, author, notebook = cppheader(text)


def main(text):
    
    ## Convert the comments to markdown cells
    ## For C++ extract main function

    if extension ==  ("C" or "c" or "cpp" or "c++") :
        #print "C++"
        ## print text
        text, functionline, mainbrace = cppfunction(text)
        ## print text

        text, counter = cpphelpers(text, functionline, mainbrace)
        text = cppcomments(text, counter)
        ## print text

    if extension == "py":
        #print "Python"
        text = pythoncomments(text)

## Add the title and header of the notebook    
    text= "# <markdowncell> \n# # %s\n# %s# \n# This notebook tutorial was automatically generated from the macro found in the ROOT repository on %s.\n# **Author:** %s \n# <codecell>\n%s" % (name.title(), description, date, author, text)

    if extension == ("C" or "c" or "cpp" or "c++"):
        text +="\n# <markdowncell> \n# Draw all canvases \n# <codecell>\ngROOT->GetListOfCanvases()->Draw()"
    if extension == "py":
        text +="\n# <markdowncell> \n# Draw all canvases \n# <codecell>\nimport ROOT \ngROOT.GetListOfCanvases().Draw()"
     
    print text
    
    nbook = v3.reads_py(text)  
    nbook = v4.upgrade(nbook)  # Upgrade v3 to v4

    jsonnometa = v4.writes(nbook) + "\n"

    json_data = json.loads(jsonnometa)

    # add the corresponding metadata
    if extension == "py": 
        json_data[u'metadata']={
       "kernelspec": {
       "display_name": "Python 2",
       "language": "python",
       "name": "python2"
      },
      "language_info": {
       "codemirror_mode": {
       "name": "ipython",
       "version": 2
       },
       "file_extension": ".py",
       "mimetype": "text/x-python",
       "name": "python",
       "nbconvert_exporter": "python",
       "pygments_lexer": "ipython2",
       "version": "2.7.10"
      }
      }
    else:
        json_data[u'metadata']={
      "kernelspec": {
       "display_name": "ROOT C++",
       "language": "c++",
       "name": "root"
      },
      "language_info": {
       "codemirror_mode": "text/x-c++src",
       "file_extension": ".C",
       "mimetype": " text/x-c++src",
       "name": "c++"
      }
     }

        ## print json.dumps(json_data)
        ## print type(json.dumps(json_data))

    ## write the json file with the metadata
    with open(outdir+outname, 'w') as f:
        json.dump(json_data, f, indent=1, sort_keys=True)

    ## The two commands to create an html version of the notebook and a notebook with the output
    ## command1 = "jupyter nbconvert --to html --execute /Users/Pau/Desktop/" + outname
    ## command2 = "jupyter nbconvert --to notebook --execute /Users/Pau/Desktop/" + outname
    #print command1
    #print command2
    #subprocess.call(command1)
    #subprocess.call(command2)
    print time.time() - starttime
    subprocess.call(["jupyter", "nbconvert","--ExecutePreprocessor.timeout=60", "--to=html", "--execute",  outdir+outname])
    subprocess.call(["jupyter", "nbconvert","--ExecutePreprocessor.timeout=60", "--to=notebook", "--execute",  outdir+outname])
    os.remove(outdir+outname)

os.environ["DYLD_LIBRARY_PATH"] = os.environ["ROOTSYS"] + "/lib"


if notebook and extension == ("C" or "c" or "cpp" or "c++" or "py"):
    main(text)
    print time.time() - starttime
else:
    print "Nope"


