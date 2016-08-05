#!/usr/bin/env python
# Author: Pau Miquel Mir
# Date: July, 2016
#
# The converttonotebooks script creates Jupyter notebooks from raw C++ or python files.
# Particulary, it is indicated to convert the ROOT tutorials found in the ROOT
# repository.
#
# The script should be called from bash with the following format:
#      python /path/to/script/converttonotebook.py /path/to/<macro>.C /path/to/outdir
#
# Indeed the script takes two arguments, the path to the macro and the path to the directory
# where the notebooks will be created
#
# For converting python tutorials it is fairly straightforward. It extracts the decription and
# author information from the header and then removes it. It also converts any comment at the
# beginning of a line into a Markdown cell.
#
# For C++ files the process is slightly more complex. The script separates the functions from the
# main code. The main function is identified as it has the smae name as the macro file. The other
# functions are considered functions. The main function is "extracted"and presented as main code.
# The helper functions are placed in their own code cell with the %%cpp -d magic to enable function
# defintion. Finally, as with Python macros, relevant informatin is extracted from the header, and
# newline comments are converted into Markdown cells (unless they are in helper functions).
#
# The script creates an .ipynb version of the macro,  with the full output included. 
# The files are named:
#     <macro>.C.nbconvert.ipynb
#
# It is called by filter.cxx, which in turn is called by doxygen when processing any file
# in the ROOT repository. filter.cxx only calls convertonotebook.py when the string \notebook
# is found in the header of the turorial, but this script checks for its presence as well.


import re
import json
import sys
import textwrap
import time
import subprocess
import os
from nbformat import v3, v4
from datetime import datetime, date

starttime = time.time()


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
            newstring+= (line[3:] + "\n")
        else:
            newstring+= (line + "\n")

    return newstring

def pythonheader(text):
    """
    Extract author and descrition from header, eliminate header from text
    """
    lines = text.splitlines()

    description=''
    author=''
    notebook=False
    jsroot=False
    nodraw=False
    for i, line in enumerate(lines):
        if line.startswith("## \\aut"):
            author = line[11:]
        elif line.startswith("## \\note"):
            notebook=True
            if "-js" in line:
                jsroot = True
            if "-nodraw" in line:
                nodrwa = True
        elif line.startswith("##"):
            if not line.startswith("## \\") and not line == "##":
                description+=(line[3:]+ '\n')
        else:
            break
    newtext=''
    for j in lines[i:]:
        newtext += (j+"\n")

    return newtext, description, author, notebook, jsroot, nodraw

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
    Extract author and description from header, eliminate header from text. Also returns notebook boolean,
    which is True if the string \notebook is present in the header

    """
    lines = text.splitlines()

    description=''
    author=''
    notebook=False
    jsroot = False
    nodraw = False
    for i, line in enumerate(lines):
        if line.startswith("/// \\aut"):
            author = line[12:]
        if line.startswith("/// \\note"):
            notebook=True
            if "-js" in line:
                jsroot = True
            if "-nodraw" in line:
                nodraw = True
        if line.startswith("///"):
            if not line.startswith("/// \\") and not line == "///":
                description+=(line[4:]+ '\n')
        else:
            break
    newtext=''
    for j in lines[i:]:
        newtext += (j+"\n")

    return newtext, description, author, notebook, jsroot, nodraw

def cppfunction(text):
    """
    Extracts main function for the function enclosure by means of regular expression
    """
    p = re.compile(r'(?<=\{).*(?=^\})',flags = re.DOTALL | re.MULTILINE)

    match = p.search(text)

    if match:
        return match.group()
    else:
        return text


def cppcomments (text):
    """
    Converts comments delimited by // and on a new line into a markdown cell. For C++ files only.
    """
    text=text.splitlines()
    newtext=''
    incomment=False
    
    for i, line in enumerate(text):
        if line.startswith("//") and not incomment: ## True if first line of comment
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

    return newtext


def split(text):
    """
    Splits the text string into main, helpers, and rest. main is the main function, i.e. the function that
    has the same name as the macro file. Helpers is a list of strings, each a helper function, i.e. any
    other function that is not the main funciton. Finally, rest is a string containing any top-level
    code outside of any function. Intended for C++ files only. 
    """
    #p = re.compile(r'^void\s\w*?\s?\([\w\n=,*\_ ]*\)\s*\{.*?^\}|^int\s\w*?\s?\([\w\n=,*\_ ]*\)\s*\{.*?^\}|^string\s\w*?\s?\([\w\n=,*\_ ]*\)\s*\{.*?^\}|^double\s\w*?\s?\([\w\n=,*\_ ]*\)\s*\{.*?^\}|^float\s\w*?\s?\([\w\n=,*\_ ]*\)\s*\{.*?^\}|^char\s\w*?\s?\([\w\n=,*\_ ]*\)\s*\{.*?^\}|^TCanvas\s\*\w*\(.*?\).*?\{.*?^\}|^TString\s\w*?\s?\([\w\n=,*\_ ]*\)\s*\{.*?^\}|^Double_t\s\w*?\s?\([\w\n=,*\_ ]*\)\s*\{.*?^\}', flags = re.DOTALL | re.MULTILINE)

    p = re.compile(r'(^void|^int|^Int_t|^TF1|^string|^bool|^double|^float|^char|^TCanvas|^TString|^TSeqCollection|^Double_t|^TFile)\s?\*?\s?\w*?\s?\([^\)]*\)\s*\{.*?^\}', flags = re.DOTALL | re.MULTILINE)
    matches = p.finditer(text)
    helpers=[]
    main = ""
    for match in matches:

        if name in match.group()[:match.group().find("\n")]: #if name is in the first line
            main = match.group()
        else:
            helpers.append(match.group())

    ## Create rest by replacing the main and helper funcitons with blank strings
    rest = text.replace(main, "")
 
    for helper in helpers:
        rest = rest.replace(helper, "")

    newhelpers=[]
    lines=text.splitlines()
    for helper in helpers:
        for i, line in enumerate(lines):
            if line.startswith(helper[:helper.find("\n")]):
                if lines[i-1].startswith("//"):
                    helperdescription = lines[i-1][2:]
                    rest = rest.replace(helperdescription, "")
                    break
                else:
                    helperdescription = "A helper funciton is created:"
                    break
                
               
        if "main" not in helper[:helper.find("\n")]: #remove void main function
            newhelpers.append("\n# <markdowncell>\n " + helperdescription + " \n# <codecell>\n%%cpp -d\n" + helper )
    

    rest = rest.rstrip() #remove newlines at the end of string
    return main, newhelpers, rest

def processmain(text):
    """
    Evaluates whether the main function returns a TCanvas or requires input. If it does then the keepfunction
    flag is True, meaning the function wont be extracted by cppfunction. If the initial condition is true then
    an extra cell is added before at the end that calls the main function is returned, and added later.
    """
    addition = ''
    keepfunction = False
    
    regex = re.compile(r'(?<=\().*?(?=\))',flags = re.DOTALL | re.MULTILINE)
    arguments = regex.search(text)
    if text: 
        if text.startswith("TCanvas") or len(arguments.group())>3: 
            keepfunction = True
            p = re.compile(r'(?<=(?<=int)|(?<=void)|(?<=TF1)|(?<=Int_t)|(?<=string)|(?<=double)|(?<=float)|(?<=char)|(?<=TString)|(?<=bool)|(?<=TSeqCollection)|(?<=TCanvas)|(?<=TFile))\s?\*?\s?[^\s]*?(?=\s?\()',flags = re.DOTALL | re.MULTILINE)

            match = p.search(text)
            functionname=match.group().strip(" *")
            addition = "\n# <markdowncell> \n# Call the main function \n# <codecell>\n%s();" %functionname
        
    return text, addition, keepfunction 

def getfile(text):
    getfilecell = ""
    if "TString dir = gSystem->UnixPathName(__FILE__);" in text:
        text = text.replace("TString dir = gSystem->UnixPathName(__FILE__);", "")
        getfilecell = "# <codecell>\n!wget -q http://root.cern.ch/files/%s.dat \nTString dir = \"%s.dat\"; " % (name, path)
        return text, getfilecell
    if "TString dat = gSystem->UnixPathName(__FILE__);" in text:
        text = text.replace("TString dat = gSystem->UnixPathName(__FILE__);", "")

        getfilecell = "# <codecell>\n!wget -q http://root.cern.ch/files/%s.dat \nTString dat = \"%s.dat\"; " % (name, name)
        return text, getfilecell
    else:
        return text, getfilecell
#-------------------------------------
#----- Preliminary definitions--------
#-------------------------------------


## Extract and define the name of the file as well as its derived names
pathname = str(sys.argv[1])
pathnoext = os.path.dirname(pathname)
filename = os.path.basename(pathname)
path = pathname.replace(filename, "")
name,extension = filename.split(".")
outname= filename + ".ipynb"
outnameconverted= filename + ".nbconvert.ipynb"


#print pathname, "**" , filename,"**" ,  name, "**" , extension,"**" ,  outname , pathname.replace(filename, "")
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

## Open the file to be converted
with open(pathname) as fpin:
    text = fpin.read()

## Extract information from header and remove header from text
if extension == "py":
    text, description, author, notebook, jsroot, nodraw = pythonheader(text)

elif extension in ("C", "c", "cpp", "C++", "cxx"):
    text, description, author, notebook, jsroot, nodraw = cppheader(text)


def mainfunction(text):
    """
    Main function. Calls all other functions, depending on whether the macro input is in python or c++.
    It adds the header information. Also, it adds a cell that draws all canvases. The working text is
    then converted to a version 3 jupyter notebook, subsequently updated to a version 4. Then, metadata
    associated with the language the macro is written in is attatched to he notebook. Finally the
    notebook is executed and output as a Jupyter notebook.
    """
    ## Modify text from macros to suit a notebook
    if extension in ("C", "c", "cpp", "C++", "cxx") :
        text, getfilecell = getfile(text)
        main, helpers, rest = split(text)
        main, addition, keepfunction = processmain(main)
        if not keepfunction:
            main = cppcomments(unindenter(cppfunction(main))) # Remove function, Unindent, and convert comments to Markdown cells
        rest = cppcomments(rest) # Convert top level code comments to Markdown cells

        ## Construct text by starting wwith top level code, then the helper functions, and finally the main function.
        if getfilecell:
            text = getfilecell
            text+= rest
        else:
            text = rest
        
        for helper in helpers:
            text+=helper

        if keepfunction:
            text+="\n# <markdowncell>\n# The main function is defined\n# <codecell>\n%%cpp -d\n"
        else:
            text+="\n# <codecell>\n"
        text+=main
        if addition:
            text+=addition
    if extension == "py":
        text = pythoncomments(text) # Convert comments into Markdown cells

    ## Add the title and header of the notebook    
    text= "# <markdowncell> \n# # %s\n# %s# \n# \n# **Author:** %s  \n# <small>This notebook tutorial was automatically generated from the macro found " \
    "in the ROOT repository on %s.</small>\n# <codecell>\n%s" % (name.title(), description, author, date, text)

    ## Add cell at the end of the notebook that draws all the canveses. Add a Markdown cell before explaining it.
    if jsroot:
        if extension == ("C" or "c" or "cpp" or "c++"):
            text +="\n# <markdowncell> \n# Draw all canvases \n# <codecell>\n%jsroot\ngROOT->GetListOfCanvases()->Draw()"
        if extension == "py":
            text +="\n# <markdowncell> \n# Draw all canvases \n# <codecell>\n%jsroot\nfrom ROOT import gROOT \ngROOT.GetListOfCanvases().Draw()"
    
    elif not nodraw:
        if extension == ("C" or "c" or "cpp" or "c++"):
            text +="\n# <markdowncell> \n# Draw all canvases \n# <codecell>\ngROOT->GetListOfCanvases()->Draw()"
        if extension == "py":
            text +="\n# <markdowncell> \n# Draw all canvases \n# <codecell>\nfrom ROOT import gROOT \ngROOT.GetListOfCanvases().Draw()"
    ## Create a notebook from the working text
    nbook = v3.reads_py(text)  
    nbook = v4.upgrade(nbook)  # Upgrade v3 to v4

    ## Load notebook string into json format, essentially creating a dictionary
    json_data = json.loads(v4.writes(nbook))

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

    ## write the json file with the metadata
    with open(outdir+outname, 'w') as f:
        json.dump(json_data, f, indent=1, sort_keys=True)

    ## The two commands to create an html version of the notebook and a notebook with the output
    print time.time() - starttime
    #subprocess.call(["jupyter", "nbconvert","--ExecutePreprocessor.timeout=60", "--to=html", "--execute",  outdir+outname])
    r = subprocess.call(["jupyter", "nbconvert","--ExecutePreprocessor.timeout=90",  "--to=notebook", "--execute",  outdir+outname])
    if r != 0:
         sys.stderr.write( "WARNING: Nbconvert failed for notebook %s \n" % outname)
    if jsroot:
        subprocess.call(["jupyter", "trust",  outdir+outnameconverted])
    os.remove(outdir+outname)

## Set DYLD_LIBRARY_PATH. When run without root access or as a different user, epecially from Mac systems,
## it is possible for security reasons that the enviornment does not include this definition, so it is manually defined.
os.environ["DYLD_LIBRARY_PATH"] = os.environ["ROOTSYS"] + "/lib"

if notebook and extension in ("C", "c", "cpp", "C++", "cxx", "py"):
    mainfunction(text)
    print time.time() - starttime
else:
    pass




