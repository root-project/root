#!/usr/bin/python

import urllib.request
import sys

# Character positions for extracting data from "mass_width_2023.txt"
MASS_WIDTH_ID_START = 0
MASS_WIDTH_ID_END = 8
MASS_WIDTH_MASS_START = 33
MASS_WIDTH_MASS_END = 51
MASS_WIDTH_WIDTH_START = 70
MASS_WIDTH_WIDTH_END = 88
MASS_WIDTH_NAME_WITH_CHARGE_START = 107
MASS_WIDTH_NAME_WITH_CHARGE_END = 128

#------------------------
# Gets the text document
#------------------------

def getFileTextFromURL(link):
    try:
        with urllib.request.urlopen(link) as f:
            text = f.read().decode("utf-8")
            lines = text.splitlines()
            return lines
    except Exception as e:
        print(f"Error while fetching data from URL: {e}")
        return None

#-------------------------------------
# Gets the correct Data for particles
#-------------------------------------

def fixedParticleData(infoLine):
    elements = infoLine.split()
    
    name = elements[1]
    trueId = int(elements[2])

    if len(elements) < 6:  # If it's an Antiparticle
        return f"{elements[0]:>5} {name:<16} {trueId:>6} {elements[3]:>7} {elements[4]:>5}\n"
    else:  # If it's not an Antiparticle
        mass = float(elements[7])
        width = float(elements[8])
            
        if 0 <= trueId < len(massWidthData):
            if (massWidthData[trueId] is not None):
                if (massWidthData[trueId].mass is not None):
                    mass = float(massWidthData[trueId].mass)
                if massWidthData[trueId].width is not None:
                    width = float(massWidthData[trueId].width)
                print(f"{float(elements[7]):<10e}  {float(massWidthData[trueId].mass):<10e}  {massWidthData[trueId].name:<10}")

        # Formatting the output line with the updated mass and width
        return f"{elements[0]:>5} {name:<16} {trueId:>6} {elements[3]:>2} {elements[4]:>3} {elements[5]:<10} {elements[6]:>2} {mass:<11e} {width:<11e} {elements[9]:>3} {elements[10]:>2} {elements[11]:>3} {elements[12]:>2} {elements[13]:>4} {elements[14]:>3}\n"

#---------------------------------
# Save the data of the mass_width
#---------------------------------

def getMassWidthValues(lines):
    data = []  # Array to store the data
    for line in lines:
        if line.startswith('*'):
            continue  # Skip comments

        # Extracting relevant information based on the character column positions
        particleId = int(line[MASS_WIDTH_ID_START:MASS_WIDTH_ID_END].strip())

        if line[MASS_WIDTH_MASS_START:MASS_WIDTH_MASS_END].strip():
            mass = line[MASS_WIDTH_MASS_START:MASS_WIDTH_MASS_END].strip()
        else:
            mass = 0.0  # No value is given
        
        if line[MASS_WIDTH_WIDTH_START:MASS_WIDTH_WIDTH_END].strip():
            width = line[MASS_WIDTH_WIDTH_START:MASS_WIDTH_WIDTH_END].strip()
        else:
            width = 0.0  # No value is given

        nameWithCharge = line[MASS_WIDTH_NAME_WITH_CHARGE_START:MASS_WIDTH_NAME_WITH_CHARGE_END].strip()
        name = nameWithCharge.split()[0]  # Extract only the name, excluding the charge

        # Storing the data in the dictionary
        if particleId >= len(data):
            while (len(data) <= particleId):
                data.append(None)
        data[particleId] = MWDataObj(float(mass), float(width), name)
        
    return data

#------------------------------
# Create updated pdg_table.txt
#------------------------------

def updateTable(massWidthData, originalTable):
    if originalTable is None or massWidthData is None:
        print("Data not available. Aborting table creation.")
        return

    outputLines = []

    for line in originalTable:
        if line.strip().startswith('#'):
            outputLines.append(f"{line}")
        else:
            if line.split()[0].isdigit() and line.split()[1].isdigit():
                outputLines.append(f"{line}")
            else:
                outputLines.append(fixedParticleData(line))

    return outputLines

#---------------------------------------
# Object for the data of the mass_width
#---------------------------------------

class MWDataObj:
    def __init__(self, mass, width, name):
        self.mass = mass
        self.width = width
        self.name = name

#------------------
# Setup everything
#------------------

if len(sys.argv)!= 3:
    print(f"Incorrect Arguments!\nUsage: {sys.argv[0]} pdgFile massWidthURL")
    exit(1)

pdgTableFile = sys.argv[1]
massUrl = sys.argv[2]

# Fetch mass_width_2023.txt content from URLs
massDataLines = getFileTextFromURL(massUrl)
if massDataLines is None:
    exit(2)
massWidthData = getMassWidthValues(massDataLines)

with open(pdgTableFile, 'r') as file:
    originalTable = file.readlines()
newTable = updateTable(massWidthData, originalTable)
with open(pdgTableFile, 'w') as file:
    file.writelines(newTable)
