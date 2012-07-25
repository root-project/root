#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <memory>
#include <string>

#include "TVirtualX.h"

#include "guibuilder.h"
#include "testframe.h"

namespace ROOT 
{
namespace CocoaTest 
{

namespace 
{

//______________________________________________________________________
void strip_line(std::string &line)
{
   std::string::size_type firstNonWS = line.find_first_not_of(" \t");
   if(firstNonWS != std::string::npos) 
   {
      std::string::size_type lastNonWS = line.find_last_not_of(" \t");
      if(lastNonWS >= firstNonWS)
         line = line.substr(firstNonWS, lastNonWS - firstNonWS + 1);
      else
         line = "";
   }
   else
      line = "";
}

//______________________________________________________________________
unsigned hex_to_decimal(char hexDigit)
{
   if(hexDigit >= '0' && hexDigit <= '9')
      return hexDigit - '0';
   
   switch (hexDigit) {
   case 'a':
      return 10;
   case 'b':
      return 11;
   case 'c':
      return 12;
   case 'd':
      return 13;
   case 'e':
      return 14;
   case 'f':
      return 15;
   }
   
   return 0;
}

}

//______________________________________________________________________
GuiBuilder::GuiBuilder()
{
}

//______________________________________________________________________
std::vector<TestFrame *> GuiBuilder::BuildGUI(std::ifstream & inputFile)
{
   if(!inputFile)
      Error("BuildWindows", "end of file while reading window declaration");

   fileData_.clear();

   std::string line(100, ' ');
   while (std::getline(inputFile, line))
      fileData_.push_back(line);

   if(!fileData_.size())
      Error("BuildWindows", "empty input file");

   topLevelWindows_.clear();//I do not care about memory management here.
   currentLine_ = fileData_.begin();

   try 
   {
      while(currentLine_ != fileData_.end())
         BuildTopLevelWindow();
   }
   catch (const std::exception & e)
   {
      std::cout<<e.what()<<std::endl;
      for(auto ptr : topLevelWindows_)
         ;//delete ptr;
      
      throw;
   }
   
   return topLevelWindows_;
}

//______________________________________________________________________
void GuiBuilder::BuildTopLevelWindow()
{
   std::auto_ptr<TestFrame> newTopLevel(BuildWindow(0));
   topLevelWindows_.push_back(newTopLevel.get());
   newTopLevel.release();
}

//______________________________________________________________________
TestFrame * GuiBuilder::BuildWindow(TestFrame * parentFrame)
{
   //window_declaration: (note, order of sub-declarations is fixed).

   //+window
   //window_geometry_declaration (mandatory)
   //color (optional)
   //input (optional)
   //grabs (optional)
   //children (optional) -> (window_declaration)+
   //-window
   
   assert(!Eof() && "BuildWindow, eof");

   std::string line(*currentLine_);
   strip_line(line);
   if(line != "+window")
      Error("BuildWindow", "'+window' tag expected");
   ++currentLine_;
   
   if(Eof())
      Error("BuildWindow", "end of file while parsing window declaration");

   const WindowGeometry newGeom = ParseGeometry();
   
   if(Eof())
      Error("BuildWindow", "end of file while '-window' tag expected");
   
   unsigned backgroundColor = 0;
   if(currentLine_->find("color") != std::string::npos)
   {
      backgroundColor = ParseColor();
      if(Eof())
         Error("BuildWindow", "end of file while '-window' tag expected");
   }
   
//   std::cout<<"background color is "<<backgroundColor<<std::endl;

   unsigned inputMask = 0;
   if(currentLine_->find("+input") != std::string::npos)
   {
      inputMask = ParseInputMask();
      if(Eof())
         Error("BuildWindow", "end of file while '-window' tag expected'");
   }
   
   //1. grabs here.
   //2. create a new window now.
   
   if(!backgroundColor)
      backgroundColor = 0xaeaeae;
   
   std::auto_ptr<TestFrame> newFrame(new TestFrame(parentFrame, newGeom.width, newGeom.height, parentFrame ? kChildFrame : kMainFrame, backgroundColor));

   gVirtualX->MoveWindow(newFrame->GetId(), newGeom.x, newGeom.y);

   if(inputMask)
   newFrame->AddInput(inputMask);
      
   if(currentLine_->find("+children") != std::string::npos)
   {
      BuildChildren(newFrame.get());
      if (Eof())
         Error("BuildWindow", "end of file while '-window' tag expected");
      
      gVirtualX->MapSubwindows(newFrame->GetId());
   }
   
   line = *currentLine_;
   strip_line(line);
   if(line != "-window")
      Error("BuildWindow", "'-window' tag expected");
      
   ++currentLine_;

   gVirtualX->MapRaised(newFrame->GetId());

   //we have children now, call map-subwindows.
   //map window itself.
   
   return newFrame.release();
}

//______________________________________________________________________
GuiBuilder::WindowGeometry GuiBuilder::ParseGeometry()
{
   //window_geometry_declaration (all fields are mandatory):

   //+geometry
   //  x : non-negative_integer
   //  y : non-negative_integer
   //  width : positive_integer
   //  height : positive_integer
   //-geometry

   //Opening tag.
   assert(!Eof() && "ReadGeometry, eof");
      
   std::string line(*currentLine_);
   strip_line(line);
   if(line != "+geometry")
      Error("ReadGeometry", "'+geometry' tag expected");
   ++currentLine_;
   
   //X field.
   if(Eof())
      Error("ReadGeometry", "end of file while 'x' field expected");
   line = *currentLine_;
   strip_line(line);
   int x = 0;
   ParseKeyValue(line, "x", x);
   if(x < 0)
      Error("ReadGeometry", "x is negative");
   ++currentLine_;
   
   //Y field.
   if(Eof())
      Error("ReadGeometry", "end of file while 'y' field expected");
   line = *currentLine_;
   strip_line(line);
   int y = 0;
   ParseKeyValue(line, "y", y);
   if (y < 0)
      Error("ReadGeometry", "y values is negative");
   ++currentLine_;
   
   //Width field.
   if(Eof())
      Error("ReadGeometry", "end of file while 'width' field expected");
   line = *currentLine_;
   strip_line(line);
   int width = 0;
   ParseKeyValue(line, "width", width);
   if(width <= 0)
      Error("ReadGeometry", "width must be a positive integer");
   ++currentLine_;
   
   //Height field.
   if(Eof())
      Error("ReadGeometry", "end of file while 'height' field expected");
   line = *currentLine_;
   strip_line(line);
   int height = 0;
   ParseKeyValue(line, "height", height);
   if(height <= 0)
      Error("ReadGeometry", "height must be a positive integer");
   ++currentLine_;
   
   //Closing tag.
   if(Eof())
      Error("ReadGeometry", "end of file while '-geometry' tag expected");
   ++currentLine_;
   
   WindowGeometry geom = {x, y, (unsigned)width, (unsigned)height};
   return geom;
}

//______________________________________________________________________
unsigned GuiBuilder::ParseInputMask()
{
   assert(!Eof() && "ParseInputMask, eof");
   
   std::string line(*currentLine_);
   strip_line(line);
   
   if(line != "+input")
      Error("ParseInputMask", "'+input' tag expected");
   ++currentLine_;
      
   unsigned inputMask = 0;

   while(true)
   {
      int val = 0;
      if(Eof())
         Error("ParseInputMask", "end of file while parsing 'input' section");
      line = *currentLine_;
      strip_line(line);
      if(line == "-input")
      {
         ++currentLine_;
         break;
      }
      
      ParseValue(line, val);

      if(val <= 0)
         Error("ParseInputMask", "illegal input bit");
      inputMask |= val;
      ++currentLine_;
   }
   
   if(!inputMask)
      Error("ParseInputMask", "Illegal input mask calculated");
   
   return inputMask;
}

//______________________________________________________________________
unsigned GuiBuilder::ParseColor()
{
   assert(!Eof() && "ParseColor, eof");
   
   std::string line(*currentLine_);
   strip_line(line);
   int backgroundColor = 0;
   ParseKeyValueHex(line, "color", backgroundColor);
   if(backgroundColor < 0)
      Error("ParseColor", "negative integer for a background color specified");
   ++currentLine_;

   return backgroundColor;
}

//______________________________________________________________________
void GuiBuilder::BuildChildren(TestFrame * parentFrame)
{
  // assert(parentFrame && "BuildChildren, parentFrame parameter is null");
   assert(!Eof() && "BuildChildren, eof");
   
   std::string line(*currentLine_);
   strip_line(line);
   
   if(line != "+children")
      Error("BuildChildren", "'+children' tag expected");

   ++currentLine_;
   if(Eof())
      Error("BuildChildren", "end of file while parsing children declarations");
   
   unsigned nChildren = 0;

   while(true)
   {
      std::string line(*currentLine_);
      strip_line(line);
      if(line == "-children")
      {
         ++currentLine_;
         break;
      }

      BuildWindow(parentFrame);
      if(Eof())
         Error("BuildChildren", "end of file, while '-children' tag expected");
   }
}

//______________________________________________________________________
bool GuiBuilder::Eof()const
{
   return currentLine_ == fileData_.end();
}

//______________________________________________________________________
void GuiBuilder::ParseValue(const std::string & line, int & value)
{
   //No checks here.
   std::istringstream stream(line);
   stream >> value;
}

//______________________________________________________________________
void GuiBuilder::ParseKeyValue(const std::string & keyValue, const std::string & key, int &value)
{
   auto pos = keyValue.find(':');
   if(pos == std::string::npos)
      Error("ParseKeyValue", "bad key : value pair found");
   
   auto testKey = keyValue.substr(0, pos);
   strip_line(testKey);
   if (testKey != key)
      Error("ParseKeyValue", "key " + key + " is not found in a string " + keyValue);
   
   ParseValue(keyValue.substr(pos + 1), value);//still can have something bad inside, but I do not check.
}

//______________________________________________________________________
void GuiBuilder::ParseKeyValueHex(const std::string & keyValue, const std::string & key, int &value)
{
   auto pos = keyValue.find(':');
   if(pos == std::string::npos)
      Error("ParseKeyValue", "bad key : value pair found");
   
   auto testKey = keyValue.substr(0, pos);
   strip_line(testKey);
   if (testKey != key)
      Error("ParseKeyValue", "key " + key + " is not found in a string " + keyValue);
   
   std::string strValue(keyValue.substr(pos + 1));
   strip_line(strValue);
   
   if(strValue.find_first_not_of("0123456789abcdef") != std::string::npos)
      Error("ParseKeyValueHex", "only hex-digits are expected");
   
   if(strValue.length() > 6)//I'm not goint to use something bigger than 24 bit colors.
      Error("ParseKeyValueHex", "hex digit is too big");
   
   unsigned res = 0;
   for(std::string::size_type i = 0, e = strValue.length(); i < e; ++i)
   {
      unsigned digit = hex_to_decimal(strValue[i]);
      res |= digit << (4 * (e - 1 - i));
   }
   
   value = (int)res;
}


//______________________________________________________________________
void GuiBuilder::Error(const std::string & where, const std::string & what)const
{
   std::ostringstream stream;
   stream<<"Error: GuiBuilder::"<<where<<" line "<<(currentLine_ - fileData_.begin()) + 1<<" -> "<<what<<std::endl;
   throw std::runtime_error(stream.str());
}


}
}
