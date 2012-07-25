#ifndef GUIBUILDER_INCLUDED
#define GUIBUILDER_INCLUDED

#include <vector>
#include <string>
#include <iosfwd>

namespace ROOT {
namespace CocoaTest {

class TestFrame;

class GuiBuilder 
{
private:
   struct WindowGeometry 
   {
      int x;
      int y;
      unsigned width;
      unsigned height;
   };

public:
   GuiBuilder();
   
   std::vector<TestFrame *> BuildGUI(std::ifstream & inputFile);

private:
   std::vector<TestFrame *> topLevelWindows_;
   std::vector<std::string> fileData_;
   std::vector<std::string>::iterator currentLine_;

   void BuildTopLevelWindow();
   TestFrame * BuildWindow(TestFrame * parentFrame);
   WindowGeometry ParseGeometry();   
   unsigned ParseInputMask();
   unsigned ParseColor();
   void BuildChildren(TestFrame * parentFrame);
   
   bool Eof()const;
   void ParseValue(const std::string & line, int & value);
   void ParseKeyValue(const std::string & keyValue, const std::string & key, int & value);
   void ParseKeyValueHex(const std::string & keyValue, const std::string & key, int &value);
   void Error(const std::string & where, const std::string & what)const;
   
   GuiBuilder(const GuiBuilder & rhs) = delete;
   GuiBuilder & operator = (const GuiBuilder & rhs) = delete;
   GuiBuilder(GuiBuilder && rhs) = delete;
   GuiBuilder & operator = (GuiBuilder && rhs) = delete;
};

}
}

#endif
