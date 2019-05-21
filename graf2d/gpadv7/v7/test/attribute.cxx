#include "gtest/gtest.h"

#include "ROOT/RAttrText.hxx"
#include "ROOT/RAttrBox.hxx"
#include "ROOT/RDrawingOptsBase.hxx"


using namespace ROOT::Experimental;

class Opts: public RDrawingOptsBase {
protected:
   Name_t GetName() const final { return "someOpts"; }

public:
   RAttrText &Text() { return Get<RAttrText>("text"); }
   RAttrBox &Box() { return Get<RAttrBox>("box"); }
};

TEST(OptsTest, AttribStrings) {
   Opts opts;

   opts.Box().Bottom().SetWidth(42.);
   opts.Text().SetSize(1.7);

   auto &holder = opts.GetHolder();

   EXPECT_FALSE(holder.AtIf("DOES_NOT_EXIST"));

   {
      EXPECT_TRUE(holder.AtIf("box"));
      auto pVal = holder.AtIf("box");
      RAttrBox *pAttrBox = dynamic_cast<RAttrBox*>(pVal);
      EXPECT_TRUE(pAttrBox);
      float val = pAttrBox->Bottom().GetWidth();
      EXPECT_FLOAT_EQ(val, 42.f);
   }

   {
      ASSERT_TRUE(holder.AtIf("text"));
      auto pVal = holder.AtIf("text");
      RAttrText *pAttrText = dynamic_cast<RAttrText*>(pVal);
      float val = pAttrText->GetSize();
      EXPECT_FLOAT_EQ(val, 1.7f);
   }
}

static bool IsFromStyle(RDrawingOptsBase &opts, const char *attrName) {
   // Value was set on this attr, not coming from style:
   auto modAttrs = opts.GetModifiedAttributeStrings();
   auto iAttr = std::find_if(modAttrs.begin(), modAttrs.end(),
      [attrName](const std::pair<std::string, std::string> &val) {
         return val.first == attrName;
      });
   return iAttr != modAttrs.end();
}

TEST(OptsTest, AttribVals) {
   Opts opts;

   auto &bottom = opts.Box().Bottom();
   opts.Box().Bottom().SetWidth(42.);

   {
      // Value was set on this attr, not coming from style:
      EXPECT_TRUE(IsFromStyle(opts, "someOpts.box.bottom.width"));

      EXPECT_FLOAT_EQ(opts.Box().Bottom().GetWidth(), 42.f);
      EXPECT_FLOAT_EQ(bottom.GetWidth(), 42.f);
   }

   {
      // Text() was not called, thus coming from style:
      EXPECT_FALSE(IsFromStyle(opts, "someOpts.text.color"));
   }

   {
      opts.Text().SetColor(RColor::kRed);
      EXPECT_EQ(opts.Text().GetColor(), RColor(RColor::kRed));

      auto modAttrs = opts.GetModifiedAttributeStrings();
      auto iAttr = std::find_if(modAttrs.begin(), modAttrs.end(),
         [](const std::pair<std::string, std::string> &val) {
            return val.first == "someOpts.text.color";
         });
      EXPECT_TRUE(iAttr != modAttrs.end());
      EXPECT_EQ(iAttr->first, "someOpts.text.color");
      EXPECT_EQ(iAttr->second, "#FF0000FF"); // red!
      EXPECT_EQ(FromAttributeString(iAttr->second, "someOpts.text.color", (RColor*)nullptr), RColor(RColor::kRed));
   }

}

TEST(OptsTest, NullAttribCompare) {
   RAttrLine al1;
   RAttrLine al2;
   EXPECT_EQ(al1, al2);
   EXPECT_EQ(al2, al1);
}

TEST(OptsTest, AttribEqual) {
   Opts opts;
   auto al1 = opts.Box().Left();
   auto al2 = opts.Box().Left();
   EXPECT_EQ(al1, al1);
   EXPECT_EQ(al1, al2);
   EXPECT_EQ(al2, al1);

   al1.SetColor(RColor::kRed);
   EXPECT_NE(al1, al2);
   EXPECT_NE(al2, al1);

   RAttrText differentLifetimeText;
   RAttrBox differentLifetimeBox;

   {
      Opts localOpts;
      localOpts.Text().SetAngle(17.);

      differentLifetimeText = localOpts.Text();
      EXPECT_EQ(differentLifetimeText, localOpts.Text());

      differentLifetimeBox = localOpts.Box();
      EXPECT_EQ(differentLifetimeBox, localOpts.Box());

      differentLifetimeBox.Bottom().SetColor(RColor::kGreen);
      EXPECT_NE(localOpts.Box(), differentLifetimeBox);
   }

   EXPECT_FLOAT_EQ(differentLifetimeText.GetAngle(), 17.);
   EXPECT_EQ(differentLifetimeBox.Bottom().GetColor(), RColor::kGreen);
}

TEST(OptsTest, AttribDiffer) {
   Opts opts1;
   Opts opts2;
   Opts opts3;
   auto al1 = opts1.Box().Left();
   auto al2 = opts2.Box().Left();
   auto al3 = opts3.Box().Left();

   al1.SetWidth(7.);
   EXPECT_NE(al1, al2);
   EXPECT_NE(al2, al1);
   EXPECT_NE(al1, al3);
   EXPECT_EQ(al2, al3);
   EXPECT_EQ(al3, al2);

   al2.SetColor(RColor::kRed);
   EXPECT_NE(al1, al2);
   EXPECT_NE(al2, al1);
   EXPECT_NE(al1, al3);
   EXPECT_NE(al2, al3);
   EXPECT_NE(al3, al2);
}


TEST(OptsTest, AttribAssign) {
   Opts opts1;
   Opts opts2;

   auto attrBox1 = opts1.Box().Border();
   auto attrBox2 = opts2.Box().Border();

   EXPECT_EQ(attrBox2, attrBox1);
   EXPECT_EQ(attrBox1, attrBox2);
   EXPECT_EQ(attrBox1, opts1.Box().Border());

   attrBox1.SetWidth(42.);
   EXPECT_NE(attrBox2, attrBox1);
   EXPECT_EQ(attrBox2, opts1.Box().Border());
   EXPECT_NE(attrBox1, opts1.Box().Border());

   attrBox2 = attrBox1;
   EXPECT_EQ(attrBox2, attrBox1);
   EXPECT_EQ(attrBox1, attrBox2);
   EXPECT_NE(attrBox1, opts1.Box().Border());
   EXPECT_NE(attrBox1, opts2.Box().Border());
   EXPECT_NE(attrBox2, opts2.Box().Border());

   // But make sure that the attributes got propagated to their options!
   opts2.Box().Border() = attrBox1;
   EXPECT_EQ(attrBox2, attrBox1);
   EXPECT_NE(opts1.Box().Border(), attrBox1);
   EXPECT_NE(opts1.Box().Border(), attrBox2);
   EXPECT_EQ(opts2.Box().Border(), attrBox1);
   EXPECT_EQ(opts2.Box().Border(), attrBox2);

   EXPECT_FLOAT_EQ(attrBox1.GetWidth(), 42.);
   EXPECT_FLOAT_EQ(attrBox2.GetWidth(), 42.);
   EXPECT_FLOAT_EQ(opts1.Box().Border().GetWidth(), 3.);
   EXPECT_FLOAT_EQ(opts2.Box().Border().GetWidth(), 42.);

   // Are the attributes disconnected from the options?
   attrBox2.SetWidth(3.);
   EXPECT_NE(opts1.Box().Border(), opts2.Box().Border());
   EXPECT_FLOAT_EQ(attrBox1.GetWidth(), 42.);
   EXPECT_FLOAT_EQ(attrBox2.GetWidth(), 3.);
   EXPECT_FLOAT_EQ(opts1.Box().Border().GetWidth(), 3.);
   EXPECT_FLOAT_EQ(opts2.Box().Border().GetWidth(), 42.);
}
