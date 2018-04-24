/// \file
/// \ingroup tutorial_v7
///
/// \macro_code
///
/// \date 2018-03-18
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!
/// \author Iliana Betsou

R__LOAD_LIBRARY(libGpad);

#include "ROOT/TCanvas.hxx"
#include "ROOT/TColor.hxx"
#include "ROOT/TText.hxx"
#include "ROOT/TLine.hxx"

void lineStyle() {
    using namespace ROOT;

    auto canvas = Experimental::TCanvas::Create("Canvas Title");
    double num = 0.3;
    double numL = 0.695;

    for (int i=10; i>0; i--){

        num = num + 0.02;
        auto text = std::make_shared<Experimental::TText>(.3, num, Form("%d", i));
        text->GetOptions().SetTextSize(13);
        text->GetOptions().SetTextAlign(33);
        text->GetOptions().SetTextFont(52);
        canvas->Draw(text);

        numL = numL - 0.02;
        auto line = std::make_shared<Experimental::TLine>(.43, numL, .8, numL);
        line->GetOptions().SetLineStyle(i);
        canvas->Draw(line);


    }

    canvas->Show();
}
