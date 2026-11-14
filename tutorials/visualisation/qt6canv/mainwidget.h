#pragma once

#include <QWidget>
#include <QTimer>
#include <array>

class QPushButton;
class QVBoxLayout;

class TCanvas;

class MainWidget : public QWidget
{
    Q_OBJECT

public:
    explicit MainWidget(QWidget *parent = nullptr);

    void button_clicked(int index);

    void toggle_timer(bool on);

signals:
    // Emitted when one of the four buttons is clicked (index 0..3).
    void buttonClicked(int index);

private:
    QWidget *m_placeholder = nullptr;   // container on the right
    QVBoxLayout *m_placeholderLayout = nullptr;

    TCanvas *m_canvas = nullptr;    // canvas embed in the placeholder

    void fill_canvas_object();

    QTimer m_timer;
    int m_counter = 0;
    int m_clicked_button = 0;
};
