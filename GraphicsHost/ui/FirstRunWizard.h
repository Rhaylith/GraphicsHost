#pragma once

#include <QWizard>

namespace gh::host {
class AppHost;
}

namespace gh::ui {

// Shown on the first start: what this computer can do, where models go, which model to begin with (and to download it
// now), and how much of the interface to show. Whatever the outcome, the wizard is not shown again.
class FirstRunWizard : public QWizard {
    Q_OBJECT

public:
    explicit FirstRunWizard(gh::host::AppHost& host, QWidget* parent = nullptr);

    void accept() override;
    void reject() override;

private:
    void markDone();
    QWizardPage* surveyPage();
    QWizardPage* folderPage();
    QWizardPage* modelPage();

    gh::host::AppHost& host_;
};

}  // namespace gh::ui
