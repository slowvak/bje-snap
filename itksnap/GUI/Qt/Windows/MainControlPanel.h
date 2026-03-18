#ifndef MAINCONTROLPANEL_H
#define MAINCONTROLPANEL_H

#include <SNAPComponent.h>

class LabelSelectionButton;
class LabelSelectionPopup;

namespace Ui {
class MainControlPanel;
}

class GlobalUIModel;
class MainImageWindow;
class QToolButton;

class MainControlPanel : public SNAPComponent
{
  Q_OBJECT
  
public:
  explicit MainControlPanel(MainImageWindow *parent = 0);
  ~MainControlPanel();

  void SetModel(GlobalUIModel *model);

protected slots:

  virtual void onModelUpdate(const EventBucket &bucket);

  void onDrawingButtonAction(QAction *);

private slots:

  void on_btnCTBrain_clicked();
  void on_btnCTAbd_clicked();
  void on_btnCTLung_clicked();
  void on_btnCTBone_clicked();

  void on_btnClearMask_clicked();
  void on_btnClearVolumeMask_clicked();

  void on_btnCursorInspector_clicked(bool checked);

  void on_btnZoomInspector_clicked(bool checked);

  void on_btnDisplayInspector_clicked(bool checked);

  void on_btnSyncInspector_clicked(bool checked);


  void on_btnPolygonInspector_clicked(bool checked);

  void on_btnPaintbrushInspector_clicked(bool checked);

  void on_btnSnakeInspector_clicked(bool checked);

  void on_btnAnnotateInspector_clicked(bool checked);

private:

  void updateLabelSelectionButton();
  void updateContrastGroupVisibility();
  void applyCTPreset(double minVal, double maxVal);
  bool isMainImageCT();

  Ui::MainControlPanel *ui;
  GlobalUIModel *m_Model;

  LabelSelectionButton *m_LabelSelectionButton;
  LabelSelectionPopup *m_LabelSelectionPopup;

  QToolButton *m_DrawingDropdownButton;
};

#endif // MAINCONTROLPANEL_H
