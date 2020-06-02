#! /bin/bash

model_name=$1
dt_margin=$2

cp ~/expers/acdc/f0/$1/pred_labels/*.* ~/expers/acdc/$1/pred_labels/
cp ~/expers/acdc/f1/$1/pred_labels/*.* ~/expers/acdc/$1/pred_labels/
cp ~/expers/acdc/f2/$1/pred_labels/*.* ~/expers/acdc/$1/pred_labels/
cp ~/expers/acdc/f3/$1/pred_labels/*.* ~/expers/acdc/$1/pred_labels/

cp ~/expers/acdc/f0/$1/pred_labels_mc/*.* ~/expers/acdc/$1/pred_labels_mc/
cp ~/expers/acdc/f1/$1/pred_labels_mc/*.* ~/expers/acdc/$1/pred_labels_mc/
cp ~/expers/acdc/f2/$1/pred_labels_mc/*.* ~/expers/acdc/$1/pred_labels_mc/
cp ~/expers/acdc/f3/$1/pred_labels_mc/*.* ~/expers/acdc/$1/pred_labels_mc/

cp ~/expers/acdc/f0/$1/umaps/*.* ~/expers/acdc/$1/umaps/
cp ~/expers/acdc/f1/$1/umaps/*.* ~/expers/acdc/$1/umaps/
cp ~/expers/acdc/f2/$1/umaps/*.* ~/expers/acdc/$1/umaps/
cp ~/expers/acdc/f3/$1/umaps/*.* ~/expers/acdc/$1/umaps/

cp ~/expers/acdc/f0/$1/dt_labels/$dt_margin/*.* ~/expers/acdc/$1/dt_labels/$dt_margin/
cp ~/expers/acdc/f1/$1/dt_labels/$dt_margin/*.* ~/expers/acdc/$1/dt_labels/$dt_margin/
cp ~/expers/acdc/f2/$1/dt_labels/$dt_margin/*.* ~/expers/acdc/$1/dt_labels/$dt_margin/
cp ~/expers/acdc/f3/$1/dt_labels/$dt_margin/*.* ~/expers/acdc/$1/dt_labels/$dt_margin/

