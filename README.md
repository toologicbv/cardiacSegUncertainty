# Automatic segmentation and detection of segmentation inaccuracies in cardiac MRI using prediction uncertainties

## Under construction - super-resolution - segmentation
Try to train a segmentation model that can segment these high resolution (z-axis) cardiac MR volumes.


## Evaluation steps

### Train MC model (for ACDC or ARVC dataset)

Currently the following models are available for the automatic segmentation of cardiac MRI:
1. Dilated Residual Network (drn or drn_mc)
2. Dilated Network (dcnn or dcnn_mc)
3. UNet (unet or unet_mc)

Please make sure that the settings specified in `datasets/data.config.py` meet your environment i.e. make
sure the `ConfigACDC` object settings are set properly. There should be a per patient directory under the 
 `short_axis_dir/` directory e.g. `~/data/ACDC/all_cardiac_phases/patient018/`. Each directory should contain
the original challenge data in nifti file format. 

A couple of configuration settings for the training are hardwired into the ```train.py``` program. **learning schedule**, 
**weight_decay**, **resample** (dcnn True/ drn, unet False), **patch_size** (151 dcnn and 128 drn, unet).
You can train the models with **train.py** in the root of the repo.
````
CUDA_VISIBLE_DEVICES=1 python train.py ~/expers/acdc/f2/dcnn_mc_brier --l=0.02 --loss=brierv2 --dataset=ACDC --network=dcnn_mc
````
Please note that for the current pipeline we expect the following directory tree:
````
~/expers/acdc/<fold>/
````
The **train.py** first parameter specifies where the details of the model are stored. A **settings.yaml** file with
the experimental settings will be stores in this directory, together with the regulary (default 10000) stored
model parameter files.
After training all folds for a specific model we expect every <fold> subdirectory to contain the same
model directory e.g. ````~/expers/acdc/f0/drn_mc````, ````~/expers/acdc/f1/drn_mc````...

**Note**: training time depends on the model, DRN takes approximately 4 hours, U-net slightly more and DCNN can
take up to 12-13 hours.

### Test MC model with and without sampling on ACDC dataset


Evaluating a model can be done without or with MC dropout enabled. Setting ````--samples```` to a value greater than 1 enables *MC dropout*

````
CUDA_VISIBLE_DEVICES=1 python test.py ~/expers/acdc/f2/dcnn_mc_brier --samples=10 --save_output --save_results --save_probs
 ````
**Important**: use ````--acdc_all```` to evaluate all patient studies (makes only sense if you want to train the 
detection network and use umaps and predicted labels for training the same fold for this task).

With the option ````--save_results```` we will store a numpy archive file in ````~/expers/acdc/f1/drn_mc```` which contains
all important evaluation metrics (e.g. DSC, HD, HD95, ASSD). Details of the arrays can be found in the TestResult.save()
method ````evaluate/test_result.py````. Actually, each dataset (ACDC, ARVC) currently has her own test result
object.
With the option ````--save_output```` the following output objects are stores:
````~/expers/acdc/f0/drn_mc/umaps/````: uncertainty maps, b-maps and e-maps in nifti, shape [z, y, x]
````~/expers/acdc/f0/drn_mc/pred_labels/````: auto segmentation mask , nifti, shape [z, y, x], multi-labels {0,1,2,3}
````~/expers/acdc/f0/drn_mc/pred_labels_mc/````: same as above but then with MC dropout enabled during testing. 
````~/expers/acdc/f0/drn_mc/pred_probs/````: probabilities per voxel per class, nifti, shape [z, nclasses, y, x]
````~/expers/acdc/f0/drn_mc/pred_probs_mc/````: same as above but then with MC dropout enabled during testing.
````~/expers/acdc/f0/drn_mc/errors/````: segmentation errors (actually currently not used), nifti, shape [z, y, x]



### Generate distance transform maps and detection labels

In order to train the second step of our segmentation and detection approach, we first determine the set of segmentation inaccuracies.
Basically, we discard segmentation errors that are too close to the tissue structures borders or are considered
to small i.e. they are not part of an errorouness region of at least size 10. 
We currently tolerate a fixed error maring of 4.6mmm and 3.1mm for segmentation errors outside resp. inside
a target structure (instructions how to change these settings can be found below). 
**NOTE**, configuration settings for the margins can be found in ````datasets/data_config.py````.
Programs can be found can be found in ````datasets/ACDC/detection/data.py```` and 
````datasets/ACDC/detection/distance_transforms.py````.

We first compute 2D distance transform maps using the reference segmentations and then generate the 
segmentation inaccuracies (for each patient volume/cardiac phase), both steps will be run in once.
You can specify ````reuse_dt_maps```` which will omit the step of generating the dt maps. You can also
specify a specific patient id with ````--patid=patient057````.
Distance transform maps and detection labels will be saved in the same directory as stated on the command line under
directories *dt_maps* resp. *dt_labels*.

**IMPORANT**: make sure that the margin settings (main criteria what defines a segmentation inaccuracy) in 
`datasets/data_config.py` (object `ConfigACDC`, property `dt_margins = (4.6, 3.1)`) are set to your needs before
you invoke the program.

`
export PYTHONPATH=~/repo/model_evaluation/
CUDA_VISIBLE_DEVICES=2 python datasets/ACDC/detection/data.py ~/expers/acdc/f3/drn_mc_ce/
`

### Copy evaluation objects of different folds into shared model directory

For each model (model+loss_function e.g. dcnn_mc_brier) we create a shared directory under ````~/expers/acdc/````
e.g. ````~/expers/acdc/unet_mc/```` with the following sub-directories: umaps, pred_labels, pred_labels_mc, 
dt_labels/fixed<margins>. Then use the ````copy_data.sh```` script in the root of the repo
to copy these objects from each fold (assuming they are four fold-directories labeled f0...f3) to the shared
directory. The script takes two parameters 

````
~/copy_data.sh drn_mc_ce fixed_46_31
````
(1) model name (2) sub-directory where the dt_labels reside. Note, we use sub-directories under dt_labels and dt_maps
which enables us to evaluate different *filtering rules* for the segmentation inaccuracies at the same time.



### Evaluate calibration of (softmax) probabilities




### Evaluate uncertainty maps using risk-coverage curves (Geifmann et al.)

In order to evaluate the *quality* of the uncertainty maps we generate so called risk-coverage curves.
First, we need to generate the risk-coverage data for the different uncertainty thresholds (values of maps are in [0,1]) 
to generate the figures.
Use the ````eval_umaps/generate_data.py```` with the following flags: 
````--seg_model [drn_mc, dcnn_mc, unet_mc] --loss_function [ce, brier, dice] --channels [bmap or emap]````
**IMPORTANT**: we assume again the dir structure ~/expers/acdc/drn_mc_ce/ with the common sub-directories (also stated below).

````
export PYTHONPATH=/home/jorg/repo/model_evaluation/
python eval_umaps/generate_data.py --seg_model=drn_mc --loss_function=ce --channels=bmap
````

The data will be stored in the shared directory for that model with the following naming convention
cov_risk_<type of map>_<ED/ES>_cropped.npz e.g. ```` cov_risk_bmap_ED_cropped.npz ````

You can also run the evaluation for one fold only (or directory). Use the followng ipython notebook
```notebooks/model_evaluation/notebooks/selective_classification.ipynb``` to do this.


#### Create actual risk-coverage figures

In the previous step we created the data necessary to generate the figures. Use the ipython notebook
```notebooks/model_evaluation/notebooks/selective_classification.ipynb``` to generate the actual figures.
You will find a cell in the notebook in which you'll have to specify the data files previously generated 
(e.g. ```` cov_risk_emap_ED_cropped.npz cov_risk_emap_ES_cropped.npz````). In one of the cells below you'll find
a tiny piece of code that calls function ```plotting.coverage_risk_curves.plot_coverage_risk_curve_per_loss```.
If **do_save** is True the output will be saved to the figures directory in *main* directory of the structure.
E.g. ````~/expers/acdc/figures````. All figures will be saved as jpeg and pdf. The following metrics will
be displayed on the y-axis: DSC, HD or number of segmentation errors (i.e. FP+FN).


### Run detection task for the different folds

The task *detection of segmentation inaccuracies* has to been run for each fold of a segmentation model (e.g. drn_mc) 
separately. Make sure you have generated the 
   + evaluated the segmentation model (i.e. you generated e- and b-maps and predicted segentation labels);
   + generated the distance transform maps for this fold;
   + generated the detection labels (supervised learning) for this fold.
   
Another important point of attention is the settings of the hyperparameters for the detection loss function.
These can be found `networks/detection/general_setup.py`. We experimented with three different models but 
finally choose a Small Residual Network for the detection task. So make sure the following parameters are set
as follows in `BaseConfig` object (*detector_cfg* property):

| Model  | loss function | umap   | `fn_penalty_weight` | `fp_penalty_weight` |
|--------|---------------|:------:|:-------------------:|:-------------------:|
| DCNN   | Brier         | emap   |  1.2                |      0.085          |
|        |               | bmap   |  1.2                |      0.085          |
|        | soft-Dice     | emap   |  1.2                |      0.085          |
|        |               | bmap   |  1.2                |      0.085          |
| DRN    | CE            | emap   |  1.2                |      0.085          |
|        |               | bmap   |  1.2                |      0.085          |
|        | soft-Dice     | emap   |  1.2                |      0.085          |
|        |               | bmap   |  1.2                |      0.085          |
| U-net  | CE            | emap   |  1.2                |      0.085          |
|        |               | bmap   |  1.2                |      0.085          |
|        | soft-Dice     | emap   |  1.2                |      0.085          |
|        |               | bmap   |  1.2                |      0.085          |

Training is very sensitive to these settings. Hence, they are different for most of the combinations of
seg\_model, loss function and uncertainty map.

```CUDA_VISIBLE_DEVICES=3 python train_detector.py ~/expers/acdc/drn_mc_ce  -l=0.00001 --network=rsn --batch_size=32 --lr_decay_after=10000 --update_visualizer_every=25 --max_iters=20000 --input_channels=umap --fold=1```

The first parameter ```~/expers/acdc/drn_mc_ce``` is a directory (as introduced earlier) that holds the necessary objects of all folds for a
particular segmentation model/loss function combination (e.g. drn_mc_dice). Make sure the directory structure
is present (i.e. under this root directory there is pred_labels/, umaps/, dt_labels/) as specified above.
The log files (e.g. settings.yaml) will be stored under the sub-directory <model_lossfunction/dt_logs/.
E.g. ```dt_logs/f0_rsn_mc_bmap_055_0085_0821_152630/```

Add ```--mc_dropout``` when using Bayesian uncertainty maps as input.

The parameter ```--input_channels ['allchannels', 'umap', 'segmask', 'mronly']``` indirectly specifies the number
of input channels:
   + ```allchannels``` uses MRI, auto segmentation mask and e- or b-map (depending whether --mc_dropout parameter is specified).
   + ```umap``` the model takes two channels as input, MRI and e- or b-map;
   + ```segmaks``` the models takes two channels as input, MRI and automatic segmentation mask;
   + ```mronly``` the model takes only the MRI as input channel.
    

### Evaluate detection model(s) 

In this step you evaluate the detection model for each fold. You can use the 
```notebooks/model_evaluation/notebooks/region_detector/test_detector_model.ipynb``` ipython notebook for the evaluation
of a separate fold or ```eval_detection/evaluate_model.py``` to run the evaluation for all four folds for one
combination of segmentation model and loss function. In the latter case you have to fill the dictionary *EXPER_DIRS*
in ```eval_detection/evaluate_model.py```. The dictionary has entries for each combination of seg-model, loss function
and uncertainty map. For each combination you specify a directory e.g. ```f0_rsn_mc_bmap_055_0085_0821_152630/```.
Then run the *evaluate_model.py* file with the following options:

````
export PYTHONPATH=/home/jorg/repo/model_evaluation/
python eval_umaps/evaluate_model.py --seg_model=drn_mc --loss_function=ce --channels=bmap --run_eval --move_files
````

If you specify ```--move_files``` the generated a) heat maps (re-sized to original image size) and b) binary ground truth
labels for the detection task (per region) will be copied to the ```dt_results/<type of map>/``` directory under
the model/loss function root directory (e.g. ```~expers/acdc/unet_mc_dice/```). 
This is actually required if you want to run the *simulation* of the correction of the detected regions described 
in the next step.
In any case the same output objects will be stored under the `dt_logs/<experiment dir>/heat_maps/` resp. 
`dt_logs/<experiment dir>/results/` directory.


### Generate figures to quantify detection performance

To evaluate the performance of the detection model(s) we generate:
   + FROC curves for the detection rate (voxel level) of segmentation inaccuracies;
   + Precision-recall curves for the slice detection performance;
   + FROC curves for the slice detection rate.
   
You can use the ipython notebook `notebooks/model_evaluation/notebooks/region_detector/compare_models.ipynb`
to generate the necessary data (based on the detection model evaluation, previous step).
All instructions can be found there. You can use the notebook to generate these figures for one model (one or all folds)
or to actually compare different combination of segmentation models, loss functions and uncertainty maps.

After loading the necessary data in the notebook you use the functions in `plotting.compare_detection_models` to 
generate the figures specified above (`compare_pr_rec_curves, compare_froc_curves_dtrate, compare_froc_curves_slices`).
The figures will be stored in `~/expers/acdc/figures/`.


### Run simulation to evaluate detection model implicitly

#### Prepare simulation

Added 06-12-2019: After evaluating the detection models per fold, assuming ``--move_files`` was enabled the resulting 
heat maps and predicted detection labels (per grid cell) are transferred to the root directory of the experiment
e.g. `~/expers/redo_expers/drn_mc_dice/`. In order to run the simulation we also need the following objects under this
root directory: `pred_labels` (also for mc evaluation), `umaps` and `dt_labels` (for e- and bmaps). 
In our new approach these files need to be copied from the specific fold directory to the main dir of the experiment.
**Note**, only for the test patients of this specific fold (remember, we generated these objects for all patients
per fold, because we needed them to train the second stage). 
To accomplish this, run following for each combination of seg-model and loss function:

````
export PYTHONPATH=/home/jorg/repo/model_evaluation/
python eval_detection/prepare_simulation.py --seg_model=drn_mc --loss_function=ce 
````
This will copy the designated files into `pred_labels`, `pred_labels_mc`, `umaps` and `dt_labels` 
under the main experiment directory e.g. `~expers/redo_expers/drn_mc_dice/`.

#### Run

We use the heat maps obtained in the previous step (*Evaluate detection model*) to correct the predicted labels 
(exchange voxels in region with reference segmentation labels) and recompute DSC and HD.
Again, first we need to generate the data that is needed for the box plots we want to generate.
You need to use `eval_detection/simulate_correction.py` with the following flags: 
`--seg_model [dcnn_mc, drn_mc, unet_mc] --loss_function [brier, ce, dice] --channels [bmap/emap] --all_errors`
If you use `--all_errors` the function generates the so called "baseline seg+detection" results for DSC and HD. 
Means, our combined seg+detection can never reach better results than these.

Remember, we filter our segmentation errors based on distance transforms and if we would correct ALL these errors, 
what is the DSC/HD. That's what these results represent. 

The results are stored in the *root* directory of the model (e.g. `~/acdc/expers/drn_mc/`) under `sim_results/<emap> or bmap/`
Numpy arrays are stored in the filename: 
   + `sim_expert_allerrors_fixed_46_31_fall_n200.npz`: baseline detection (correction of all possible segmentation inaccuracies);
   + `sim_expert_base_fall_n200.npz`: segmentation-only results (without detection);
   + `sim_expert_fixed_46_31_bmap_fall_n200.npz`: segmentation & detection performance (simulation).

and remember we need to run this for bmaps and emaps because enabling MC dropout during evaluation produces slightly
different segmentation masks and errors.

`
export PYTHONPATH=/home/jorg/repo/model_evaluation/
python eval_detection/simulate_correction.py --seg_model=drn_mc --loss_function=ce --channels=bmap  (--all_errors)
`

Finally, you can use the ipython notebook `notebooks/region_detector/simulate_correction.ipynb` to generate 
the box plot figures. See instructions in notebook.

### Generate latex table entries journal article

Use the following jupyter notebook `notebooks/generate_latex_table.ipynb` to generate the latex table entries
for the journal article.

### Visualize detection results 

You can use jupyter notebook `notebooks/detection/visualize_detection_results.ipynb` to gain some qualitative insights
into the detection results. The function `plot_slices_per_phase` in `plotting/visualize_set_det_results.py`
can be used to visualize for a specific patient and phase the different results of the combined approach.
I.e., overlay the MR image with e.g. uncertainty information, segmentation errors, segmentation failures to be detected,
heat maps generated by the detection network etc. 

