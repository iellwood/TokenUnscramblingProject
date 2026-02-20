<img src="https://github.com/iellwood/MatchAndControlPaper/blob/main/Match_and_Control_Image_For_GitHub.jpg" alt="Illustration of the Match-and-Control principle" width="600">

### Token unscrambling in fixed-weight biological models of transformer attention

#### Ian T. Ellwood, Department of Neurobiology and Behavior, Cornell University

This repository contains the code to implement the model described in the paper, "Token unscrambling in fixed-weight biological models of transformer attention". While we have made every attempt to make sure that all the files needed to reproduce the paper's figures are available, please contact the author with any questions or concerns about the code or aspects of the model. Some of the files produced by the stimulations are very large. If the complete datafiles are required, a link to a dropbox containing the PyCharm project can be made available upon request.

#### Running the scripts

All the scripts were run in python version 3.9.13 on Windows 11. Required libraries are `numpy`, `matplotlib`, `scipy`, `torch`, and `pickle`. PyCharm community edition was used for the management of the project and running the scripts. 

#### Organization of the scripts for figure generation

To reproduce the results of the paper, the code must be run in the following order

1) `createGermanAndEnglishSentenceFiles.py` followed by `createtokenizer.py`. These scripts download the iwslt dataset (German to English) and create an appropriate tokenizer for the two languages.
2) `TrainScrambledModels` trains 15 models with a scrambled attention layer. This is enough to reproduce every figure except Fig. 2 panels A & B. These panels require running the script again with `number_of_replicates = 4` and the lines of code:
   ```
   train_model(
        'FAModel_SeparateScrambleMatrices_replicate_' + str(replicate_number),
        use_modified_layer=True,
        separate_scramble_matrix_for_K_and_V=True,
        print_header='M_K != M_V. Replicate = ' + str(replicate_number)
    )
   ```

   replaced with

   ```
   train_model(
        'FAModel_Control_replicate_' + str(replicate_number),
        use_modified_layer=False,
        separate_scramble_matrix_for_K_and_V=False,
        print_header='Unmodified transformer. Replicate = ' + str(replicate_number)
    )
   ```
  
   
3) `AnalyzeCheckpoints.py`: This script produces the bulk of the analysis of the paper and saves it in the folder `ModelData/` which must exist for the script to run successfully. 
4) `GetTestNLLs.py`: This script is essential only for Figure 2 panels A & B and requires that the control transformer models have been trained.
5) `PlotScripts/PlotDataFiguresForPaper.py` This produces all of the plots of the paper and saves them in the folder `PlotScripts/Plots/`. Note that this script requires that the control models have been trained, but this requirement can be eliminated by setting `plot_test_loss_for_scrambled_and_unscrambled_models = False`. This file also performs the same PCA analysis shown in the paper for the value scramble matrices if one sets `plot_PCA_analysis_for_value_scramble_matrix = True`.


