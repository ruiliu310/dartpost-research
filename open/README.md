# Dartpost Research Project

## Directory structure and Files

- **code**: All the scripts to run the statistical models and FIG scores in the paper.
- **data**: data
- **output**: all the output from the models
  - *alluvial*: alluvial plot (Figure 4) (.pdf)
  - *table2*: results table (Table 2) (.tex)


## Requirements

- R 3.6.3
- Python >= 3.8
  + networkx >= 2.0
  + numpy
  + scipy
  + pandas
  + seaborn >= 0.11
  + matplotlib
  + tqdm

##	Executing the Script

- **code/execute.sh**: will run the models and create the plot and table for impression statistical analysis. In root directory, use command `bash ./code/execute.sh` to run the script.
- **code/pr_script.sh**: will run the FIG score computations. In root directory, use command `bash ./code/pr_script.sh` to run all the FIG1 and FIG2 compuations.
- **code/pr_compute.py**: includes the procedure to compute FIG1 or FIG2 based on user input. See help text with command `python ./code/pr_compute.py --help`.

