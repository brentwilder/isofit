#!/usr/bin/env bash

#SBATCH -J EMIT                                                          # job name
#SBATCH -o /store/bawilder/isofit/local/logs/log_slurm.o%j               # output and error file name (%j expands to jobID)
#SBATCH -N 1                                                             # Number of nodes
#SBATCH --ntasks 42                                                      # Number of tasks 
#SBATCH -t 00-30:00:00                                                   # run time (d-hh:mm:ss)  


python /store/bawilder/isofit/isofit/scripts/run_oe.py