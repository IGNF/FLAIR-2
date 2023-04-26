#!/bin/bash

###SBATCH --account=tel@v100     # sur quel projet imputer

#SBATCH --job-name=test_UTAE       # nom du job
###  -C v100-32g          # de-commente pour la partition gpu_p2
#SBATCH --nodes=3                  # on demande 3 noeuds
#SBATCH --ntasks-per-node=4          # avec 4 tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:4                 # nombre de GPU-4
#SBATCH --cpus-per-task=10        # nombre de coeurs CPU par tache

#SBATCH --time=20:00:00             # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --hint=nomultithread

#SBATCH --output=jobs/stouterr_%j.out      # nom du fichier de sortie
#SBATCH --error=jobs/stouterr_%j.out       # nom du fichier d'erreur (ici commun avec la sortie)

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
module load python

conda deactivate
conda activate TEST-NODIR
set -x

export TORCH_HOME=$HOME/.cache/torch/
export TMPDIR=$SCRATCH/torch_tmp
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_SHOW_CPP_STACKTRACES=1


# execution du code
srun python $WORK/FLAIR2LIB-LAST/main.py --config_file $WORK/FLAIR2LIB-LAST/flair-2-config.yml
