#!/bin/bash -l
#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J MPITest
## Liczba alokowanych węzłów
#SBATCH -N 2
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=8
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 5GB na rdzeń)
#SBATCH --mem-per-cpu=5GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=08:00:00 
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plgar2022-cpu
## Specyfikacja partycji
#SBATCH -p plgrid
## Plik ze standardowym wyjściem
#SBATCH --output="output.out"
## Plik ze standardowym wyjściem błędów
#SBATCH --error="error.err"
#SBATCH -n 16
srun /bin/hostname
## Zaladowanie modulu IntelMPI
module load scipy-bundle/2021.10-intel-2021b
## przejscie do katalogu z ktorego wywolany zostal sbatch
cd $SLURM_SUBMIT_DIR



for i in {1 ... 16}
do
    mpiexec -n $i python ./lab3.py 10000 2 20
done