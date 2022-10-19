#!/bin/bash -l
#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J MPITest
## Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=8
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 5GB na rdzeń)
#SBATCH --mem-per-cpu=5GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=01:00:00 
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plgar2022-cpu
## Specyfikacja partycji
#SBATCH -p plgrid-testing
## Plik ze standardowym wyjściem
#SBATCH --output="output.out"
## Plik ze standardowym wyjściem błędów
#SBATCH --error="error.err"

srun /bin/hostname

## Zaladowanie modulu IntelMPI
module add plgrid/tools/impi/2021.1.1-intel-compilers-2021.1.2
## przejscie do katalogu z ktorego wywolany zostal sbatch
cd $SLURM_SUBMIT_DIR

mpiexec ./sito1.py 1000


