# HWMAID

This repository contains two Python codes:
- `hwmid`: to evaluate the intensity and duration of heatwaves from daily gridded maximum temperature
- `hwens`: to calculate mean, std and probability from the results of hwmid

## Dependencies

An Anaconda environment with all the required dependencies can be created using `environment.yml`:
```bash
conda env create -f environment.yml
```
To activate the environment, use:
```bash
conda activate hwmaid
```

The installation time of the required dependencies should not exceed 3 minutes using a stable and standard internet connection

## Usage

`hwmid` takes 4 input files:
- `analysis_file`: the daily gridded maximum temperature to be analyzed
- `per90_file`: the multi-year daily running 90th percentile of the reference data
- `t25_file`: the 25th percentile values over all year max of the reference data
- `t75_file`: the 75th percentile values over all year max of the reference data

```bash
usage: hwmid.py [-h] [-d NDAYS_TMAX] [-o OUTDIR] [-p NPROCS] [--outfile1 OUTFILE1] [--outfile2 OUTFILE2] [--outfile3 OUTFILE3] [-t TIMERANGE] [-s SHOW_PBARS] analysis_file per90_file t25_file t75_file

Process heat wave index data.

positional arguments:
  analysis_file         Path to the analysis file
  per90_file            Path to the percentiles file
  t25_file              Path to the t25 file
  t75_file              Path to the t75 file

optional arguments:
  -h, --help            show this help message and exit
  -d NDAYS_TMAX, --ndays-tmax NDAYS_TMAX
                        Number of consecutive days with tmax
  -o OUTDIR, --outdir OUTDIR
                        Path to the output directory
  -p NPROCS, --nprocs NPROCS
                        Number of processes
  --outfile1 OUTFILE1   Filename of the output file 1
  --outfile2 OUTFILE2   Filename of the output file 2
  --outfile3 OUTFILE3   Filename of the output file 3
  -t TIMERANGE, --timerange TIMERANGE
                        Selected time range (date1,date2)
  -s SHOW_PBARS, --show-pbars SHOW_PBARS
                        Comma speparated indices of the processesfor which progress bars have to be shown
```

`hwens` takes 3 types of input files:
- the `duration` files created by `hwmid`for all members
- the `intensity` files created by `hwmid`for all members
- the `cumintensity` files created by `hwmid`for all members

```bash
usage: hwens.py [-h] [-d DURATION [DURATION ...]] [-i INTENSITY [INTENSITY ...]] [-c CUMINTENSITY [CUMINTENSITY ...]] [-o OUTNAME] [-n N_CHUNKS]

optional arguments:
  -h, --help            show this help message and exit
  -d DURATION [DURATION ...], --duration DURATION [DURATION ...]
                        Duration files
  -i INTENSITY [INTENSITY ...], --intensity INTENSITY [INTENSITY ...]
                        Intensity files
  -c CUMINTENSITY [CUMINTENSITY ...], --cumintensity CUMINTENSITY [CUMINTENSITY ...]
                        Cumulative intensity files
  -o OUTNAME, --outname OUTNAME
                        Output filename
  -n N_CHUNKS, --n_chunks N_CHUNKS
                        Number of chunks in time
```

## License

`HWMAID` is licensed under the terms of the GNU General Public License Version 3.
