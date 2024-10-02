import argparse
import numpy as np
import xarray as xr
import pandas as pd
import cftime
from itertools import groupby
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def parse_range(arg):

    if arg is None:
        return None
    else:
        timerange = arg.split(",")
        assert len(timerange) == 2
        return slice(*timerange)

def parse_pbars(arg):
    
    if arg is None:
        return []
    else:
        return [int(i) for i in arg.split(",")]

def to_datetime64(time):

    times = []
    for t in time:
        if isinstance(t, cftime.datetime):
            times.append(pd.Timestamp(t.year, t.month, t.day, t.hour, t.minute, t.second))
        elif isinstance(t, (pd.Timestamp, np.datetime64)):
            times.append(pd.Timestamp(t))
        else:
            print("Unsupported time type:", type(t))
    return np.array(times)

def get_dims(da):
    ddims = {"time": "time"}
    dims = da.dims
    for key in ("lon", "lat"):
        for dim in dims:
            if key in dim:
                ddims[key] = dim

    assert len(ddims.keys()) == 3

    return ddims

def get_da(fname, variable_names, timerange=None, nprocs=None, keep_da=False, squeeze=False):

    ds = xr.open_dataset(fname)
    if timerange:
        ds = ds.sel(time=timerange)

    for var_name in variable_names:
        if var_name in ds:
            da = ds[var_name]
            da.attrs.clear()
            ddims = get_dims(da)
            da = da.transpose(*list(ddims.values()))
            ds.close()
            if keep_da:
                return da, ddims
            else:
                shape = da.shape
                da = da.values.reshape(shape[0], -1)
                axis = 1
                if squeeze:
                    da = da.squeeze()
                    axis = 0
                if nprocs:
                    da = np.array_split(da, nprocs, axis=axis)
                return da
    raise ValueError(f"Variable(s) {', '.join(variable_names)} not found in file: {ds.attrs.get('filename', 'unknown')}")

def create_netcdf(da, sizes, filename, data, variable_name, units, long_name):
    new_da = da.rename(variable_name)
    new_da.values = np.concatenate(data, axis=1).reshape(sizes)
    ds = xr.Dataset({variable_name: new_da}, attrs={'units': units, 'long_name': long_name})
    ds.to_netcdf(filename)

def rle(val):
    return zip(*[(key, len(list(group))) for key, group in groupby(val)])

def get_durations(values, lengths):
    return [lengths[l] for l in range(len(lengths)) if values[l]]

def calc_d90(hide_pbar, dmax, p90, doy_analysis, doy_perc):

    shape = dmax.shape
    d90 = np.full(shape, False)
    for t in tqdm(range(shape[0]), disable=hide_pbar):
        perc_day = np.where(doy_analysis[t] == doy_perc)[0]
        for i in range(shape[1]):
            if dmax[t, i] > p90[perc_day, i]:
                d90[t, i] = True

    return d90

def calc_counter(hide_pbar, d90, ndays_tmax):

    shape = d90.shape
    counter = np.full(shape, False)
    for t in tqdm(range(shape[0] - ndays_tmax + 1), disable=hide_pbar):
        trange = slice(t, t + ndays_tmax)
        for i in range(shape[1]):
            if np.all(d90[trange, i]):
                counter[trange, i] = True

    return counter

def calc_md(hide_pbar, dmax, counter, t25, t75):

    shape = dmax.shape
    md = np.zeros(shape) * np.nan
    for i in tqdm(range(shape[1]), disable=hide_pbar):
        t = np.where(counter[:, i])[0]
        for n in range(len(t)):
            if dmax[t[n], i] > t25[i]:
                md[t[n], i] = (dmax[t[n], i] - t25[i]) / (t75[i] - t25[i])
            else:
                md[t[n], i] = 0

    return md

def calc_valdur(hide_pbar, counter):

    shape = counter.shape
    valdur = []
    for i in tqdm(range(shape[1]), disable=hide_pbar):
        # Reduce the (non-)occurrences HWs (counter) into values (True/False) and consecutive days (lengths) in chronological order
        values, lengths = rle(counter[:, i])
        # Durations for the HW episodes only
        durations = get_durations(values, lengths)
        t = np.where(counter[:, i])[0]
        valdur.append((values, lengths, durations, t))

    return valdur

def calc_cdur(hide_pbar, valdur, shape):

    cdur = np.zeros(shape) * np.nan
    for i in tqdm(range(shape[1]), disable=hide_pbar):
        # Duration of HWs since starting day (cumulative value)
        cdur[valdur[i][3], i] = [l for e, c in enumerate(valdur[i][2]) for l in range(c)]

    return cdur

def calc_mda(hide_pbar, valdur, md):

    shape = md.shape
    mda = np.zeros(shape) * np.nan
    for i in tqdm(range(shape[1]), disable=hide_pbar):
        if len(valdur[i][3]) > 0:
            s = np.split(np.arange(sum(valdur[i][1])), np.cumsum(valdur[i][1])[:-1])
            hw = np.where(np.array(valdur[i][0]))[0]
            len_hw = len(hw)

            for n in range(len_hw):
                start = s[hw[n]][0]
                stop = s[hw[n]][valdur[i][2][n] - 1]
                mda[start:stop + 1, i] = np.cumsum(md[start:stop + 1, i])

    return mda

def main():
    parser = argparse.ArgumentParser(description='Process heat wave index data.')
    parser.add_argument('analysis_file', type=str, help='Path to the analysis file')
    parser.add_argument('per90_file', type=str, help='Path to the percentiles file')
    parser.add_argument('t25_file', type=str, help='Path to the t25 file')
    parser.add_argument('t75_file', type=str, help='Path to the t75 file')
    parser.add_argument("-d", '--ndays-tmax', type=int, default=3, help='Number of consecutive days with tmax')
    parser.add_argument("-o", '--outdir', type=str, default="." , help='Path to the output directory')
    parser.add_argument("-p", '--nprocs', type=int, default=None , help='Number of processes')
    parser.add_argument('--outfile1', type=str, default="HW_intensity.nc", help='Filename of the output file 1')
    parser.add_argument('--outfile2', type=str, default="cummulative_HW_intensity.nc", help='Filename of the output file 2')
    parser.add_argument('--outfile3', type=str, default="HW_duration.nc", help='Filename of the output file 3')
    parser.add_argument("-t", '--timerange', type=parse_range, default=None, help='Selected time range (date1,date2)')
    parser.add_argument("-s", '--show-pbars', type=parse_pbars, default=None, help='Comma speparated indices of the processes'\
                                                                            'for which progress bars have to be shown')
    args = parser.parse_args()

    temperature_names = ["tas", "tasmax", "tx"]
    nprocs = cpu_count()
    if args.nprocs:
        nprocs = min(args.nprocs, nprocs)

    hide_pbars = [False if i in args.show_pbars else True for i in range(nprocs)]

    print("* Loading the data") 

    da, ddims = get_da(args.analysis_file, temperature_names, timerange=args.timerange, keep_da=True)
    sizes = da.shape
    dmax = da.values.reshape(sizes[0], -1)
    shape = dmax.shape
    dmax = np.array_split(dmax, nprocs, axis=1)
    analysis_time = to_datetime64(da['time'].values)

    p90 = get_da(args.per90_file, temperature_names, nprocs=nprocs)

    doy_perc = np.arange(1, 367)
    doy_analysis = np.array([t.dayofyear for  t in analysis_time])

    t25 = get_da(args.t25_file, temperature_names, nprocs=nprocs, squeeze=True)
    t75 = get_da(args.t75_file, temperature_names, nprocs=nprocs, squeeze=True)

    with Pool(processes=nprocs) as pool:

        print("* Computing d90")
        inps = [(hide_pbars[c], dmax[c], p90[c], doy_analysis, doy_perc) for c in range(nprocs)]
        d90 = pool.starmap(calc_d90, inps)

        print("* Counting the heat days")
        inps = [(hide_pbars[c], d90[c], args.ndays_tmax) for c in range(nprocs)]
        counter = pool.starmap(calc_counter, inps)

        print("* Evaluating the intensity")
        inps = [(hide_pbars[c], dmax[c], counter[c], t25[c], t75[c]) for c in range(nprocs)]
        md = pool.starmap(calc_md, inps)

        print("* Evaluating the duration")
        inps = [(hide_pbars[c], counter[c]) for c in range(nprocs)]
        valdur = pool.starmap(calc_valdur, inps)

        print("* Evaluating the cumulative duration")
        inps = [(hide_pbars[c], valdur[c], counter[c].shape) for c in range(nprocs)]
        cdur = pool.starmap(calc_cdur, inps)

        print("* Evaluating the cumulative intensity")
        inps = [(hide_pbars[c], valdur[c], md[c]) for c in range(nprocs)]
        mda = pool.starmap(calc_mda, inps)

    outdir = args.outdir + "/"
    create_netcdf(da, sizes, outdir + args.outfile1, md, 'hw', 'degrees_C', 'Heat_Wave')
    create_netcdf(da, sizes, outdir + args.outfile2, mda, 'chw', 'degrees_C', 'Cummulative_Heat_Wave')
    create_netcdf(da, sizes, outdir + args.outfile3, cdur, 'dur', 'days', 'Heat_Wave_duration')

if __name__ == "__main__":
    main()
