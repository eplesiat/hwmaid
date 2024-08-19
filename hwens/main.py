import argparse
import xarray as xr
from dask.diagnostics import ProgressBar

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", '--duration', type=str, help='Duration files', nargs="+")
    parser.add_argument("-i", '--intensity', type=str, help='Intensity files', nargs="+")
    parser.add_argument("-c", '--cumintensity', type=str, help='Cumulative intensity files', nargs="+")
    parser.add_argument('-o', '--outname', type=str, default="", help='Output filename')
    args = parser.parse_args()

    outname = args.outname.split("/")
    if outname == [""]:
        dirname, rootname = "", ""
    else:
        dirname = "/".join(outname[:-1]) + "/"
        rootname = "_" + outname[-1].replace(".nc", "")

    for type in ('intensity', 'cumintensity', 'duration'):

        print("* Treating", type)
        files = getattr(args, type)
        ds = xr.open_mfdataset(files, combine='nested', concat_dim='member')

        with ProgressBar():
            ds.mean(dim="member").to_netcdf(dirname + type + rootname + "_mean.nc")
            ds.std(dim="member").to_netcdf(dirname + type + rootname + "_std.nc")

    ds_prob = ~ds.isnull()
    ds_prob = ds_prob.mean(dim="member")

    ds_prob.to_netcdf(dirname + type + rootname + "_prob.nc")

if __name__ == "__main__":
    main()