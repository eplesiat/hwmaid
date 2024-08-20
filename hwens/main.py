import argparse
import xarray as xr
from dask.diagnostics import ProgressBar
from dask.distributed import Client

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", '--duration', type=str, help='Duration files', nargs="+")
    parser.add_argument("-i", '--intensity', type=str, help='Intensity files', nargs="+")
    parser.add_argument("-c", '--cumintensity', type=str, help='Cumulative intensity files', nargs="+")
    parser.add_argument('-o', '--outname', type=str, default="", help='Output filename')
    parser.add_argument('-n', '--n_chunks', type=int, default=10, help='Number of chunks in time')
    args = parser.parse_args()
    
    client = Client()
    ProgressBar().register()

    outname = args.outname.split("/")
    if outname == [""]:
        dirname, rootname = "", ""
    else:
        dirname = "/".join(outname[:-1]) + "/"
        rootname = "_" + outname[-1].replace(".nc", "")

    for type in ('intensity', 'cumintensity', 'duration'):

        print("* Treating", type)
        files = getattr(args, type)

        
        ds = xr.open_mfdataset(files, combine='nested', concat_dim='member', chunks={"time": args.n_chunks})
        ds_ = ds.mean(dim="member")
        ds_ = ds_.compute()
        ds_.to_netcdf(dirname + type + rootname + "_mean.nc")
        ds_ = ds.std(dim="member")
        ds_ = ds_.compute()
        ds_.to_netcdf(dirname + type + rootname + "_std.nc")

    print("* Calculating probability")
    ds_ = ~ds.isnull()
    ds_ = ds_.mean(dim="member")
    ds_ = ds_.compute()

    ds_.to_netcdf(dirname + type + rootname + "_prob.nc")

if __name__ == "__main__":
    main()
