# Copyright 2024 ACCESS-NRI (https://www.access-nri.org.au/)
# See the top-level COPYRIGHT.txt file for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# Created by: Chermelle Engel <Chermelle.Engel@anu.edu.au>
# Modified in 2026 by: Ashley Barnes <ashley.barnes@monash.edu>

import numpy as np
from pathlib import Path
from dask.distributed import Client
from distributed import wait
from datetime import datetime
from datetime import timedelta
import xarray as xr
import dask
import subprocess
import argparse
import pandas
import time

_t = time.time()
def tick(label=""):
    global _t
    now = time.time()
    print(f"{label}: {now - _t:.2f}s")
    _t = now


def longitude_slicer(data, longitude_extent, longitude_coords):
    """Slice a dataset in longitude, handling periodicity and domain seams.

    Correctly clips datasets whose longitude coordinate may use any
    convention (e.g. ``[0, 360]`` or ``[-180, 180]``) and where the
    requested ``longitude_extent`` may straddle the wrap-around seam.

    The algorithm proceeds in five steps:

    1. Determine the integer multiple of 360° needed to shift the midpoint
       of ``longitude_extent`` into the range covered by ``data``.
    2. Roll the dataset so that its centre aligns with the midpoint of the
       target extent.
    3. Rebuild a monotonically increasing longitude coordinate that removes
       the seam introduced by the roll.
    4. Index out the required number of longitude points symmetrically
       around the new centre.
    5. Re-centre the coordinate values to match the target domain.

    Parameters
    ----------
    data : xarray.Dataset
        Global (or at least periodic) dataset to slice.  The longitude
        coordinate must be uniformly spaced.
    longitude_extent : array-like of float
        Target longitude bounds ``(west, east)`` in degrees, in increasing
        order.
    longitude_coords : str or list of str
        Name(s) of the longitude coordinate(s) in ``data`` along which to
        slice.

    Returns
    -------
    xarray.Dataset
        Dataset sliced to ``longitude_extent``, with the longitude
        coordinate re-centred to match the target domain.

    Raises
    ------
    AssertionError
        If any named longitude coordinate is not uniformly spaced.
    """

    if isinstance(longitude_coords, str):
        longitude_coords = [longitude_coords]

    for lon in longitude_coords:

        central_longitude = np.mean(longitude_extent)  ## Midpoint of target domain

        ## Find a corresponding value for the intended domain midpoint in our data.
        ## It's assumed that data has equally-spaced longitude values.

        lons = data[lon].data
        dlons = lons[1] - lons[0]
        lon_span = lons[-1] - lons[0] + dlons

        assert np.allclose(
            np.diff(lons), dlons * np.ones(np.size(lons) - 1), atol=1e-4, rtol=0
        ), "provided longitude coordinate must be uniformly spaced"

        is_longitude_extent_in_data = (
            False  # This boolean checks if the 360 + i adjustment isn't found
        )
        for i in range(-1, 2, 1):
            if data[lon][0] <= central_longitude + 360 * i <= data[lon][-1]:
                is_longitude_extent_in_data = True

                ## Shifted version of target midpoint; e.g., could be -90 vs 270
                ## integer i keeps track of what how many multiples of 360 we need to shift entire
                ## grid by to match central_longitude
                _central_longitude = central_longitude + 360 * i

                ## Midpoint of the data
                central_data = data[lon][data[lon].shape[0] // 2].values

                ## Number of indices between the data midpoint and the target midpoint.
                ## Sign indicates direction needed to shift.
                shift = int(
                    -(data[lon].shape[0] * (_central_longitude - central_data))
                    // lon_span
                )

                ## Shift data so that the midpoint of the target domain is the middle of
                ## the data for easy slicing.
                new_data = data.roll({lon: 1 * shift}, roll_coords=True)

                ## Create a new longitude coordinate.
                ## We'll modify this to remove any seams (i.e., jumps like -270 -> 90)
                new_lon = new_data[lon].values.copy()

                ## Take the 'seam' of the data, and either backfill or forward fill based on
                ## whether the data was shifted F or west
                if shift > 0:
                    new_seam_index = shift

                    new_lon[0:new_seam_index] -= 360

                if shift < 0:
                    new_seam_index = data[lon].shape[0] + shift

                    new_lon[new_seam_index:] += 360

                ## new_lon is used to re-centre the midpoint to match that of target domain
                new_lon -= i * 360

                new_data = new_data.assign_coords({lon: new_lon})

                ## Choose the number of lon points to take from the middle, including a buffer.
                ## Use this to index the new global dataset
                num_lonpoints = abs(
                    int(
                        int(
                            data[lon].shape[0]
                            * (central_longitude - longitude_extent[0])
                        )
                        // lon_span
                    )
                )

        if not is_longitude_extent_in_data:
            raise ValueError(
                "The longitude of the data doesn't seem to include the longitude of the grid."
            )

        data = new_data.isel(
            {
                lon: slice(
                    data[lon].shape[0] // 2 - num_lonpoints,
                    data[lon].shape[0] // 2 + num_lonpoints,
                )
            }
        )

    return data

def replacement_era5grib(starttime, endtime,outdir,region=None,tasks = 52):
    client = Client(n_workers = tasks)

    ## Constants
    era5_vars = {
        "skt" : {"stype" : "single", "code" : 235},
        "sp" : {"stype" : "single", "code" : 134},
        "ci" : {"stype" : "single", "code" : 31},
        "sst" : {"stype" : "single", "code" : 34},
        "sd" : {"stype" : "single", "code" : 141},
        "stl1" : {"stype" : "single", "code" : 139},
        "stl2" : {"stype" : "single", "code" : 170},
        "stl3" : {"stype" : "single", "code" : 183},
        "stl4" : {"stype" : "single", "code" : 236},
        "swvl1" : {"stype" : "single", "code" : 39},
        "swvl2" : {"stype" : "single", "code" : 40},
        "swvl3" : {"stype" : "single", "code" : 41},
        "swvl4" : {"stype" : "single", "code" : 42},
        "u" : {"stype" : "pressure", "code" : 131},
        "v" : {"stype" : "pressure", "code" : 132},
        "t" : {"stype" : "pressure", "code" : 130},
        "q" : {"stype" : "pressure", "code" : 133},
        "lsm" : {"stype" : "single", "code" : 172},
        "z" : {"stype" : "single", "code" : 129}
        }

    ERADIR = "/g/data/rt52/era5/"

    ### HELPER FUNCTIONS FOR PARALLEL PROCESSING
    def write_single_file(ds,outdir):
        """
        Write a single hour of ERA5 data with correct name to pass to the cdo command later
        """
        time_string = datetime.strftime(
        ds.time[0].values.astype('datetime64[s]').astype(object), # Converts to a datetime object so we can format string
        '%Y%m%d%H%M.t+000'
        )
        ds.to_netcdf(Path(outdir) / f"raw_{time_string}.nc")
        return time_string

    def preprocess_subset(ds):
        """
        Preprocess data on xarray_mfdataset read to subset each file before it's read in
        """
        return longitude_slicer(
            ds.sel(latitude = slice(region[3],region[2]), time = slice(starttime,endtime)),
            [region[0],region[1]],
            "longitude"
            )

    def preprocess(ds):
        """
        Same but with no spatial subsetting
        """
        return ds.isel(time = slice(starttime,endtime))

    # Function to call in parallel with dask
    def run_cdo(command):
        """
        Wrapper for running the cdo command in parallel
        """
        subprocess.run(
            command,
            shell=True
        )


    ### END HELPER FUNCTIONS ###

    era5_months = [starttime.month]
    era5_years = [starttime.year]

    # If the time period straddles more than one month, add this month to read
    if (
        (starttime.month + 1 == endtime.month and starttime.year == endtime.year) or
        (starttime.month == endtime.month + 11 and starttime.year + 1 == endtime.year) # Account for December -> Jan
    ):
        era5_months.append(endtime.month)
        era5_years.append(endtime.year)

        # Raise an error otherwise. Making boundary conditions with more than a months worth of data would 
        # result in incredibly huge files anyway - not something we intend to support.
    elif starttime.month == endtime.month and starttime.year == endtime.year:
        pass

    else:
        raise ValueError(f"Starttime of {starttime} can only be a maximum of 1 month ahead of endtime {endtime}.")

    # Get a list of all files we'll read across every month and variable

    era5_files = []
    for month,year in  zip(era5_months,era5_years):
        for var in era5_vars:
            stype, eccode = era5_vars[var]["stype"], era5_vars[var]["code"]
            filedir = ERADIR + f"/{stype}-levels/reanalysis/{var}/{year}/"
            yyyymm = "%4.4d%2.2d"%(year, month)
            era5_files.append(next(Path(filedir).glob(f"*_{yyyymm}*")))


    tick("Starting to lazily load data")
    # Load all files and subset in memory to just your time and region
    if region == None:
        alldata = xr.open_mfdataset(
            era5_files,
            compat="override",
            coords = "minimal",
            preprocess = preprocess,
            parallel=True,
            chunks = {"time":"auto","longitude":-1,"latitude":-1}).rename({"siconc":"ci"}).persist()
    else:
        # Note that slicing of latitude is backwards since era5 data is upside down
        alldata = xr.open_mfdataset(
            era5_files,
            compat="override",
            coords = "minimal",
            preprocess = preprocess_subset,
            parallel=True,
            chunks = {"time":"auto","longitude":-1,"latitude":-1}).rename({"siconc":"ci"}).persist()

    tick("Loading all data into memory in parallel...")
    wait(alldata)
    tick("Data loaded. Now writing a .nc file for each hour")
    # Do some metadata stuff here
    for var in era5_vars:
        alldata[var].attrs["code"] = era5_vars[var]["code"]
        alldata[var].attrs["table"] = 128


    ## Save all of the individual raw netcdf files before grib conversion. 
    time_strings = []
    write_nc_tasks = []
    for i in range(len(alldata.time)):
        write_nc_tasks.append(
            dask.delayed(write_single_file)(alldata.isel(time = [i]),outdir)
        )

    time_strings = dask.compute(write_nc_tasks)[0]

    ## Perform all the grib conversion in parallel on each file
    cdo_command_tasks = []
    for t in time_strings:
        cdo_command_tasks.append(f"cdo -L --eccodes -f grb1 copy {str(outdir)}/raw_{t}.nc {str(outdir)}/ec_grib_{t}")

    tick("All files written. Now performing cdo conversion")

    dask.compute(
        [dask.delayed(run_cdo)(command) for command in cdo_command_tasks]
    )
    tick("cdo conversion finished.")

    return


def main():
    """
    The main function that creates a worker pool and generates single GRIB files 
    for requested date/times in parallel.

    Parameters
    ----------
    None.  The arguments are given via the command-line

    Returns
    -------
    None.  The GRIB file are written to the output directory
    """


    # Parse the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path,help = "output path")
    parser.add_argument('--start', required=True, type=pandas.to_datetime,help="Date of first output file")
    parser.add_argument('--count', default=1, type=int,
                        help="Number of files to create, one every `freq`")
    parser.add_argument('--freq', default=60*60, type=lambda x: int(x),help="How often to save grib files in seconds. Default: 3600 (hourly)")
    parser.add_argument('--region', default=None, type=str,
                         help="Optional lon1,lon2,lat1,lat2 bounding box to subset "
                              "with CDO's sellonlatbox before conversion. "
                              "Default: no subsetting (global domain).")
    parser.add_argument('--tasks', default=52, type=int,
                         help="Number of worker processes to run in parallel. "
                              "Default: 4.")
    args = parser.parse_args() 

    # Parse the lat/lon region
    if args.region:
        parts = [p.strip() for p in args.region.split(",")]
        if len(parts) != 4:
            raise argparse.ArgumentTypeError(
                "--region expects four comma-separated values: lon1,lon2,lat1,lat2"
            )
        try:
            region = tuple(float(p) for p in parts)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "--region values must be numeric: lon1,lon2,lat1,lat2"
            )
    else:
        # Default case just slices Nones down the track in xarray
        region = (None,None,None,None)


    # Create a list of the requested date/times
    starttime = args.start
    endtime = starttime + timedelta(seconds = args.count * args.freq)

    # Pass to the era5 -> grib function
    replacement_era5grib(starttime, endtime,args.output,region,args.tasks)
    return

if __name__ == '__main__':
    main()

