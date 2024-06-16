# Data manipulation packages
import numpy as np
import pandas as pd

# Raster manipulation
import rioxarray as rxr

# Sort out stuck issue with pytorch
import os


def array_creation(data_array, value, switch=False):
    """
    @Definition:
                A function to create an array of coordinate
    @References:
                None
    @Arguments:
                data_array (array):
                                        Original array with full data
                value (string):
                                        Name of value - 'x' or 'y'
                switch (boolean):
                                        Switch the shape of array (shape x,y or shape y,x)
    @Returns:
                new_array (array):
                                        An array of coordinate
    """
    # Read x, y values into arrays
    arr_x = data_array.x.values
    arr_y = data_array.y.values

    if switch:
        # Create zero array
        new_array = np.zeros((arr_y.shape[0], arr_x.shape[0]))
    else:
        # Create zero array
        new_array = np.zeros((arr_x.shape[0], arr_y.shape[0]))

    # Get full number of x or y values
    if value == "x":
        for i in range(arr_x.shape[0]):
            for j in range(arr_y.shape[0]):
                new_array[j, i] = arr_x[i]
        return new_array

    else:
        for i in range(arr_x.shape[0]):
            for j in range(arr_y.shape[0]):
                new_array[j, i] = arr_y[j]
        return new_array


def xyz_array(
    dataset_z_rxr,
    switch=False,
    z=False,
    no_value=False
):
    """
    @Definition:
                A function to create an array of coordinate
    @References:
                None
    @Arguments:
                dataset_z_rxr (array):
                                    Rioxarray array that contains elevation or flowdepth values including padding
    @Returns:
                new_array (array):
                                        An array of coordinate
    """

    # Create full number of values of x, y, z coordinates
    array_x = array_creation(dataset_z_rxr, 'x', switch)
    array_y = array_creation(dataset_z_rxr, 'y', switch)
    if z:
        array_z = dataset_z_rxr.z.values
    elif no_value:
        array_z = dataset_z_rxr.values
    else:
        array_z = dataset_z_rxr.isel(band=0).values

    # Flatten x, y, z arrays
    flatten_x = array_x.flatten()
    flatten_y = array_y.flatten()
    flatten_z = array_z.flatten()

    # Put all x, y, z into one array
    full_dataset = np.vstack((flatten_x, flatten_y, flatten_z)).transpose()

    return full_dataset


def raster_conversion(
    x_func, y_func, z_func
):
    """
    @Definition:
                A function to convert !D dimension of array into raster array.
                Particularly, convert 1D x, 1D y, and 1D z into multi-dimensional arrays of x, y, z
    @References:
                https://stackoverflow.com/questions/41897544/make-a-contour-plot-by-using-three-1d-arrays-in-python
                https://stackoverflow.com/questions/41815079/pandas-merge-join-two-data-frames-on-multiple-columns
    @Arguments:
                x_func, y_func, z_func (arrays):
                                            1D x, 1D y, 1D z datasets
    @Returns:
                x_func, y_func, z_func (arrays):
                                            Multi-dimensional arrays of x, y, z
    """
    # Gather x, y, z datasets into a pandas dataframe (DATAFRAME of ACTUAL DATA)
    pd_dataframe_actual = pd.DataFrame(dict(
        x=x_func,
        y=y_func,
        z=z_func
    ))

    # Assign dataframe column names into variables
    xcol, ycol, zcol = 'x', 'y', 'z'

    # Sort actual dataframe according to x then y values
    pd_dataframe_sorted = pd_dataframe_actual.sort_values(by=[xcol, ycol])

    # Getting values of x, y, z under raster array format
    # unique() function is used to remove duplicates
    x_values_func = pd_dataframe_sorted[xcol].unique()
    y_values_func = pd_dataframe_sorted[ycol].unique()
    z_values_func = pd_dataframe_sorted[zcol].values.reshape(len(x_values_func), len(y_values_func)).T

    return x_values_func, y_values_func, z_values_func


def value_change(shapefile_func, file_need_changing_func, value_func, inside=True):
    """
    @Definition:
                A function to change pixel values inside or outside polygons
    @References:
                https://corteva.github.io/rioxarray/html/rioxarray.html
                https://corteva.github.io/rioxarray/stable/examples/convert_to_raster.html
                https://gis.stackexchange.com/questions/414194/changing-raster-pixel-values-outside-of-polygon-box-using-rioxarray
                https://automating-gis-processes.github.io/CSC/notebooks/L2/geopandas-basics.html
                https://corteva.github.io/rioxarray/stable/getting_started/nodata_management.html
                https://gdal.org/programs/gdal_translate.html
                https://gis.stackexchange.com/questions/390438/replacing-nodata-values-by-a-constant-using-gdal-cli-tools
    @Arguments:
                shapefile_func (polygon):
                                            Polygon boundaries
                file_need_changing_func (string):
                                            File contains values that need changing
                value_func (int or float):
                                            Values used to replace
                inside (boolean):
                                            If True, change values inside, else, change values outside
    @Returns:
                None.
    """
    # Set up value changing command
    if inside:
        # Change values inside polygons
        inside_command = fr"gdal_rasterize -burn {value_func} {shapefile_func} {file_need_changing_func}"
        os.system(inside_command)
    else:
        # Change values outside polygons
        outside_command = fr"gdal_rasterize -i -burn {value_func} {shapefile_func} {file_need_changing_func}"
        os.system(outside_command)



# Ref: https://stackoverflow.com/questions/16856788/slice-2d-array-into-smaller-2d-arrays
def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))



# Ref: https://stackoverflow.com/questions/16873441/form-a-big-2d-array-from-multiple-smaller-2d-arrays/16873755#16873755
def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))



def zo_to_n(path_in, path_out, H):
    """
    @Definition:
                A function to create a raster file of Manning's n from roughness length
    @References:
                https://doi.org/10.1080/15715124.2017.1411923
    @Arguments:
                number_simulation (string):
                                            A string to identify the order of simulation (should be angle, x, y)
                H (int):
                                            Value of depth
    @Returns:
                A raster of Manning's n
    """
    # Extract roughness length
    zo = rxr.open_rasterio(
        fr"{path_in}"
    )

    # Convert roughness length (zo) to Manning's n
    manning_n = (0.41 * (H ** (1 / 6)) * ((H / zo) - 1)) / (np.sqrt(9.80665) * (1 + (H / zo) * (np.log(H / zo) - 1)))

    # Calibrate manning's n
    new_manning_n = manning_n * 1.7

    # Fill nan
    new_manning_n_interpolation = new_manning_n.interpolate_na(dim="x", method="linear")

    # Write out Manning's n
    new_manning_n_interpolation.rio.to_raster(
        fr"{path_out}"
    )

    return new_manning_n_interpolation.values.flatten()

