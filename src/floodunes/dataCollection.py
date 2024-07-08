# Raster manipulation packages
import rioxarray as rxr
from scipy.interpolate import griddata
import xarray as xr
import richdem as rd # for slope

# Data manipulation packages
import numpy as np
import pandas as pd

# Path package
import pathlib

# Image packages
from skimage import feature
from scipy.ndimage import generate_binary_structure
from scipy import ndimage
from xrspatial import proximity

# Other packages
from .rasterArray import xyz_array, zo_to_n


class dataCollection:
    def __init__(
            self,
            general_folder,
            geometry_domain_list,
            path_elev,
            path_wd,
            path_wse,
            path_proportion=None,
            path_manning=None,
            path_roughness=None,
            path_sd=None
    ):
        # Create a folder
        pathlib.Path(f"{general_folder}").mkdir(parents=True, exist_ok=True)
        self.general_folder = general_folder

        # Geometry domain for all inputs
        geometry_domain = [{
            'type': 'Polygon',
            'coordinates': [[
                [geometry_domain_list[0], geometry_domain_list[1]],
                [geometry_domain_list[0], geometry_domain_list[3]],
                [geometry_domain_list[2], geometry_domain_list[3]],
                [geometry_domain_list[2], geometry_domain_list[1]],
                [geometry_domain_list[0], geometry_domain_list[1]]
            ]]
        }]
        self.geometry_domain = geometry_domain

        # Elevation
        dem = rxr.open_rasterio(
            fr"{path_elev}"
        )
        dem.rio.write_crs(2193, inplace=True)
        dem_domain = dem.z.rio.clip(self.geometry_domain)
        dem_domain.rio.write_nodata(-9999)
        dem_domain.attrs['_FillValue'] = -9999
        dem_domain.rio.to_raster(fr"{self.general_folder}/dem_input_domain.nc")
        self.dem_domain = dem_domain

        # Water depth
        wd = rxr.open_rasterio(
            fr"{path_wd}"
        )
        wd.rio.write_crs(2193, inplace=True)
        wd_domain_origin = wd.rio.clip(self.geometry_domain)
        wd_domain_filter = wd_domain_origin.where(
            wd_domain_origin >= 0.1,
            other=0
        )
        wd_domain_filter.rio.write_nodata(-9999)
        wd_domain_filter.rio.to_raster(fr"{self.general_folder}/wd_input_domain.nc")
        self.wd_domain = wd_domain_filter

        # Water surface elevation
        wse = rxr.open_rasterio(
            fr"{path_wse}"
        )
        wse.rio.write_crs(2193, inplace=True)
        wse_domain_origin = wse.rio.clip(self.geometry_domain)
        wse_domain = wse_domain_origin.where(
            self.wd_domain >= 0.1,
            other=-9999
        )
        wse_domain.rio.write_nodata(-9999)
        wse_domain.rio.to_raster(fr"{self.general_folder}/wse_input_domain.nc")
        self.wse_domain = wse_domain

        # Proportion
        self.path_proportion = path_proportion
        if self.path_proportion != None:
            proportion = rxr.open_rasterio(
                fr"{self.path_proportion}"
            )
            proportion.rio.write_crs(2193, inplace=True)
            proportion_domain = proportion.rio.clip(self.geometry_domain)
            proportion_domain.rio.write_nodata(-9999)
            proportion_domain.rio.to_raster(fr"{self.general_folder}/proportion_domain.nc")
            self.proportion_domain = proportion_domain
        else:
            pass

        # Manning
        self.path_manning = path_manning
        if self.path_manning != None:
            manning = rxr.open_rasterio(
                fr"{self.path_manning}"
            )
            manning.rio.write_crs(2193, inplace=True)
            manning_domain = manning.rio.clip(self.geometry_domain)
            manning_domain.rio.write_nodata(-9999)
            manning_domain.rio.to_raster(fr"{self.general_folder}/manning_input_domain.nc")
            self.manning_domain = manning_domain
        else:
            pass

        # Roughness
        self.path_roughness = path_roughness
        if self.path_roughness != None:
            roughness = rxr.open_rasterio(
                fr"{self.path_roughness}"
            )
            roughness.rio.write_crs(2193, inplace=True)
            roughness_domain = roughness.zo.rio.clip(self.geometry_domain)
            roughness_domain.rio.write_nodata(-9999)
            roughness_domain.rio.to_raster(fr"{self.general_folder}/roughness_input_domain.nc")
            self.roughness_domain = roughness_domain

        else:
            pass

        # SD
        self.path_sd = path_sd
        if self.path_sd != None:
            sd = rxr.open_rasterio(
                fr"{self.path_sd}"
            )
            sd.rio.write_crs(2193, inplace=True)
            sd_domain = sd.rio.clip(self.geometry_domain)
            sd_domain.rio.write_nodata(-9999)
            sd_domain.rio.to_raster(fr"{self.general_folder}/sd_domain.nc")
            self.sd_domain = sd_domain
        else:
            pass



    def dem_input(self):

        # Get x, y, z values from dem
        elev_arr = xyz_array(self.dem_domain, switch=True, no_value=True)

        # Put them into pandas dataframe
        elev_dict = {
            "x": elev_arr[:, 0],
            "y": elev_arr[:, 1],
            "elev": elev_arr[:, 2]
        }

        # Convert into dataframe
        elev_df = pd.DataFrame(data=elev_dict)

        # Write out tiff file for other input generation
        self.dem_domain.rio.write_crs('epsg:2193', inplace=True)
        self.dem_domain.rio.write_nodata(-9999)
        self.dem_domain.rio.to_raster(fr"{self.general_folder}/for_richdem.tiff")

        return elev_df

    def depth_input(self, sd=False):

        if sd == False:
            return self.wd_domain.values.flatten()

        else:
            # Get x, y, z values from dem
            elev_arr = xyz_array(self.dem_domain, switch=True, no_value=True)

            # Put them into pandas dataframe
            depth_dict = {
                "x": elev_arr[:, 0],
                "y": elev_arr[:, 1],
                "depth": self.wd_domain.values.flatten()
            }

            # Convert into dataframe
            depth_df = pd.DataFrame(data=depth_dict)

            return depth_df



    def hanf_input(self):

        # Filter wse where the wd is larger than 0.1
        wse_filter = self.wse_domain.where(
            self.wd_domain >= 0.1,
            other=np.nan
        )

        # Convert into xyz columns
        wse_array = xyz_array(wse_filter, True)

        # Filter values (rows of array) not nan
        wse_notnan = wse_array[~np.isnan(wse_array).any(axis=1), :]

        # Filter values (rows of array) nan
        wse_nan = wse_array[np.isnan(wse_array).any(axis=1) == True]

        # Interpolate using scipy griddata
        interpolated_z = griddata(
            points = wse_notnan[:, 0:2],            # Points in 2d array, first column is x and second column is y
            values = wse_notnan[:, 2],              # Point values for each xy coordinates (same shape as above)
            xi=(wse_nan[:, 0], wse_nan[:, 1]),      # xy coordinates of points needing interpolated
            method='nearest'
        )

        # Create a new 3D array for nan values that have just been interpolated
        wse_interpolated_nan = np.vstack((
            wse_nan[:, 0],
            wse_nan[:, 1],
            interpolated_z
        )).transpose()

        # Create two dataframes two merge two 3d arrays to get the full z values
        # Dataframe 1: from data array used for interpolation
        data_array_usedforinterpolation = {
            'x': wse_array[:, 0],
            'y': wse_array[:, 1],
            'z': wse_array[:, 2]
        }
        # Create dataframe1
        df_usedforinterpolation = pd.DataFrame(data=data_array_usedforinterpolation)

        # Dataframe 2: from interpolated data array
        data_array_interpolated = {
            'x': wse_interpolated_nan[:, 0],
            'y': wse_interpolated_nan[:, 1],
            'z': wse_interpolated_nan[:, 2]
        }
        # Create dataframe2
        df_interpolated = pd.DataFrame(data=data_array_interpolated)

        # Merge two dataframes
        combined_df = pd.merge(df_usedforinterpolation, df_interpolated,
                               left_on=['x', 'y'], right_on=['x', 'y'],
                               how='left')  # There will be 4 columns in the results, 'x', 'y', 'z_x' with NaN, 'z_y' with NaN. We need to combine z_x and z_y to remove NaNs
        combined_df.z_x.fillna(combined_df.z_y,
                               inplace=True)  # Combine two 'z_x' and 'z_y' columns for removing NaNs. Ref: https://stackoverflow.com/questions/29177498/python-pandas-replace-nan-in-one-column-with-value-from-corresponding-row-of-sec

        # Change combined_df to new dataframe with only 'x', 'y', new 'z'
        new_df = combined_df[['x', 'y', 'z_x']].copy(
            deep=True)  # Create a new dataframe by selected columns from another dataframe. Ref: https://stackoverflow.com/questions/34682828/extracting-specific-selected-columns-to-new-dataframe-as-a-copy
        new_df.rename(columns={'z_x': 'z'}, inplace=True)  # Rename 'z_x' column into 'z'

        # Sort values
        new_df_sort = new_df.sort_values(by=['x', 'y'])  # Sort. Be careful!!!! This method needs sorting

        # Get raster values
        x_raster = new_df_sort.x.unique()  # Get x coordinates
        y_raster = new_df_sort.y.unique()  # Get y coordinates
        z_raster = new_df_sort.z.to_numpy().reshape(len(x_raster), len(y_raster)).T  # Get z values

        # Write to interpolated raster array
        interpolated_raster_array = xr.DataArray(
            data=z_raster,
            dims=['y', 'x'],
            coords={
                'x': (['x'], x_raster),
                'y': (['y'], y_raster)
            }
        )

        # Set up crs and nodata
        interpolated_raster_array.rio.write_crs('epsg:2193', inplace=True)

        # Get hanf
        hanf = self.dem_domain - interpolated_raster_array

        # Create hanf array
        hanf_arr = xr.DataArray(
            data=hanf.values[0], # as after clipping, actual cell will give only 0
            dims=['y', 'x'],
            coords={
                'x': (['x'], self.dem_domain.x.values),
                'y': (['y'], self.dem_domain.y.values)
            },
            attrs=self.dem_domain.attrs
        )

        hanf_arr.rio.write_crs("epsg:2193", inplace=True)
        hanf_arr.rio.write_nodata(-9999)
        hanf_arr.rio.to_raster(fr"{self.general_folder}/hanf_input_domain.nc")

        # Get x, y, z values again (not use the above one, just to make sure)
        hanf_xyz = xyz_array(hanf_arr, True, False, True)

        return hanf_xyz[:, 2]


    # INPUTS
    def depthlabel_input(self):

        # Depth label
        depthlabel_value = self.wd_domain.values.copy()
        depthlabel_value[(depthlabel_value > 0)] = 1

        # Create array
        depthlabel_arr = xr.DataArray(
            data=depthlabel_value[0],
            dims=['y', 'x'],
            coords={
                'x': (['x'], self.dem_domain.x.values),
                'y': (['y'], self.dem_domain.y.values)
            },
            attrs=self.dem_domain.attrs
        )

        # Write out
        depthlabel_arr.rio.write_crs(2193)
        depthlabel_arr.rio.write_nodata(-9999)
        depthlabel_arr.rio.to_raster(fr"{self.general_folder}/depthlabel_input_domain.nc")

        return depthlabel_value.flatten()



    def slope_input(self, type='slope_degrees'):

        # Read DEM
        dem_slope = rd.LoadGDAL(fr"{self.general_folder}/for_richdem.tiff")

        # Get slope
        slope = rd.TerrainAttribute(dem_slope, attrib=type)

        # Reshape the slope
        slope_reshape = slope.reshape(-1, slope.shape[0], slope.shape[1])

        # Create array
        slope_arr = xr.DataArray(
            data=slope_reshape[0],
            dims=['y', 'x'],
            coords={
                'x': (['x'], self.dem_domain.x.values),
                'y': (['y'], self.dem_domain.y.values)
            },
            attrs=self.dem_domain.attrs
        )

        # Write out
        slope_arr.rio.write_crs(2193)
        slope_arr.rio.write_nodata(-9999)
        slope_arr.rio.to_raster(fr"{self.general_folder}/slope_input_domain.nc")

        return slope_reshape.flatten()



    def manning_input(self):

        if self.path_manning == None:
            # Convert roughness to manning
            manning_domain = zo_to_n(
                fr"{self.general_folder}/roughness_input_domain.nc",
                fr"{self.general_folder}/manning_input_domain.nc",
                1
            )

        else:
            manning_domain = self.manning_domain.values.flatten()

        return manning_domain


    def roughness_input(self):

        if self.path_roughness != None:

            return self.roughness_domain.values.flatten()
        else:
            pass



    def floodproximity_input(self):

        # Copy array values
        flood_proximity_data = self.wd_domain.values[0].copy()

        # Filter
        flood_proximity_data[flood_proximity_data < 0.1] = 0

        # Create array
        flood_proximity_arr = xr.DataArray(
            data=flood_proximity_data,
            dims=['y', 'x'],
            coords={
                'x': (['x'], self.dem_domain.x.values),
                'y': (['y'], self.dem_domain.y.values)
            },
            attrs=self.dem_domain.attrs
        )

        # Calculate proximity
        flood_proximity = proximity(flood_proximity_arr)

        # Write out
        flood_proximity.rio.write_crs(2193)
        flood_proximity.rio.write_nodata(-9999)
        flood_proximity.rio.to_raster(fr"{self.general_folder}/floodproximity_input_domain.nc")

        return flood_proximity.values.flatten()



    def sobeledgelabel_input(self):

        # Depth label
        depthlabel_value = self.wd_domain.values.copy()
        depthlabel_value[(depthlabel_value > 0)] = 1

        # Get depth
        sobel = depthlabel_value[0]

        # Calculate sobel edge
        sobel_h = ndimage.sobel(sobel, 0) # horizontal gradient
        sobel_v = ndimage.sobel(sobel, 1) # vertical gradient
        magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
        magnitude *= 255 / np.max(magnitude)

        # Create array
        sobel_arr = xr.DataArray(
            data=magnitude,
            dims=['y', 'x'],
            coords={
                'x': (['x'], self.wd_domain.x.values),
                'y': (['y'], self.wd_domain.y.values)
            },
            attrs=self.dem_domain.attrs
        )

        # Write out
        sobel_arr.rio.write_crs(2193)
        sobel_arr.rio.write_nodata(-9999)
        sobel_arr.rio.to_raster(fr"{self.general_folder}/sobeledgelabel_input_domain.nc")

        return magnitude.flatten()


    def sobeledgevalue_input(self):

        # Depth value
        depth_value = self.wd_domain.values.copy()

        # Get depth only
        sobel = depth_value[0]

        # Calculate sobel edge
        sobel_h = ndimage.sobel(sobel, 0)  # horizontal gradient
        sobel_v = ndimage.sobel(sobel, 1)  # vertical gradient
        magnitude = np.sqrt(sobel_h ** 2 + sobel_v ** 2)
        magnitude *= 255 / np.max(magnitude)  # normalization

        # Create array
        sobel_arr = xr.DataArray(
            data=magnitude,
            dims=['y', 'x'],
            coords={
                'x': (['x'], self.wd_domain.x.values),
                'y': (['y'], self.wd_domain.y.values)
            },
            attrs=self.dem_domain.attrs
        )

        # Write out
        sobel_arr.rio.write_crs(2193)
        sobel_arr.rio.write_nodata(-9999)
        sobel_arr.rio.to_raster(fr"{self.general_folder}/sobeledgevalue_input_domain.nc")

        return magnitude.flatten()


    def curvature_input(self):

        # Read DEM
        dem_curvature = rd.LoadGDAL(fr"{self.general_folder}/for_richdem.tiff")

        # Get curvature
        curvature = rd.TerrainAttribute(dem_curvature, attrib='curvature')

        # Reshape the curvature
        curvature_reshape = curvature.reshape(-1, curvature.shape[0], curvature.shape[1])

        # Create array
        curvature_arr = xr.DataArray(
            data=curvature_reshape[0],
            dims=['y', 'x'],
            coords={
                'x': (['x'], self.dem_domain.x.values),
                'y': (['y'], self.dem_domain.y.values)
            },
            attrs=self.dem_domain.attrs
        )

        # Write out
        curvature_arr.rio.write_crs(2193)
        curvature_arr.rio.write_nodata(-9999)
        curvature_arr.rio.to_raster(fr"{self.general_folder}/curvature_input_domain.nc")

        return curvature_reshape.flatten()



    def flowaccumulation_input(self):

        # Read DEM
        dem_flow_accumulation = rd.LoadGDAL(fr"{self.general_folder}/for_richdem.tiff")

        # Get flow accumulation
        flow_accumulation = rd.FlowAccumulation(dem_flow_accumulation, method='D4')

        # Reshape the curvature
        flow_accumulation_reshape = flow_accumulation.reshape(-1, flow_accumulation.shape[0], flow_accumulation.shape[1])

        # Create array
        flow_accumulation_arr = xr.DataArray(
            data=flow_accumulation_reshape[0],
            dims=['y', 'x'],
            coords={
                'x': (['x'], self.dem_domain.x.values),
                'y': (['y'], self.dem_domain.y.values)
            },
            attrs=self.dem_domain.attrs
        )

        # Write out
        flow_accumulation_arr.rio.write_crs(2193)
        flow_accumulation_arr.rio.write_nodata(-9999)
        flow_accumulation_arr.rio.to_raster(fr"{self.general_folder}/flowaccumulation_input_domain.nc")

        return flow_accumulation_reshape.flatten()


    def proportionlabel_train_input(self, fortrain=True):

        if self.path_proportion != None:

            # Get proportion values
            proportion_label = self.proportion_domain.values.copy()

            if fortrain == True:
                # Convert values < 100 to 1
                proportion_label[(proportion_label>0) & (proportion_label<100)] = 1
                # Convert values = 100 to 2
                proportion_label[proportion_label==100] = 2
                # Convert values -9999 to 0
                proportion_label[proportion_label<0] = 0
            else:
                pass

            # Create array
            proportion_label_arr = xr.DataArray(
                data=proportion_label[0],
                dims=['y', 'x'],
                coords={
                    'x': (['x'], self.dem_domain.x.values),
                    'y': (['y'], self.dem_domain.y.values)
                },
                attrs=self.dem_domain.attrs
            )

            # Write out
            proportion_label_arr.rio.write_crs(2193)
            proportion_label_arr.rio.write_nodata(-9999)
            proportion_label_arr.rio.to_raster(fr"{self.general_folder}/proportionlabel_regression_input_domain.nc")

            return proportion_label.flatten()

        else:
            pass

    def proportionproximity_input(self):

        if self.path_proportion != None:

            # Get proportion values
            proportion_label = self.proportion_domain.values.copy()

            # Replace -9999 with 0
            proportion_label_no0 = proportion_label.copy()
            proportion_label_no0[proportion_label_no0 == -9999] = 0

            # Create array
            proportion_label_no0_arr = xr.DataArray(
                data=proportion_label_no0[0],
                dims=['y', 'x'],
                coords={
                    'x': (['x'], self.dem_domain.x.values),
                    'y': (['y'], self.dem_domain.y.values)
                },
                attrs=self.dem_domain.attrs
            )

            # Proximity
            proportion_label_proximity = proximity(proportion_label_no0_arr)

            # Write out
            proportion_label_proximity.rio.write_crs(2193)
            proportion_label_proximity.rio.write_nodata(-9999)
            proportion_label_proximity.rio.to_raster(
                fr"{self.general_folder}/proportionproximity_regression_input_domain.nc"
            )

            return proportion_label_proximity.values.flatten()
        else:
            pass

    def proximitydifference_input(self):

        if self.path_proportion != None:

            # Copy array values
            flood_proximity_data = self.wd_domain.values[0].copy()

            # Filter
            flood_proximity_data[flood_proximity_data < 0.1] = 0

            # Create array
            flood_proximity_arr = xr.DataArray(
                data=flood_proximity_data,
                dims=['y', 'x'],
                coords={
                    'x': (['x'], self.dem_domain.x.values),
                    'y': (['y'], self.dem_domain.y.values)
                },
                attrs=self.dem_domain.attrs
            )

            # Calculate proximity
            flood_proximity = proximity(flood_proximity_arr)

            # Get proportion values
            proportion_label = self.proportion_domain.values.copy()

            # Replace -9999 with 0
            proportion_label_no0 = proportion_label.copy()
            proportion_label_no0[proportion_label_no0 == -9999] = 0

            # Create array
            proportion_label_no0_arr = xr.DataArray(
                data=proportion_label_no0[0],
                dims=['y', 'x'],
                coords={
                    'x': (['x'], self.dem_domain.x.values),
                    'y': (['y'], self.dem_domain.y.values)
                },
                attrs=self.dem_domain.attrs
            )

            # Proximity
            proportion_label_proximity = proximity(proportion_label_no0_arr)

            # Get difference
            proximity_difference = proportion_label_proximity - flood_proximity

            # Write out
            proximity_difference.rio.write_crs(2193)
            proximity_difference.rio.write_nodata(-9999)
            proximity_difference.rio.to_raster(fr"{self.general_folder}/proportionproximity_regression_input_domain.nc")

            return proximity_difference.values.flatten()

        else:
            pass


    def cannyedge_input(self):

        # Depth value
        depth_value = self.wd_domain.values.copy()

        # Get depth only
        depth_value_only = depth_value[0]

        # Compute the Canny filter for two values of sigma
        canny_edge = feature.canny(depth_value_only, sigma=2)

        # Put it into xarray
        canny_edge_arr = xr.DataArray(
            data=(canny_edge * 1).copy(),
            dims=['y', 'x'],
            coords={
                'x': (['x'], self.wd_domain.x.values),
                'y': (['y'], self.wd_domain.y.values)
            },
            attrs=self.dem_domain.attrs
        )

        # Write out
        canny_edge_arr.rio.write_crs(2193)
        canny_edge_arr.rio.write_nodata(-9999)
        canny_edge_arr.attrs['_FillValue'] = -9999
        canny_edge_arr.rio.to_raster(
            fr"{self.general_folder}/cannyedge_input_domain.nc", dtype=np.float64
        )


        return canny_edge_arr.values.flatten()


    def laplace_input(self):

        # Depth value
        depth_value = self.wd_domain.values.copy()

        # Get depth only
        depth_value_only = depth_value[0]

        # Get laplace
        laplace_magnitude = ndimage.laplace(depth_value_only)

        # Put it into xarray
        laplace_arr = xr.DataArray(
            data=laplace_magnitude,
            dims=['y', 'x'],
            coords={
                'x': (['x'], self.wd_domain.x.values),
                'y': (['y'], self.wd_domain.y.values)
            },
            attrs=self.dem_domain.attrs
        )

        # Write out
        laplace_arr.rio.write_crs(2193)
        laplace_arr.rio.write_nodata(-9999)
        laplace_arr.rio.to_raster(fr"{self.general_folder}/laplace_input_domain.nc")

        return laplace_magnitude.flatten()

    def morphologicallaplace_input(self):

        # Depth value
        depth_value = self.wd_domain.values.copy()

        # Get morphological laplace
        square = generate_binary_structure(rank=2, connectivity=3)
        morphological_laplace_conversion = ndimage.morphological_laplace(depth_value[0],
                                                                         structure=square)
        # Get depth data
        morphological_laplace_arr = xr.DataArray(
            data=morphological_laplace_conversion,
            dims=['y', 'x'],
            coords={
                'x': (['x'], self.wd_domain.x.values),
                'y': (['y'], self.wd_domain.y.values)
            },
            attrs=self.dem_domain.attrs
        )

        # Write out
        morphological_laplace_arr.rio.write_crs(2193)
        morphological_laplace_arr.rio.write_nodata(-9999)
        morphological_laplace_arr.rio.to_raster(fr"{self.general_folder}/morphologicallaplace_input_domain.nc")

        return morphological_laplace_conversion.flatten()

    def gaussiangradient_input(self):

        # Depth value
        depth_value = self.wd_domain.values.copy()

        # Get gaussian gradient
        gaussian_gradient_conversion = ndimage.gaussian_gradient_magnitude(depth_value[0],
                                                                           sigma=.3)

        # Put it into xarray
        gaussian_gradient_arr = xr.DataArray(
            data=gaussian_gradient_conversion,
            dims=['y', 'x'],
            coords={
                'x': (['x'], self.wd_domain.x.values),
                'y': (['y'], self.wd_domain.y.values)
            },
            attrs=self.dem_domain.attrs
        )

        # Write out
        gaussian_gradient_arr.rio.write_crs(2193)
        gaussian_gradient_arr.rio.write_nodata(-9999)
        gaussian_gradient_arr.rio.to_raster(fr"{self.general_folder}/gaussiangradient_input_domain.nc")

        return gaussian_gradient_conversion.flatten()



    # OUTPUTS
    def sd_output(self):

        if self.path_sd != None:
            # Get sd
            sd_values = self.sd_domain.values.copy()

            # Get sd only
            sd_values_only = sd_values[0]
            sd_values_only[sd_values_only == -9999] = 0

            # Convert sd
            sd_values_conversion = sd_values_only * 100

            # Put it into xarray
            sd_arr = xr.DataArray(
                data=sd_values_conversion,
                dims=['y', 'x'],
                coords={
                    'x': (['x'], self.wd_domain.x.values),
                    'y': (['y'], self.wd_domain.y.values)
                },
                attrs=self.dem_domain.attrs
            )

            # Write out
            sd_arr.rio.write_crs(2193)
            sd_arr.rio.write_nodata(-9999)
            sd_arr.rio.to_raster(fr"{self.general_folder}/sdconversion_onput_domain.nc")

            return sd_values_conversion.flatten()

        else:
            pass


    def proportion_output(self):

        if self.path_proportion != None:
            # Get proportion values
            proportion_values = self.proportion_domain.values.copy()

            # Change -9999 to 0 values
            proportion_values[proportion_values == -9999] = 0

            return proportion_values.flatten()
        else:
            pass


    def proportionlabel_output(self):

        if self.path_proportion != None:
            # Get proportion values
            proportion_label = self.proportion_domain.values.copy()

            # Convert values < 100 to 1
            proportion_label[(proportion_label>0) & (proportion_label<100)] = 1
            # Convert values = 100 to 2
            proportion_label[proportion_label==100] = 2
            # Convert values -9999 to 0
            proportion_label[proportion_label<0] = 0

            # Create array
            proportion_label_arr = xr.DataArray(
                data=proportion_label[0],
                dims=['y', 'x'],
                coords={
                    'x': (['x'], self.dem_domain.x.values),
                    'y': (['y'], self.dem_domain.y.values)
                },
                attrs=self.dem_domain.attrs
            )

            # Write out
            proportion_label_arr.rio.write_crs(2193)
            proportion_label_arr.rio.to_raster(fr"{self.general_folder}/proportionlabel_classification_output_domain.nc")

            return proportion_label.flatten()

        else:
            pass

    
    
    def loadpara_into_dataframe_classification(self, name_csv='train'):
        
        # Call out each para
        # DEM - include 3 columns - x, y, dem - so the next para will be counted from 3
        elev_IN = self.dem_input()
        # 3. HANF
        hanf_IN = self.hanf_input()
        # 4. Depth label
        depthlabel_IN = self.depthlabel_input()
        # 5. Slope
        slope_IN = self.slope_input()
        # 6. Manning
        manning_IN = self.manning_input()
        # 7. Flood proximity
        floodproximity_IN = self.floodproximity_input()
        # 8. Sobel edge label
        sobeledgelabel_IN = self.sobeledgelabel_input()
        # 9. Curvature
        curvature_IN = self.curvature_input()
        # 10. Flow accumulation
        flowaccumulation_IN = self.flowaccumulation_input()
        # 11. Proportion label
        proportionlabel_OUT = self.proportionlabel_output()

        # Create dataframe
        ml_df = elev_IN.copy(deep=True)
        # Get name list for each para. Start from 'hanf'
        para_names_list = [
            'hanf',
            'depthlabel',
            'slope',
            'manning',
            'floodproximity',
            'sobeledgelabel',
            'curvature',
            'flowaccumulation',
            'proportionlabel'
        ]
        # Get list of para values. Start from 'hanf'
        para_values_list = [
            hanf_IN,
            depthlabel_IN,
            slope_IN,
            manning_IN,
            floodproximity_IN,
            sobeledgelabel_IN,
            curvature_IN,
            flowaccumulation_IN,
            proportionlabel_OUT
        ]

        # Load all para into dataframe
        for i in range(len(para_values_list)):
            para_order = i + 3
            ml_df.insert(
                loc=para_order,
                column=para_names_list[i],
                value=para_values_list[i]
            )

        # Write out csv
        ml_df.to_csv(fr"{self.general_folder}/ml_full_{name_csv}_df_classification_proportion.csv", index=False)

        return ml_df

    def loadpara_into_dataframe_regression_proportion(self, name_csv='train'):

        # Call out each para
        # DEM - include 3 columns - x, y, dem - so the next para will be counted from 3
        elev_IN = self.dem_input()
        # 3. HANF
        hanf_IN = self.hanf_input()
        # 4. Depth
        depth_IN = self.depth_input()
        # 5. Slope
        slope_IN = self.slope_input('slope_riserun')*100
        # 6. Roughness
        roughness_IN = self.roughness_input()
        # 7. Curvature
        curvature_IN = self.curvature_input()*10
        # 8. Flow accumulation
        old_flowaccumulation_IN = self.flowaccumulation_input()
        flowaccumulation_IN = ((old_flowaccumulation_IN - 1)/(old_flowaccumulation_IN.max() - 1))*(20000 - 1) + 1

        # 9. Proportion label
        if name_csv == 'train':
            proportionlabel_IN = self.proportionlabel_train_input()
        else:
            proportionlabel_IN = self.proportionlabel_train_input(fortrain=False)
        # 10. Proportion proximity
        proportionproximity_IN = self.proportionproximity_input()
        # 11. Proximity difference
        proximitydifference_IN = self.proximitydifference_input()
        # 12. Proportion
        proportion_OUT = self.proportion_output()

        # Flood proximity for filtering out results
        floodproximity_IN = self.floodproximity_input()

        # Create dataframe
        ml_df = elev_IN.copy(deep=True)
        # Get name list for each para. Start from 'hanf'
        para_names_list = [
            'hanf',
            'depth',
            'slope',
            'roughness',
            'curvature',
            'flowaccumulation',
            'proportionlabel',
            'proportionproximity',
            'proximitydifference',
            'proportion'
        ]
        # Get list of para values. Start from 'hanf'
        para_values_list = [
            hanf_IN,
            depth_IN,
            slope_IN,
            roughness_IN,
            curvature_IN,
            flowaccumulation_IN,
            proportionlabel_IN,
            proportionproximity_IN,
            proximitydifference_IN,
            proportion_OUT
        ]

        # Load all para into dataframe
        for i in range(len(para_values_list)):
            para_order = i + 3
            ml_df.insert(
                loc=para_order,
                column=para_names_list[i],
                value=para_values_list[i]
            )

        # Write out csv
        ml_df.to_csv(fr"{self.general_folder}/ml_full_{name_csv}_df_regression_proportion.csv", index=False)

        return ml_df


    def loadpara_into_dataframe_regression_sd(self, name_csv='train'):

        # Call out each para
        elev_IN = self.dem_input() # This is too create richdem files
        # Depth includes 3 columns - x, y, depth - so the next para will be counted from 3
        depth_IN = self.depth_input(sd=True)
        # 3. Slope
        slope_IN = self.slope_input('slope_riserun')
        # 4. Manning
        manning_IN = self.manning_input()
        # 5. Sobel edge values
        sobeledgevalue_IN = self.sobeledgevalue_input()
        # 6. Canny edge
        cannyedge_IN = self.cannyedge_input()
        # 7. Laplace
        laplace_IN = self.laplace_input()
        # 8. Morphological laplace
        morphologicallaplace_IN = self.morphologicallaplace_input()
        # 9. Gaussian gradient
        gaussiangradient_IN = self.gaussiangradient_input()
        # 10. SD
        sd_OUT = self.sd_output()

        # Flood proximity for filtering out results
        floodproximity_IN = self.floodproximity_input()

        # Create dataframe
        ml_df = depth_IN.copy(deep=True)
        # Get name list for each para. Start from 'hanf'
        para_names_list = [
            'slope',
            'manning',
            'sobeledgevalue',
            'cannyedge',
            'laplace',
            'morphologicallaplace',
            'gaussiangradient',
            'sd'
        ]
        # Get list of para values. Start from 'hanf'
        para_values_list = [
            slope_IN,
            manning_IN,
            sobeledgevalue_IN,
            cannyedge_IN,
            laplace_IN,
            morphologicallaplace_IN,
            gaussiangradient_IN,
            sd_OUT
        ]

        # Load all para into dataframe
        for i in range(len(para_values_list)):
            para_order = i + 3
            ml_df.insert(
                loc=para_order,
                column=para_names_list[i],
                value=para_values_list[i]
            )

        # Write out csv
        ml_df.to_csv(fr"{self.general_folder}/ml_full_{name_csv}_df_regression_sd.csv", index=False)

        return ml_df



    def loadpara_into_dataframe_testextra(self,
                                          type,
                                          name_csv='testextra'):

        if type == 'classification_proportion':

            # Call out each para
            # DEM - include 3 columns - x, y, dem - so the next para will be counted from 3
            elev_IN = self.dem_input()
            # 3. HANF
            hanf_IN = self.hanf_input()
            # 4. Depth label
            depthlabel_IN = self.depthlabel_input()
            # 5. Slope
            slope_IN = self.slope_input()
            # 6. Manning
            manning_IN = self.manning_input()
            # 7. Flood proximity
            floodproximity_IN = self.floodproximity_input()
            # 8. Sobel edge label
            sobeledgelabel_IN = self.sobeledgelabel_input()
            # 9. Curvature
            curvature_IN = self.curvature_input()
            # 10. Flow accumulation
            flowaccumulation_IN = self.flowaccumulation_input()
            # 11. Proportion label
            proportionlabel_OUT = self.proportionlabel_output()

            # Flood proximity for filtering out results
            floodproximity_IN = self.floodproximity_input()

            # Create dataframe
            ml_df = elev_IN.copy(deep=True)
            # Get name list for each para. Start from 'hanf'
            para_names_list = [
                'hanf',
                'depthlabel',
                'slope',
                'manning',
                'floodproximity',
                'sobeledgelabel',
                'curvature',
                'flowaccumulation',
                'proportionlabel'
            ]
            # Get list of para values. Start from 'hanf'
            para_values_list = [
                hanf_IN,
                depthlabel_IN,
                slope_IN,
                manning_IN,
                floodproximity_IN,
                sobeledgelabel_IN,
                curvature_IN,
                flowaccumulation_IN,
                proportionlabel_OUT
            ]

        elif type == 'regression_proportion':

            # Call out each para
            # DEM - include 3 columns - x, y, dem - so the next para will be counted from 3
            elev_IN = self.dem_input()
            # 3. HANF
            hanf_IN = self.hanf_input()
            # 4. Depth
            depth_IN = self.depth_input()
            # 5. Slope
            slope_IN = self.slope_input('slope_riserun')
            # 6. Roughness
            roughness_IN = self.roughness_input()
            # 7. Curvature
            curvature_IN = self.curvature_input()
            # 8. Flow accumulation
            flowaccumulation_IN = self.flowaccumulation_input()
            # 9. Proportion label
            if name_csv == 'train':
                proportionlabel_IN = self.proportionlabel_train_input()
            else:
                proportionlabel_IN = self.proportionlabel_train_input(fortrain=False)
            # 10. Proportion proximity
            proportionproximity_IN = self.proportionproximity_input()
            # 11. Proximity difference
            proximitydifference_IN = self.proximitydifference_input()
            # 12. Proportion
            proportion_OUT = self.proportion_output()

            # Create dataframe
            ml_df = elev_IN.copy(deep=True)
            # Get name list for each para. Start from 'hanf'
            para_names_list = [
                'hanf',
                'depth',
                'slope',
                'roughness',
                'curvature',
                'flowaccumulation',
                'proportionlabel',
                'proportionproximity',
                'proximitydifference',
                'proportion'
            ]
            # Get list of para values. Start from 'hanf'
            para_values_list = [
                hanf_IN,
                depth_IN,
                slope_IN,
                roughness_IN,
                curvature_IN,
                flowaccumulation_IN,
                proportionlabel_IN,
                proportionproximity_IN,
                proximitydifference_IN,
                proportion_OUT
            ]

        else:

            # Call out each para
            elev_IN = self.dem_input() # For creating richdemte
            # Depth includes 3 columns - x, y, depth - so the next para will be counted from 3
            depth_IN = self.depth_input(sd=True)
            # 3. Slope
            slope_IN = self.slope_input('slope_riserun')
            # 4. Manning
            manning_IN = self.manning_input()
            # 5. Sobel edge values
            sobeledgevalue_IN = self.sobeledgevalue_input()
            # 6. Canny edge
            cannyedge_IN = self.cannyedge_input()
            # 7. Laplace
            laplace_IN = self.laplace_input()
            # 8. Morphological laplace
            morphologicallaplace_IN = self.morphologicallaplace_input()
            # 9. Gaussian gradient
            gaussiangradient_IN = self.gaussiangradient_input()
            # 10. SD
            sd_OUT = self.sd_output()

            # Create dataframe
            ml_df = depth_IN.copy(deep=True)
            # Get name list for each para. Start from 'hanf'
            para_names_list = [
                'slope',
                'manning',
                'sobeledgevalue',
                'cannyedge',
                'laplace',
                'morphologicallaplace',
                'gaussiangradient',
                'sd'
            ]
            # Get list of para values. Start from 'hanf'
            para_values_list = [
                slope_IN,
                manning_IN,
                sobeledgevalue_IN,
                cannyedge_IN,
                laplace_IN,
                morphologicallaplace_IN,
                gaussiangradient_IN,
                sd_OUT
            ]


        # Load all para into dataframe
        for i in range(len(para_values_list)):
            para_order = i + 3
            ml_df.insert(
                loc=para_order,
                column=para_names_list[i],
                value=para_values_list[i]
            )

        # Write out csv
        ml_df.to_csv(fr"{self.general_folder}/ml_full_{name_csv}_df_regression_sd.csv", index=False)

        return ml_df




    def loadpara_into_dataframe_estimate(self,
                                         type,
                                         name_csv='estimate'):

        if type == 'classification_proportion':

            # Call out each para
            # DEM - include 3 columns - x, y, dem - so the next para will be counted from 3
            elev_IN = self.dem_input()
            # 3. HANF
            hanf_IN = self.hanf_input()
            # 4. Depth label
            depthlabel_IN = self.depthlabel_input()
            # 5. Slope
            slope_IN = self.slope_input()
            # 6. Manning
            manning_IN = self.manning_input()
            # 7. Flood proximity
            floodproximity_IN = self.floodproximity_input()
            # 8. Sobel edge label
            sobeledgelabel_IN = self.sobeledgelabel_input()
            # 9. Curvature
            curvature_IN = self.curvature_input()
            # 10. Flow accumulation
            flowaccumulation_IN = self.flowaccumulation_input()

            # Create dataframe
            ml_df = elev_IN.copy(deep=True)
            # Get name list for each para. Start from 'hanf'
            para_names_list = [
                'hanf',
                'depthlabel',
                'slope',
                'manning',
                'floodproximity',
                'sobeledgelabel',
                'curvature',
                'flowaccumulation'
            ]
            # Get list of para values. Start from 'hanf'
            para_values_list = [
                hanf_IN,
                depthlabel_IN,
                slope_IN,
                manning_IN,
                floodproximity_IN,
                sobeledgelabel_IN,
                curvature_IN,
                flowaccumulation_IN
            ]

        elif type == 'regression_proportion':

            # Call out each para
            # DEM - include 3 columns - x, y, dem - so the next para will be counted from 3
            elev_IN = self.dem_input()
            # 3. HANF
            hanf_IN = self.hanf_input()
            # 4. Depth
            depth_IN = self.depth_input()
            # 5. Slope
            slope_IN = self.slope_input('slope_riserun')
            # 6. Roughness
            roughness_IN = self.roughness_input()
            # 7. Curvature
            curvature_IN = self.curvature_input()
            # 8. Flow accumulation
            flowaccumulation_IN = self.flowaccumulation_input()
            # 9. Proportion label
            if name_csv == 'train':
                proportionlabel_IN = self.proportionlabel_train_input()
            else:
                proportionlabel_IN = self.proportionlabel_train_input(fortrain=False)
            # 10. Proportion proximity
            proportionproximity_IN = self.proportionproximity_input()
            # 11. Proximity difference
            proximitydifference_IN = self.proximitydifference_input()

            # Flood proximity for filtering out results
            floodproximity_IN = self.floodproximity_input()

            # Create dataframe
            ml_df = elev_IN.copy(deep=True)
            # Get name list for each para. Start from 'hanf'
            para_names_list = [
                'hanf',
                'depth',
                'slope',
                'roughness',
                'curvature',
                'flowaccumulation',
                'proportionlabel',
                'proportionproximity',
                'proximitydifference'
            ]
            # Get list of para values. Start from 'hanf'
            para_values_list = [
                hanf_IN,
                depth_IN,
                slope_IN,
                roughness_IN,
                curvature_IN,
                flowaccumulation_IN,
                proportionlabel_IN,
                proportionproximity_IN,
                proximitydifference_IN
            ]

        else:

            # Call out each para
            # Depth includes 3 columns - x, y, depth - so the next para will be counted from 3
            depth_IN = self.depth_input(sd=True)
            # 3. Slope
            slope_IN = self.slope_input('slope_riserun')
            # 4. Manning
            manning_IN = self.manning_input()
            # 5. Sobel edge values
            sobeledgevalue_IN = self.sobeledgevalue_input()
            # 6. Canny edge
            cannyedge_IN = self.cannyedge_input()
            # 7. Laplace
            laplace_IN = self.laplace_input()
            # 8. Morphological laplace
            morphologicallaplace_IN = self.morphologicallaplace_input()
            # 9. Gaussian gradient
            gaussiangradient_IN = self.gaussiangradient_input()

            # Flood proximity for filtering out results
            floodproximity_IN = self.floodproximity_input()

            # Create dataframe
            ml_df = depth_IN.copy(deep=True)
            # Get name list for each para. Start from 'hanf'
            para_names_list = [
                'slope',
                'manning',
                'sobeledgevalue',
                'cannyedge',
                'laplace',
                'morphologicallaplace',
                'gaussiangradient'
            ]
            # Get list of para values. Start from 'hanf'
            para_values_list = [
                slope_IN,
                manning_IN,
                sobeledgevalue_IN,
                cannyedge_IN,
                laplace_IN,
                morphologicallaplace_IN,
                gaussiangradient_IN
            ]


        # Load all para into dataframe
        for i in range(len(para_values_list)):
            para_order = i + 3
            ml_df.insert(
                loc=para_order,
                column=para_names_list[i],
                value=para_values_list[i]
            )

        # Write out csv
        ml_df.to_csv(fr"{self.general_folder}/ml_full_{name_csv}_df_regression_sd.csv", index=False)

        return ml_df
