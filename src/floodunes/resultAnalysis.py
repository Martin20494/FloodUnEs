import os
from pathlib import Path

import numpy as np
import pandas as pd

import rioxarray as rxr
import xarray as xr
import geopandas as gpd

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, \
                                                  BboxPatch
import seaborn as sns

from sklearn.metrics import confusion_matrix, mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error, classification_report
import scipy.stats as stats


import json


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


def mark_inset_noconnectedline(parent_axes, inset_axes, zorder=None, **kwargs):
    """
    @Definition:
                A function to create inset plot represent a zooming place in map
    @Arguments:
                parent_axes (matplotlib axis):
                            Axis of matplotlib subplot (the original axis)
                inset_axes (matplotlib axis):
                            Axis of matplotlib inset plot
                zorder (int):
                            Oridinal number of plot layer
                **kwargs:
                            Any other arguments
    @Returns:
                None
    """
    # Get the coordinates of the box based on the parent axes
    rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)

    # Create the box
    pp = BboxPatch(rect, fill=False, zorder=zorder, **kwargs)

    # Add to parent axes
    parent_axes.add_patch(pp)


def remove_values_outside_floodplain(
    path_to_floodproximity,
    path_to_prediction,
    path_to_foldersavingthefile,
    filter_value=20
):

    # Get rasters
    proximity = rxr.open_rasterio(
        fr"{path_to_floodproximity}"
    )
    prediction = rxr.open_rasterio(
        fr"{path_to_prediction}"
    )

    # Remove values
    new_prediction = prediction.where(
        proximity.values < filter_value, 0
    )

    # Write out
    new_prediction.rio.to_raster(
        fr"{path_to_foldersavingthefile}"
    )

def comparison_plot(
    actual_filter,
    predicted_filter,
    difference,
    zoom_coord,
    inset_axes_position,
    path_tosavingfolder,
    title,
    color_cmap='viridis'
):
    # Ref: https://stackoverflow.com/questions/33602042/how-to-move-a-colorbar-label-downward
    # Set up fig
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3,
                                        figsize=(13, 4))

    fig.subplots_adjust(wspace=.05, hspace=0)

    # Actual
    act = actual_filter.plot(
        add_colorbar=False,
        cmap=color_cmap,
        ax=ax1
    )
    ax1.set_title('Monte Carlo result', pad=20)
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    act_cbar = fig.colorbar(act, ax=ax1, location="bottom")
    act_cbar.set_label(title, labelpad=20)
    act_cbar.ax.tick_params(rotation=-90, length=3)

    # Zoom - actual
    act_zoom = ax1.inset_axes(inset_axes_position)
    actual_filter.plot(
        add_colorbar=False,
        cmap=color_cmap,
        ax=act_zoom
    )
    # Zoom - coordinates
    act_zoom.set_xlim(zoom_coord[0], zoom_coord[1])
    act_zoom.set_ylim(zoom_coord[2], zoom_coord[3])
    # Remove all labels and ticks of zoom
    act_zoom.set_title('')
    act_zoom.set_xticks([])
    act_zoom.set_yticks([])
    act_zoom.set_xlabel('')
    act_zoom.set_ylabel('')
    # Remove black edge of plot background
    for spine in act_zoom.spines.values():
        spine.set_edgecolor('fuchsia')
        spine.set_linewidth(1)
    # Plot zoom
    mark_inset_noconnectedline(
        ax1,
        act_zoom,
        fc="none", ec="fuchsia", linewidth=1
    )

    # Text
    ax1.text(0.06, 0.9, 'a)',
             fontsize=15,
             color='white',
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax1.transAxes)

    ax2.set_xticks([])
    ax2.set_yticks([])


    # predicted_filter
    pre = predicted_filter.plot(
        add_colorbar=False,
        cmap=color_cmap,
        ax=ax2
    )
    ax2.set_title('BNNBB result', pad=20)
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    ax2.xaxis.set_tick_params(labelbottom=False)
    pre_cbar = fig.colorbar(pre, ax=ax2, location="bottom")
    pre_cbar.set_label(label=title, labelpad=20)
    pre_cbar.ax.tick_params(rotation=-90, length=3)

    # Zoom - predicted_proportion_filter
    pre_zoom = ax2.inset_axes(inset_axes_position)
    predicted_filter.plot(
        add_colorbar=False,
        cmap=color_cmap,
        ax=pre_zoom
    )
    # Zoom - coordinates
    pre_zoom.set_xlim(zoom_coord[0], zoom_coord[1])
    pre_zoom.set_ylim(zoom_coord[2], zoom_coord[3])
    # Remove all labels and ticks of zoom
    pre_zoom.set_title('')
    pre_zoom.set_xticks([])
    pre_zoom.set_yticks([])
    pre_zoom.set_xlabel('')
    pre_zoom.set_ylabel('')
    # Remove black edge of plot background
    for spine in pre_zoom.spines.values():
        spine.set_edgecolor('fuchsia')
        spine.set_linewidth(1)
    # Plot zoom
    mark_inset_noconnectedline(
        ax2,
        pre_zoom,
        fc="none", ec="fuchsia", linewidth=1
    )

    # Text
    ax2.text(0.06, 0.9, 'b)',
             fontsize=15,
             color='white',
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax2.transAxes)

    ax2.set_xticks([])
    ax2.set_yticks([])

    # difference
    diff = difference.plot(
        add_colorbar=False,
        cmap='seismic',
        ax=ax3
    )
    ax3.set_title('Differences between\nMonte Carlo and BNNBB results', pad=20)
    ax3.set_xlabel('')
    ax3.set_ylabel('')
    ax3.xaxis.set_tick_params(labelbottom=False)
    diff_cbar = fig.colorbar(diff, ax=ax3, location="bottom")
    diff_cbar.set_label(label=title, labelpad=20)
    diff_cbar.ax.tick_params(rotation=-90, length=3)

    # Zoom - predicted_filter
    dif_zoom = ax3.inset_axes(inset_axes_position)
    difference.plot(
        add_colorbar=False,
        cmap='seismic',
        ax=dif_zoom
    )
    # Zoom - coordinates
    dif_zoom.set_xlim(zoom_coord[0], zoom_coord[1])
    dif_zoom.set_ylim(zoom_coord[2], zoom_coord[3])
    # Remove all labels and ticks of zoom
    dif_zoom.set_title('')
    dif_zoom.set_xticks([])
    dif_zoom.set_yticks([])
    dif_zoom.set_xlabel('')
    dif_zoom.set_ylabel('')
    # Remove black edge of plot background
    for spine in dif_zoom.spines.values():
        spine.set_edgecolor('fuchsia')
        spine.set_linewidth(1)
    # Plot zoom
    mark_inset_noconnectedline(
        ax3,
        dif_zoom,
        fc="none", ec="fuchsia", linewidth=1
    )

    # Text
    ax3.text(0.06, 0.9, 'c)',
             fontsize=15,
             color='black',
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax3.transAxes)

    ax3.set_xticks([])
    ax3.set_yticks([])

    # Save fig
    fig.savefig(
        fr"{path_tosavingfolder}\comparisonplot.jpg",
        bbox_inches='tight', dpi=600
    )

def scatterplot(
    data_df_remove0,
    path_to_savingfolder,
    indent_list
):

    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=(5, 5))

    ident = indent_list
    ax.plot(ident, ident, linestyle='--', color='black')

    sns.regplot(
        data=data_df_remove0,
        x='BNNBB_result',
        y='Monte_Carlo_result',
        fit_reg=False,
        marker='o', color='red', scatter_kws={'s': 1},
        ax=ax
    )

    # Save fig
    fig.savefig(
        fr"{path_to_savingfolder}/scatterplot.jpg",
        bbox_inches='tight', dpi=600
    )

def histogram(
    proportion_filter,
    path_to_savingfolder,
    name,
    zoom=None
):
    # Get data to plot
    all_values = proportion_filter.values.flatten().copy()
    all_values_extraction = all_values[all_values != 0]
    hist, bins = np.histogram(all_values_extraction, bins=100)
    width = 1 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2

    # Zoom
    if zoom != None:
        plt.xlim(zoom)
    else:
        pass

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(center, hist, align='center', width=width)

    # Save fig
    fig.savefig(
        fr"{path_to_savingfolder}/{name}.jpg",
        bbox_inches='tight', dpi=600
    )

def confusion_matrix_plot(
    actual_proportion_filter,
    predicted_proportion_filter,
    path_to_savingfolder
):
    # Monte Carlo
    actual_label = actual_proportion_filter.values.flatten().copy()
    actual_label[(actual_label > 0) & (actual_label < 100)] = 1
    actual_label[actual_label == 100] = 2

    # Prediction
    prediction_label = predicted_proportion_filter.values.flatten().copy()
    prediction_label[(prediction_label > 0) & (prediction_label < 100)] = 1
    prediction_label[prediction_label == 100] = 2

    # Confusion matrix calculation
    np.bool = np.bool_
    cfm = confusion_matrix(actual_label,
                           prediction_label)
    cfm_pc = np.stack([(cfm[i, :] / np.sum(cfm[i, :])) for i in range(3)])
    df_cfm_pc = pd.DataFrame(cfm_pc, index=[i for i in ['No flood', 'Maybe flood', 'Yes flood']],
                             columns=['No flood', 'Maybe flood', 'Yes flood'])
    group_counts = ['{0:0.0f}'.format(value) for value in cfm.flatten()]
    calculate_percentages = np.concatenate([(cfm[i, :] / np.sum(cfm[i, :])) for i in range(3)])
    group_percentages = ['{0:.2%}'.format(value) for value in calculate_percentages]
    labels = [f'{v2}\n{v3}' for v2, v3 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(3, 3)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    g = sns.heatmap(df_cfm_pc, annot=labels, cmap="rainbow", ax=ax, fmt='', cbar=False, annot_kws={"size": 20})
    ax.set_xlabel('LABEL based on Machine Learning result', fontsize=15, labelpad=15)
    ax.set_ylabel('LABEL based on Monte Carlo result', fontsize=15, labelpad=15)
    g.set_xticklabels(g.get_xmajorticklabels(), fontsize=16)

    # Save fig
    fig.savefig(
        fr"{path_to_savingfolder}/confusion_matrix.jpg",
        bbox_inches='tight', dpi=600
    )


class resultCalculation():

    def __init__(self,
                 result_path,
                 filter_value_outside=20
                 ):

        # Set up parameters for getting data
        with open(result_path, "r") as result_path_r:
            self.result_path = json.load(result_path_r)

        # Main path
        self.main_path = self.result_path['general_folder']

        # Create result folder
        self.result_folder = fr"{self.main_path}/results/{self.result_path['type_prediction']}/{self.result_path['version']}"
        Path(self.result_folder).mkdir(parents=True, exist_ok=True)

        # Remove outside of floodplain area
        remove_values_outside_floodplain(
            fr"{self.main_path}/{self.result_path['type_test']}/floodproximity_input_domain.nc",
            fr"{self.main_path}/{self.result_path['type_test']}/model_{self.result_path['type_prediction']}/prediction/{self.result_path['name_prediction_file']}",
            fr"{self.result_folder}/{self.result_path['type_prediction']}_removeoutside.nc",
            filter_value_outside
        )


    def resultProportion(
        self,
        zoom_coord_forcomparisonplot,
        inset_axis_position_forcomparisonplot,
        zoom_list_forhistogram=None,
    ):

        # Get save path
        save_path = fr"{self.result_folder}"

        # GET ALL NECESSARY PATHS
        # Get actual data
        actual_proportion_path = fr"{self.main_path}/{self.result_path['actual_file_name']}"
        actual_proportion = rxr.open_rasterio(f"{actual_proportion_path}")
        actual_proportion = actual_proportion.where(actual_proportion.values != -9999, 0)
        actual_proportion.rio.to_raster(
            fr"{save_path}/actual_proportion.tiff")  # Convert to tiff to remove river and sea
        actual_proportion_path_new = fr"{save_path}/actual_proportion.tiff"
        # Get predicted data
        predicted_proportion_path = fr"{save_path}/{self.result_path['type_prediction']}_removeoutside.nc"
        predicted_proportion = rxr.open_rasterio(f"{predicted_proportion_path}")
        # Filter values
        predicted_values = predicted_proportion.values
        predicted_values[predicted_values >= 99.5] = 100
        predicted_values[(predicted_values > 99) & (predicted_values <= 99.5)] = 99
        predicted_values[predicted_values < 0.5] = 0
        predicted_values[(predicted_values >= 0.5) & (predicted_values < 1)] = 1
        # Change values
        predicted_proportion_arr = xr.DataArray(
            data=predicted_values[0],
            dims=['y', 'x'],
            coords={
                'x': (['x'], actual_proportion.x.values),
                'y': (['y'], actual_proportion.y.values)
            },
            attrs=actual_proportion.attrs
        )
        predicted_proportion_arr.rio.to_raster(
            fr"{save_path}/predicted_proportion.tiff")  # Convert to tiff to remove river and sea
        predicted_proportion_path_new = fr"{save_path}/predicted_proportion.tiff"
        # Get sea
        sea_polygon = gpd.read_file(fr"{save_path}/polygon_angle_0.0_x_0.0_y_0.0.shp")
        sea_polygon.to_file(fr"{save_path}/sea_polygon.shp")
        sea_polygon_path = fr"{save_path}/sea_polygon.shp"

        # REMOVE SEA
        # Remove sea for ACTUAL
        value_change(sea_polygon_path, actual_proportion_path_new, 0, True)
        # Remove sea for PREDICTED
        value_change(sea_polygon_path, predicted_proportion_path_new, 0, True)

        # CONVERT TIFF TO NC
        # Actual
        actual_proportion_filter = rxr.open_rasterio(fr"{actual_proportion_path_new}")
        actual_proportion_filter.rio.to_raster(fr"{save_path}/actual_proportion_filter.nc")
        # Predicted
        predicted_proportion_filter = rxr.open_rasterio(fr"{predicted_proportion_path_new}")
        predicted_proportion_filter.rio.to_raster(fr"{save_path}/predicted_proportion_full_filter.nc")

        # DIFFERENCE
        difference = actual_proportion_filter - predicted_proportion_filter
        difference.rio.to_raster(fr"{save_path}/difference_proportion_filter.nc")

        # COMPARISON PLOT
        comparison_plot(
            actual_proportion_filter,
            predicted_proportion_filter,
            difference,
            zoom_coord_forcomparisonplot,
            inset_axis_position_forcomparisonplot,
            save_path,
            title='Proportion (%)'
        )

        # METRICS INCLUDE ZERO
        # Calculate MSE
        mse_include_zero = mean_squared_error(
            actual_proportion_filter.values.flatten(),
            predicted_proportion_filter.values.flatten()
        )
        # Calculate RMSE
        rmse_include_zero = mean_squared_error(
            actual_proportion_filter.values.flatten(),
            predicted_proportion_filter.values.flatten(),
            squared=False
        )
        # Calculate MAE
        mae_include_zero = mean_absolute_error(
            actual_proportion_filter.values.flatten(),
            predicted_proportion_filter.values.flatten()
        )
        # Calculate R2
        r2_include_zero = r2_score(
            actual_proportion_filter.values.flatten(),
            predicted_proportion_filter.values.flatten()
        )

        # METRICS EXCLUDE ZERO
        data_full = {
            'Monte_Carlo_result': actual_proportion_filter.values.flatten(),
            'BNNBB_result': predicted_proportion_filter.values.flatten()
        }
        data_df_full = pd.DataFrame(data=data_full)
        data_df_remove0 = data_df_full.query('0 < Monte_Carlo_result < 100')
        # Calculate MSE
        mse_remove0 = mean_squared_error(
            data_df_remove0.Monte_Carlo_result,
            data_df_remove0.BNNBB_result
        )
        # Calculate RMSE
        rmse_remove0 = mean_squared_error(
            data_df_remove0.Monte_Carlo_result,
            data_df_remove0.BNNBB_result,
            squared=False
        )
        # Calculate MAE
        mae_remove0 = mean_absolute_error(
            data_df_remove0.Monte_Carlo_result,
            data_df_remove0.BNNBB_result
        )
        # Calculate R2
        r2_remove0 = r2_score(
            data_df_remove0.Monte_Carlo_result,
            data_df_remove0.BNNBB_result
        )

        # CORRELATION INCLUDES ZERO
        # Calculating Pearson's correlation
        corrp_include_zero, _ = stats.pearsonr(data_df_full.Monte_Carlo_result, data_df_full.BNNBB_result)
        # Calculating Spearman's correlation
        corrs_include_zero, _ = stats.spearmanr(data_df_full.Monte_Carlo_result, data_df_full.BNNBB_result)

        # CORRELATION EXCLUDES ZERO
        # Calculating Pearson's correlation
        corrp_remove0, _ = stats.pearsonr(data_df_remove0.Monte_Carlo_result, data_df_remove0.BNNBB_result)
        # Calculating Spearman's correlation
        corrs_remove0, _ = stats.spearmanr(data_df_remove0.Monte_Carlo_result, data_df_remove0.BNNBB_result)

        # METRICS TABLE
        row_name = [
            'mse', 'rmse', 'mae', 'r2', 'pearson', 'spearman'
        ]
        include_zero = [
            mse_include_zero, rmse_include_zero, mae_include_zero, r2_include_zero, corrp_include_zero,
            corrs_include_zero
        ]
        remove_zero = [
            mse_remove0, rmse_remove0, mae_remove0, r2_remove0, corrp_remove0, corrs_remove0
        ]
        dict_metrics = {
            'metrics': row_name,
            'include_zero': include_zero,
            'remove_zero': remove_zero
        }
        df_metrics = pd.DataFrame(data=dict_metrics)
        df_metrics.to_csv(fr"{save_path}/metrics.csv", index=False)


        # SCATTERPLOT
        scatterplot(
            data_df_remove0,
            save_path,
            [0, 100]
        )

        # HISTOGRAM - NO ZERO
        # Actual
        histogram(
            actual_proportion_filter,
            save_path,
            'actual_histogram',
        )
        # Prediction
        histogram(
            predicted_proportion_filter,
            save_path,
            'predicted_histogram',
        )
        # Difference
        histogram(
            difference,
            save_path,
            'difference_histogram',
        )
        if zoom_list_forhistogram != None:
            # Difference zoom
            histogram(
                difference,
                save_path,
                'difference_histogram',
                zoom=zoom_list_forhistogram
            )
        else:
            pass

        # CONFUSION MATRIX
        confusion_matrix_plot(
            actual_proportion_filter,
            predicted_proportion_filter,
            save_path
        )

        # CONFUSION MATRIX REPORT
        # Ref: https://stackoverflow.com/questions/39662398/scikit-learn-output-metrics-classification-report-into-csv-tab-delimited-format
        # Monte Carlo
        actual_label = actual_proportion_filter.values.flatten().copy()
        actual_label[(actual_label > 0) & (actual_label < 100)] = 1
        actual_label[actual_label == 100] = 2
        # Prediction
        prediction_label = predicted_proportion_filter.values.flatten().copy()
        prediction_label[(prediction_label > 0) & (prediction_label < 100)] = 1
        prediction_label[prediction_label == 100] = 2
        # Report
        confusion_report = classification_report(
            actual_label,
            prediction_label,
            output_dict=True
        )
        df_confusion = pd.DataFrame(confusion_report).transpose()
        df_confusion.to_csv(fr"{save_path}/confusion_metrics.csv", index=False)


    def resultSD(
        self,
        zoom_coord_forcomparisonplot,
        inset_axis_position_forcomparisonplot,
        zoom_list_forhistogram=None,
    ):

        # Get save path
        save_path = fr"{self.result_folder}"

        # GET ALL NECESSARY PATHS
        # Get actual data
        actual_sd_path = fr"{self.main_path}/{self.result_path['actual_file_name']}"
        actual_sd = rxr.open_rasterio(f"{actual_sd_path}")
        actual_sd = actual_sd.where(actual_sd.values != -9999, 0)
        actual_sd.rio.to_raster(
            fr"{save_path}/actual_sd.tiff")  # Convert to tiff to remove river and sea
        actual_sd_path_new = fr"{save_path}/actual_sd.tiff"
        # Get predicted data
        predicted_sd_path = fr"{save_path}/{self.result_path['type_prediction']}_removeoutside.nc"
        predicted_sd = rxr.open_rasterio(f"{predicted_sd_path}")
        # Filter values
        predicted_sd = predicted_sd.where(predicted_sd.values >= 0.01, 0)
        # Change coordinates
        predicted_sd_arr = xr.DataArray(
            data=predicted_sd.values[0],
            dims=['y', 'x'],
            coords={
                'x': (['x'], actual_sd.x.values),
                'y': (['y'], actual_sd.y.values)
            },
            attrs=actual_sd.attrs
        )
        predicted_sd_arr.rio.to_raster(fr"{save_path}/predicted_sd.tiff")  # Convert to tiff to remove river and sea
        predicted_sd_path_new = fr"{save_path}/predicted_sd.tiff"
        # Get river
        river_polygon = gpd.read_file(fr"{save_path}/river_polygon.geojson")
        river_polygon.to_file(fr"{save_path}/river_polygon.shp")  # Convert to shapfile to remove river and sea
        river_polygon_path = fr"{save_path}/river_polygon.shp"
        # Get sea
        sea_polygon = gpd.read_file(fr"{save_path}/polygon_angle_0.0_x_0.0_y_0.0.shp")
        sea_polygon.to_file(fr"{save_path}/sea_polygon.shp")
        sea_polygon_path = fr"{save_path}/sea_polygon.shp"

        # Remove river and sea for ACTUAL
        # River
        value_change(river_polygon_path, actual_sd_path_new, 0, True)
        # Sea
        value_change(sea_polygon_path, actual_sd_path_new, 0, True)

        # Remove river and sea for PREDICTED
        # River
        value_change(river_polygon_path, predicted_sd_path_new, 0, True)
        # Sea
        value_change(sea_polygon_path, predicted_sd_path_new, 0, True)

        # CONVERT TIFF TO NC
        # Actual
        actual_sd_filter = rxr.open_rasterio(fr"{actual_sd_path_new}")
        actual_sd_filter.rio.to_raster(fr"{save_path}/actual_sd_filter.nc")
        # Predicted
        predicted_sd_filter = rxr.open_rasterio(fr"{predicted_sd_path_new}")
        predicted_sd_filter.rio.to_raster(fr"{save_path}/predicted_sd_full_filter.nc")

        # DIFFERENCE
        difference = actual_sd_filter - predicted_sd_filter
        difference.rio.to_raster(fr"{save_path}/difference_sd_filter.nc")

        # COMPARISON PLOT
        comparison_plot(
            actual_sd_filter,
            predicted_sd_filter,
            difference,
            zoom_coord_forcomparisonplot,
            inset_axis_position_forcomparisonplot,
            save_path,
            title='Sd (m)',
            color_cmap='plasma_r'
        )

        # METRICS INCLUDE ZERO
        # Calculate MSE
        mse_include_zero = mean_squared_error(
            actual_sd_filter.values.flatten(),
            predicted_sd_filter.values.flatten()
        )
        # Calculate RMSE
        rmse_include_zero = mean_squared_error(
            actual_sd_filter.values.flatten(),
            predicted_sd_filter.values.flatten(),
            squared=False
        )
        # Calculate MAE
        mae_include_zero = mean_absolute_error(
            actual_sd_filter.values.flatten(),
            predicted_sd_filter.values.flatten()
        )
        # Calculate R2
        r2_include_zero = r2_score(
            actual_sd_filter.values.flatten(),
            predicted_sd_filter.values.flatten()
        )

        # METRICS EXCLUDE ZERO
        data_full = {
            'Monte_Carlo_result': actual_sd_filter.values.flatten(),
            'BNNBB_result': predicted_sd_filter.values.flatten()
        }
        data_df_full = pd.DataFrame(data=data_full)
        data_df_remove0 = data_df_full.query('0 < Monte_Carlo_result < 100')
        # Calculate MSE
        mse_remove0 = mean_squared_error(
            data_df_remove0.Monte_Carlo_result,
            data_df_remove0.BNNBB_result
        )
        # Calculate RMSE
        rmse_remove0 = mean_squared_error(
            data_df_remove0.Monte_Carlo_result,
            data_df_remove0.BNNBB_result,
            squared=False
        )
        # Calculate MAE
        mae_remove0 = mean_absolute_error(
            data_df_remove0.Monte_Carlo_result,
            data_df_remove0.BNNBB_result
        )
        # Calculate R2
        r2_remove0 = r2_score(
            data_df_remove0.Monte_Carlo_result,
            data_df_remove0.BNNBB_result
        )

        # CORRELATION INCLUDES ZERO
        # Calculating Pearson's correlation
        corrp_include_zero, _ = stats.pearsonr(data_df_full.Monte_Carlo_result, data_df_full.BNNBB_result)
        # Calculating Spearman's correlation
        corrs_include_zero, _ = stats.spearmanr(data_df_full.Monte_Carlo_result, data_df_full.BNNBB_result)

        # CORRELATION EXCLUDES ZERO
        # Calculating Pearson's correlation
        corrp_remove0, _ = stats.pearsonr(data_df_remove0.Monte_Carlo_result, data_df_remove0.BNNBB_result)
        # Calculating Spearman's correlation
        corrs_remove0, _ = stats.spearmanr(data_df_remove0.Monte_Carlo_result, data_df_remove0.BNNBB_result)

        # METRICS TABLE
        row_name = [
            'mse', 'rmse', 'mae', 'r2', 'pearson', 'spearman'
        ]
        include_zero = [
            mse_include_zero, rmse_include_zero, mae_include_zero, r2_include_zero, corrp_include_zero,
            corrs_include_zero
        ]
        remove_zero = [
            mse_remove0, rmse_remove0, mae_remove0, r2_remove0, corrp_remove0, corrs_remove0
        ]
        dict_metrics = {
            'metrics': row_name,
            'include_zero': include_zero,
            'remove_zero': remove_zero
        }
        df_metrics = pd.DataFrame(data=dict_metrics)
        df_metrics.to_csv(fr"{save_path}/metrics.csv", index=False)


        # SCATTERPLOT
        scatterplot(
            data_df_remove0,
            save_path,
            [0, 2]
        )

        # HISTOGRAM - NO ZERO
        # Actual
        histogram(
            actual_sd_filter,
            save_path,
            'actual_histogram',
        )
        # Prediction
        histogram(
            predicted_sd_filter,
            save_path,
            'predicted_histogram',
        )
        # Difference
        histogram(
            difference,
            save_path,
            'difference_histogram',
        )
        if zoom_list_forhistogram != None:
            # Difference zoom
            histogram(
                difference,
                save_path,
                'difference_histogram',
                zoom=zoom_list_forhistogram
            )
        else:
            pass