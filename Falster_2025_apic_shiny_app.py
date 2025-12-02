import xarray as xr
import pandas as pd
from datetime import datetime
import numpy as np

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import cartopy.crs as ccrs
import matplotlib as mpl
from cartopy.io.shapereader import natural_earth, Reader
import math

from shiny import App, ui, reactive, render
import shinyswatch
from shinywidgets import output_widget, render_widget

import folium

# adjust directory as necessary
fpath = ""
#fpath = "C:/Users/georg/Dropbox/~python_working/aus_isotopes/shiny_app/APIC_shiny_app/"

# monthly data
d2H = xr.open_dataset(f"{fpath}netcdfs/aus_prec.d2H_v1_196201-202312_monthly_median.nc")
d18O = xr.open_dataset(f"{fpath}netcdfs/aus_prec.d18O_v1_196201-202312_monthly_median.nc")
dxs = xr.open_dataset(f"{fpath}netcdfs/aus_prec.dxs_v1_196201-202312_monthly_median.nc")

# annual data (Jan-Dec)
d2H_ann = xr.open_dataset(f"{fpath}netcdfs/aus_prec.d2H_v1_1962-2023_ann_median.nc")
d18O_ann = xr.open_dataset(f"{fpath}netcdfs/aus_prec.d18O_v1_1962-2023_ann_median.nc")
dxs_ann = xr.open_dataset(f"{fpath}netcdfs/aus_prec.dxs_v1_1962-2023_ann_median.nc")

years_cal = d2H_ann.time.dt.year.values

# we'll also need the precipitation amount data for if users want to specific time periods
prec = xr.open_dataset(f"{fpath}netcdfs/prec/aus_prec_v1_195901-202312_monthly_1.nc")
prec = prec["prec"].sel(time=slice("1962-01-01", None))

# long-term mean (calendar year)
d2H_mean = xr.open_dataset(f"{fpath}netcdfs/aus_prec.d2H_v1_1962-2023_long-term-annual-mean_median.nc")
d18O_mean = xr.open_dataset(f"{fpath}netcdfs/aus_prec.d18O_v1_1962-2023_long-term-annual-mean_median.nc")
dxs_mean = xr.open_dataset(f"{fpath}netcdfs/aus_prec.dxs_v1_1962-2023_long-term-annual-mean_median.nc")
   
# define pop-up information windows
modal_ts = ui.modal(
    ui.markdown(
        """To extract data from your chosen location, enter the coordinates and define the temporal resolution. 
        You have the option to select a date range and assign a site name 
        which will be appended to the data download, although these are not required. <b>All values are 
        precipitation amount-weighted means</b>. Values will only be calculated when you click the `Extract and plot values` button.
        <br><br>All running means are the average of <i>n</i> months up to and including the index month (i.e., right-aligned). 
        For values averaged over July-June and December-February, the year index applies to the calendar year at the <i>start</i> of the 
        averaging period (e.g., the 1990 DJF values represent December 1990 and January-February 1991).
        <br><br>Timeseries of the δ²H, δ¹⁸O, and <i>dxs</i> values will appear in the window to the right, at the selected temporal resolution. 
        Below is a map showing the location of your lat/lon selection (check that it is where you expect!), and a local meteoric water line for that location. 
        <b>If you update any of the parameters you will need to click the `Extract and plot values` button again to re-calculate the values</b>. 
        You can download the data to a csv file by clicking the button below the timeseries plot. 
        <br><br> It is important to note that these are modelled values, not primary observations.
        """
    ),
    title = "Extract timeseries",
    easy_close = True,
    footer = ui.div(ui.div(
        ui.modal_button("Close window"),
        class_="text-center"),
        class_="w-100"),
    size = "xl"
)
modal_spatial = ui.modal(
    ui.markdown(
        """To identify possible source regions for a sample, choose an isotope system and enter the value.
        If the measured material was not precipitation (or you haven't already calculated an equivalent source water value), you can enter 
        an expected offset and this will be applied to your sample value. You can also enter an expected range (uncertainty) around your specific value 
        (the default is +/- 2‰ but you should almost certaintly change this - it can also be zero).
        <br><br>You can choose to search for potential location matches in the long-term (1962-2023) mean <i>or</i> over a particular time period. The latter is useful if 
        you have an idea of when your sample might have formed. If you need a more tailored search, please consider working with the raw data 
        files (see link in the sidebar).
        <br><br>After entering your parameters and clicking `Find my sample`, a map will appear showing your results.
        <br><br> It is important to note that these are modelled values, not primary observations.
        """
    ),
    title = "Identify potential source water locations",
    easy_close = True,
    footer = ui.div(ui.div(
        ui.modal_button("Close window"),
        class_="text-center"),
        class_="w-100"),
    size = "xl"
)

# DEFINE USER INTERFACE
app_ui = ui.page_fluid(

    ui.tags.head(
        ui.tags.script(
            src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
        ),
        # JavaScript to make the pop-up windows work
        ui.tags.script(
            """
            document.addEventListener("DOMContentLoaded", function() {
                document.querySelectorAll('.nav-link').forEach(function(tab) {
                    tab.addEventListener('click', function() {
                        const selectedTab = this.innerText.trim();
                        Shiny.setInputValue('active_tab', selectedTab, {priority: "event"});
                    });
                });
            });
            """
        )
    ),
    # size of the intro text
    ui.tags.style("""
                  .custom-text {
                  font-size: 0.9rem;
                  }
                  """),

    # in the spatial search - make the image appear at the top
#    ui.tags.style("""
#                  .grid-top-align {
#                  align-items: start !important;
#                  justify-content-center;
#                  }
#                  """),

    # update the title so the panel is bigger, and coloured, and to include some more info
    ui.panel_title(
        "Australian precipitation isotope calculator"
        ),

    ui.div(
        ui.markdown(
            """This online calculator allows users to extract modelled monthly or annual precipitation isotope δ²H, δ¹⁸O, and <i>dxs</i> values 
             for any location on the Australian continent, within the time period January 1962 to December 2023. 
             <br><br>Below are two tabs for different data extraction types: timeseries or location search. On the `Extract timeseries` page you can enter a location (latitude and
             longitude), choose a temporal resolution and optional date range, then view the precipitation isotope δ²H, δ¹⁸O, and <i>dxs</i> values for your chosen location. 
             You will have the option to download the data to csv. On the 'Spatial search' page, you 
             can enter a δ²H, δ¹⁸O, or <i>dxs</i> value as well as an optional expected offest from precipitation δ²H/δ¹⁸O/<i>dxs</i> and time period of interest. 
             You will then see a map of locations where that sample could have come from. 
             <br><br>When choosing a tab, an information window will appear with further important details. To make the information window reappear, click the relevant tab. 
             <br><br>If using data from this online calculator, please cite 
             the <a href="https://egusphere.copernicus.org/preprints/2025/egusphere-2025-2458/" target="_blank">original publication</a>. Please also see the 
             publication for all details of how the precipitation isotope values were calculated. It is important to note 
             that these are modelled values, not primary observations. If you encounter problems with this web app, please get in touch with Georgy Falster.  
            """
            
        ), class_="custom-text"
    ),

    ui.tags.style(
    """
    .smaller-text {
        font-size: 12px;
        line-height: 1.5;
    }
    """
    ),
    
    # top-level UI set-up: timeseries or spatial search?
    ui.page_navbar(
        # first panel: extract and plot timeseries
        ui.nav_panel("Extract timeseries",ui.layout_sidebar(
            # set up the sidebar
            ui.sidebar(
                # required inputs: lat and lon
                ui.card(
                    ui.card_header(
                        ui.tags.h3("Required inputs", style="font-weight: bold; font-size: 20px;")
                        ),
                    ui.input_numeric("lat", "Latitude (decimal degrees)", min=-45, max=-10, value=-28),
                    ui.input_numeric("lon", "Longitude (decimal degrees)", min=112, max=154, value=134),
                    ui.input_select("time_res", "Temporal resolution",
                    choices = {"monthly": "Monthly", "ann": "Annual (Jan-Dec)", "ann_trop":"Annual (Jul-Jun)",
                           "DJF":"Annual (DJF)", "MAM":"Annual (MAM)", "JJA":"Annual (JJA)",
                           "SON":"Annual (SON)", "3mrm":"3-month running mean", "6mrm":"6-month running mean",
                           "12mrm":"12-month running mean"},
                    selected="monthly")
                ),

                # optional inputs: date range, site name for download
                ui.card(
                    ui.card_header(
                        ui.tags.h3("Optional inputs", style="font-weight: bold; font-size: 20px;")
                    ),
                    ui.input_date_range("date_range", "Select date range", start = "1962-01-01", end = "2023-12-31",
                                    min = "1962-01-01", max = "2023-12-31"),
                    ui.input_text("site_name",
                        ui.HTML("Site name <br><i>resets when lat and/or lon are changed</i>")),
                ),
    
                # button to extract and plot the values
                ui.input_action_button("run_calcs", "Extract and plot values",
                    style=
                    "background: linear-gradient(to bottom, rgb(90, 174, 240), rgb(50, 134, 200)); "
                    #"background:rgb(80, 164, 230); "
                    "color: white; "
                    "box-shadow: 0px 0px 8px rgba(0, 0, 0, 0.2); "
                    "border: none; "
                    "border-radius: 5px; "
                    "padding: 12px 20px; "
                    "font-size: 18px; "),

                # some css to make the button 'squish' when pressed
                ui.tags.style("""
                    #run_calcs:hover {
                    background: linear-gradient(to bottom, rgb(100, 184, 250), rgb(60, 144, 210));
                    transform: scale(1.03); 
                    box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.3);
                    color: rgba(240, 240, 240, 0.84);    
                    }
                    #run_calcs:active {
                        background: linear-gradient(to bottom, rgb(80, 164, 230), rgb(40, 124, 190));
                        transform: scale(0.98);
                        box-shadow: inset 0px 4px 6px rgba(0, 0, 0, 0.3); 
                    }
                """),

                # card describing/linking to the original publication, disclaimer etc
                ui.card(
                    ui.card_header(
                    ui.tags.h3("Dataset details", style="font-weight: bold; font-size: 20px;") 
                    ),
                    ui.markdown("""Please read the below-linked publication for all details as to how these precipitation 
                                δ²H, δ¹⁸O, and <i>dxs</i> values 
                                were produced. If you use data from this calculator, 
                                please cite the paper below.
                                """),
                    ui.a("Go to publication", href="https://egusphere.copernicus.org/preprints/2025/egusphere-2025-2458/", target="_blank", class_="btn btn-secondary")
                ),

                # link to zenodo repo for users to download the netcdfs
                ui.card(
                    ui.card_header(
                        ui.tags.h3("Download netcdf files", style="font-weight: bold; font-size: 20px;")
                        ),
                    ui.markdown(
                        """<a href="https://doi.org/10.5281/zenodo.15486277" target="_blank">This Zenodo repository</a> holds netcdf files 
                        with monthly precipitation isotope data across the Australian continent, at 0.25° spatial resolution. 
                        The data are available at monthly and annual temporal resolution.
                """
                    )
                ),

                # make the sidebar a bit wider than the default
                width = 350,

                # we don't want a collapsible sidebar (maybe later modify for mobile phones)
                open = "always",
                ),

            # main panel in this page shows the timeseries, download button, location map, and LMWL
            ui.layout_columns(
                # first card shows the timeseries and has the download button
                ui.card(
                    # card header
                    ui.card_header("Values for selected location and temporal resolution",
                                style="text-align: center; font-size: 20px; font-weight: bold;"),
                    output_widget("plot_ts"),
                    ui.download_button("download_csv", "Click here to download data (after selecting location/resolution and extracting the data)",
                        class_="btn btn-secondary"),
                    height = "700px"
                ),
                # now, two cards side by side with the location map and the LMWL
                # left card
                ui.card(
                    ui.card_header("Selected location",
                                style="text-align: center; font-size: 20px; font-weight: bold;"),
                    ui.output_ui("loc_map"),
                    height = "400px"
                ),
                # right card
                ui.card(
                    ui.card_header("Local meteoric water line",
                                style="text-align: center; font-size: 20px; font-weight: bold;"),
                    output_widget("lmwl"),
                    height = "400px"
                ),
                col_widths = (12, 7, 5), # the Shiny CSS has 12 columns: so this wraps around to two rows
                # heights_equal = "row"
                # row_heights = (1.5,1),
                # height = "600px"
                ),
            ),
        ),
        # and now the second tab: the map search
        ui.nav_panel("Spatial search", ui.layout_sidebar(
            ui.sidebar(
                # required inputs: system, value, long-term mean or time search -> years and months
                ui.card(
                    ui.card_header(
                        ui.tags.h3("Required inputs", style="font-weight: bold; font-size: 20px;")
                        ),
                        ui.input_select("isotope", "Isotope system",
                                        choices = {"d2H": "δ²H", "d18O": "δ¹⁸O", "dxs":"dxs"}
                        ),
                    ui.input_numeric("input_val", "Value (‰ VSMOW)", value = 0
                    ),

                    # define the search type. If the user wants a particular search period: define it!
                    ui.input_radio_buttons("search_type", "Search type:",
                                           choices = ["Long-term mean", "Mean over time period"], 
                                           selected = "Long-term mean"),
                    ui.panel_conditional("input.search_type === 'Mean over time period'",
                        ui.layout_columns(
                            ui.input_numeric("year_start", "Start year", value=1962, min=1962, max=2023),
                            ui.input_numeric("year_end", "End year", value=2023, min=1962, max=2023),
                            col_widths = (6,6)
                        ),
                        ui.input_checkbox_group("months_spatial", "Months", choices={"1": "Jan", "2": "Feb", "3": "Mar", "4": "Apr",
                                                                             "5": "May", "6": "Jun", "7": "Jul", "8": "Aug",
                                                                             "9": "Sep", "10": "Oct", "11": "Nov", "12": "Dec"},
                                                                             selected=[str(i) for i in range(1, 13)],inline=True)
                        )  
                ),

                # next, optional inputs: offset, range
                ui.card(
                    ui.card_header(
                        ui.tags.h3("Optional inputs", style="font-weight: bold; font-size: 20px;")
                    ),
                    ui.input_numeric("offset", "Offset (‰)", value=0),
                    ui.input_numeric("input_range", "Range (+/- ‰)", value=2),
                ),
    
                # button to extract and plot the values
                ui.input_action_button("run_spatial_search", "Find my sample",
                    style=
                    "background: linear-gradient(to bottom, rgb(90, 174, 240), rgb(50, 134, 200)); "
                    #"background:rgb(80, 164, 230); "
                    "color: white; "
                    "box-shadow: 0px 0px 8px rgba(0, 0, 0, 0.2); "
                    "border: none; "
                    "border-radius: 5px; "
                    "padding: 12px 20px; "
                    "font-size: 18px; "),

                # some css to make the button 'squish' when pressed
                ui.tags.style("""
                    #run_spatial_search:hover {
                    background: linear-gradient(to bottom, rgb(100, 184, 250), rgb(60, 144, 210));
                    transform: scale(1.03); 
                    box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.3);
                    color: rgba(240, 240, 240, 0.84);    
                    }
                    #run_spatial_search:active {
                        background: linear-gradient(to bottom, rgb(80, 164, 230), rgb(40, 124, 190));
                        transform: scale(0.98);
                        box-shadow: inset 0px 4px 6px rgba(0, 0, 0, 0.3); 
                    }
                """),

                # card describing/linking to the original publication, disclaimer etc
                ui.card(
                    ui.card_header(
                    ui.tags.h3("Dataset details", style="font-weight: bold; font-size: 20px;") 
                    ),
                    ui.markdown("""Please read the below-linked publication for all details as to how these precipitation 
                                δ²H, δ¹⁸O, and <i>dxs</i> values 
                                were produced. If you use data from this calculator, 
                                please cite the paper below.
                                """),
                    ui.a("Go to publication", href="https://egusphere.copernicus.org/preprints/2025/egusphere-2025-2458/", target="_blank", class_="btn btn-secondary")
                ),

                # link to zenodo repo for users to download the netcdfs
                ui.card(
                    ui.card_header(
                        ui.tags.h3("Download netcdf files", style="font-weight: bold; font-size: 20px;")
                        ),
                    ui.markdown(
                        """<a href="https://doi.org/10.5281/zenodo.15486277" target="_blank">This Zenodo repository</a> holds netcdf files 
                        with monthly precipitation isotope data across the Australian continent, at 0.25° spatial resolution. 
                        The data are available at monthly and annual temporal resolution.
                """
                    )
                ),
                # match sidebar display features to the timeseries tab
                width = 350,
                open = "always",
                ),
            ui.layout_columns(
                    # just one card on this tab
                    
                    ui.card(
                        # card header
                        ui.card_header("Matching locations",
                                    style="text-align: center; font-size: 20px; font-weight: bold;"),  
                        ui.output_plot("plot_matches"),
                        style="margin-top: 0px; width: 100%"
                    ),
                col_widths=(12, 12)
            ),
        )),
    
    ),
    # define theme
    theme = shinyswatch.get_theme('cerulean')
)
    

# NOW THE SERVER
def server(input, output, session):
    # reset site name when lat or lon are changed
    @reactive.Effect
    @reactive.event(input.lat, input.lon)
    def reset_inputs():
        ui.update_text("site_name", value="")

    # make pop-up information window appear when tab is selected
    @reactive.Effect
    def _():
        if input.active_tab() == "Extract timeseries":
            ui.modal_show(modal_ts)
        elif input.active_tab() == "Spatial search":
            ui.modal_show(modal_spatial)
    
    # when the app is first opened, show info window for the timeseries
    @session.on_flush
    def show_modal_on_load():
        ui.modal_show(modal_ts)
    
    # helper function to check lats/lons
    def is_valid_point(ds, lat, lon):
        try:
            da = ds.sel(lat=lat, lon=lon, method="nearest")
        except Exception:
            return False

        return not np.all(np.isnan(da.to_array()))
    # and one to show a modal
    def show_error_modal(message):
        ui.modal_show(
            ui.modal(
                ui.h4("Invalid lat/lon. Please check your coordinates."),
                ui.p(message),
                easy_close=True,
                footer=ui.input_action_button("dismiss_error", "OK")
            )
        )

    # TIMESERIES: function to get data at selected point
    def extract_timeseries(lat, lon):
        # extract relevant timeseries
        if input.time_res() == "ann":

            # annual data (Jan-Dec)
            d2H_ann = xr.open_dataset(f"{fpath}netcdfs/aus_prec.d2H_v1_1962-2023_ann_median.nc")
            d18O_ann = xr.open_dataset(f"{fpath}netcdfs/aus_prec.d18O_v1_1962-2023_ann_median.nc")
            dxs_ann = xr.open_dataset(f"{fpath}netcdfs/aus_prec.dxs_v1_1962-2023_ann_median.nc")

            site_name = input.site_name() if input.site_name() else "site"
            site_name = site_name.replace(" ", "_")
            d2H_vals = d2H_ann.sel(lat=lat, lon=lon, method="nearest").d2Hp.values
            d18O_vals = d18O_ann.sel(lat=lat, lon=lon, method="nearest").d18Op.values
            dxs_vals = dxs_ann.sel(lat=lat, lon=lon, method="nearest").dxsp.values
            time = d18O_ann.time.values

            return pd.DataFrame({'site': site_name, 'year': time, 'lat': lat, 'lon': lon, 'd2H': d2H_vals, 'd18O': d18O_vals, 'dxs': dxs_vals})
        elif input.time_res() == "ann_trop":

            # annual (Jul-Jun)
            H_ann_trop = xr.open_dataset(f"{fpath}netcdfs/aus_prec.d2H_v1_1962-2022_ann-trop.nc")
            O_ann_trop = xr.open_dataset(f"{fpath}netcdfs/aus_prec.d18O_v1_1962-2022_ann-trop.nc")
            d_ann_trop = xr.open_dataset(f"{fpath}netcdfs/aus_prec.dxs_v1_1962-2022_ann-trop.nc")

            H_ann_trop = H_ann_trop.rename({'year': 'time'})
            O_ann_trop = O_ann_trop.rename({'year': 'time'})
            d_ann_trop = d_ann_trop.rename({'year': 'time'})

            new_time_trop = [pd.Timestamp(year=year, month=7, day=1) for year in years_cal]
            new_time_trop = new_time_trop[:-1]

            
            H_ann_trop = H_ann_trop.assign_coords(time=("time", new_time_trop))
            O_ann_trop = O_ann_trop.assign_coords(time=("time", new_time_trop))
            d_ann_trop = d_ann_trop.assign_coords(time=("time", new_time_trop))

            site_name = input.site_name() if input.site_name() else "site"
            site_name = site_name.replace(" ", "_")
            d2H_vals = H_ann_trop.sel(lat=lat, lon=lon, method="nearest").d2Hp.values
            d18O_vals = O_ann_trop.sel(lat=lat, lon=lon, method="nearest").d18Op.values
            dxs_vals = d_ann_trop.sel(lat=lat, lon=lon, method="nearest").dxsp.values
            time = H_ann_trop.time.values

            return pd.DataFrame({'site': site_name, 'year': time, 'lat': lat, 'lon': lon, 'd2H': d2H_vals, 'd18O': d18O_vals, 'dxs': dxs_vals})
        elif input.time_res() == "DJF":

            # annual (DJF)
            H_djf = xr.open_dataset(f"{fpath}netcdfs/aus_prec.d2H_v1_1962-2022_ann-djf.nc")
            O_djf = xr.open_dataset(f"{fpath}netcdfs/aus_prec.d18O_v1_1962-2022_ann-djf.nc")
            d_djf = xr.open_dataset(f"{fpath}netcdfs/aus_prec.dxs_v1_1962-2022_ann-djf.nc")
     
            H_djf = H_djf.rename({'year': 'time'})
            O_djf = O_djf.rename({'year': 'time'})
            d_djf = d_djf.rename({'year': 'time'})

            new_time_djf = [pd.Timestamp(year=year, month=12, day=1) for year in years_cal]
            new_time_djf = new_time_djf[:-1]

            H_djf = H_djf.assign_coords(time=("time", new_time_djf))
            O_djf = O_djf.assign_coords(time=("time", new_time_djf))
            d_djf = d_djf.assign_coords(time=("time", new_time_djf))

            site_name = input.site_name() if input.site_name() else "site"
            site_name = site_name.replace(" ", "_")
            d2H_vals = H_djf.sel(lat=lat, lon=lon, method="nearest").d2Hp.values
            d18O_vals = O_djf.sel(lat=lat, lon=lon, method="nearest").d18Op.values
            dxs_vals = d_djf.sel(lat=lat, lon=lon, method="nearest").dxsp.values
            time = H_djf.time.values

            return pd.DataFrame({'site': site_name, 'year': time, 'lat': lat, 'lon': lon, 'd2H': d2H_vals, 'd18O': d18O_vals, 'dxs': dxs_vals})
        elif input.time_res() == "MAM":

            # annual (MAM)
            H_mam = xr.open_dataset(f"{fpath}netcdfs/aus_prec.d2H_v1_1962-2023_ann-mam.nc")
            O_mam = xr.open_dataset(f"{fpath}netcdfs/aus_prec.d18O_v1_1962-2023_ann-mam.nc")
            d_mam = xr.open_dataset(f"{fpath}netcdfs/aus_prec.dxs_v1_1962-2023_ann-mam.nc")
            
            H_mam = H_mam.rename({'year': 'time'})
            O_mam = O_mam.rename({'year': 'time'})
            d_mam = d_mam.rename({'year': 'time'})

            new_time_mam = [pd.Timestamp(year=year, month=5, day=31) for year in years_cal]

            
            H_mam = H_mam.assign_coords(time=("time", new_time_mam))
            O_mam = O_mam.assign_coords(time=("time", new_time_mam))
            d_mam = d_mam.assign_coords(time=("time", new_time_mam))

            site_name = input.site_name() if input.site_name() else "site"
            site_name = site_name.replace(" ", "_")
            d2H_vals = H_mam.sel(lat=lat, lon=lon, method="nearest").d2Hp.values
            d18O_vals = O_mam.sel(lat=lat, lon=lon, method="nearest").d18Op.values
            dxs_vals = d_mam.sel(lat=lat, lon=lon, method="nearest").dxsp.values
            time = H_mam.time.values

            return pd.DataFrame({'site': site_name, 'year': time, 'lat': lat, 'lon': lon, 'd2H': d2H_vals, 'd18O': d18O_vals, 'dxs': dxs_vals})
        elif input.time_res() == "JJA":

            # annual (JJA)
            H_jja = xr.open_dataset(f"{fpath}netcdfs/aus_prec.d2H_v1_1962-2023_ann-jja.nc")
            O_jja = xr.open_dataset(f"{fpath}netcdfs/aus_prec.d18O_v1_1962-2023_ann-jja.nc")
            d_jja = xr.open_dataset(f"{fpath}netcdfs/aus_prec.dxs_v1_1962-2023_ann-jja.nc")

            new_time_jja = [pd.Timestamp(year=year, month=8, day=31) for year in years_cal]

            H_jja = H_jja.rename({'year': 'time'})
            O_jja = O_jja.rename({'year': 'time'})
            d_jja = d_jja.rename({'year': 'time'})

            H_jja = H_jja.assign_coords(time=("time", new_time_jja))
            O_jja = O_jja.assign_coords(time=("time", new_time_jja))
            d_jja = d_jja.assign_coords(time=("time", new_time_jja))

            site_name = input.site_name() if input.site_name() else "site"
            site_name = site_name.replace(" ", "_")
            d2H_vals = H_jja.sel(lat=lat, lon=lon, method="nearest").d2Hp.values
            d18O_vals = O_jja.sel(lat=lat, lon=lon, method="nearest").d18Op.values
            dxs_vals = d_jja.sel(lat=lat, lon=lon, method="nearest").dxsp.values
            time = H_jja.time.values

            return pd.DataFrame({'site': site_name, 'year': time, 'lat': lat, 'lon': lon, 'd2H': d2H_vals, 'd18O': d18O_vals, 'dxs': dxs_vals})
        elif input.time_res() == "SON":

            # annual (SON)
            H_son = xr.open_dataset(f"{fpath}netcdfs/aus_prec.d2H_v1_1962-2023_ann-son.nc")
            O_son = xr.open_dataset(f"{fpath}netcdfs/aus_prec.d18O_v1_1962-2023_ann-son.nc")
            d_son = xr.open_dataset(f"{fpath}netcdfs/aus_prec.dxs_v1_1962-2023_ann-son.nc")

            H_son = H_son.rename({'year': 'time'})
            O_son = O_son.rename({'year': 'time'})
            d_son = d_son.rename({'year': 'time'})  

            new_time_son = [pd.Timestamp(year=year, month=11, day=30) for year in years_cal]

            H_son = H_son.assign_coords(time=("time", new_time_son))
            O_son = O_son.assign_coords(time=("time", new_time_son))
            d_son = d_son.assign_coords(time=("time", new_time_son))

            site_name = input.site_name() if input.site_name() else "site"
            site_name = site_name.replace(" ", "_")
            d2H_vals = H_son.sel(lat=lat, lon=lon, method="nearest").d2Hp.values
            d18O_vals = O_son.sel(lat=lat, lon=lon, method="nearest").d18Op.values
            dxs_vals = d_son.sel(lat=lat, lon=lon, method="nearest").dxsp.values
            time = H_son.time.values

            return pd.DataFrame({'site': site_name, 'year': time, 'lat': lat, 'lon': lon, 'd2H': d2H_vals, 'd18O': d18O_vals, 'dxs': dxs_vals})
        elif input.time_res() == "3mrm":

            # three-month running mean
            H_3m = xr.open_dataset(f"{fpath}netcdfs/aus_prec.d2H_v1_1962-2023_3-month-running-mean.nc")
            O_3m = xr.open_dataset(f"{fpath}netcdfs/aus_prec.d18O_v1_1962-2023_3-month-running-mean.nc")
            d_3m = xr.open_dataset(f"{fpath}netcdfs/aus_prec.dxs_v1_1962-2023_3-month-running-mean.nc")

            site_name = input.site_name() if input.site_name() else "site"
            site_name = site_name.replace(" ", "_")
            d2H_vals = H_3m.sel(lat=lat, lon=lon, method="nearest").d2Hp.values
            d18O_vals = O_3m.sel(lat=lat, lon=lon, method="nearest").d18Op.values
            dxs_vals = d_3m.sel(lat=lat, lon=lon, method="nearest").dxsp.values
            time = H_3m.time.values

            return pd.DataFrame({'site': site_name, 'date': time, 'lat': lat, 'lon': lon, 'd2H': d2H_vals, 'd18O': d18O_vals, 'dxs': dxs_vals})
        elif input.time_res() == "6mrm":

            # six-month running mean
            H_6m = xr.open_dataset(f"{fpath}netcdfs/aus_prec.d2H_v1_1962-2023_6-month-running-mean.nc")
            O_6m = xr.open_dataset(f"{fpath}netcdfs/aus_prec.d18O_v1_1962-2023_6-month-running-mean.nc")
            d_6m = xr.open_dataset(f"{fpath}netcdfs/aus_prec.dxs_v1_1962-2023_6-month-running-mean.nc")

            site_name = input.site_name() if input.site_name() else "site"
            site_name = site_name.replace(" ", "_")
            d2H_vals = H_6m.sel(lat=lat, lon=lon, method="nearest").d2Hp.values
            d18O_vals = O_6m.sel(lat=lat, lon=lon, method="nearest").d18Op.values
            dxs_vals = d_6m.sel(lat=lat, lon=lon, method="nearest").dxsp.values
            time = H_6m.time.values

            return pd.DataFrame({'site': site_name, 'date': time, 'lat': lat, 'lon': lon, 'd2H': d2H_vals, 'd18O': d18O_vals, 'dxs': dxs_vals})
        elif input.time_res() == "12mrm":

            # twelve-month running mean
            H_12m = xr.open_dataset(f"{fpath}netcdfs/aus_prec.d2H_v1_1962-2023_12-month-running-mean.nc")
            O_12m = xr.open_dataset(f"{fpath}netcdfs/aus_prec.d18O_v1_1962-2023_12-month-running-mean.nc")
            d_12m = xr.open_dataset(f"{fpath}netcdfs/aus_prec.dxs_v1_1962-2023_12-month-running-mean.nc")

            site_name = input.site_name() if input.site_name() else "site"
            site_name = site_name.replace(" ", "_")
            d2H_vals = H_12m.sel(lat=lat, lon=lon, method="nearest").d2Hp.values
            d18O_vals = O_12m.sel(lat=lat, lon=lon, method="nearest").d18Op.values
            dxs_vals = d_12m.sel(lat=lat, lon=lon, method="nearest").dxsp.values
            time = H_12m.time.values

            return pd.DataFrame({'site': site_name, 'date': time, 'lat': lat, 'lon': lon, 'd2H': d2H_vals, 'd18O': d18O_vals, 'dxs': dxs_vals})
        else:

            # monthly data
            d2H = xr.open_dataset(f"{fpath}netcdfs/aus_prec.d2H_v1_196201-202312_monthly_median.nc")
            d18O = xr.open_dataset(f"{fpath}netcdfs/aus_prec.d18O_v1_196201-202312_monthly_median.nc")
            dxs = xr.open_dataset(f"{fpath}netcdfs/aus_prec.dxs_v1_196201-202312_monthly_median.nc")

            site_name = input.site_name() if input.site_name() else "site"
            site_name = site_name.replace(" ", "_")
            d2H_vals = d2H.sel(lat=lat, lon=lon, method="nearest").d2Hp.values
            d18O_vals = d18O.sel(lat=lat, lon=lon, method="nearest").d18Op.values
            dxs_vals = dxs.sel(lat=lat, lon=lon, method="nearest").dxsp.values
            time = d18O.time.values
            
            return pd.DataFrame({'site_name': site_name, 'date': time, 'lat': lat, 'lon': lon, 'd2H': d2H_vals, 'd18O': d18O_vals, 'dxs': dxs_vals})

    # TIMESERIES: we only want to run the actions when the button is clicked
    @reactive.event(input.run_calcs)
    # get the timeseries data for the specified location
    def selected_location_data():
        lat = input.lat()
        lon = input.lon()

        # check ther lat/lon choice is valid
        ds_check = d18O_ann
        if not is_valid_point(ds_check, lat, lon):
            ui.notification_show(f"Lat/lon ({lat}, {lon}) is outside the grid area. Please check your coordinates and try again",type="error",duration=None)
            return pd.DataFrame()

        data = extract_timeseries(lat, lon)

        if input.date_range():
            start_date = pd.Timestamp(input.date_range()[0])
            end_date = pd.Timestamp(input.date_range()[1])

            if input.time_res() in ["ann", "ann_trop", "DJF", "MAM", "JJA", "SON"]:
                data = data[(data['year'] >= start_date) & (data['year'] <= end_date)]
            else:
                data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]

        return data

    # TIMESERIES: make the timeseries plots
    @output
    @render_widget
    @reactive.event(input.run_calcs)
    def plot_ts():
        data = selected_location_data()
        
        time_ax = "year" if input.time_res() in ["ann", "ann_trop", "DJF", "MAM", "JJA", "SON"] else "date"
        time_ax_title = "Year" if input.time_res() in ["ann", "ann_trop", "DJF", "MAM", "JJA", "SON"] else "Date"

        # plotly/shiny together are weird about dates...
        if np.issubdtype(data[time_ax].dtype, np.datetime64):
            data[time_ax] = data[time_ax].dt.strftime("%Y-%m")
    
        # initialise plotly figure
        fig = go.Figure()

        d2H_col = "#193333"
        d18O_col = "#3e91c7"
        dxs_col = "#729a7e"

        fig.add_trace(go.Scatter(x=data[time_ax], y=data['d2H'], mode='lines+markers',name="δ²H", line=dict(color=d2H_col), yaxis="y1"))
        fig.add_trace(go.Scatter(x=data[time_ax], y=data['d18O'], mode='lines+markers', name="δ¹⁸O", line=dict(color=d18O_col), yaxis="y2"))
        fig.add_trace(go.Scatter(x=data[time_ax], y=data['dxs'], mode='lines+markers', name="dxs", line=dict(color=dxs_col), yaxis="y3"))

        fig.update_layout(
            title=None,
            xaxis_title=time_ax_title,
            showlegend=False, 
            xaxis=dict(
                anchor='y3',
                tickformat="%Y" if input.time_res() in ["ann", "ann_trop", "DJF", "MAM", "JJA", "SON"] else "%Y-%m"
            ),
            yaxis=dict(
                title=dict(
                    text=r"$\delta^{2}\mathrm{H}\ (\text{‰}_{\text{VSMOW}})$",
                    font=dict(color=d2H_col)
            ),
            domain=[0.7, 1],
            tickfont=dict(color=d2H_col),
),
            yaxis2=dict(
                title=dict(
                    text=r"$\delta^{18}\mathrm{O}\ (\text{‰}_{\text{VSMOW}})$",
                    font=dict(color=d18O_col) 
                ),
            domain=[0.35, 0.7],
            tickfont=dict(color=d18O_col),
            ),
            yaxis3=dict(
                title=dict(
                    text=r"$\mathit{dxs}\ (\text{‰}_{\text{VSMOW}})$",
                    font=dict(color=dxs_col)
                ),
            domain=[0, 0.35],
            tickfont=dict(color=dxs_col),
            ),
            template="simple_white"
                )

        return fig

    # TIMESERIES: location map
    @output
    @render.ui
    @reactive.event(input.run_calcs)
    def loc_map():
        lat, lon = input.lat(), input.lon()
        
        # this chunk creates a basic map then sets the map background
        m = folium.Map(location=[-28, 134], zoom_start=3, tiles = "CartoDB positron", name = "positron", show = True)

        # now add a marker showing the point
        folium.Marker(location=[lat, lon], tooltip=f"User-defined point: ({lat}, {lon})").add_to(m)
        folium.LayerControl().add_to(m)

        return ui.HTML(
            f"""
            <div style="width: 100%; height: 100%; overflow: hidden; position: relative;">
                {m._repr_html_()}
            </div>
            """
        )   
    
    # TIMESERIES: scatter plot (LMWL)
    @output
    @render_widget
    @reactive.event(input.run_calcs)
    def lmwl():
        data = selected_location_data()
        if input.time_res() in ["ann", "ann_trop", "DJF", "MAM", "JJA", "SON"]:
            resolution = "annual"
        else:
            resolution = input.time_res()
        site_label = input.site_name() or f"[{input.lat()}, {input.lon()}]"

        if resolution in ["monthly", "3mrm", "6mrm", "12mrm"]:
            months = data['date'].dt.month
            colours = months
            hover_text = data['date'].dt.strftime("%b %Y")
            colour_scale = "twilight"
        else:
            years = data['year'].dt.year
            colours = years
            hover_text = data['year'].dt.strftime("%Y")
            colour_scale = 'viridis'

        fig = go.Figure()

        x_line = [data["d18O"].min(), data["d18O"].max()]
        y_line = [8 * x + 10 for x in x_line]
        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            line=dict(color="black", width = 1),
            name="",
            hoverinfo="skip",
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x = data["d18O"],
            y = data["d2H"],
            mode = "markers",
            marker = dict(
                size = 10,
                color = colours,
                colorscale = colour_scale,
                colorbar=dict(title="Month" if resolution == "monthly" else "Year",
                              thickness = 10),
            ),
            showlegend = False,
            text = hover_text,
            hovertemplate=(
                #"δ18O: %{x:.2f}‰<br>"
                #"δ2H: %{y:.2f}‰<br>"
                "%{text}<extra></extra>"
            )
        ))

        fig.update_layout(
            title=f"{site_label}: {resolution} resolution",
            xaxis_title=r"$\delta^{18}\mathrm{O}\ (\text{‰}_{\text{VSMOW}})$",
            yaxis_title=r"$\delta^{2}\mathrm{H}\ (\text{‰}_{\text{VSMOW}})$",
            template="simple_white"
            )

        return fig  

    # TIMESERIES: download a csv with values for the selected location
    @output
    @render.download(filename=lambda: generate_csv_fname())
    def download_csv():

        # metadata for the csv header
        metadata = [
          f"# Data downloaded {datetime.now().strftime('%Y-%m-%d')}",
          "# Please see Falster et al 2025 (HESS) for reference and data details"
        ]

        if input.time_res() in ["ann", "ann_trop", "DJF", "MAM", "JJA", "SON"]:
            data = selected_location_data()
            data['year'] = pd.to_datetime(data['year'], format='%Y').dt.year
        else:
            data = selected_location_data() 

        if not input.site_name():
            data['site_name'] = 'no_sitename_specified'

        for line in metadata:
            yield line + "\n"
            
        yield data.to_csv(index=False)
    
    # function to generate the csv filename
    def generate_csv_fname():
        site_name = f"{input.site_name()}_" if input.site_name() else ""
        lat = input.lat()
        lon = input.lon()
        start_date = input.date_range()[0] if input.date_range() else "1962-01-01"
        end_date = input.date_range()[1] if input.date_range() else "2023-12-31"

        start_date = start_date.strftime("%Y%m%d")
        end_date = end_date.strftime("%Y%m%d")

        resolution = input.time_res()

        # update to add a readme tab
        filename = f"{site_name}lat{lat}_lon{lon}_{resolution}_{start_date}-{end_date}.csv"
        filename = filename.replace("/", "_").replace("\\", "_").replace(" ", "")

        return filename
    
    # SPATIAL SEARCH: a function to select the appropriate dataset

    @reactive.calc
    def get_chosen_system():
        if input.isotope() == "d2H":
            return d2H.d2Hp, d2H_ann.d2Hp, d2H_mean.d2Hp
        if input.isotope() == 'd18O':
            return d18O.d18Op, d18O_ann.d18Op, d18O_mean.d18Op
        if input.isotope() == 'dxs':
            return dxs.dxsp, dxs_ann.dxsp, dxs_mean.dxsp

    # SPATIAL SEARCH: a function to update the time inputs
    @reactive.calc
    def get_time_inputs():
        these_months = input.months_spatial()
        months = [int(m) for m in these_months]
        year_start = input.year_start()
        year_end = input.year_end()

        return months, year_start, year_end
        
    # SPATIAL SEARCH: perform the spatial search
    @reactive.calc
    def get_mapdata():
        dat_mth, dat_ann, dat_mean = get_chosen_system()

        input_val = input.input_val()
        input_range = input.input_range()
        offset = input.offset()

        input_val_adj = input_val-offset
        input_lwr = input_val_adj-input_range
        input_upr = input_val_adj+input_range

        # do we need to to any calculations:
        if input.search_type() =="Long-term mean":
            exact_match = dat_mean.where((dat_mean >= input_lwr) & (dat_mean <= input_upr))
            return exact_match
        else:
            months, year_start, year_end = get_time_inputs()


            dat_red = dat_mth.where(
                ((dat_mth['time.year'] >= year_start) & (dat_mth['time.year'] <= year_end)) &
                (dat_mth['time'].dt.month.isin(months)), drop=True)
            
            prec_red = prec.where(
                ((prec['time.year'] >= year_start) & (prec['time.year'] <= year_end)) &
                (prec['time'].dt.month.isin(months)), drop=True)
            
            # amount-weight the values
            PREC_mth = prec_red.groupby('time.year')
            PREC_ann = prec_red.groupby('time.year').sum()
                
            dat_wtd = (dat_red*(PREC_mth/PREC_ann)).resample(time='YE').sum()
            dat_wtd = dat_wtd.where(dat_wtd != 0.)
            dat_wtd_mean = dat_wtd.mean(dim="time") 

            # find matches
            exact_match = dat_wtd_mean.where((dat_wtd_mean >= input_lwr) & (dat_wtd_mean <= input_upr)) 
            return exact_match
    
    # SPATIAL SEARCH: make the plot
    @output
    @render.plot
    @reactive.event(input.run_spatial_search)
    def plot_matches():
        # functions for the plotting
        def make_titles(search_type, chosen_system, input_lwr, input_upr, year_start, year_end, months):
    
            if chosen_system == 'd2H':
                system_str = r"$\delta^{2}\mathrm{H}$"
            if chosen_system == 'd18O':
                system_str = r"$\delta^{18}\mathrm{O}$"
            if chosen_system == 'dxs':
                system_str = r"$\mathit{dxs}$"

            months_str = ", ".join(str(m) for m in months)

            if search_type == "Long-term mean":
                title = f"Locations where precipitation {system_str} is between {input_lwr}‰ and {input_upr}‰ in the long-term annual mean"
                subtitle = f"{year_start} to {year_end}"
                label = f"Precipitation {system_str} (‰VSMOW)" 
            else:
                title = f"Locations where precipitation {system_str} is between {input_lwr}‰ and {input_upr}‰ in the long-term mean"
                subtitle = f"{year_start} to {year_end}, including months {months_str}"
                label = f"Precipitation {system_str} (‰VSMOW)"

            return title, subtitle, label 

        def get_value_lims(search_type, input_lwr, input_upr):
            if search_type == "Long-term mean":
                vmin = math.ceil(input_lwr)
                vmax = math.floor(input_upr)  
                extend_type = "both"
                cmap = "copper_r"
            if search_type == "Mean over time period":
                vmin = math.ceil(input_lwr)
                vmax = math.floor(input_upr) 
                extend_type = "both"
                cmap = "copper_r"
        
            return vmin, vmax, extend_type, cmap
    
        map_dat = get_mapdata()

        input_val = input.input_val()
        input_range = input.input_range()
        offset = input.offset()

        input_val_adj = input_val-offset
        input_lwr = input_val_adj-input_range
        input_upr = input_val_adj+input_range

        year_start = input.year_start()
        year_end = input.year_end()
        these_months = input.months_spatial()
        #months_int = [int(m) for m in these_months]

        # first set the various parameters and get plotting inputs
        mpl.rcParams['font.family'] = 'Arial'
        mpl.rcParams['text.color'] = 'black'
        mpl.rcParams['axes.labelcolor'] = 'black'
        mpl.rcParams['xtick.color'] = 'black'
        mpl.rcParams['ytick.color'] = 'black'

        new_proj = ccrs.PlateCarree()
        dat_proj = ccrs.PlateCarree()

        title, subtitle, label = make_titles(input.search_type(), input.isotope(), input_lwr, input_upr, year_start, year_end, input.months_spatial())

        vmin, vmax, extend_type, cmap = get_value_lims(input.search_type(), input_lwr, input_upr)

        # now make the graphic
        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': new_proj})
        
        im = map_dat.plot(ax=ax, transform=dat_proj, cmap=cmap,
                         add_colorbar=False, vmin=vmin, vmax=vmax, add_labels=False)
        
        ax.set_extent([110, 155, -45, -10], crs=ccrs.PlateCarree())

        shpfilename = natural_earth(resolution='10m',category='cultural',name='admin_0_countries')
        reader = Reader(shpfilename)
        australia_geom = [record.geometry for record in reader.records()
                          if record.attributes['NAME_LONG'] == 'Australia']

        ax.add_geometries(australia_geom, crs=ccrs.PlateCarree(),edgecolor='black', facecolor='none', linewidth=0.5)

        ax.set_title(title, fontname='Arial', color='black', fontsize=12, loc="left", pad=20)
        ax.text(0, 0.99, subtitle, ha='left', va='bottom', transform=ax.transAxes,
                fontname='Arial', color='black', fontsize=10)

        ax.axis('off')

        im = ax.collections[0] 

        cbar = fig.colorbar(im, orientation='vertical', fraction=0.02, pad=0.04, extend=extend_type)
        cbar.set_label(label, fontsize=10)

        return fig
    
# create the Shiny app
app = App(app_ui, server)