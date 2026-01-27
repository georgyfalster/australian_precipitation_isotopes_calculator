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

# we'll also need the precipitation amount data for if users want to specific time periods
prec = xr.open_dataset(f"{fpath}netcdfs/prec/aus_prec_v1_195901-202312_monthly_1.nc")
prec = prec["prec"].sel(time=slice("1962-01-01", None))

# long-term mean (calendar year)
d2H_mean = xr.open_dataset(f"{fpath}netcdfs/aus_prec.d2H_v1_1962-2023_long-term-annual-mean_median.nc")
d18O_mean = xr.open_dataset(f"{fpath}netcdfs/aus_prec.d18O_v1_1962-2023_long-term-annual-mean_median.nc")
dxs_mean = xr.open_dataset(f"{fpath}netcdfs/aus_prec.dxs_v1_1962-2023_long-term-annual-mean_median.nc")

# this is for the location check
non_null_points = d18O_ann.d18Op.notnull().all(dim='time').stack(point=('lat', 'lon'))
non_null_points = non_null_points.where(non_null_points, drop=True)

valid_x = non_null_points.lon.values
valid_y = non_null_points.lat.values
   
# define pop-up information windows
modal_ts = ui.modal(
    ui.markdown(
        """**Note that this tab is currently showing a minimal version (without plots). This is temporary while Georgy deals with a recent Shiny update 
        that broke the plotting functions.** 
        <br><br>To download data from your chosen location, enter the coordinates and define the temporal resolution in the sidebar to the left, then
        press the download button. 
        You have the option to assign a site name which will be appended to the data download, although this is not required.
        <b>All values are precipitation amount-weighted means</b>.
        For values averaged over July-June, the year index applies to the calendar year at the <i>start</i> of the 
        averaging period.
        <br><br>For more complicated use cases e.g., extrating data for many sites: if you are not comfortable using 
        netcdf files (which are available for download from Zenodo; see details in sidebar) please email Georgy with the request (georgina.falster@adelaide.edu.au).
        <br><br>It is important to note that these are modelled values, not primary observations.
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
modal_isoscape = ui.modal(
    ui.markdown(
        """These maps show the precipitation amount-weighted long-term (1962-2023) annual mean δ²H<sub>P</sub>, δ¹⁸O<sub>P</sub>, and <i>dxs</i><sub>P</sub>
         values across the Australian continent.
        <br><br> It is important to note that these are modelled values, not primary observations.
        """
    ),
    title = "Long-term mean annual isoscapes",
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
            """This online calculator allows users to extract modelled precipitation isotope δ²H, δ¹⁸O, and <i>dxs</i> values 
             for any location on the Australian continent, within the time period January 1962 to December 2023. 
             <br><br>The first two tabs below are for different data extraction types: timeseries or location search. On the `Extract timeseries` page 
             you can enter a location (latitude and
             longitude), choose a temporal resolution and optional date range, then view and download the precipitation isotope δ²H, δ¹⁸O, and <i>dxs</i> values 
             for your chosen location. On the 'Spatial search' page, you 
             can enter a δ²H, δ¹⁸O, or <i>dxs</i> value as well as an optional expected offest from precipitation δ²H/δ¹⁸O/<i>dxs</i> and time period of interest. 
             You will then see a map of locations where that sample could have come from. 
             <br><br>The third tab provides simple maps of long-term mean precipitation δ²H/δ¹⁸O/<i>dxs</i> values across the continent 
             (equivalent to previously-published long-term mean isoscapes).
             <br><br>When choosing a tab, an information window will appear with further important details. To make the information window reappear, click the relevant tab. 
             <br><br>If using data from this online calculator, please cite 
             the <a href="https://hess.copernicus.org/articles/30/289/2026/hess-30-289-2026.html" target="_blank">original publication</a>. Please also see the 
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
    
    # top-level UI set-up: timeseries or spatial search? or just the images of the long-term mean
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
                    choices = {"monthly": "Monthly", "ann": "Annual (Jan-Dec)", "ann_trop":"Annual (Jul-Jun)"},
                    selected="monthly")
                ),

                # optional inputs: date range, site name for download
                ui.card(
                    ui.card_header(
                        ui.tags.h3("Optional inputs", style="font-weight: bold; font-size: 20px;")
                    ),
                    ui.input_text("site_name",
                        ui.HTML("Site name <br><i>resets when lat and/or lon are changed</i>")),
                ),
    

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
                    ui.a("Go to publication", href="https://hess.copernicus.org/articles/30/289/2026/hess-30-289-2026.html", target="_blank", class_="btn btn-secondary")
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
                    ui.card_header("Download values for selected location and temporal resolution",
                                style="text-align: center; font-size: 20px; font-weight: bold;"),
                    ui.download_button("download_csv", "Click here to download data (after selecting location/resolution to the left)",
                        class_="btn btn-secondary"),
                    height = "700px"
                ),

                col_widths = (12, 12), # the Shiny CSS has 12 columns: so this wraps around to two rows
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
                    ui.a("Go to publication", href="https://hess.copernicus.org/articles/30/289/2026/hess-30-289-2026.html", target="_blank", class_="btn btn-secondary")
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
        ui.nav_panel("Long-term mean isoscapes", ui.layout_sidebar(
            ui.sidebar(
                ui.card(
                    ui.card_header(
                        ui.tags.h3("Inputs", style="font-weight: bold; font-size: 20px;")
                        ),
                        ui.input_radio_buttons("isotope_scape", "",
                                        choices = {"d2H": "δ²H   ", "d18O": "δ¹⁸O   ", "dxs":"dxs   "},
                                        selected = "d18O", inline=True
                        ),
                    ui.input_select("cmap_isoscape", "Colormap", 
                                    choices = {"bone":"Blues", "viridis":"Viridis", "copper":"Copper"},
                                    selected = "bone"
                    )
                ),
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
                    ui.a("Go to publication", href="https://hess.copernicus.org/articles/30/289/2026/hess-30-289-2026.html", target="_blank", class_="btn btn-secondary")
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
                ui.card(
                    # card header
                    ui.card_header("Long-term mean isoscapes",
                                   style="text-align: center; font-size: 20px; font-weight: bold;"),
                        ui.output_plot("plot_isoscapes"),style="margin-top: 0px; width: 100%"
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
        elif input.active_tab() == "Long-term mean isoscapes":
            ui.modal_show(modal_isoscape)
    
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

    def plot_isoscape_maps(fig, ax, dat, dat_proj, new_proj, title,vmin, vmax, cmap, cbar_lab):
        im = dat.plot(ax=ax,transform=dat_proj,cmap=cmap,add_colorbar=False,vmin=vmin,vmax=vmax)

        ax.set_extent([110, 155, -45, -10], crs=ccrs.PlateCarree())

        # Australia outline
        shpfilename = natural_earth(resolution="10m",category="cultural",name="admin_0_countries")
        reader = Reader(shpfilename)
        australia_geom = [
            rec.geometry for rec in reader.records()
            if rec.attributes["NAME_LONG"] == "Australia"
        ]

        states_shp = natural_earth(resolution='10m',category='cultural',name='admin_1_states_provinces')
        states_reader = Reader(states_shp)
        aus_states = [
            rec.geometry for rec in states_reader.records()
            if rec.attributes.get('admin') == 'Australia'
        ]

        ax.add_geometries(aus_states,crs=ccrs.PlateCarree(),edgecolor='black',facecolor='none',linewidth=0.5, zorder=3)
        ax.add_geometries(australia_geom, crs=ccrs.PlateCarree(),edgecolor='black', facecolor='none', linewidth=0.8, zorder=4)

        #ax.set_title(title, fontsize=14)

        ax.axis("off")

        cbar = fig.colorbar(im,ax=ax,orientation="vertical",shrink=0.4,pad=0.02, extend = "both")
        cbar.set_label(cbar_lab, fontsize=11)

        return im

    # the easy one (just show long-term mean maps; not reactive in any way)
    @output
    @render.plot
    def plot_isoscapes():

        which_iso = input.isotope_scape()
        this_cmap = input.cmap_isoscape()

        if which_iso == "d18O":
            da, vmin, vmax, lab = (d18O_mean.d18Op, -7, -3, "δ¹⁸O (‰ VSMOW)")
            title = r"Long-term mean $\delta^{18}\mathrm{O}_{\mathrm{p}}$ isoscape (1962–2023)"

        elif which_iso == "d2H":
            da, vmin, vmax, lab = (d2H_mean.d2Hp, -45, -5, "δ²H (‰ VSMOW)")
            title = r"Long-term mean $\delta^{2}\mathrm{H}_{\mathrm{p}}$ isoscape (1962–2023)"

        elif which_iso == "dxs":
            da, vmin, vmax, lab = (dxs_mean.dxsp, 5, 16, r"$\mathit{dxs}$")
            title = r"Long-term mean annual $\mathit{dxs}$ isoscape (1962–2023)"

        mpl.rcParams['font.family'] = 'Arial'
        mpl.rcParams['text.color'] = 'black'
        mpl.rcParams['axes.labelcolor'] = 'black'
        mpl.rcParams['xtick.color'] = 'black'
        mpl.rcParams['ytick.color'] = 'black'

        dat_proj = new_proj = ccrs.PlateCarree()

        fig, ax = plt.subplots(figsize=(10, 7),subplot_kw={"projection": ccrs.PlateCarree()})

        plot_isoscape_maps(fig, ax, da, dat_proj, new_proj, "",vmin, vmax, cmap=this_cmap, cbar_lab=lab)


        fig.suptitle(title, fontsize=14, y=0.98)
        fig.tight_layout()
        return fig
    
    # TIMESERIES: function to get data at selected point
    def extract_timeseries(lat, lon):
        # extract relevant timeseries
        if input.time_res() == "ann":

            site_name = input.site_name() if input.site_name() else "site"
            site_name = site_name.replace(" ", "_")
            d2H_vals = d2H_ann.sel(lat=lat, lon=lon, method="nearest").d2Hp.values
            d18O_vals = d18O_ann.sel(lat=lat, lon=lon, method="nearest").d18Op.values
            dxs_vals = dxs_ann.sel(lat=lat, lon=lon, method="nearest").dxsp.values
            time = d18O_ann.time.values

            return pd.DataFrame({'site': site_name, 'year': time, 'lat': lat, 'lon': lon, 'd2H': d2H_vals, 'd18O': d18O_vals, 'dxs': dxs_vals})
        elif input.time_res() == "ann_trop":

            site_name = input.site_name() if input.site_name() else "site"
            site_name = site_name.replace(" ", "_")
            d2H_vals = H_ann_trop.sel(lat=lat, lon=lon, method="nearest").d2Hp.values
            d18O_vals = O_ann_trop.sel(lat=lat, lon=lon, method="nearest").d18Op.values
            dxs_vals = d_ann_trop.sel(lat=lat, lon=lon, method="nearest").dxsp.values
            time = H_ann_trop.time.values

            return pd.DataFrame({'site': site_name, 'year': time, 'lat': lat, 'lon': lon, 'd2H': d2H_vals, 'd18O': d18O_vals, 'dxs': dxs_vals})
        
        else:

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

        distances = np.sqrt((valid_y - lat)**2 + (valid_x - lon)**2)

        idx = distances.argmin()
        lat = valid_y[idx]
        lon = valid_x[idx]

        if not is_valid_point(d18O_ann, lat, lon):
            ui.notification_show(f"Lat/lon ({lat}, {lon}) is outside the grid area. Please check your coordinates and try again",type="error",duration=None)
            return pd.DataFrame()

        data = extract_timeseries(lat, lon)

        return data



    # TIMESERIES: download a csv with values for the selected location
    @output
    @render.download(filename=lambda: generate_csv_fname())
    def download_csv():
        lat = input.lat()
        lon = input.lon()

        # nearest valid point
        distances = np.sqrt((valid_y - lat)**2 + (valid_x - lon)**2)
        idx = distances.argmin()
        lat = valid_y[idx]
        lon = valid_x[idx]

        if not is_valid_point(d18O_ann, lat, lon):
            ui.notification_show(f"Lat/lon ({lat}, {lon}) is outside the grid area. Please check your coordinates and try again",
                                 type="error", duration=None)
            return

        # extract data (runs automatically now)
        if input.time_res() in ["ann", "ann_trop", "DJF", "MAM", "JJA", "SON"]:
            data = extract_timeseries(lat, lon)
            data['year'] = pd.to_datetime(data['year'], format='%Y').dt.year
        else:
            data = extract_timeseries(lat, lon)


        if not input.site_name():
            data['site_name'] = 'no_sitename_specified'

        # metadata header
        metadata = [
            f"# Data downloaded {datetime.now().strftime('%Y-%m-%d')}",
            "# Please see Falster et al 2026 (HESS) for reference and data details"
        ]

        for line in metadata:
            yield line + "\n"
    
        yield data.to_csv(index=False)
    
    # function to generate the csv filename
    def generate_csv_fname():
        site_name = f"{input.site_name()}_" if input.site_name() else ""
        lat = input.lat()
        lon = input.lon()

        resolution = input.time_res()

        # update to add a readme tab
        filename = f"{site_name}lat{lat}_lon{lon}_{resolution}_19620101-20231231.csv"
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
                title = f"Locations where precipitation {system_str} is between {input_lwr:.2f}‰ and {input_upr:.2f}‰ in the long-term annual mean"
                subtitle = f"{year_start} to {year_end}"
                label = f"Precipitation {system_str} (‰VSMOW)" 
            else:
                title = f"Locations where precipitation {system_str} is between {input_lwr:.2f}‰ and {input_upr:.2f}‰ in the long-term mean"
                subtitle = f"{year_start} to {year_end}, including months {months_str}"
                label = f"Precipitation {system_str} (‰VSMOW)"

            return title, subtitle, label 

        def get_value_lims(search_type, input_lwr, input_upr):
            if search_type == "Long-term mean":
                #vmin = math.ceil(input_lwr) # this can skew the colorbar a bit - we want the desired value in the middle
                #vmax = math.floor(input_upr) 
                vmin = input_lwr
                vmax = input_upr 
                extend_type = "both"
                cmap = "twilight"
            if search_type == "Mean over time period":
                #vmin = math.ceil(input_lwr)
                #vmax = math.floor(input_upr) 
                vmin = input_lwr
                vmax = input_upr 
                extend_type = "both"
                cmap = "twilight"
        
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
        
        states_shp = natural_earth(resolution='10m',category='cultural',name='admin_1_states_provinces')
        states_reader = Reader(states_shp)
        aus_states = [
            rec.geometry for rec in states_reader.records()
            if rec.attributes.get('admin') == 'Australia'
            ]

        ax.add_geometries(aus_states,crs=ccrs.PlateCarree(),edgecolor='black',facecolor='none',linewidth=0.5, zorder=3)
        ax.add_geometries(australia_geom, crs=ccrs.PlateCarree(),edgecolor='black', facecolor='none', linewidth=0.8, zorder=4)

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

#app.run()