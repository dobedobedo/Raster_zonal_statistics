#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 09:54:12 2018

@author: uqytu1
"""

import os
import sys
import glob
import gc
from difflib import SequenceMatcher
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename, askdirectory, asksaveasfilename
from tkinter import messagebox
import numpy as np
from osgeo import ogr, osr, gdal
import pandas as pd

AOI_filetype = [('Shapefile', ['*.shp']), ('Geopackage', ['*.gpkg']), ('GeoJSON', ['*.geojson', '*.json'])]
Raster_filetype = '.tif'
Out_filetypes=[('Microsoft Excel',['*.xls','*.xlsx'])]

def Create_Masked_Image(lyr, total_FIDs, Image, *udm):
    # Define a function to extract the extent of input vector geometries
    def ExtractGeomExtent(geom, coordTrans):
        try:
            geom.Transform(coordTrans)
        except TypeError:
    # If any of the coordinate system have nothing, assume they're in the same CSR and apply no transform
            pass
    # Get extent of feat
        geom = feat.GetGeometryRef()
        if (geom.GetGeometryName() == 'MULTIPOLYGON'):
            count = 0
            pointsX = []; pointsY = []
            for polygon in geom:
                geomInner = geom.GetGeometryRef(count)
                ring = geomInner.GetGeometryRef(0)
                numpoints = ring.GetPointCount()
                for p in range(numpoints):
                        x, y, z = ring.GetPoint(p)
                        pointsX.append(x)
                        pointsY.append(y)
                count += 1
        elif (geom.GetGeometryName() == 'POLYGON'):
            ring = geom.GetGeometryRef(0)
            numpoints = ring.GetPointCount()
            pointsX = []; pointsY = []
            for p in range(numpoints):
                    x, y, z = ring.GetPoint(p)
                    pointsX.append(x)
                    pointsY.append(y)
        else:
            Message = 'Geometry needs to be either Polygon or Multipolygon'
            ErrorBox('Error!', Message)
            sys.exit('ERROR: Geometry needs to be either Polygon or Multipolygon')
        xmin = min(pointsX)
        xmax = max(pointsX)
        ymin = min(pointsY)
        ymax = max(pointsY)
        
        return xmin, xmax, ymin, ymax
    
    # Open raster files
    InputFile = gdal.Open(Image)
    
    # Open usable data mask raster if there is any
    if len(udm) > 0:
        udm_config = udm[1]
        udm_File = gdal.Open(udm[0])
        if len(udm_config) > 0:
            udm_channels_selected = list()
            udm_criterion = list()
            for _channel, _criteria in udm_config.items():
                udm_channels_selected.append(_channel)
                udm_criterion.append(_criteria)
        else:
            udm_channels = udm_File.RasterCount
            udm_channels_selected = CheckBox(list(map(lambda x: f'Band {str(x+1)}', range(udm_channels))), 
                                             'Which udm bands to use?')
            udm_criterion = list()
            for _udm_channel in udm_channels_selected:
                udm_criterion.append(UserInputBox(
                    f'{_udm_channel} mask value', 
                    f'Please enter usable value for {_udm_channel}\nTips: Type >, <, >=, or <= to set threshold (e.g. >= 1)', 
                    allow_none=False, isnum=True))
            udm_channels_selected = list(map(lambda x: int(x.split(' ')[1]), udm_channels_selected))
            for _idx, _channel in enumerate(udm_channels_selected):
                udm_config[_channel] = udm_criterion[_idx]
            
    # Get raster georeference info
    GeoTransform = InputFile.GetGeoTransform()
    xOrigin = GeoTransform[0]
    yOrigin = GeoTransform[3]
    pixelWidth = GeoTransform[1]
    pixelHeight = GeoTransform[5]
    channels = InputFile.RasterCount
    
    # Get usable data mast georefernece info if there is any
    try:
        GeoTransform_udm = udm_File.GetGeoTransform()
        xOrigin_udm = GeoTransform_udm[0]
        yOrigin_udm = GeoTransform_udm[3]
        pixelWidth_udm = GeoTransform_udm[1]
        pixelHeight_udm = GeoTransform_udm[5]
    except NameError:
        pass
    
    # Reproject vector geometry to same projection as raster
    sourceSR = lyr.GetSpatialRef()
    targetSR = osr.SpatialReference()
    targetSR.ImportFromWkt(InputFile.GetProjectionRef())
    coordTrans = osr.CoordinateTransformation(sourceSR,targetSR)
    
    # Calculate the transformation to usable data mask raster
    try:
        targetSR_udm = osr.SpatialReference()
        targetSR_udm.ImportFromWkt(udm_File.GetProjectionRef())
        coordTrans_udm = osr.CoordinateTransformation(sourceSR, targetSR_udm)
    except NameError:
        pass
    
    Image_set = dict()

    # Loop through features
    for FID in total_FIDs:
        feat = lyr.GetFeature(FID)
        geom = feat.GetGeometryRef()
        
        xmin, xmax, ymin, ymax = ExtractGeomExtent(geom, coordTrans)
        
    # Calculate the extents of vector on the udm raster if there is any
        try:
            xmin_udm, xmax_udm, ymin_udm, ymax_udm = ExtractGeomExtent(geom, coordTrans_udm)
        except NameError:
            pass
        
    # Specify offset and rows and columns to read
        xoff = int((xmin - xOrigin)/pixelWidth)
        yoff = int((yOrigin - ymax)/pixelWidth)
        xcount = int((xmax - xmin)/pixelWidth)+1
        ycount = int((ymax - ymin)/pixelWidth)+1
        
    # Skip the feature if the extent is outside the image extent
        if xoff < 0 or yoff <0:
            continue
        
    # Specify offset and rows and columns of udm to read if there is any
        try:
            xoff_udm = int((xmin_udm - xOrigin_udm)/pixelWidth_udm)
            yoff_udm = int((yOrigin_udm - ymax_udm)/pixelWidth_udm)
            xcount_udm = int((xmax_udm - xmin_udm)/pixelWidth_udm)+1
            ycount_udm = int((ymax_udm - ymin_udm)/pixelWidth_udm)+1
        except NameError:
            pass
        
    # Create memory target raster
        target_ds = gdal.GetDriverByName('MEM').Create('', xcount, ycount, channels, gdal.GDT_Byte)
        target_ds.SetGeoTransform((xmin, pixelWidth, 0, 
                                   ymax, 0, pixelHeight, ))
        
    # Create for target raster the same projection as for the input raster
        target_ds.SetProjection(targetSR.ExportToWkt())
        
    # Rasterize zone polygon to raster
        gdal.RasterizeLayer(target_ds, [1], lyr, burn_values=[1])
        
    # Create mask
        bandmask = target_ds.GetRasterBand(1)
        datamask = bandmask.ReadAsArray(0, 0, xcount, ycount).astype(float)
        
    # Create the udm mask if there is any
        try:
            target_ds_udm = gdal.GetDriverByName('MEM').Create('', xcount_udm, ycount_udm, 1, gdal.GDT_Byte)
            target_ds_udm.SetGeoTransform((xmin_udm, pixelWidth_udm, 0, 
                                           ymax_udm, 0, pixelHeight_udm, ))
            
    # Create for target raster the same projection as for the udm raster
            target_ds_udm.SetProjection(targetSR_udm.ExportToWkt())
            
    # Rasterize zone polygon to raster
            gdal.RasterizeLayer(target_ds_udm, [1], lyr, burn_values=[1])
            
    # Create mask for udm
            udm_zoneraster = list()
            bandmask_udm = target_ds_udm.GetRasterBand(1)
            datamask_udm = bandmask_udm.ReadAsArray(0, 0, xcount_udm, ycount_udm).astype(float)
            for _idx in udm_channels_selected:
                udm_band = udm_File.GetRasterBand(_idx)
                udm_ndv = udm_band.GetNoDataValue()
                udm_image = udm_band.ReadAsArray(xoff_udm, yoff_udm, xcount_udm, ycount_udm).astype(float)
                if udm_ndv is not None:
                    udm_image_ndv = np.ma.masked_values(udm_image, udm_ndv)
            
    # If the feature is outside the udm extent, skip it
                    if udm_image_ndv[~udm_image_ndv.mask].shape[0] == 0:
                        raise NameError
                else:
                    udm_image_ndv = udm_image
                udm_zonal_data = np.ma.masked_array(udm_image_ndv, np.logical_not(datamask_udm))
                
                # If the whole feature is smaller than a pixel size, use the pixel value
                if udm_zonal_data[~udm_zonal_data.mask].shape[0] == 0:
                    if geom.Area() < abs(pixelWidth_udm*pixelHeight_udm):
                        udm_zonal_data = udm_image_ndv
                    else:
                        raise NameError
                    
                udm_zoneraster.append(udm_zonal_data)
                
            # Create the udm mask
            for _i, udm_criteria in enumerate(udm_criterion):
                udm_cur = udm_zoneraster[_i]
                udm_value, udm_cond = udm_criteria
                if udm_cond == '=':
                    np.ma.masked_where(udm_cur != udm_value, udm_cur, copy=False)
                elif udm_cond == '>':
                    np.ma.masked_where(udm_cur <= udm_value, udm_cur, copy=False)
                elif udm_cond == '<':
                    np.ma.masked_where(udm_cur >= udm_value, udm_cur, copy=False)
                elif udm_cond == '>=':
                    np.ma.masked_where(udm_cur < udm_value, udm_cur, copy=False)
                elif udm_cond == '<=':
                    np.ma.masked_where(udm_cur > udm_value, udm_cur, copy=False)
                udm_zoneraster[_i] = udm_cur
                
    # Use or logic to stak the udm mask
            udm_mask = udm_zoneraster[0].mask
            if len(udm_zoneraster) > 1:
                for _i in range(1, len(udm_zoneraster)):
                    udm_mask = np.logical_or(udm_mask, udm_zoneraster[_i].mask)
            
        except NameError:
            pass
        
        except AttributeError:
            continue
        
    # Read raster as numpy array
        zoneraster = list()
        try:
            for _idx in range(channels):
                band = InputFile.GetRasterBand(_idx+1)
                ndv = band.GetNoDataValue()
                image = band.ReadAsArray(xoff, yoff, xcount, ycount).astype(float)
                if ndv is not None:
                    image_ndv = np.ma.masked_values(image, ndv)
                
                # If the feature is outside the raster extent, skip it
                    if image_ndv[~image_ndv.mask].shape[0] == 0:
                        raise AttributeError
                else:
                    image_ndv = image
                    
                # Try to apply udm mask and datamask together. Apply only datamask if there is no udm mask
                try:
                    zonal_data = np.ma.masked_array(image_ndv, np.logical_or(udm_mask, np.logical_not(datamask)))
                except NameError:
                    zonal_data = np.ma.masked_array(image_ndv,  np.logical_not(datamask))
                
                # If the whole feature is smaller than a pixel size, use the pixel value
                if zonal_data[~zonal_data.mask].shape[0] == 0:
                    if geom.Area() < abs(pixelWidth*pixelHeight):
                        zonal_data = image_ndv
                    else:
                        raise AttributeError
    # Mask zone of raster
                zoneraster.append(zonal_data)
            Image_set[FID] = zoneraster
        
        except AttributeError:
            continue
        
    # Close raster file
    InputFile = None
    target_ds = None
    udm_File = None
    
    if len(udm) > 0:
        return Image_set, udm_config
    else:
        return Image_set

def RasterStats(Image_set):
    stats = ['count', 'min', 'max', 'mean', 'median', 'std', 'LQ', 'UQ']
    stat_sheets = list()
    indice = list()
    for FID in Image_set.keys():
        feature = Image_set[FID]
        stat_sheet = list()
        header = list()
        for idx, band in enumerate(feature):
            count = band.count()
            Min = band.min()
            Max = band.max()
            Mean = band.mean()
            Median = np.percentile(band[~band.mask], 50)
            Std = band.std()
            LQ = np.percentile(np.array([x for x in band[~band.mask] if x<=Median]), 50)
            UQ = np.percentile(np.array([x for x in band[~band.mask] if x>=Median]), 50)
            stat_sheet.append([count, Min, Max, Mean, Median, Std, LQ, UQ])
            prefix = 'band{}_'.format(idx+1)
            header.append(list(map(''.join([prefix, '{}']).format, stats)))
        header = [col for sublist in header for col in sublist]
        stat_sheets.append([stat for sublist in stat_sheet for stat in sublist])
        indice.append(FID)
    stat_sheet = np.stack(stat_sheets)
    stat_sheet = pd.DataFrame(stat_sheet, columns=header, index=indice)
    return stat_sheet

def Save_Excel_Image(features, OutFile):
    OutputFile = pd.ExcelWriter(OutFile)
    for FID in features.keys():
        Masked_image = features[FID]
        Masked_image_1d = list()
        headers = list()
        for _idx, band in enumerate(Masked_image):
            Masked_image_1d.append(band[~band.mask].reshape(-1))
            headers.append('band{}'.format(_idx+1))
        Array = pd.DataFrame(Masked_image_1d).T
        Array.to_excel(OutputFile, sheet_name=str(FID), header=headers, index=False)
    OutputFile.save()
    OutputFile.close()

def Save_Excel_Stats(Stat_sheet, total_FIDs, OutFile):
    OutputFile = pd.ExcelWriter(OutFile)
    files = sorted(list(Stat_sheet.keys()))
    for FID in total_FIDs:
        indice = list()
        templist = list()
        for file in files:
            try:
                templist.append(Stat_sheet[file].loc[FID])
                indice.append(file)
            except KeyError:
                continue
        temppd = pd.DataFrame(templist, index=indice)
        temppd.to_excel(OutputFile, sheet_name=str(FID))
    OutputFile.save()
    OutputFile.close()

def UserInputBox(_property, text, allow_none=True, isnum=False):
    class popupWindow(tk.Tk):
        def __init__(self, _property, allow_none, isnum):
            tk.Tk.__init__(self)
            self.resizable(width=False, height=False)
            self.title(_property)
            self.protocol('WM_DELETE_WINDOW', self.on_exit)
            
            # Set variable
            self.input = tk.StringVar(self, name='input')
            self.cond = tk.StringVar(self, name='cond')
            
            # Label                 
            self.l = ttk.Label(self, text=text)
            self.l.grid(column=0, row=0, padx=5, pady=5)

            # Entry for suffix input
            self.e = ttk.Entry(self, textvariable=self.input, width=30)
            self.e.bind('<Return>', self.on_exit_ok)
            self.e.bind('<KP_Enter>', self.on_exit_ok)
            self.e.grid(column=0, row=4, padx=5, pady=5)

            # Ok Button
            self.b = ttk.Button(self,text='Ok', width=40)
            self.b.bind('<Button-1>', self.on_exit_ok)
            self.b.grid(column=0, row=5, padx=5, pady=5)
            
            # Make popup window at the centre
            self.update_idletasks()
            w = self.winfo_screenwidth()
            h = self.winfo_screenheight()
            size = tuple(int(_) for _ in self.geometry().split('+')[0].split('x'))
            x = int(w/2 - size[0]/2)
            y = int(h/2 - size[1]/2)
            self.geometry('+{}+{}'.format(x, y))
        
        def on_exit(self):
        # When you click x to exit, this function is called
            if User_Confirm("Exit", "Do you want to skip the input?") or allow_none:
                self.input.set('')
                self.quit()
            else:
                if User_Confirm('Empty input', 'Input is empty\nDo you want to quit?'):
                    self.destroy()
                    sys.exit(0)
                
        def on_exit_ok(self, event):
            if allow_none:
                self.quit()
            elif len(self.getvar('input'))>0 and isnum:
                _input = self.getvar('input').strip()
                self.setvar(name='cond', value='=')
                if not _input.isnumeric():
                    # Test if the input is floating number
                    try:
                        float(_input)
                        self.quit()
                    except ValueError:
                        # Check the conditional expression
                        for _cond in ['>=', '<=', '>', '<']:
                            _new_input = _input.lstrip(_cond).strip()
                            if _new_input.isnumeric():
                                break
                            else:
                                # Test if the input is floating number
                                try:
                                    float(_new_input)
                                    break
                                except ValueError:
                                    continue
                        if not _new_input.isnumeric():
                            # Test if the input is floating number
                            try:
                                float(_new_input)
                                self.setvar(name='input', value=_new_input)
                                self.setvar(name='cond', value=_cond)
                                self.quit()
                            except ValueError:
                                ErrorBox('Warning!', 'Input contains unacceptable characters!')
                        else:
                            self.setvar(name='input', value=_new_input)
                            self.setvar(name='cond', value=_cond)
                            self.quit()
                else:
                    self.quit()
                    
            else:
                ErrorBox('Warning!', 'Input cannot be blank!')

    # Run the inputbox app
    InputBox = popupWindow(_property, allow_none=allow_none, isnum=isnum)
    InputBox.mainloop()
    _input = InputBox.getvar('input')
    _cond = InputBox.getvar('cond')
    CleanWidget(InputBox)
    InputBox.destroy()
    InputBox = None
    gc.collect()
    
    if not isnum:
        return _input
    else:
        return float(_input), _cond
    
def CheckBox(picks, text, allow_none=False):
    class checkboxfromlist(ttk.Frame):
        def __init__(self, parent, picks):
            ttk.Frame.__init__(self, parent)
            self.vars = []
            max_width = len(max(picks, key=len))
            total_pick = len(picks)
            # Change row every three columns
            total_row = total_pick // 3 + (total_pick % 3 > 0)

            # Map the checkboxes
            for _row, _column in [(_row, _column) for _row in range(total_row) for _column in range(3)]:
                if _row * 3 + _column < total_pick:
                    var = tk.BooleanVar()
                    chk = ttk.Checkbutton(self, text='{:<{width}}'.format(picks[_row*3+_column], width=max_width),
                                          variable=var, onvalue=True, offvalue=False)
                    chk.grid(column=_column, row=_row, padx=5, pady=5, sticky='ew')
                    self.vars.append(var)
        
        def state(self):
            return list(map((lambda var: var.get()), self.vars))

    class MainWindow(tk.Tk):
        def __init__(self, picks, text, allow_none=False):
            tk.Tk.__init__(self)
            self.resizable(width=False, height=False)
            self.title('Please select below options')
            self.protocol('WM_DELETE_WINDOW', self.on_exit)

            # Label
            self.l = ttk.Label(self, text=text)
            self.l.grid(column=0, row=0, padx=5, pady=10)

            current_row = 1

            # Checkboxes frame
            self.ChkBox = checkboxfromlist(self, picks)
            self.ChkBox.grid(column=0, row=current_row, padx=5, pady=10)
            current_row += 1

            # Ok button
            self.Button = ttk.Button(self, text='Ok', width=40, command=self.get_value)
            self.Button.grid(column=0, row=current_row, padx=5, pady=5)
            
            # Make popup window at the centre
            self.update_idletasks()
            w = self.winfo_screenwidth()
            h = self.winfo_screenheight()
            size = tuple(int(_) for _ in self.geometry().split('+')[0].split('x'))
            x = int(w/2 - size[0]/2)
            y = int(h/2 - size[1]/2)
            self.geometry('+{}+{}'.format(x, y))

            # Initialise
            self.mask = None
            self.result = None
            
        def on_exit(self):
        # When you click x to exit, this function is called
            if User_Confirm("Exit", "Do you want to quit the application?"):
                self.destroy()
                sys.exit(0)

        def get_value(self):
            self.mask = self.ChkBox.state()
            if any(self.mask) or allow_none:
                selected = [item for index, item in enumerate(picks) if self.mask[index]]
                self.result = selected
                self.quit()
            else:
                InfoBox('No selection', 'Need to select at least one item.')

    # Run checkbox
    App = MainWindow(picks, text, allow_none=allow_none)
    App.mainloop()
    result = App.result
    CleanWidget(App)
    App.destroy()
    App = None
    gc.collect()

    return result

def User_Confirm(title, question):
    root = tk.Tk()
    root.withdraw()
    answer = messagebox.askyesno(title, question)
    root.destroy()

    return answer

def InfoBox(title, info):
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(title, info)
    root.destroy()
    
def ErrorBox(title, error):
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(title, error)
    root.destroy()

def AskOpenFile(title, filetype):
    root = tk.Tk()
    root.withdraw()
    filename = askopenfilename(title=title, filetypes=filetype)
    root.destroy()

    return filename

def AskDirectory(title):
    root = tk.Tk()
    root.withdraw()
    directory = askdirectory(title=title)
    root.destroy()
        
    return directory

def AskSaveFile(title, defaultname, filetypes, defaultext):
    root = tk.Tk()
    root.withdraw()
    savefilename = asksaveasfilename(title=title, 
                                     initialdir=ImagePath, 
                                     initialfile=defaultname, 
                                     filetypes=filetypes, 
                                     defaultextension=defaultext)
    
    return savefilename

def CleanWidget(parent):
    # This function tries to clean all the widgets to avoid garbages
    for child in parent.winfo_children():
        wtype = child.winfo_class()
        if wtype in ('TFrame', 'TLabelframe'):
            CleanWidget(child)
            child = None
        else:
            child = None

if __name__ == '__main__':
    # Ask user for the input vector file
    AOI = AskOpenFile('Choose a vector file for cropping', AOI_filetype)
    AOI_name = os.path.split(AOI)[1]
    AOI_name = os.path.splitext(AOI_name)[0]
    
    # Ask user for the input image folder
    ImagePath = AskDirectory(title='Select your image folder')
    basename = os.path.split(ImagePath.rstrip(os.sep))[1]
    
    # Ask for optional image filename suffix
    _suffix = UserInputBox('input images', 
                           'Please enter the filename suffix of input images (optional)')
    Images = sorted(glob.glob(os.path.join(ImagePath, f'*{_suffix}{Raster_filetype}')))
    
    # Ask for optional usable data mask input
    if User_Confirm('Usable data mask?', 'Do you have usable data mask images?'):
        udmPath = AskDirectory(title='Select the folder containing udm images')
        _suffix = UserInputBox('usable data mask images', 
                               'Please enter the filename suffix of usable data mask images (optional)')
        udms = sorted(glob.glob(os.path.join(udmPath, f'*{_suffix}{Raster_filetype}')))
        InfoBox('Usable data mask will be applied', 
                'The program will automatically use the file that has the longest matched filename as the usable data mask image')
    else:
        udms = []
        
    # Initialise statistics dictionary
    Stat_sheet = dict()
    
    # Open shapefile
    shp = ogr.Open(AOI)
    lyr = shp.GetLayer()
    featList = range(lyr.GetFeatureCount())
    total_FIDs = list()
    for FID in featList:
        total_FIDs.append(lyr.GetNextFeature().GetFID())
        
    lyr.ResetReading()
    
    # Loop for images
    udm_config = dict()
    for Image in Images:
        Image_Name = os.path.split(Image)[1]
        Image_Name, Image_Ext = os.path.splitext(Image_Name)
        if len(udms) > 0:
            Matches = list()
            for udm in udms:
                udm_Name = os.path.split(udm)[1]
                udm_Name, udm_Ext = os.path.splitext(udm_Name)
                Matches.append(
                    SequenceMatcher(
                        None, Image_Name, udm_Name).find_longest_match(0, None, 0, None).size)
            largest_match_size = max(Matches)
            udm_index = [ _idx for _idx, _match_size in enumerate(Matches) if _match_size == largest_match_size]
            
            # If there are two files having the same match string length, use the one that has higher porportion of filename string
            if len(udm_index) > 1:
                best_matches = list()
                for _index in udm_index:
                    udm_Name = os.path.split(udm[_index])[1]
                    udm_Name, udm_Ext = os.path.splitext(udm_Name)
                    best_matches.append(Matches[_index]/len(udm_Name))
                best_match = best_matches.index(max(best_matches))
                udm = udms[udm_index[best_match]]
            else:
                udm = udms[udm_index[0]]
            
            masked_image_set, udm_config = Create_Masked_Image(lyr, total_FIDs, Image, udm, udm_config)
                
        else:
            masked_image_set = Create_Masked_Image(lyr, total_FIDs, Image)

        # Only process the images that at least one feature is inside the image scene
        if len(masked_image_set) > 0:
            Filename = os.path.join(ImagePath, Image_Name)
            Filename = '_'.join([Filename, AOI_name])
            Save_Excel_Image(masked_image_set, '.'.join([Filename, 'xlsx']))
            Image_Stat = RasterStats(masked_image_set)
            Stat_sheet[Image_Name] = Image_Stat
    
    # Close shapefile
    lyr = None
    shp = None
    
    # Only save data when there is more than one statistics
    if len(Stat_sheet) > 0:
        OutFile = AskSaveFile('Enter your statistic result Excel file name', 
                              '_'.join([basename, 'Raster_Stats']), 
                              Out_filetypes, '.xlsx')
        
        Save_Excel_Stats(Stat_sheet, total_FIDs, OutFile)
    else:
        InfoBox('No statistics result', 'The input images didn\'t satisfy the condition to calculate any statistics.')
