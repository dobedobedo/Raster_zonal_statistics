#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 09:54:12 2018

@author: uqytu1
"""

import os
import sys
import glob
import tkinter as tk
from tkinter.filedialog import askopenfilename, askdirectory, asksaveasfilename
from tkinter import messagebox
import numpy as np
from osgeo import ogr, osr, gdal
import pandas as pd

AOI_filetype = [('Shapefile', ['*.shp'])]
Raster_filetype = '*.tif'
Out_filetypes=[('Microsoft Excel',['*.xls','*.xlsx'])]

def Create_Masked_Image(lyr, Image):
    # Open vector and raster files
    InputFile = gdal.Open(Image)
            
    # Get raster georeference info
    GeoTransform = InputFile.GetGeoTransform()
    xOrigin = GeoTransform[0]
    yOrigin = GeoTransform[3]
    pixelWidth = GeoTransform[1]
    pixelHeight = GeoTransform[5]
    channels = InputFile.RasterCount
    
    # Reproject vector geometry to same projection as raster
    sourceSR = lyr.GetSpatialRef()
    targetSR = osr.SpatialReference()
    targetSR.ImportFromWkt(InputFile.GetProjectionRef())
    coordTrans = osr.CoordinateTransformation(sourceSR,targetSR)
    
    Image_set = list()
    featList = range(lyr.GetFeatureCount())
    FIDs = list()
    # Loop through features
    for FID in featList:
        feat = lyr.GetFeature(FID)
        geom = feat.GetGeometryRef()
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
            messagebox.showerror('Error!', Message)
            sys.exit('ERROR: Geometry needs to be either Polygon or Multipolygon')
        xmin = min(pointsX)
        xmax = max(pointsX)
        ymin = min(pointsY)
        ymax = max(pointsY)
        
    # Specify offset and rows and columns to read
        xoff = int((xmin - xOrigin)/pixelWidth)
        yoff = int((yOrigin - ymax)/pixelWidth)
        xcount = int((xmax - xmin)/pixelWidth)+1
        ycount = int((ymax - ymin)/pixelWidth)+1
        
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
        datamask = bandmask.ReadAsArray(0, 0, xcount, ycount).astype(np.float)
        
    # Read raster as numpy array
        zoneraster = list()
        try:
            for _idx in range(channels):
                band = InputFile.GetRasterBand(_idx+1)
                #ndv = band.GetNoDataValue()
                ndv = -10000
                image = band.ReadAsArray(xoff, yoff, xcount, ycount).astype(np.float)
                image_ndv = np.ma.masked_values(image, ndv)
                if image_ndv[~image_ndv.mask].shape[0] == 0:
                    raise AttributeError
                zonal_data = np.ma.masked_array(image_ndv,  np.logical_not(datamask))
                if zonal_data[~zonal_data.mask].shape[0] == 0:
                    raise AttributeError
    # Mask zone of raster
                zoneraster.append(zonal_data)
            Image_set.append(zoneraster)
            FIDs.append(FID)
        
        except AttributeError:
            continue
        
    # Close raster file
    InputFile = None
    target_ds = None
    
    return Image_set, FIDs

def RasterStats(Image_set, FIDs):
    stats = ['count', 'min', 'max', 'mean', 'median', 'std', 'LQ', 'UQ']
    stat_sheets = list()
    indice = list()
    for index, feature in enumerate(Image_set):
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
        indice.append(FIDs[index])
    stat_sheet = np.stack(stat_sheets)
    stat_sheet = pd.DataFrame(stat_sheet, columns=header, index=indice)
    return stat_sheet

def Save_Excel_Image(features, FIDs, OutFile):
    OutputFile = pd.ExcelWriter(OutFile)
    for index, feature in enumerate(features):
        Masked_image = feature
        Masked_image_1d = list()
        headers = list()
        for _idx, band in enumerate(Masked_image):
            Masked_image_1d.append(band[~band.mask].reshape(-1))
            headers.append('band{}'.format(_idx+1))
        Array = pd.DataFrame(Masked_image_1d).T
        Array.to_excel(OutputFile, sheet_name=str(FIDs[index]), header=headers, index=False)
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

if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    
    # Ask user input and output file names
    AOI = askopenfilename(title='Choose a vector file for cropping', filetypes=AOI_filetype)
    AOI_name = os.path.split(AOI)[1]
    AOI_name = os.path.splitext(AOI_name)[0]
    ImagePath = askdirectory(title='Select your image folder')
    basename = os.path.split(ImagePath.rstrip(os.sep))[1]
    Images = sorted(glob.glob(os.path.join(ImagePath, Raster_filetype)))
    Stat_sheet = dict()
    
    # Open shapefile
    shp = ogr.Open(AOI)
    lyr = shp.GetLayer()
    featList = range(lyr.GetFeatureCount())
    total_FIDs = list()
    for FID in featList:
        total_FIDs.append(FID)
    
    # Loop for images
    for Image in Images:
        Image_Name = os.path.split(Image)[1]
        Image_Name, Image_Ext = os.path.splitext(Image_Name)
        masked_image_set, FIDs = Create_Masked_Image(lyr, Image)
        Filename = os.path.join(ImagePath, Image_Name)
        Filename = '_'.join([Filename, AOI_name])
        Save_Excel_Image(masked_image_set, FIDs, '.'.join([Filename, 'xlsx']))
        Image_Stat = RasterStats(masked_image_set, FIDs)
        Stat_sheet[Image_Name] = Image_Stat
    
    # Close shapefile
    lyr = None
    shp = None
    
    OutFile = asksaveasfilename(title='Enter your statistic result Excel file name', 
                                initialdir=ImagePath, 
                                initialfile='_'.join([basename, 'Raster_Stats']), 
                                filetypes=Out_filetypes, 
                                defaultextension='.xlsx')
    
    Save_Excel_Stats(Stat_sheet, total_FIDs, OutFile)
    
    
