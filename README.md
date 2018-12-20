# Raster zonal statistics
Calculate zonal statistics of raster files using a shapefile
  
This script use polygon or multipolygon shapefile and calculate basic statistics including count,minimum, maximum, mean, median, standard deviation, uppoer quartile, and lower quartile for every features for multiple raster files. The pixel values for each feature from one raster file will be preserved and saved to an Excel spreadsheet. The final statistical result for all the raster will be saved as another Excel spreadsheet for easier comparison.  
  
The input raster format must be GeoTiff with the file extension \*.tif. The program will prompt the user to select the polygon shapefile, then the folder that contains all the desired images. When the process finishs, another prompt window will appear to ask the user for the location and file name for the Excel result.
