# -*- coding: utf-8 -*-
"""WAM_Network_Process_without_argis.ipynb"""

# !pip install gdal

# !pip install dbfread

import os
import sys
from osgeo import gdal, ogr
from osgeo_utils.gdal2xyz import gdal2xyz
import geopandas as gpd
import pandas as pd


def wam_model_builder(
    working_path: str,
    land_use_shapefile_name: str = 'Land_Use.shp',
    stream_node_raster_name: str = 'Streamnode.asc',
    uk_raster_name: str = 'UK_TP_Existing.tif',
    nutrient: str = 'P',
    output_watershed_subbasin_land_use_name: str = 'Watershed_Subbasin_LU_TP.xlsx',
    ):
    gdal.PushErrorHandler('CPLQuietErrorHandler')

    ######## Step 01: Run gdal_polygonize ################ 
    print("Step 1 - Run gdal_polygonize")

    Wam_path = os.path.join(working_path, 'WAM')
    Inputs_path = os.path.join(Wam_path, 'Inputs')
    Outputs_path = os.path.join(Wam_path, 'Outputs')

    #create a new folder named 'WAM' under the current working folder if it does not exist
    if not os.path.exists(Wam_path):
        print(f"Create {Wam_path} in the current working directory!")
        os.makedirs(Wam_path)

    #create a new folder named 'Inputs' under the 'WAM' folder if it does not exist
    if not os.path.exists(Inputs_path):
        print(f"Create {Inputs_path} in the current working directory!")
        os.makedirs(Inputs_path)

    #create another new folder named Outputs under the 'WAM' folder if it does not exist
    if not os.path.exists(Outputs_path):
        print(f"Create {Outputs_path} in the current working directory!")
        os.makedirs(Outputs_path)

    vector_path = "Subbasins.shp"
    vector_path = os.path.join(Outputs_path, vector_path)
    # #if file exist, then delete it
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(vector_path):
        driver.DeleteDataSource(vector_path)  # Remove existing file

    raster_path = stream_node_raster_name
    raster_path = os.path.join(Inputs_path, raster_path)
    raster = gdal.Open(raster_path)
    srcBand = raster.GetRasterBand(1)  # Use Band 1 for polygonization
    maskBand = srcBand.GetMaskBand()  # Create a bitmask for NoData pixels

    geotransform = raster.GetGeoTransform()
    projection = raster.GetProjection()

    # # Create a new shapefile
    vector_ds = driver.CreateDataSource(vector_path)
    out_layername = "Subbasins" # The name of the layer created to hold the polygon features.

    layer = vector_ds.CreateLayer(out_layername, 
                                geom_type=ogr.wkbPolygon)


    # ADD ATTRIBUTE FIELDS 
    field_id = ogr.FieldDefn("ID", ogr.OFTInteger64)  # Unique integer ID
    field_id.SetWidth(10)
    layer.CreateField(field_id) # leave it empty for now

    newfield = ogr.FieldDefn("GRIDCODE", ogr.OFTInteger64)
    newfield.SetWidth(10)
    layer.CreateField(newfield) # leave it empty for now


    # # Polygonize the raster file
    gdal.Polygonize(srcBand, 
                    maskBand,  # Mask band
                    layer, # Output layer
                    1) # the field index of the new layer indicating the feature attribute into which the pixel value of the polygon should be written

    # ASSIGN UNIQUE ID TO EACH REMAINING POLYGON
    id_counter = 1
    for feature in layer:
        feature.SetField("ID", id_counter)  # Assign incremental ID
        layer.SetFeature(feature)
        id_counter += 1 

    # # Close files
    feature = None
    vector_ds = None
    raster = None
    layer = None

    ######## Step 02: Dissolve ################ 
    print(10*"-", "Run gdal_dissolve", 10*"-" )

    # # Open the original shapefile
    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.Open(vector_path, 0)  # Read-only mode

    # # Create a new shapefile to store dissolved output
    dissolve_vector_path = "Dissolved_subbasins.shp"
    dissolve_vector_path = os.path.join(Outputs_path, 
                                        dissolve_vector_path)
    # #if file exist, then delete it
    if os.path.exists(dissolve_vector_path):
        driver.DeleteDataSource(dissolve_vector_path)  # Remove existing file

    out_ds = driver.CreateDataSource(dissolve_vector_path)

    # # Perform dissolve (merging by the 'Value' field)
    sql = "SELECT ST_Union(geometry) AS geometry, GRIDCODE FROM Subbasins GROUP BY GRIDCODE"
    result_layer = ds.ExecuteSQL(sql, 
                                dialect="SQLITE")

    # # Copy the result layer to the new shapefile
    out_layer = out_ds.CopyLayer(result_layer, 
                "Dissolved_subbasins")


    # # Release the result
    out_layer = None
    out_ds = None
    ds.ReleaseResultSet(result_layer)
    ds = None

    ######## Step 03: Intersection ################

    ######### Step 03-01: Fix geometries - Land_Use.shp ################ 

    print(10*"-", "Fix geometries - Land_Use_Valid.shp", 10*"-")

    fix_vector_path = "Land_Use_Valid.shp"
    fix_vector_path = os.path.join(Inputs_path, 
                                fix_vector_path)
    #if file exist, then delete it
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(fix_vector_path):
        driver.DeleteDataSource(fix_vector_path)  # Remove existing file

    ds_path = land_use_shapefile_name
    ds_path = os.path.join(Inputs_path, ds_path)

    ds = ogr.Open(ds_path, 1)  # Return a ogr.DataSource object
    layer = ds.GetLayer()  # Return a ogr.Layer object  

    # #Create a new shapefile
    out_ds = driver.CreateDataSource(fix_vector_path)  # # Return a ogr.DataSource object, write

    # # copy the layer' CRS to the new shapefile
    crs = layer.GetSpatialRef()
    out_layer = out_ds.CreateLayer("Land_Use_Valid", 
                                geom_type=layer.GetGeomType(), srs=crs)

    # # Copy fields, but modify SHAPE_Area field
    layer_defn = layer.GetLayerDefn()
    for i in range(layer_defn.GetFieldCount()):
        field_def = layer_defn.GetFieldDefn(i)
        if field_def.GetName() == "SHAPE_Area":
            new_field_def = ogr.FieldDefn("SHAPE_Area", ogr.OFTReal)
            new_field_def.SetWidth(23)  # Increase field width
            new_field_def.SetPrecision(11)  # Increase decimal places
            out_layer.CreateField(new_field_def)
        else:
            out_layer.CreateField(field_def)

    for feature in layer:
        new_feature = ogr.Feature(out_layer.GetLayerDefn())
        # print("Get geometry")
        geom = feature.GetGeometryRef()
        # check if the geometry is not valid
        if not geom.IsValid() and geom.GetGeometryType() == ogr.wkbPolygon:
            #fix using makeValid using STRUCTURE method, quite mode
            # print("Invalid geometry found")
            geom = geom.MakeValid(["METHOD=STRUCTURE"])

        # print("Set geometry")    
        new_feature.SetGeometry(geom)
        
        # print("Set fields")
        for i in range(layer_defn.GetFieldCount()):
            new_feature.SetField(i, feature.GetField(i))
        # print("Create feature")
        out_layer.CreateFeature(new_feature)
        new_feature = None

    # Close resources
    feature = None
    layer = None
    ds = None
    out_layer = None
    out_ds = None


    ######### Step 03-02: Fix geometries - Dissolved_subbasins.shp ################ 

    print(10*"-", "Fix geometries - New_dissolved_subbasins.shp", 10*"-")
    fix_vector_path = "Dissolved_subbasins_valid.shp"
    fix_vector_path = os.path.join(Outputs_path, 
                                fix_vector_path)
    # #if file exist, then delete it
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(fix_vector_path):
        driver.DeleteDataSource(fix_vector_path)  # Remove existing file

    ds_path = "Dissolved_subbasins.shp"
    ds_path = os.path.join(Outputs_path, ds_path)

    ds = ogr.Open(ds_path, 1)  # Return a ogr.DataSource object
    layer = ds.GetLayer()  # Return a ogr.Layer object  

    # #Create a new shapefile with encoding UTF-8
    driver = ogr.GetDriverByName("ESRI Shapefile")
    out_ds = driver.CreateDataSource(fix_vector_path)  # # Return a ogr.DataSource object, write

    # # copy the layer' CRS to the new shapefile
    crs = ds.GetLayer().GetSpatialRef()
    out_layer = out_ds.CreateLayer("Dissolved_subbasins_valid", 
                                geom_type=layer.GetGeomType(), 
                                srs=crs)

    # # Copy fields, but modify SHAPE_Area field
    layer_defn = layer.GetLayerDefn()
    for i in range(layer_defn.GetFieldCount()):
        field_def = layer_defn.GetFieldDefn(i)
        out_layer.CreateField(field_def)

    for feature in layer:
        new_feature = ogr.Feature(out_layer.GetLayerDefn())
        geom = feature.GetGeometryRef()
        #check if the geometry is not valid
        if not geom.IsValid() and geom.GetGeometryType() == ogr.wkbPolygon:
            #fix using makeValid using STRUCTURE method
            # print("Invalid geometry found")
            geom = geom.MakeValid(["METHOD=STRUCTURE"])
        
        # print("Set geometry")
        new_feature.SetGeometry(geom)
        
        # print("Set fields")
        for i in range(layer_defn.GetFieldCount()):
            new_feature.SetField(i, feature.GetField(i))
        # print("Create feature")
        out_layer.CreateFeature(new_feature)
        new_feature = None 

    feature = None
    layer = None
    ds = None
    out_layer = None
    out_ds = None

    print(10*"-", "Extract the overlapping portions of polygons", 10*"-")

    overlapping_vector_path = "Land_Subbasin_Intersection.shp"
    overlapping_vector_path = os.path.join(Outputs_path, 
                                        overlapping_vector_path)

    # #if file exist, then delete it
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(overlapping_vector_path):
        driver.DeleteDataSource(overlapping_vector_path)  # Remove existing file

    land_usage_path =  "Land_Use_Valid.shp"
    land_usage_path = os.path.join(Inputs_path, land_usage_path)
    try:
        land_usage_ds = ogr.Open(land_usage_path, 0)  # Read-only
    except Exception as e:
        print(f"Could not open the {land_usage_path} file")
        print(e)
        sys.exit(1)
    land_layer = land_usage_ds.GetLayer()
    crs = land_layer.GetSpatialRef()

    subbasins_ds_path = "Dissolved_subbasins_valid.shp"
    subbasins_ds_path = os.path.join(Outputs_path, subbasins_ds_path)
    try:
        subbasins_ds = ogr.Open(subbasins_ds_path, 0)
    except Exception as e:
        print(f"Could not open the {subbasins_ds_path} file")
        print(e)
        sys.exit(1)
    subbasin_layer = subbasins_ds.GetLayer()

    # # # Create output shapefile for the intersected data
    driver = ogr.GetDriverByName("ESRI Shapefile")
    out_ds = driver.CreateDataSource(overlapping_vector_path)
    out_layer = out_ds.CreateLayer("Land_Subbasin_Intersection", 
                                geom_type=land_layer.GetGeomType(),
                                srs=crs)

    # # Perform the intersection
    land_layer.Intersection(subbasin_layer, 
                            out_layer, (["METHOD_PREFIX=B_",
                                        "PRETEST_CONTAINMENT=YES",
                                        "KEEP_LOWER_DIMENSION_GEOMETRIES=NO"   
                                        ]))

    # # # Close files
    land_layer = None
    subbasin_layer = None
    land_usage_ds = None
    subbasins_ds = None
    out_layer = None
    out_ds = None

    ######## Step 04: Convert the raster file  UK_TP_Existing.tif to a Point based shape file ################ 
    print(10*"-", "Convert the raster file UK_TP_Existing.tif to a shape file ", 10*"-")

    vector_path = "Uk_tp_pt.shp"
    vector_path = os.path.join(Outputs_path,
                            vector_path)

    driver = ogr.GetDriverByName("ESRI Shapefile")
    # #if file exist, then delete it
    if os.path.exists(vector_path):
        driver.DeleteDataSource(vector_path)  # Remove existing file

    raster_path = uk_raster_name  # Replace with your actual file path
    raster_path = os.path.join(Inputs_path, raster_path)
    raster_ds = gdal.Open(raster_path) # return a gdal.Dataset object
    TP_band = raster_ds.GetRasterBand(1)
    no_data_value = TP_band.GetNoDataValue()

    # # wkbPoint
    # driver = ogr.GetDriverByName("ESRI Shapefile")
    vector_ds = driver.CreateDataSource(vector_path)
    out_layername = "Uk_tp_pt" # The name of the layer created to hold the POint features.
    # # Get the CRS (Coordinate Reference System) from raster and assign it to the shapefile
    crs = raster_ds.GetSpatialRef()

    layer = vector_ds.CreateLayer(out_layername, 
                                geom_type=ogr.wkbPoint,
                                srs=crs)


    # ADD ATTRIBUTE FIELDS 
    field_id = ogr.FieldDefn("POINTID", ogr.OFTInteger)  # Unique integer ID
    field_id.SetWidth(6)
    layer.CreateField(field_id) # leave it empty for now

    newfield = ogr.FieldDefn("GRIDCODE", ogr.OFTReal)
    newfield.SetWidth(17)
    newfield.SetPrecision(8)
    layer.CreateField(newfield) # leave it empty for now

    all_geo_x, all_geo_y, all_data, nodata = gdal2xyz(srcfile=raster_ds,
                        band_nums=1,
                        skip_nodata=True,
                        src_nodata=no_data_value,
                        return_np_arrays=True)

    # # print the size of all_geo_x numpy array
    # print("Size of all_geo_x numpy array:", all_geo_x.shape)
    # print("Size of all_geo_y numpy array:", all_geo_y.shape)
    # print("Size of all_data numpy array:", all_data.shape)
    # print("Size of all_data numpy array:", all_data.shape)
    # print("NoData Value:", nodata)

    # # process the all_data numpy array by
    # # first, transpose it.
    # # then convert it to a 1D array

    all_data = all_data.transpose()
    all_data = all_data.flatten()
    # print("Size of all_data numpy array:", all_data.shape)

    id_counter = 1
    for idx in range(len(all_geo_x)):
        # point
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(all_geo_x[idx], all_geo_y[idx])
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(point)
        feature.SetField("POINTID", id_counter)  # Assign incremental ID
        feature.SetField("GRIDCODE", float(all_data[idx]))
        layer.CreateFeature(feature)
        id_counter += 1

    # Close files
    layer = None
    vector_ds = None
    raster_ds = None

    # # ###############Step 05: Batch-wise Spatial join using Geopandas ############

    print(10*"-", "Spatial Join", 10*"-")
    overlapping_vector_path = "Sub_LU_TP.shp"
    overlapping_vector_path = os.path.join(Outputs_path, 
                                        overlapping_vector_path)

    # #if file exist, then delete it
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(overlapping_vector_path):
        driver.DeleteDataSource(overlapping_vector_path)  # Remove existing file

    # Input-1: Land_Subbasin_Intersection.shp
    land_usage_path =  "Land_Subbasin_Intersection.shp"
    land_usage_path = os.path.join(Outputs_path, land_usage_path)

    land_usage_ds = gpd.read_file(land_usage_path)  # Read-only


    # Input-2: Uk_tp_pt.shp
    phosphorus_path = "Uk_tp_pt.shp"
    phosphorus_path = os.path.join(Outputs_path, phosphorus_path)

    phosphorus_ds = gpd.read_file(phosphorus_path)


    """# Add TARGET_FID"""
    land_usage_ds["TARGET_FID"] = land_usage_ds.index

    # Get the CRS from subbasins_land_use
    subbasins_crs = land_usage_ds.crs

    # Transform the geometry of uk_tp_pt to the CRS of subbasins_land_use
    phosphorus_ds = phosphorus_ds.to_crs(subbasins_crs)

    subbasins_land_use_uk_tp_pt = gpd.sjoin(land_usage_ds,
                                            phosphorus_ds,
                                            how='left',
                                            predicate='intersects',
                                            )

    subbasins_land_use_uk_tp_pt["POINTID"] = subbasins_land_use_uk_tp_pt["POINTID"].fillna(0).astype(int)

    subbasins_land_use_uk_tp_pt['index_right'] = subbasins_land_use_uk_tp_pt['index_right'].fillna(0).astype(int)

    subbasins_land_use_uk_tp_pt['GRIDCODE'] = subbasins_land_use_uk_tp_pt['GRIDCODE'].fillna(0).astype(int)

    #subbasins_land_use_uk_tp_pt.head(1)

    #subbasins_land_use_uk_tp_pt.tail(1)

    # Create a new column 'JOIN_COUNT' and set it to 0
    subbasins_land_use_uk_tp_pt['Join_Count'] = 0

    # update the 'Join_Count' column 
    # for each TARGET_FID, count the number of occurrences in the 'TARGET_FID' column. 
    # If there are multiple occurrences, set the 'Join_Count' to the number of occurrences.

    subbasins_land_use_uk_tp_pt['Join_Count'] = subbasins_land_use_uk_tp_pt.groupby('TARGET_FID')['TARGET_FID'].transform('count')


    # remove the duplicates
    temp1 = subbasins_land_use_uk_tp_pt.drop_duplicates(subset=['TARGET_FID'], keep='first') # keep the first occurrence

    # # Convert 'SHAPE_Area' column to a String 
    temp1['SHAPE_Area'] = temp1['SHAPE_Area'].astype(str)

    temp1.to_file(overlapping_vector_path, 
                driver="ESRI Shapefile")


    # ############### Step 06: Computing Area ################
    print(10*"-", "Area", 10*"-")
    out_path = "Sub_LU_TP_Area.shp"
    out_path = os.path.join(Outputs_path, out_path)
    # # #if file exist, then delete it
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(out_path):
        driver.DeleteDataSource(out_path)  # Remove existing file

    ds_path = "Sub_LU_TP.shp"
    ds_path = os.path.join(Outputs_path, ds_path)

    ds = ogr.Open(ds_path, 1)  # Write; Return a ogr.DataSource object
    layer = ds.GetLayer()  # Return a ogr.Layer object

    driver = ogr.GetDriverByName("ESRI Shapefile")
    out_ds = driver.CreateDataSource(out_path)
    out_layer = out_ds.CopyLayer(layer, "Sub_LU_TP_Area")

    # # Add a new field for the area
    new_field = ogr.FieldDefn("AREA", ogr.OFTReal)
    new_field.SetWidth(23)
    new_field.SetPrecision(11)
    out_layer.CreateField(new_field)

    # # Calculate the area of each polygon
    for feature in out_layer:
        geom = feature.GetGeometryRef()
        area = geom.GetArea()
        # round up to 3 decimal places
        # area = round(area, 3)
        feature.SetField("AREA", area)
        out_layer.SetFeature(feature)

    # # Close files
    feature = None
    layer = None
    ds = None
    out_layer = None
    out_ds = None


    # ####################Step 07: Group By####################################
    print(10*"-", "Group by the New_Sub_LU_TP.shp by LUID, B_GRIDCODE and compute the sum of AREA, and GRIDCODE ", 10*"-")
    # export to a Excel file
    out_path = output_watershed_subbasin_land_use_name
    out_path = os.path.join(Outputs_path, out_path)

    # Open the shapefile
    ds_path = "Sub_LU_TP_Area.shp"
    ds_path = os.path.join(Outputs_path, ds_path)

    ds = ogr.Open(ds_path, 1)  # Write; Return a ogr.DataSource object
    layer = ds.GetLayer()  # Return a ogr

    # Create a dictionary to store the results
    results = {}
    for feature in layer:
        luid = feature.GetField("LUID")
        gridcode = feature.GetField("B_GRIDCODE")
        area = feature.GetField("AREA")

        if luid not in results:
            results[luid] = {}
        
        if gridcode not in results[luid]:
            results[luid][gridcode] = {"Area_ft2": 0, f"T{nutrient}_kg/ha": 0}
        
        results[luid][gridcode]["Area_ft2"] += area
        results[luid][gridcode][f"T{nutrient}_kg/ha"] += gridcode

    # close the shapefile
    feature = None
    layer = None
    ds = None

    # Create a pandas DataFrame from the dictionary
    df = pd.DataFrame.from_dict({(i,j): results[i][j] 
                                for i in results.keys() 
                                for j in results[i].keys()},
                                orient='index')

    print(df.info())
    # create a new column Area_acres
    df["Area_ac"] = df["Area_ft2"] * 0.0000229568

    # Reset the index
    df.reset_index(inplace=True)
    df.rename(columns={"level_0": "LUID", "level_1": "REACH"}, inplace=True)

    # re-arrange the column orders: "LUID", "REACH", "Area_ft2", "Area_ac", "TP_kg/ha"
    df = df[["LUID", "REACH", "Area_ft2", "Area_ac", f"T{nutrient}_kg/ha"]]

    # Save the DataFrame to an Excel file without the integer index
    df.to_excel(out_path, index=False)




    ##SBATCH --ntasks=1                     # Number of tasks (processes)
    #SBATCH --cpus-per-task=1              # Number of CPU cores per task
    #SBATCH --mem=16G                      # Memory per node
    #SBATCH --time=20:00:00                # Time limit (hh:mm:ss)
    #SBATCH --partition=snsm_itn19 
    #SBATCH --qos=snsm19_special

    # create a SLURM's srun command to run the script


if __name__ == '__main__':
    # working_path provided
    if len(sys.argv) < 2:
        print('Usage: python WAM_Model_Builder.py <working_path> <land_use_shapefile_name.shp> <stream_node_raster_name.asc> <uk_raster_name.tif> <nutrient> <output_watershed_subbasin_land_use_name.xlsx>')
        print('Using current working directory as the working path')
        print('Using default file names')
        working_path = os.getcwd()
        file_names = {}
    # working_path and file names provided
    elif len(sys.argv) == 7:
        working_path = sys.argv[1]
        file_names = {
            'land_use_shapefile_name': sys.argv[2],
            'stream_node_raster_name': sys.argv[3],
            'uk_raster_name': sys.argv[4],
            'nutrient': sys.argv[5],
            'output_watershed_subbasin_land_use_name': sys.argv[6]
        }
    # Invalid number of arguments provided
    else:
        print("Usage: python WAM_Model_Builder.py <working_path> <land_use_shapefile_name.shp> <stream_node_raster_name.asc> <uk_raster_name.tif> <nutrient> <output_watershed_subbasin_land_use_name.xlsx>")
        print('Invalid number of arguments')
        print('Exiting...')
        sys.exit(1)
    
    # Run the WAM_Model_Builder function
    wam_model_builder(working_path, **file_names)
