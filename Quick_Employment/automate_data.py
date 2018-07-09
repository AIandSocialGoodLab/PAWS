
import os
import sys
import gdal
import processing
import pandas as pd
import numpy as np
from qgis.core import * 
from qgis.gui import *
from qgis.utils import *
from qgis import analysis
from math import ceil
from osgeo import ogr
from PyQt4.QtCore import QVariant
import numpy as np
import matplotlib.pyplot as plt
import csv
from qchainage import chainagetool

####################################################################################################################################################  
####################################################################################################################################################
####################################################################################################################################################  
####################################################################################################################################################
#
#         General Overview
#
#   Module 0 : Specifying paths and file names
#   Module 1 : Open source implementation of NNJoin plugin
#   Module 2 : Open source implementation of MMQGIS's export to csv file 
#   Module 3 : Currently not in code, used to be for QChainage implemenation
#               might be needed dependent on system dependencies
#               if needed, find open source implementation online
#   Module 4 : Actual processing stage
#             4.1 : Discretizing the conservation site and obtaining centroids
#             4.2 : Obtaining "is-" features for each grid cell
#             4.3 : Obtaining "dist-" fefatures for each grid cell
#             4.4 : Obtaining slope and elevation data
#                   https://www.coolutils.com/online/TIFF-Combine/
#             4.5 : Obtaining Patrol Length
#             4.6 : Putting all the csv files together, adding normalized columns for
#                   the non "is-" features
#

####################################################################################################################################################  
####################################################################################################################################################
####################################################################################################################################################  
####################################################################################################################################################
#
#   Module 0 : Specifying paths and file names
#
#   Important : use shapefiles of the same CRS
#
#   Subtle problem: qgis only allows attributes of maximum 10 character length,
#   so attribute names will be restricted based off the beginning of the shapefile names
#
#   The current entries are placeholders/examples

#usually in meters, might differ based off CRS being used
gridWidth = 200
gridHeight = 200

#set lowerleft and upper right coordinates of discretization if necessary
#set to [-1, -1] and [-1, -1] if you don't want to customize the boundary
#and simply want to use the tightest square discretization
lowerleft = [-1, -1]
upperright = [-1, -1]

#specify the name off the boundary file, you do not need to specify the path to this file as
#long as all input shapefiles are in the same path described by input_path
boundary_file = "TN_Core_AOI.shp"

#import paths
input_path = "/example/path/to/input/"
output_path_shapefiles = "/example/path/to/output shapefiles/"
output_path_excel = "/example/path/to/output excel/"

#specify layers we want to find distances from 
dist_layers = ['EXAMPLES.shp', 'Lake.shp', 'Rivers_HS.shp', 'TN_AllBasecamp.shp']

#specify layers we want to find intersections with
int_layers = ['EXAMPLES.shp', 'Drainage_clip.shp', 'Rivers_HS.shp', 'TN_AllBasecamp.shp']

#specify raster files
raster_layers = ['elevation.tif', 'slope.tif']

#patrolling layers (shapefiles of lines)
patrols = ['patrol.shp']
#specify level of splitting patrol lines into points (higher -> less points)
POINT_DISTANCE = 100


####################################################################################################################################################    
####################################################################################################################################################
####################################################################################################################################################    
####################################################################################################################################################
#
# Module 1 - importing implementation for NNjoin plugin 
#

from qgis.core import QgsMessageLog
from qgis.core import QGis
#from qgis.core import QgsWkbTypes
from qgis.core import QgsVectorLayer, QgsFeature, QgsSpatialIndex
from qgis.core import QgsFeatureRequest, QgsField, QgsGeometry
from qgis.core import QgsRectangle, QgsCoordinateTransform

#QGIS 3
#from qgis.PyQt import QtCore
#from qgis.PyQt.QtCore import QCoreApplication, QVariant

#QGIS 2
from PyQt4 import QtCore
from PyQt4.QtCore import QCoreApplication, QVariant


class Worker(QtCore.QObject):
    '''The worker that does the heavy lifting.
    /* QGIS offers spatial indexes to make spatial search more
     * effective.  QgsSpatialIndex will find the nearest index
     * (approximate) geometry (rectangle) for a supplied point.
     * QgsSpatialIndex will only give correct results when searching
     * for the nearest neighbour of a point in a point data set.
     * So something has to be done for non-point data sets
     *
     * Non-point join data set:
     * A two pass search is performed.  First the index is used to
     * find the nearest index geometry (approximation - rectangle),
     * and then compute the distance to the actual indexed geometry.
     * A rectangle is constructed from this (maximum minimum)
     * distance, and this rectangle is used to find all features in
     * the join data set that may be the closest feature to the given
     * point.
     * For all the features is this candidate set, the actual
     * distance to the given point is calculated, and the nearest
     * feature is returned.
     *
     * Non-point input data set:
     * First the centroid of the non-point input geometry is
     * calculated.  Then the index is used to find the nearest
     * neighbour to this point (using the approximate index
     * geometry).
     * The distance vector to this feature, combined with the
     * bounding rectangle of the input feature is used to create a
     * search rectangle to find the candidate join geometries.
     * For all the features is this candidate set, the actual
     * distance to the given feature is calculated, and the nearest
     * feature is returned.
     *
     * Joins involving multi-geometry data sets are not supported
     * by a spatial index.
     *
    */
    '''
    # Define the signals used to communicate back to the application
    progress = QtCore.pyqtSignal(float)  # For reporting progress
    status = QtCore.pyqtSignal(str)      # For reporting status
    error = QtCore.pyqtSignal(str)       # For reporting errors
    # Signal for sending over the result:
    finished = QtCore.pyqtSignal(bool, object)

    def __init__(self, inputvectorlayer, joinvectorlayer,
                 outputlayername, joinprefix,
                 distancefieldname="distance",
                 approximateinputgeom=False,
                 usejoinlayerapproximation=False,
                 usejoinlayerindex=True,
                 selectedinputonly=True,
                 selectedjoinonly=True):
        """Initialise.

        Arguments:
        inputvectorlayer -- (QgsVectorLayer) The base vector layer
                            for the join
        joinvectorlayer -- (QgsVectorLayer) the join layer
        outputlayername -- (string) the name of the output memory
                           layer
        joinprefix -- (string) the prefix to use for the join layer
                      attributes in the output layer
        distancefieldname -- name of the (new) field where neighbour
                             distance is stored
        approximateinputgeom -- (boolean) should the input geometry
                                be approximated?  Is only be set for
                                non-single-point layers
        usejoinlayerindexapproximation -- (boolean) should the index
                             geometry approximations be used for the
                             join?
        usejoinlayerindex -- (boolean) should an index for the join
                             layer be used.
        """

        QtCore.QObject.__init__(self)  # Essential!
        # Set a variable to control the use of indexes and exact
        # geometries for non-point input geometries
        self.nonpointexactindex = usejoinlayerindex
        # Creating instance variables from the parameters
        self.inpvl = inputvectorlayer
        self.joinvl = joinvectorlayer
        self.outputlayername = outputlayername
        self.joinprefix = joinprefix
        self.approximateinputgeom = approximateinputgeom
        self.usejoinlayerapprox = usejoinlayerapproximation
        self.selectedinonly = selectedinputonly
        self.selectedjoonly = selectedjoinonly
        # Check if the layers are the same (self join)
        self.selfjoin = False
        if self.inpvl is self.joinvl:
            # This is a self join
            self.selfjoin = True
        # The name of the attribute for the calculated distance
        self.distancename = distancefieldname
        # Creating instance variables for the progress bar ++
        # Number of elements that have been processed - updated by
        # calculate_progress
        self.processed = 0
        # Current percentage of progress - updated by
        # calculate_progress
        self.percentage = 0
        # Flag set by kill(), checked in the loop
        self.abort = False
        # Number of features in the input layer - used by
        # calculate_progress (set when needed)
        self.feature_count = 1
        # The number of elements that is needed to increment the
        # progressbar (set when needed)
        self.increment = 0

    def run(self):
        try:
            # Check if the layers look OK
            if self.inpvl is None or self.joinvl is None:
                #self.status.emit('Layer is missing!')
                self.finished.emit(False, None)
                return
            # Check if there are features in the layers
            incount = 0
            if self.selectedinonly:
                incount = self.inpvl.selectedFeatureCount()
            else:
                incount = self.inpvl.featureCount()
            joincount = 0
            if self.selectedjoonly:
                joincount = self.joinvl.selectedFeatureCount()
            else:
                joincount = self.joinvl.featureCount()
            if incount == 0 or joincount == 0:
                #self.status.emit('Layer without features!')
                print "Layer without features"
                self.finished.emit(False, None)
                return
            # Check the geometry type and prepare the output layer
            geometryType = self.inpvl.geometryType()
            geometrytypetext = 'Point'
            if geometryType == QGis.Point:
                geometrytypetext = 'Point'
            elif geometryType == QGis.Line:
                geometrytypetext = 'LineString'
            elif geometryType == QGis.Polygon:
                geometrytypetext = 'Polygon'
            # Does the input vector contain multi-geometries?
            # Try to check the first feature
            # This is not used for anything yet
            self.inputmulti = False
            if self.selectedinonly:
                feats = self.inpvl.selectedFeaturesIterator()
            else:
                feats = self.inpvl.getFeatures()
            if feats is not None:
                testfeature = next(feats)
                feats.rewind()
                feats.close()
                if testfeature is not None:
                    if testfeature.geometry() is not None:
                        if testfeature.geometry().isMultipart():
                            self.inputmulti = True
                            geometrytypetext = 'Multi' + geometrytypetext
                        else:
                            pass
                    else:
                        #self.status.emit('No geometry!')
                        print "No geometry"
                        self.finished.emit(False, None)
                        return
                else:
                    #self.status.emit('No input features!')
                    print "No input features!"
                    self.finished.emit(False, None)
                    return
            else:
                #self.status.emit('getFeatures returns None for input layer!')
                print "getFeatures returns None for input layer"
                self.finished.emit(False, None)
                return
            geomttext = geometrytypetext
            # Set the coordinate reference system to the input
            # layer's CRS using authid (proj4 may be more robust)
            crstext = "PROJ4:" + str(self.inpvl.crs().toProj4())
            # If the authid is valid (EPSG), use it.
            if "EPSG" in str(self.inpvl.crs().authid()):
                crstext = self.inpvl.crs().authid()
            if self.inpvl.crs() is not None:
                geomttext = (geomttext + "?crs=" +
                              crstext)
            # Retrieve the fields from the input layer
            outfields = self.inpvl.pendingFields().toList()
            # Retrieve the fields from the join layer
            if self.joinvl.pendingFields() is not None:
                jfields = self.joinvl.pendingFields().toList()
                for joinfield in jfields:
                    outfields.append(QgsField(self.joinprefix +
                                     str(joinfield.name()),
                                     joinfield.type()))
            else:
                #self.status.emit('Unable to get any join layer fields')
                print "Unable to get any join layer fields"
            # Add the nearest neighbour distance field
            # Check if there is already a "distance" field
            # (should be avoided in the user interface)
            # Try a new name if there is a collission
            collission = True
            while collission:   # Iterate until there are no collissions
                collission = False
                for field in outfields:
                    if field.name() == self.distancename:
                        #self.status.emit(
                              #'Distance field already exists - renaming!')
                        print "Distance field already exists - renaming!"
                        #self.abort = True
                        #self.finished.emit(False, None)
                        #break
                        collission = True
                        self.distancename = self.distancename + '1'
            outfields.append(QgsField(self.distancename, QVariant.Double))
            # Create a memory layer
            self.mem_joinl = QgsVectorLayer(geomttext,
                                            self.outputlayername,
                                            "memory")
            self.mem_joinl.startEditing()
            # Add the fields
            for field in outfields:
                self.mem_joinl.dataProvider().addAttributes([field])
            # For an index to be used, the input layer has to be a
            # point layer, the input layer geometries have to be
            # approximated to centroids, or the user has to have
            # accepted that a join layer index is used (for
            # non-point input layers).
            # (Could be extended to multipoint)
            if (self.inpvl.wkbType() == QGis.WKBPoint or
                    self.inpvl.wkbType() == QGis.WKBPoint25D or
                    self.approximateinputgeom or
                    self.nonpointexactindex):
                # Create a spatial index to speed up joining
                ##self.status.emit('Creating join layer index...')
                print "Creating join layer index"
                # Number of features in the input layer - used by
                # calculate_progress
                if self.selectedjoonly:
                    self.feature_count = self.joinvl.selectedFeatureCount()
                else:
                    self.feature_count = self.joinvl.featureCount()
                # The number of elements that is needed to increment the
                # progressbar - set early in run()
                self.increment = self.feature_count // 1000
                self.joinlind = QgsSpatialIndex()
                if self.selectedjoonly:
                    for feat in self.joinvl.selectedFeaturesIterator():
                        # Allow user abort
                        if self.abort is True:
                            break
                        self.joinlind.insertFeature(feat)
                        self.calculate_progress()
                else:
                    for feat in self.joinvl.getFeatures():
                        # Allow user abort
                        if self.abort is True:
                            break
                        self.joinlind.insertFeature(feat)
                        self.calculate_progress()
                ##self.status.emit('Join layer index created!')
                print "Join layer index created"
                self.processed = 0
                self.percentage = 0
                #self.calculate_progress()
            # Does the join layer contain multi geometries?
            # Try to check the first feature
            # This is not used for anything yet
            self.joinmulti = False
            if self.selectedjoonly:
                feats = self.joinvl.selectedFeaturesIterator()
            else:
                feats = self.joinvl.getFeatures()
            if feats is not None:
                testfeature = next(feats)
                feats.rewind()
                feats.close()
                if testfeature is not None:
                    if testfeature.geometry() is not None:
                        if testfeature.geometry().isMultipart():
                            self.joinmulti = True
                    else:
                        #self.status.emit('No join geometry!')
                        print "No join geometry"
                        self.finished.emit(False, None)
                        return
                else:
                    #self.status.emit('No join features!')
                    print "No join features"
                    self.finished.emit(False, None)
                    return
            # Prepare for the join by fetching the layers into memory
            # Add the input features to a list
            self.inputf = []
            if self.selectedinonly:
                for f in self.inpvl.selectedFeaturesIterator():
                    self.inputf.append(f)
            else:
                for f in self.inpvl.getFeatures():
                    self.inputf.append(f)
            # Add the join features to a list
            self.joinf = []
            if self.selectedjoonly:
                for f in self.joinvl.selectedFeaturesIterator():
                    self.joinf.append(f)
            else:
                for f in self.joinvl.getFeatures():
                    self.joinf.append(f)
            self.features = []
            # Do the join!
            # Number of features in the input layer - used by
            # calculate_progress
            if self.selectedinonly:
                self.feature_count = self.inpvl.selectedFeatureCount()
            else:
                self.feature_count = self.inpvl.featureCount()
            # The number of elements that is needed to increment the
            # progressbar - set early in run()
            self.increment = self.feature_count // 1000
            # Using the original features from the input layer
            for feat in self.inputf:
                # Allow user abort
                if self.abort is True:
                    break
                self.do_indexjoin(feat)
                self.calculate_progress()
            self.mem_joinl.dataProvider().addFeatures(self.features)
            ##self.status.emit('Join finished')
            print "Join finished"
        except:
            #print "2"
            import traceback
            print traceback.format_exc()
            #self.error.emit(traceback.format_exc())
            self.finished.emit(False, None)
            if self.mem_joinl is not None:
                self.mem_joinl.rollBack()
        else:
            #print "3"
            self.mem_joinl.commitChanges()
            if self.abort:
                self.finished.emit(False, None)
            else:
                #self.status.emit('Delivering the memory layer...')
                self.finished.emit(True, self.mem_joinl)

    def calculate_progress(self):
        '''Update progress and emit a signal with the percentage'''
        self.processed = self.processed + 1
        # update the progress bar at certain increments
        if (self.increment == 0 or
                self.processed % self.increment == 0):
            # Calculate percentage as integer
            perc_new = (self.processed * 100) / self.feature_count
            if perc_new > self.percentage:
                self.percentage = perc_new
                self.progress.emit(self.percentage)

    def kill(self):
        '''Kill the thread by setting the abort flag'''
        self.abort = True

    def do_indexjoin(self, feat):
        '''Find the nearest neigbour using an index, if possible

        Parameter: feat -- The feature for which a neighbour is
                           sought
        '''
        infeature = feat
        # Get the feature ID
        infeatureid = infeature.id()
        # Get the feature geometry
        inputgeom = QgsGeometry(infeature.geometry())
        # Shall approximate input geometries be used?
        if self.approximateinputgeom:
            # Use the centroid as the input geometry
            inputgeom = QgsGeometry(infeature.geometry()).centroid()
        # Check if the coordinate systems are equal, if not,
        # transform the input feature!
        if (self.inpvl.crs() != self.joinvl.crs()):
            try:
                inputgeom.transform(QgsCoordinateTransform(
                    self.inpvl.crs(), self.joinvl.crs()))
            except:
                import traceback
                self.error.emit(self.tr('CRS Transformation error!') +
                                ' - ' + traceback.format_exc())
                self.abort = True
                return
        ## Find the closest feature!
        nnfeature = None
        mindist = float("inf")
        if (self.approximateinputgeom or
                self.inpvl.wkbType() == QGis.WKBPoint or
                self.inpvl.wkbType() == QGis.WKBPoint25D):
            # The input layer's geometry type is point, or has been
            # approximated to point (centroid).
            # Then a join index will always be used.
            if (self.usejoinlayerapprox or
                    self.joinvl.wkbType() == QGis.WKBPoint or
                    self.joinvl.wkbType() == QGis.WKBPoint25D):
                # The join index nearest neighbour function can
                # be used without refinement.
                if self.selfjoin:
                    # Self join!
                    # Have to get the two nearest neighbours
                    nearestids = self.joinlind.nearestNeighbor(
                                             inputgeom.asPoint(), 2)
                    if nearestids[0] == infeatureid and len(nearestids) > 1:
                        # The first feature is the same as the input
                        # feature, so choose the second one
                        if self.selectedjoonly:
                            nnfeature = next(
                                self.joinvl.selectedFeaturesIterator(
                                    QgsFeatureRequest(nearestids[1])))
                        else:
                            nnfeature = next(self.joinvl.getFeatures(
                                QgsFeatureRequest(nearestids[1])))
                    else:
                        # The first feature is not the same as the
                        # input feature, so choose it
                        if self.selectedjoonly:
                            nnfeature = next(
                                self.joinvl.selectedFeaturesIterator(
                                    QgsFeatureRequest(nearestids[0])))
                        else:
                            nnfeature = next(self.joinvl.getFeatures(
                                QgsFeatureRequest(nearestids[0])))
                else:
                    # Not a self join, so we can search for only the
                    # nearest neighbour (1)
                    nearestid = self.joinlind.nearestNeighbor(
                                           inputgeom.asPoint(), 1)[0]
                    if self.selectedjoonly:
                        nnfeature = next(self.joinvl.selectedFeaturesIterator(
                                 QgsFeatureRequest(nearestid)))
                    else:
                        nnfeature = next(self.joinvl.getFeatures(
                                 QgsFeatureRequest(nearestid)))
                mindist = inputgeom.distance(nnfeature.geometry())
            elif (self.joinvl.wkbType() == QGis.WKBPolygon or
                  self.joinvl.wkbType() == QGis.WKBPolygon25D or
                  self.joinvl.wkbType() == QGis.WKBLineString or
                  self.joinvl.wkbType() == QGis.WKBLineString25D):
                # Use the join layer index to speed up the join when
                # the join layer geometry type is polygon or line
                # and the input layer geometry type is point or an
                # approximation (point)
                nearestindexid = self.joinlind.nearestNeighbor(
                    inputgeom.asPoint(), 1)[0]
                # Check for self join
                if self.selfjoin and nearestindexid == infeatureid:
                    # Self join and same feature, so get the
                    # first two neighbours
                    nearestindexes = self.joinlind.nearestNeighbor(
                                             inputgeom.asPoint(), 2)
                    nearestindexid = nearestindexes[0]
                    if (nearestindexid == infeatureid and
                                  len(nearestindexes) > 1):
                        nearestindexid = nearestindexes[1]
                if self.selectedjoonly:
                    nnfeature = next(self.joinvl.selectedFeaturesIterator(
                        QgsFeatureRequest(nearestindexid)))
                else:
                    nnfeature = next(self.joinvl.getFeatures(
                        QgsFeatureRequest(nearestindexid)))
                mindist = inputgeom.distance(nnfeature.geometry())
                px = inputgeom.asPoint().x()
                py = inputgeom.asPoint().y()
                closefids = self.joinlind.intersects(QgsRectangle(
                    px - mindist,
                    py - mindist,
                    px + mindist,
                    py + mindist))
                for closefid in closefids:
                    if self.abort is True:
                        break
                    # Check for self join and same feature
                    if self.selfjoin and closefid == infeatureid:
                        continue
                    if self.selectedjoonly:
                        closef = next(self.joinvl.selectedFeaturesIterator(
                            QgsFeatureRequest(closefid)))
                    else:
                        closef = next(self.joinvl.getFeatures(
                            QgsFeatureRequest(closefid)))
                    thisdistance = inputgeom.distance(closef.geometry())
                    if thisdistance < mindist:
                        mindist = thisdistance
                        nnfeature = closef
                    if mindist == 0:
                        break
            else:
                # Join with no index use
                # Go through all the features from the join layer!
                for inFeatJoin in self.joinf:
                    if self.abort is True:
                        break
                    joingeom = QgsGeometry(inFeatJoin.geometry())
                    thisdistance = inputgeom.distance(joingeom)
                    # If the distance is 0, check for equality of the
                    # features (in case it is a self join)
                    if (thisdistance == 0 and self.selfjoin and
                            infeatureid == inFeatJoin.id()):
                        continue
                    if thisdistance < mindist:
                        mindist = thisdistance
                        nnfeature = inFeatJoin
                    # For 0 distance, settle with the first feature
                    if mindist == 0:
                        break
        else:
            # non-simple point input geometries (could be multipoint)
            if (self.nonpointexactindex):
                # Use the spatial index on the join layer (default).
                # First we do an approximate search
                # Get the input geometry centroid
                centroid = QgsGeometry(infeature.geometry()).centroid()
                centroidgeom = centroid.asPoint()
                # Find the nearest neighbour (index geometries only)
                nearestid = self.joinlind.nearestNeighbor(centroidgeom, 1)[0]
                # Check for self join
                if self.selfjoin and nearestid == infeatureid:
                    # Self join and same feature, so get the two
                    # first two neighbours
                    nearestindexes = self.joinlind.nearestNeighbor(
                        centroidgeom, 2)
                    nearestid = nearestindexes[0]
                    if nearestid == infeatureid and len(nearestindexes) > 1:
                        nearestid = nearestindexes[1]
                if self.selectedjoonly:
                    nnfeature = next(self.joinvl.selectedFeaturesIterator(
                        QgsFeatureRequest(nearestid)))
                else:
                    nnfeature = next(self.joinvl.getFeatures(
                        QgsFeatureRequest(nearestid)))
                mindist = inputgeom.distance(nnfeature.geometry())
                # Calculate the search rectangle (inputgeom BBOX
                inpbbox = infeature.geometry().boundingBox()
                minx = inpbbox.xMinimum() - mindist
                maxx = inpbbox.xMaximum() + mindist
                miny = inpbbox.yMinimum() - mindist
                maxy = inpbbox.yMaximum() + mindist
                #minx = min(inpbbox.xMinimum(), centroidgeom.x() - mindist)
                #maxx = max(inpbbox.xMaximum(), centroidgeom.x() + mindist)
                #miny = min(inpbbox.yMinimum(), centroidgeom.y() - mindist)
                #maxy = max(inpbbox.yMaximum(), centroidgeom.y() + mindist)
                searchrectangle = QgsRectangle(minx, miny, maxx, maxy)
                # Fetch the candidate join geometries
                closefids = self.joinlind.intersects(searchrectangle)
                # Loop through the geometries and choose the closest
                # one
                for closefid in closefids:
                    if self.abort is True:
                        break
                    # Check for self join and identical feature
                    if self.selfjoin and closefid == infeatureid:
                        continue
                    if self.selectedjoonly:
                        closef = next(self.joinvl.selectedFeaturesIterator(
                            QgsFeatureRequest(closefid)))
                    else:
                        closef = next(self.joinvl.getFeatures(
                            QgsFeatureRequest(closefid)))
                    thisdistance = inputgeom.distance(closef.geometry())
                    if thisdistance < mindist:
                        mindist = thisdistance
                        nnfeature = closef
                    if mindist == 0:
                        break
            else:
                # Join with no index use
                # Check all the features of the join layer!
                mindist = float("inf")  # should not be necessary
                for inFeatJoin in self.joinf:
                    if self.abort is True:
                        break
                    joingeom = QgsGeometry(inFeatJoin.geometry())
                    thisdistance = inputgeom.distance(joingeom)
                    # If the distance is 0, check for equality of the
                    # features (in case it is a self join)
                    if (thisdistance == 0 and self.selfjoin and
                            infeatureid == inFeatJoin.id()):
                        continue
                    if thisdistance < mindist:
                        mindist = thisdistance
                        nnfeature = inFeatJoin
                    # For 0 distance, settle with the first feature
                    if mindist == 0:
                        break
        if not self.abort:
            # Collect the attribute
            atMapA = infeature.attributes()
            atMapB = nnfeature.attributes()
            attrs = []
            attrs.extend(atMapA)
            attrs.extend(atMapB)
            attrs.append(mindist)
            # Create the feature
            outFeat = QgsFeature()
            # Use the original input layer geometry!:
            outFeat.setGeometry(QgsGeometry(infeature.geometry()))
            # Use the modified input layer geometry (could be
            # centroid)
            #outFeat.setGeometry(QgsGeometry(inputgeom))
            # Add the attributes
            outFeat.setAttributes(attrs)
            #self.calculate_progress()
            self.features.append(outFeat)
            #self.mem_joinl.dataProvider().addFeatures([outFeat])

    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('NNJoinEngine', message)


####################################################################################################################################################    
####################################################################################################################################################
####################################################################################################################################################    
####################################################################################################################################################
#
# Module 2 - importing implementation for MMQGIS export attibutes to csv function
#

def mmqgis_attribute_export(qgis, outfilename, layername, attribute_names, field_delimiter, line_terminator, decimal_mark):
    # Error checks

    if (not outfilename) or (len(outfilename) <= 0):
        return "No output CSV file given"

    # this line is changed by Tianyu
    # layer = mmqgis_find_layer(layername)
    layer = layername
    if layer == None:
        return "Layer " + layername + " not found"

    # print("Delimiter: ", field_delimiter, " Decimal mark: ", decimal_mark)

    # Find attribute indices
    if (not attribute_names) or (len(attribute_names) <= 0):
        attribute_indices = layer.pendingAllAttributesList()

        # http://www.secnetix.de/olli/Python/list_comprehensions.hawk
        # attribute_names = map(layer.attributeDisplayName, attribute_indices)
        attribute_names = [layer.attributeDisplayName(x) for x in attribute_indices]

    else:
        attribute_indices = []
        for name in attribute_names:
            index = layer.fieldNameIndex(name)
            if index < 0:
                return "Layer " + "layername" + " has no attribute " + name
            attribute_indices.append(index)

    # Create the CSV file
    try:
        outfile = open(outfilename, 'w')
    except:
        return "Failure opening " + outfilename

    writer = csv.writer(outfile, delimiter = field_delimiter, lineterminator = line_terminator)

    # Encoding is forced to UTF-8 because CSV writer doesn't support Unicode
    writer.writerow([field.encode("utf-8") for field in attribute_names])

    # Iterate through each feature in the source layer
    feature_count = layer.featureCount()
    for index, feature in enumerate(layer.getFeatures()):
        if (index % 50) == 0:
            qgis.mainWindow().statusBar().showMessage \
                ("Exporting feature " + unicode(feature.id()) + " of " + unicode(feature_count))
        attributes = feature.attributes()

        row = []
        for column in attribute_indices:
            # print unicode(column) + " (" + decimal_mark + "): " + type(attributes[column]).__name__

            if attributes[column] == None:
                row.append("")

            
                
            else:
                row.append(attributes[column])

        writer.writerow([unicode(field).encode("utf-8") for field in row])

    del writer

    #mmqgis_completion_message(qgis, unicode(feature_count) + " records exported")

    return None 


####################################################################################################################################################    
####################################################################################################################################################
####################################################################################################################################################    
####################################################################################################################################################
#
# Module 4 - Our actual preprocessing code
#

#only use this class for creating discretized grid, doesn't work for centroids layer
class Crea_layer(object):
    def __init__(self,name,type):
        self.type=type
        self.name = name
        self.layer =  QgsVectorLayer(self.type, self.name , "memory")
        self.layer.dataProvider().addAttributes([QgsField("X", QVariant.Double), QgsField("Y", QVariant.Double)])
        self.layer.updateFields()
        self.pr =self.layer.dataProvider() 
    def create_poly(self, points, x, y):
        self.seg = QgsFeature()  
        self.seg.setGeometry(QgsGeometry.fromPolygon([points]))
        fields = self.layer.pendingFields()
        self.seg.setFields( fields, True )
        self.seg['X'] = x
        self.seg['Y'] = y
        self.pr.addFeatures( [self.seg] )
        self.layer.updateExtents()
    @property
    def disp_layer(self):
        QgsMapLayerRegistry.instance().addMapLayers([self.layer])
    def get_layer(self):
        return self.layer

#annotated importation of vector layer
def import_layer(filename, label):

    temp = ogr.Open(filename, 0)
    if temp is None:
        print 'Failed to open: ', filename
    else:
        print 'Opened: ', filename
    temp = QgsVectorLayer(filename, label, "ogr")
    return temp

#annotated importation of raster layer
def import_layer_raster(filename, label):
    
    temp = gdal.Open(filename, 0)
    if temp is None:
        print 'Failed to open: ', filename
    else:
        print 'Opened: ', filename
    return QgsRasterLayer(filename, label, "gdal")

#function to add a field when determining is- features
def add_is_field(layer, name, val):
    res = layer.dataProvider().addAttributes([QgsField(name, QVariant.Int)])
    layer.updateFields()
    fieldIndex = layer.dataProvider().fieldNameIndex(name)
    attrFeatMap = {}
    attrMap = { fieldIndex : val }
    for feature in layer.getFeatures():
        attrFeatMap[ feature.id() ] = attrMap
    layer.dataProvider().changeAttributeValues( attrFeatMap )

####################################################################################################################################################
#
# Module 4.1 - Creating the Discretized grid and its centroids
#

#insert path to conservation site 
conservation_site = import_layer(input_path + boundary_file, "conservation site")
QgsMapLayerRegistry.instance().addMapLayers([conservation_site])

#obtain the discretized grid
canvas= qgis.utils.iface.mapCanvas()
xmin,ymin,xmax,ymax = conservation_site.extent().toRectF().getCoords()

#set boundary coordinates to customable points if specified
if lowerleft[0] != -1:
    xmin = lowerleft[0]
if lowerleft[1] != -1:
    ymin = lowerleft[1]
if upperright[0] != -1:
    xmax = upperright[0]
if upperright[1] != -1:
    ymax = upperright[1]

rows = ceil((ymax-ymin)/gridHeight)
cols = ceil((xmax-xmin)/gridWidth)
Xleft = xmin
Xright = xmin + gridWidth
YbottomOrigin = ymin 
YtopOrigin = ymin + gridHeight

grid = Crea_layer("grid", "Polygon")

for i in range(int(cols)):
    Ybottom = YbottomOrigin
    Ytop = YtopOrigin
    for j in range(int(rows)):
        cur_block = [QgsPoint(Xleft, Ytop), \
                        QgsPoint(Xright, Ytop), \
                        QgsPoint(Xright, Ybottom), \
                        QgsPoint(Xleft, Ybottom), 
                        QgsPoint(Xleft, Ytop)]
        grid.create_poly(cur_block, Xleft, Ybottom)
        Ybottom += gridHeight
        Ytop += gridHeight
    Xleft += gridWidth
    Xright += gridWidth

#create intersection betweengrid cells and conservation site
site_layer = QgsVectorLayer(input_path + boundary_file, "conservation site", "ogr")
overlayAnalyzer = qgis.analysis.QgsOverlayAnalyzer() 
overlayAnalyzer.intersection(grid.get_layer(), site_layer, output_path_shapefiles + "discretized_grid.shp") 

#create DN attribute to uniquely identify grid cells
grid = import_layer(output_path_shapefiles + "discretized_grid.shp", "temp grid")
grid.startEditing()
grid.dataProvider().addAttributes([QgsField("DN", QVariant.Int)])
grid.updateFields()

#enumerate DN based on ordering of x and y coordinates
pos_id_list = [(feat.geometry().asPoint().x(), feat.geometry().asPoint().y(), feat.id()) for feat in grid.getFeatures()]
pos_id_list.sort()  

i = 1
#get DN index
dnindex = grid.fieldNameIndex('DN')  
for feat in pos_id_list:
    #set DN value
    grid.changeAttributeValue(feat[2], dnindex, i)  
    i += 1

grid.commitChanges()  # save changed and stop editing

#create centroids
points = QgsVectorLayer("Point", "point grid", "memory")
pr = points.dataProvider()
points.startEditing()
points.dataProvider().addAttributes([QgsField("DN", QVariant.Int), QgsField("X", QVariant.Double), QgsField("Y", QVariant.Double)])
points.updateFields()

features = grid.getFeatures()
num_features = 0
for feature in features:

    cur_point = QgsFeature()
    cur_point.setGeometry(feature.geometry().centroid())
    fields = points.pendingFields()
    cur_point.setFields( fields, True )

    cur_point['DN'] = feature.attributes()[grid.fieldNameIndex('DN')]
    cur_point['X'] = feature.attributes()[grid.fieldNameIndex('X')]
    cur_point['Y'] = feature.attributes()[grid.fieldNameIndex('Y')]
    pr.addFeatures( [ cur_point ] )

    num_features += 1

print("Total number of cells created: ", num_features)
points.commitChanges()

#save the centroids
QgsVectorFileWriter.writeAsVectorFormat(points, output_path_shapefiles + "point_grid.shp", "utf-8", None, "ESRI Shapefile")

#load the centroids and grid
centroids = import_layer(output_path_shapefiles + "point_grid.shp", "point grid")
grid = import_layer(output_path_shapefiles + "discretized_grid.shp", "discretized grid")

QgsMapLayerRegistry.instance().addMapLayers([grid, centroids])

####################################################################################################################################################
#
# Module 4.2 calculating "is-" features
#

for int_layer in int_layers:
  areas = []
  cur_layer = import_layer(input_path + int_layer, int_layer)
  for line_feature in cur_layer.getFeatures():
      cands = grid.getFeatures(QgsFeatureRequest().setFilterRect(line_feature.geometry().boundingBox()))
      for grid_block in cands:
          if line_feature.geometry().intersects(grid_block.geometry()):
              areas.append(grid_block.id())
  grid.select(areas)

  #save true features
  layer = iface.activeLayer()
  res = QgsVectorFileWriter.writeAsVectorFormat( layer, output_path_shapefiles + "(1)is-" + int_layer, "utf-8", layer.crs(), "ESRI Shapefile", 1)

  if res != QgsVectorFileWriter.NoError:
    print 'Error number:', res, ' Input Layer (1): ', int_layer
  else:
    print "Saved (1): ", int_layer

  #save false features
  grid.invertSelection()
  layer = iface.activeLayer()
  res = QgsVectorFileWriter.writeAsVectorFormat( layer, output_path_shapefiles + "(0)is-" + int_layer, "utf-8", layer.crs(), "ESRI Shapefile", 1)

  if res != QgsVectorFileWriter.NoError:
    print 'Error number:', res, ' Input Layer (0): ', int_layer
  else:
    print "Saved (0): ", int_layer

  #add field to true features
  len_name = min((len(int_layer) - 4), 6)
  is1 = import_layer(output_path_shapefiles + "(1)is-" + int_layer, "is1_file")
  add_is_field(is1, "is-" + int_layer[0 : len_name], 1)
  is0 = import_layer(output_path_shapefiles + "(0)is-" + int_layer, "is0_file")
  add_is_field(is0, "is-" + int_layer[0 : len_name], 0)

  processing.runalg("qgis:mergevectorlayers", \
            output_path_shapefiles + "(0)is-" + int_layer + ";" + \
            output_path_shapefiles + "(1)is-" + int_layer, \
            output_path_shapefiles + "is-" + int_layer)

  final_layer = import_layer(output_path_shapefiles + "is-" + int_layer, "is-" + int_layer)

  #export as csv
  mmqgis_attribute_export(iface, \
        output_path_excel + "is-" + int_layer[0 : (len(int_layer) - 4)] + ".csv", \
        final_layer, ["DN", "is-" + int_layer[0 : len_name]], \
        field_delimiter = ',', line_terminator = '\n', decimal_mark = '.')

####################################################################################################################################################
#
# Module 4.3 calculating the "dist-" features
#

for dist_layer in dist_layers:

  name = dist_layer[0 : min((len(dist_layer) - 4),4)]

  cur_layer = import_layer(input_path + dist_layer, name)
  worker = Worker(centroids, cur_layer, "dist-" + name, "join_", selectedinputonly=False, selectedjoinonly=False)
  worker.run()

  #save the distance layer
  res = QgsVectorFileWriter.writeAsVectorFormat(worker.mem_joinl, output_path_shapefiles + "dist-" + dist_layer, "utf-8", None, "ESRI Shapefile")
  if res != QgsVectorFileWriter.NoError:
    print 'Error number:', res, ' Input Layer: ', dist_layer
  else:
    print "Saved: dist_", dist_layer

  actual_layer = import_layer(output_path_shapefiles + "dist-" + dist_layer, name)

  #export as csv
  mmqgis_attribute_export(iface, \
        output_path_excel + "dist-" + dist_layer[0 : (len(dist_layer) - 4)] + ".csv", \
        actual_layer, ["DN", "distance"], \
        field_delimiter = ',', line_terminator = '\n', decimal_mark = '.')

####################################################################################################################################################
#
# Module 4.4 obtaining slope and elevation data
#

for raster_name in raster_layers:

    feature_name = raster_name[0:(len(raster_name) - 4)]
    raster_layer = import_layer_raster(input_path + raster_name, feature_name)

    raster_provider = raster_layer.dataProvider()

    pr = centroids.dataProvider()
    centroids.startEditing()
    centroids.dataProvider().addAttributes([QgsField(feature_name, QVariant.Double)])
    centroids.updateFields()

    features = centroids.getFeatures()
    for feat in features:

      point = QgsPoint((feat['X']), (feat['Y']))

      raster_value = raster_provider.identify(point, QgsRaster.IdentifyFormatValue).results()[1]

      fields = centroids.pendingFields()
      old_fields = {}

      for field in fields:
        if field.name() != feature_name:
          old_fields[field.name()] = feat.attributes()[centroids.fieldNameIndex(field.name())]

      feat.setFields( fields, True )

      for field in fields:
        if field.name() == feature_name:
          feat[feature_name] = raster_value
        else:
          feat[field.name()] = old_fields[field.name()]

      centroids.updateFeature(feat)

    centroids.commitChanges()

    mmqgis_attribute_export(iface, \
            output_path_excel + feature_name + ".csv", \
            centroids, ["DN", feature_name], \
            field_delimiter = ',', line_terminator = '\n', decimal_mark = '.')


####################################################################################################################################################
#
# Module 4.5 obtaining patrol length
#

for patrol in patrols:

  patrol_layer = import_layer(input_path + patrol, "patrol layer temp")
  features = patrol_layer.getFeatures()

  points = QgsVectorLayer("Point", "points", "memory")
  pr = points.dataProvider()
  points.startEditing()

  for feat in features:

    if feat.geometry() is not None:

      geom = feat.geometry()
      length = geom.length()

      distance = POINT_DISTANCE

      iter = distance

      while iter <= length:
        cur_point = QgsFeature()
        cur_point.setGeometry(feat.geometry().interpolate(iter))
        iter += distance
        pr.addFeatures( [ cur_point ] )

  points.commitChanges()

  export_points = output_path_shapefiles + "/points-" + patrol
  layer = points
  res = QgsVectorFileWriter.writeAsVectorFormat( layer, export_points, "utf-8", None, "ESRI Shapefile")
  if res != QgsVectorFileWriter.NoError:
    print 'Error number:', res, ' Input Layer: ', patrol
  else:
    print "Saved: points-", patrol

  Result = output_path_shapefiles + "/num-" + patrol
  processing.runalg("qgis:countpointsinpolygon", input_path + "rectangle-grid-abs.shp", export_points, 'NUMPOINTS', Result)

  result_layer = import_layer(Result, "result patrol")

  mmqgis_attribute_export(iface, \
        output_path_excel + "num" + patrol[0:(len(patrol)-4)] + ".csv", \
        result_layer, ["DN", "NUMPOINTS"], \
        field_delimiter = ',', line_terminator = '\n', decimal_mark = '.')

####################################################################################################################################################
#
# Module 4.6 combining csv files
#

files = [output_path_excel + "elevation.csv", output_path_excel + "slope.csv"]
column_names = ["elevation", "slope"]

for int_layer in int_layers:
  feature = int_layer[0:(len(int_layer) - 4)]
  #field = int_layer[0:min(len(int_layer) - 4, 6)]
  files.append(output_path_excel + "is-" + feature + ".csv")
  #column_names.append("is-" + field)
  column_names.append("is-" + feature)

for dist_layer in dist_layers:
  feature = dist_layer[0:(len(dist_layer) - 4)]
  #field = dist_layer[0:min(len(dist_layer) - 4, 4)]
  files.append(output_path_excel + "dist-" + feature + ".csv")
  #column_names.append("dist-" + field)
  column_names.append("dist-" + feature)


raw_df_list = []

# read files into dataframe
for f in files:
  print (f)
  raw_df_list.append(pd.read_csv(f))

# get the DN as a dataframe
DN_df = raw_df_list[0][['DN']].sort_values(by=['DN'])
DN_df.reset_index(inplace=True)

# rename columns and sort based on DN
select_df_list = []
for i in range(0,len(raw_df_list)):
  # rename columns
  col_names = ['DN',column_names[i]]
  raw_df_list[i].columns = col_names

  # sort by DN
  cur_sorted_df = raw_df_list[i].sort_values(by=['DN'])
  cur_sorted_df.reset_index(inplace=True)

  # select revelant columns
  cur_select_df = cur_sorted_df[[column_names[i]]]

  # normalize the selected columns
  cur_normalized_df = (cur_select_df - cur_select_df.min())/(cur_select_df.max()-cur_select_df.min())
  cur_normalized_df.columns = ["normal-"+column_names[i]]

  select_df_list.append(cur_select_df)
  if column_names[i][0:3] != 'is-': 
    select_df_list.append(cur_normalized_df)


# concatenate columns
select_df_list = [DN_df] + select_df_list
comb_DN_ABC = pd.concat(select_df_list, axis=1)
comb_DN_ABC.sort_values(by=["DN"],inplace=True)
comb_DN_ABC.drop(['index'], axis=1)
comb_DN_ABC.to_csv(output_path_excel + "final.csv")

print("Finished")



