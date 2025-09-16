from shapely import Polygon
import pyproj
from shapely.ops import transform
import numpy as np
from typing import Optional, List, Dict, Any
from shapely.strtree import STRtree
import geopandas as gpd

class FootprintFeaturesCalculator:
    UNPROJECTED_CRS = 4326
    #Mercator
    PROJECTED_CRS = 3857
    
    @staticmethod
    def calculate_unprojected_area(
        footprint: Polygon,
        original_crs:int= 4326)->float:
        """
        Calculate the unprojected area of a footprint. If no CRS is provided, 
        the function assumes the footprint is in EPSG:4326.
        
        Parameters
        ----------
        footprint
            The polygon footprint as a shapely Polygon object.
        
        Returns
        -------
        area
            The unprojected area of the footprint.
        """
        
        if(original_crs == FootprintFeaturesCalculator.UNPROJECTED_CRS):
            area = footprint.area
            return area
        
        else:
            #Use pyproj to reproject the coordinates of the polygon
            start_crs = pyproj.CRS(original_crs)
            end_crs = pyproj.CRS(FootprintFeaturesCalculator.UNPROJECTED_CRS)
            project = pyproj.Transformer.from_crs(
                start_crs,
                end_crs,
                always_xy = True).transform
            reprojected_footprint = transform(project, footprint)
            
            area = reprojected_footprint.area
            return area
    
    @staticmethod
    def calculate_projected_area(
        footprint:Polygon,
        original_crs:int = 4326)->float:
        
        """
        Calculate the projected area of a footprint in square meters. 
        
        Para
        
        """
        if(original_crs == FootprintFeaturesCalculator.PROJECTED_CRS):
            area = footprint.area
            return area
        else:
            start_crs = pyproj.CRS(original_crs)
            end_crs = pyproj.CRS(FootprintFeaturesCalculator.PROJECTED_CRS)
            project = pyproj.Transformer.from_crs(
                start_crs,
                end_crs,
                always_xy = True).transform
            reprojected_footprint = transform(project, footprint)
            area = reprojected_footprint.area
            return area
    
    @staticmethod
    def longitude_difference(
        footprint:Polygon)->float:
        """
        Calculate the difference between the maximum and minimum longitude.
        The method assumes the polygon is in EPSG:4326.
        Parameters
        ----------
        footprint
            The polygon footprint as a shapely Polygon object. 
        
        Returns
        -------
        diff_lon
            The difference betweeen max_longitude and min_longitude.
        """
        
        poly_coords = list(footprint.exterior.coords)
        #We assume the tuples are (longitude, latitude)
        longitude_values = [x[0] for x in poly_coords]
        max_longitude = float("-inf")
        min_longitude = float("inf")
        for i in range(0,len(longitude_values)):
            longitude = longitude_values[i]
            if(longitude > max_longitude):
                max_longitude = longitude
            if(longitude < min_longitude):
                min_longitude = longitude
        
        diff_lon = max_longitude - min_longitude
        return diff_lon

    @staticmethod
    def latitude_difference(
        footprint:Polygon
    )-> float:
        """
        Calculates the difference between max_latitude and min_latitude.
        The method assumes that the polygon is in EPSG:4326. 
        
        Parameters
        ----------
        footprint
            Shapely polygon of the building footprint.
        
        Returns
        -------
        
        latitude_diff
            difference between max_lat and min_lat.
        """
        
        poly_coords = list(footprint.exterior.coords)
        latitude_values = [x[1] for x in poly_coords]
        max_lat = float("-inf")
        min_lat = float("inf")
        
        for i in range(0,len(poly_coords)):
            latitude = latitude_values[i]
            if(latitude > max_lat):
                max_lat = latitude
            if(latitude < min_lat):
                min_lat = latitude
            
        
        latitude_diff = max_lat - min_lat
        return latitude_diff

    @staticmethod
    def n_vertices(
        footprint:Polygon
    )-> int:
        """
        Calculate the number of vertices in a polygon footprint.
        
        Parameters
        ----------
        footprint
            The polygon footprint as a shapely Polygon object.
        
        Returns
        -------
        n_vertices
            The number of vertices in the polygon footprint.
        """
        
        poly_coords = list(footprint.exterior.coords)
        #Exclude the repeated last vertex
        n_vertices = len(poly_coords) - 1
        return n_vertices
    
    @staticmethod
    def shape_length(
        footprint:Polygon
    )->float:
        """
        
        Calculate the length of the polygon footprint.
        
        Parameters
        ----------
        footprint
            Polygon of the building's footprint. 
        
        Returns
        -------
        shape_length
            Gets the perimeter of the building's polygon.
        """
        
        poly_coords = list(footprint.exterior.coords)
        shape_length = 0
        for i in range(0,len(poly_coords) - 1):
            current_vertex = poly_coords[i]
            next_vertex = poly_coords[i+1]
            
            dst = np.sqrt((next_vertex[0] - current_vertex[0])**2 + (next_vertex[1] - current_vertex[1])**2)
            shape_length += dst
        
        
        return shape_length

    @staticmethod
    def complexity(
        footprint:Polygon, 
        crs:int = 4326
    )-> float:
        """
        Calculate the footprint's complexity defined as perimeter/area.
        
        Parameters
        ----------
        footprint
            Shapely polygon of the building's footprint.
        crs
            Optional crs. Default is 4326 (longitude, latitude). 
        
        Returns
        ------
        complexity
            Complexity of the building's footprint.
        """
        
        area = FootprintFeaturesCalculator.calculate_unprojected_area(footprint, original_crs = crs)
        length = FootprintFeaturesCalculator.shape_length(footprint)
        
        complexity = length/area
        return complexity


    @staticmethod
    def inverse_average_segment_length(
        footprint:Polygon
    )->float:
        
        """
        Calculates the inverse average segment length. 
        
        Parameters
        ----------
        footprint
            Shapely polygon of the building's footprint. 
        
        
        Returns
        -------
        iasl 
            Inverse average segment length. 
        """
        
        #Let's get the segment lengths
        coordinates = list(footprint.exterior.coords)
        segment_lengths = [np.sqrt((coordinates[i+1][0] - coordinates[i][0])**2 +(coordinates[i+1][1] - coordinates[i][1])**2) for i in range(0,len(coordinates) - 1)]
        avg = 0
        n_segments = len(segment_lengths)
        for i in range(0,n_segments):
            avg += (segment_lengths[i]/n_segments)
        
        if(avg != 0):
            return 1/avg
        else:
            return 0
    
    
    @staticmethod
    def vertices_per_area(
        footprint:Polygon,
        crs:int = 4326
    )-> float:
        """
        Calculates vertices per area.
        
        Parameters
        ----------
        footprint
            Shapely polygon of the building's footprint. 
        
        
        Returns
        -------
        vpa
            Number of vertices per unprojected area. 
        

        """

        n_vertices = FootprintFeaturesCalculator.n_vertices(footprint)
        area = FootprintFeaturesCalculator.calculate_unprojected_area(footprint, original_crs = crs)
        
        vpa = n_vertices/area
        return vpa
    
    @staticmethod
    def average_complexity_per_segment(
        footprint:Polygon,
        crs:int = 4326
    )->float:

        """
        Obtains the average complexity per segment, which is defined as (1/(A*n_vertices))*sum(Li) where Li is the
        length of the i-th segment. 
        
        Parameters
        ----------
        footprint
            Shapely polygon of the building's footprint.
        
        Returns
        -------
        acps
            Average complexity per segment.
        """
        
        coordinates = list(footprint.exterior.coords)
        segment_lengths = [np.sqrt((coordinates[i+1][0] - coordinates[i][0])**2 + (coordinates[i+1][1] - coordinates[i][1])**2) for i in range(0,len(coordinates) - 1)]
        area = FootprintFeaturesCalculator.calculate_unprojected_area(footprint, original_crs = crs)
        acps = 0
        n_segments = len(segment_lengths)
        
        for i in range(0,n_segments):
            acps = acps + (1/(area*n_segments))*segment_lengths[i]
        
        
        return acps
    
    @staticmethod
    def isoperimetric_quotient(
        footprint:Polygon, 
        crs:int = 4326
        )->float:
        
        """
        Obtain the isoperimetric quotient of a polygon's footprint. This quotient
        is defined as 4*pi*Area/(Perimeter^2).
        
        Parameters
        ----------
        footprint
            Shapely polygon of the building's footprint.
        
        Returns
        -------
        iq
            Isoperimetric quotient of the polygon footprint.
        """
        #Get the area
        area = FootprintFeaturesCalculator.calculate_unprojected_area(footprint, original_crs = crs)
        #Get the perimeter
        perimeter = FootprintFeaturesCalculator.shape_length(footprint)
        
        iq = (4*np.pi*area)/(perimeter**2)
        
        return iq
    
    @staticmethod
    def get_footprints_within_radius(
        footprints:List[Polygon], 
        center_footprint:Polygon, 
        radius:float = 600 #Obtain footprints within 600 meters.
    )-> List[Polygon]:
        """
        Filters a list of footprints to obtain those within a given radius (in meters).
        The function assumes that the footprints are in a EPSG:4326. 
        
        Parameters
        ----------
        footprints
            List of shapely polygons to filter.
        center_footprint
            Footprint at the center of the search area. 
        radius
            Radius in meters to search for nearby footprints. Default is 600 meters.
            
        
        Returns
        -------
        projected_centered_footprint
            Center footprint projected to a mercator projection.
        buffered_footprints
            List of footprints intersecting the search area.
        """
        
        #Convert to a mercator projection.
        #We will use geopandas for this. 
        
        gdf = gpd.GeoDataFrame(geometry = footprints, crs = "EPSG:4326")
        gdf = gdf.to_crs(epsg = 3857)
        
        projected_footprints = gdf["geometry"].tolist()
        #Creat the tree
        tree = STRtree(projected_footprints)
        #Project the center footprint
        gdf_center = gpd.GeoDataFrame(geometry = [center_footprint], crs = "EPSG:4326")
        gdf_center = gdf_center.to_crs(epsg = 3857)
        projected_centered_footprint = gdf_center["geometry"].tolist()[0]
        
        buffer = projected_centered_footprint.buffer(radius)
        candidates_idx = tree.query(buffer)  # returns indices
        buffered_footprints = [
            projected_footprints[i] for i in candidates_idx if projected_footprints[i].intersects(buffer)
        ]
        
        #Check if the center footprint is in the list. If it is remove it.
        if(projected_centered_footprint in buffered_footprints):
            buffered_footprints.remove(projected_centered_footprint)

        return projected_centered_footprint, buffered_footprints
        

    @staticmethod
    def neighbor_count(
        footprints:List[Polygon], 
        center_footprint:Polygon,
        radius:float = 600
    )->int:
        """
        Compute the number of neighbors of a footprint within a given radius.
        
        Parameters
        ----------
        footprints
            Complete list of footprints to search for neighbors. 
        center_footprint
            Footprint at the center of the search area.
        radius
            Search radius in meters. Default is 600 meters.
            
        Returns
        -------
        n_neighbors
            Number of neighboring footprints found within the search area.
        """

        projected_centered_footprint, filtered_footprints = FootprintFeaturesCalculator.get_footprints_within_radius(
            footprints,
            center_footprint,
            radius
        )
        n_neighbors = len(filtered_footprints)
        return n_neighbors
    
    @staticmethod
    def mean_distance_to_neighbors(
        footprints:List[Polygon], 
        center_polygon:Polygon,
        radius:float = 600
    )->float:
        """
        Compute the mean distance from a footprint to its neighbors. The vicinity is defined by
        a buffer radius in meters. 
        
        Parameters
        ----------
        footprints
            Complete list of footprints.
        center_footprint
            Footprint at the center of the neighborhood area.
        radius
            Radius that defines the neighborhood area (in meters). Default is 600 meters.
        
        Returns
        -------
        mdn
            Mean distance to neighbors.
        """
        
        projected_footprint, neighbors = FootprintFeaturesCalculator.get_footprints_within_radius(
            footprints, 
            center_polygon, 
            radius
        )
        
        centroid_center_polygon = projected_footprint.centroid.coords
        coords_neighbors = [x.centroid.coords for x in neighbors]
        
        mdn = 0
        n_neighbors = len(coords_neighbors)
        for i in range(0, n_neighbors):
            coords = coords_neighbors[i]
            dst = np.sqrt((coords[0] - centroid_center_polygon[0]**2) + (coords[1] - centroid_center_polygon[1])**2)
            
            mdn += (1/n_neighbors)*dst
        
        return mdn
    @staticmethod
    def expected_nearest_neighbor_distance(
        footprints:List[Polygon],
        center_footprint:Polygon,
        radius:float = 600
    )->float:
        n = FootprintFeaturesCalculator.neighbor_count(
            footprints,
            center_footprint,
            radius
        )
        area = np.pi*(radius**2)
        l = n/area
        if(l > 0):
            expected_distance = 1/(2*np.sqrt(l))
        else:
            expected_distance = float("inf")
        
        return expected_distance

    @staticmethod
    def expected_distance_to_neighbors(
        radius:float = 600
    )->float:
        """
        As far as i understand this thing only depends on radius. If you assume a homogeneous distribution
        of buildings the expected distance would be 2/3 R.
        
        Parameters
        ----------
        radius
            Neighborhood radius.
    
        Returns
        -------
        expected_distance
            Expected distance given a uniform distribution of buildings (2/3 R). 
        """
        
        expected_distance = (2/3)*radius
        return expected_distance

    @staticmethod
    def nearest_neighbor_distance(
        footprints:List[Polygon], 
        center_footprint:Polygon,
        radius:float = 600
    )->float:
        """
        Compute the distance from a building's polygon to its nearest neighbor.
        
        Parameters
        ----------
        footprints
            Complete list of footprints.
        center_footprint
            Footprint at the center of the neighborhood. 
        radius
            Radius of the neighborhood.
        
        Returns
        -------
        nnd
            Distance to nearest neighbor (in meters).
        """
        
        projected_polygon, neighbors = FootprintFeaturesCalculator.get_footprints_within_radius(
            footprints,
            center_footprint,
            radius
        )
        
        nnd = float("inf")
        polygon_coords = projected_polygon.exterior.centroid.coords[0]
        for i in range(0,len(neighbors)):
            neighbor = neighbors[i]
            neighbor_coords = neighbor.exterior.centroid.coords[0]
            dst = np.sqrt((neighbor_coords[0] - polygon_coords[0])**2 + (neighbor_coords[1] - polygon_coords[1])**2)
            if(dst > 0 and dst < nnd):
                nnd = dst
        
        return nnd

    @staticmethod
    def n_size_mean(
        footprints:List[Polygon], 
        central_footprint:Polygon, 
        radius:float = 600
    )->float:
        """
        Obtains the mean footprint area in a neighborhood aroud the building. 
        
        Parameters
        ----------
        footprints
            Complete list of footprints.
        central_footprint
            Footprint at the center of the neighborhood.
        radius
            Radius of the neighborhood.
        
        Returns
        -------
        n_size_mean
            Mean area of buildings within the neighborhood.
        """
        
        projected_footprint, neighbors = FootprintFeaturesCalculator.get_footprints_within_radius(
            footprints,
            central_footprint,
            radius
        )
        
        #For this one we are going to average over all buildings
        neighbors.append(projected_footprint)
        n_size_mean = 0
        for i in range(0,len(neighbors)):
            n_size_mean += (1/len(neighbors))*neighbors[i].area
        
        return n_size_mean
    
    @staticmethod
    def n_size_std(
        footprints:List[Polygon], 
        central_footprint:Polygon, 
        radius:float = 600
    )->float:
        """
        Obtains the standard deviation of footprint areas in a neighborhood around the building.
        
        Parameters
        ----------
        footprints
            Complete list of footprints.
        central_footprint
            Footprint at the center of the neighborhood.
        radius
            Radius of the neighborhood.
        
        Returns
        -------
        n_size_std
            Standard deviation of footprint areas within the neighborhood.
        """
        
        projected_footprint, neighbors = FootprintFeaturesCalculator.get_footprints_within_radius(
            footprints,
            central_footprint,
            radius
        )
        
        #Get the standard deviation of the areas.
        neighbors.append(projected_footprint)
        areas = [x.area for x in neighbors]
        n_size_std = np.std(areas)
        
        return n_size_std
    
    @staticmethod
    def n_size_min(
        footprints:List[Polygon], 
        central_footprint:Polygon, 
        radius:float = 600
    )->float:
        """
        Obtains the minimum footprint area in a neighborhood around the building.
        
        Parameters
        ----------
        footprints
            Complete list of footprints.
        central_footprint
            Footprint at the center of the neighborhood.
        radius
            Radius of the neighborhood.
        
        Returns
        -------
        n_size_min
            Minimum footprint area within the neighborhood.
        """
        
        projected_footprint, neighbors = FootprintFeaturesCalculator.get_footprints_within_radius(
            footprints,
            central_footprint,
            radius
        )
        
        #Get the minimum area of the buildings in the neighborhood
        neighbors.append(projected_footprint)
        areas = [x.area for x in neighbors]
        n_size_min = np.min(areas)
        
        return n_size_min
    
    @staticmethod
    def n_size_max(
        footprints:List[Polygon], 
        central_footprint:Polygon, 
        radius:float = 600
    )->float:
        """
        Obtains the maximum footprint area in a neighborhood around the building.
        
        Parameters
        ----------
        footprints
            Complete list of footprints.
        central_footprint
            Footprint at the center of the neighborhood.
        radius
            Radius of the neighborhood.
        
        Returns
        -------
        n_size_max
            Maximum footprint area within the neighborhood.
        """
        
        projected_footprint, neighbors = FootprintFeaturesCalculator.get_footprints_within_radius(
            footprints,
            central_footprint,
            radius
        )
        
        #Get the maximum area of the buildings in the neighborhood
        neighbors.append(projected_footprint)
        areas = [x.area for x in neighbors]
        n_size_max = np.max(areas)
        
        return n_size_max
    
    
    @staticmethod
    def n_size_cv(
        footprints:List[Polygon], 
        central_footprint:Polygon,
        radius:float = 600
    )->float:
        """
        Obtains the coefficient of variation of footprint areas in a neighborhood around the building.
        
        Parameters
        ----------
        footprints
            Complete list of footprints.
        central_footprint
            Footprint at the center of the neighborhood.
        radius
            Radius of the neighborhood.
        
        Returns
        -------
        n_size_cv
            Coefficient of variation of footprint areas within the neighborhood.
        """
        
        projected_footprint, neighbors = FootprintFeaturesCalculator.get_footprints_within_radius(
            footprints,
            central_footprint,
            radius
        )
        
        #Get the coefficient of variation of the areas.
        neighbors.append(projected_footprint)
        areas = [x.area for x in neighbors]
        n_size_cv = np.std(areas)/np.mean(areas)
        
        return n_size_cv
    @staticmethod
    def nni(
        footprints:List[Polygon], 
        center_building:Polygon,
        radius:float = 600
    )->float:
        """
        Compute the nearest neighbor index (NNI) for a building's footprint. The NNI is defined as the ratio
        of the observed nearest neighbor distance to the expected nearest neighbor distance in a random distribution.
        
        Parameters
        ----------
        footprints
            Complete list of footprints.
        center_building
            Footprint at the center of the neighborhood.
        radius
            Radius of the neighborhood. Default is 600 meters.
        
        Returns
        -------
        nni
            Nearest neighbor index.
        """
        
        ond = FootprintFeaturesCalculator.nearest_neighbor_distance(
            footprints, 
            center_building,
            radius)
        
        end = FootprintFeaturesCalculator.expected_nearest_neighbor_distance(
            footprints, 
            center_building,
            radius
        )
        
        if(end > 0 and end != float("inf")):
            nni = ond/end
        else:
            nni = float("inf")

        return nni


        
        


    
            
        
            
        
    
    
        

        
        
        
        
            
    
        
        
            
    
    
    
            
        
        
                

            
        
        
        
        
        
        
        
        
        
    