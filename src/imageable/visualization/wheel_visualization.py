
from typing import List
import numpy as np
from shapely import Polygon
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from PIL import Image


class WheelElement:

    def __init__(self,
                 data: float|np.ndarray|str|Polygon,
                 group_name:str,
                 feature_name:str
                 ):

        self.data = data
        self.group_name = group_name
        self.feature_name = feature_name


    def assign_matrix_indices(self, group_index:int,
                              feature_index:int) -> None:

        self.group_index =group_index
        self.feature_index = feature_index

    def assign_position(self, x:float, y:float)->None:
        self.x = x
        self.y = y
    def assign_angular_position(self, radius:float, angle:float)->None:
        self.radius = radius
        self.angle = angle

    def assign_size(self, size:float)->None:
        self.size = size

    def draw(self,
             ax:plt.Axes,
             total_radius:float,
             delta_angle:float,
             delta_r:float,
             color:str = "#000000",
             label:str = None,
             gamma:float = 0.8)->None:
        # numeric -> circular segment (thick arc)
        if isinstance(self.data, (float, int)):
            radius = self.radius - delta_r/2
            initial_theta = self.angle - delta_angle/2
            final_theta = self.angle + delta_angle/2
            arc_width = delta_r*gamma

            arc = Arc(
                (0, 0),
                width=2*radius,
                height=2*radius,
                theta1=initial_theta*180/np.pi,
                theta2=final_theta*180/np.pi,
                color=color,
                lw=arc_width
            )
            ax.add_patch(arc)

            if label is not None:
                text_radius = self.radius
                text_x = text_radius*np.cos(self.angle)
                text_y = text_radius*np.sin(self.angle)
                angle_deg = (np.degrees(self.angle) + 90) % 360
                if 90 < angle_deg < 270:
                    angle_deg += 180
                ax.text(
                    text_x,
                    text_y,
                    label,
                    ha="center",
                    va="center",
                    rotation=angle_deg,
                    rotation_mode="anchor"
                )

        # image -> show at element position, square with side = 2*size
        elif isinstance(self.data, np.ndarray):
            #Resize image to fit in square of side size

            image = Image.fromarray(self.data)
            size = self.size/3
            image = image.resize((int(2*size), int(2*size)))
            ax.imshow(
                self.data,
                extent=[
                    self.x - size, self.x + size,
                    self.y - size, self.y + size
                ],
                aspect="equal"
            )

        # text -> write at angular position, oriented
        elif isinstance(self.data, str):
            text_radius = self.radius
            text_x = text_radius*np.cos(self.angle)
            text_y = text_radius*np.sin(self.angle)
            angle_deg = (np.degrees(self.angle) + 90) % 360
            if 90 < angle_deg < 270:
                angle_deg += 180
            ax.text(
                text_x,
                text_y,
                self.data,
                ha="center",
                va="center",
                rotation=angle_deg,
                rotation_mode="anchor"
            )

        # polygon -> just fill it
        elif isinstance(self.data, Polygon):
            #Create a                                                                                                                                                                                                                                                                                                                                                                                       
            xs, ys = self.data.exterior.xy
            centroid_x = 0
            centroid_y = 0
            for x, y in zip(xs, ys):
                centroid_x += x/len(xs)
                centroid_y += y/len(ys)
            

            #We will ge tthe vectors from each point to the centroid and scale them
            scaled_xs = [x - centroid_x for x in xs]
            scaled_ys = [y - centroid_y for y in ys]
            norms = [np.sqrt(sx**2 + sy**2) for sx, sy in zip(scaled_xs, scaled_ys)]
            max_norm = np.max(norms)
            scaled_xs = [sx / max_norm * self.size/2 for sx in scaled_xs]
            scaled_ys = [sy / max_norm * self.size/2 for sy in scaled_ys]

            new_xs = [self.x + sx for sx in scaled_xs]
            new_ys = [self.y + sy for sy in scaled_ys]
            ax.fill(new_xs, new_ys, color=color, alpha=0.7)



class WheelVisualization:
    def __init__(self,
                 elements:List[WheelElement],
                 radius_size:float = 10.0,
                 reduction_factor:float = 0.8
                 ) -> None:

        self.elements = elements
        #Let's get the number of groups (clusters) and features
        #groups correspond to angular divisions, features to radial divisions
        def unique_preserve_order(seq):
            seen = set()
            result = []
            for x in seq:
                if x not in seen:
                    seen.add(x)
                    result.append(x)
            return result

        self.group_names = unique_preserve_order([e.group_name for e in elements])
        self.feature_names = unique_preserve_order([e.feature_name for e in elements])
        self.n_groups = len(self.group_names)
        self.n_features = len(self.feature_names)
        self.radius_size = radius_size
        self.reduction_factor = reduction_factor

        self._assign_matrix_indices()
        self._assign_spatial_coordinates()
        self._assign_sizes(reduction_factor = reduction_factor)


    def _assign_matrix_indices(self)->None:
        for e in self.elements:
            group_index = self.group_names.index(e.group_name)
            feature_index = self.feature_names.index(e.feature_name)
            e.assign_matrix_indices(group_index, feature_index)

    def _assign_spatial_coordinates(self)->None:
        delta_theta = 2*np.pi/self.n_groups
        delta_r = self.radius_size/self.n_features
        self.initial_and_final_radii = []
        for e in self.elements:
            theta = delta_theta*(e.group_index + 1 + 1/2)
            radius = delta_r*(e.feature_index + 1 + 1/2)
            x = radius*np.cos(theta)
            y = radius*np.sin(theta)
            e.assign_position(x, y)
            e.assign_angular_position(radius, theta)
            initial_radius = delta_r*e.feature_index
            final_radius = delta_r*(e.feature_index + 1)
            self.initial_and_final_radii.append((initial_radius, final_radius))


    def _assign_sizes(self, reduction_factor:float = 0.8)-> None:
        index = 0
        delta_angle = 2*np.pi/self.n_groups
        for e in self.elements:
            initial_radius, final_radius = self.initial_and_final_radii[index]
            index += 1
            max_area =(1/2)*(final_radius**2 - initial_radius**2)*delta_angle
            size = np.sqrt(max_area/np.pi)*reduction_factor
            e.assign_size(size)

    def draw_wheel(self,
                   cmap:plt.Colormap = plt.cm.viridis,
                   contour_color:str|list[str] = "red",
                   figure_width:float = 8.0,
                   figure_height:float = 8.0,
                   circle_linewidth:float = 0.5,
                   label_features:bool = True)->None:        
        #First let's draw all countours
        center = (0, 0)
        radius_delta = self.radius_size/self.n_features
        radius_list = [i*radius_delta for i in range(self.n_features + 2)]
        fig, ax = plt.subplots(figsize=(figure_width, figure_height))

        # Determine whether contour_color is a single color or a list
        if isinstance(contour_color, (list, tuple)):
            # Ensure the list has enough colors
            if len(contour_color) < len(radius_list):
                raise ValueError("Not enough colors provided for each contour circle.")
            circle_colors = contour_color
        else:
            # Single color case
            circle_colors = [contour_color] * len(radius_list)

        # Draw circles with correct colors
        for r, c in zip(radius_list, circle_colors):
            circle = plt.Circle(
                center,
                r,
                color=c,
                fill=False,
                lw=circle_linewidth
            )
            ax.add_patch(circle)

        
        #We also draw radial lines
        # We also draw radial lines
        delta_theta = 2*np.pi/self.n_groups
        angle_list = [i*delta_theta for i in range(self.n_groups)]

        # Choose a single color for the radial lines
        if isinstance(contour_color, (list, tuple)):
            line_color = contour_color[0]
        else:
            line_color = contour_color

        for angle in angle_list:
            x_end = (self.radius_size + radius_delta) * np.cos(angle)
            y_end = (self.radius_size + radius_delta) * np.sin(angle)
            ax.plot(
                [0, x_end],
                [0, y_end],
                color=line_color,
                lw=circle_linewidth,
                linestyle="--"
            )


        min_feature_values = np.zeros(len(self.feature_names))
        max_feature_values = np.zeros(len(self.feature_names))
        counted_feature_instances = np.zeros(len(self.feature_names), dtype=int)
        for i, feature_name in enumerate(self.feature_names):
            feature_values = [e.data for e in self.elements if e.feature_name == feature_name and isinstance(e.data, (float, int))]
            if len(feature_values) > 0:
                min_feature_values[i] = np.min(feature_values)
                max_feature_values[i] = np.max(feature_values)
            else:
                min_feature_values[i] = 0
                max_feature_values[i] = 1  # Avoid division by zero later

        #Finally we draw each element
        for e in self.elements:

            if(isinstance(e.data, (float, int))):
                label = None
                #We will only label the first instance of each feature
                if label_features:
                    feature_index = self.feature_names.index(e.feature_name)
                    if counted_feature_instances[feature_index] == 0:
                        label = e.feature_name
                        counted_feature_instances[feature_index] += 1
                
                #Get color based on normalized value
                feature_index = self.feature_names.index(e.feature_name)
                if max_feature_values[feature_index] - min_feature_values[feature_index] > 0:
                    normalized_value = (e.data - min_feature_values[feature_index]) / (max_feature_values[feature_index] - min_feature_values[feature_index])
                else:
                    normalized_value = 0.5  # If all values are the same, use middle of colormap
                color = cmap(normalized_value)

                e.draw(
                    ax,
                    total_radius = self.radius_size,
                    delta_angle = delta_theta,
                    delta_r = radius_delta,
                    color = color,
                    label = label
                )
            elif(isinstance(e.data, np.ndarray)):
                e.draw(
                    ax,
                    total_radius = self.radius_size,
                    delta_angle = delta_theta,
                    delta_r = radius_delta,
                    color = "black"
                )
            
            elif(isinstance(e.data, str)):
                e.draw(
                    ax,
                    total_radius = self.radius_size,
                    delta_angle = delta_theta,
                    delta_r = radius_delta,
                    color = "black"
                )
            elif(isinstance(e.data, Polygon)):
                e.draw(
                    ax,
                    total_radius = self.radius_size,
                    delta_angle = delta_theta,
                    delta_r = radius_delta,
                    color = "blue"
                )
        ax.set_axis_off()
        ax.set_aspect('equal')
        ax.set_xlim(-self.radius_size - radius_delta, self.radius_size + radius_delta)
        ax.set_ylim(-self.radius_size - radius_delta, self.radius_size + radius_delta)



                



















