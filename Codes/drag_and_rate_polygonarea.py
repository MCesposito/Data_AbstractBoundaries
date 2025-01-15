import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt 
import seaborn as sns 
from scipy.spatial import ConvexHull
import os 
from scipy import stats 

general_path = "/home/data/"    ## behav

    
##### PART 1.
# Define the coordinates of the polygon vertices)
polygon_vertices =np.array( [(1/5.8, 21/33), (2/5.8, 1/33), (5.8/5.8, 13/33), (4.4/5.8, 33/33)])

################ Proportion of area that is being shown #####################

# Function to calculate the area of a polygon using the shoelace formula
def polygon_area(vertices):
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n  # Next vertex
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    area = abs(area) / 2.0
    return area

# Calculate the area of the polygon
area_polygon = polygon_area(polygon_vertices)

# Area of the bounding box (assuming space is [0, 1] in both dimensions)
area_bounding_box = 1.0  # Since it's a square from (0, 0) to (1, 1) --> width * height --> adjust according to what partiicpants were shown!!!

# Calculate the proportion of the area
proportion_area = area_polygon / area_bounding_box

print(f"Area of the polygon: {area_polygon:.4f}")

print(f"Percentage of area occupied by the polygon: {proportion_area*100:.4f}%")

print(f"Probability of all points falling inside: {proportion_area**16}")

################################################################################################
################################ part 2 - original placement data at per subject level 

people = [1,2, 3, 6, 4, 7, 8, 9, 10,11, 12, 13,14,15, 17, 18, 19, 20, 21, 23, 24, 26, 27, 28, 30, 31, 32, 33, 34]
flipped = [17, 18, 19, 20, 21, 23, 24, 26, 27, 28, 30, 33, 34]
unflipped = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 31, 32]



def getting_placement(participants_set, cond): 
    all_parts_placement = pd.DataFrame()
    all_parts_borders = pd.DataFrame()

    if cond == "flipped":
        for j in participants_set: 
            placement = pd.read_csv(f"/home/mariachiara/Escritorio/fMRI/behavior/Meadows/p{j}.csv",sep=",")
            placement["participant"] = j
            all_parts_placement= pd.concat([all_parts_placement, placement])
            
            placement["stimulus"].iloc[[2, 7, 19, 6]] = ["calabacines", "fresas", "manzana", "piña"]
            borders = placement.iloc[[2, 7, 19, 6]].reset_index()
            
            all_parts_borders = pd.concat([all_parts_borders, borders])   #flipped
            
    else: 
        for j in participants_set:
            placement = pd.read_csv(f"/home/mariachiara/Escritorio/fMRI/behavior/Meadows/p{j}.csv",sep=",")
            placement["participant"] = j
            all_parts_placement= pd.concat([all_parts_placement, placement])
            
            placement["stimulus"].iloc[[4, 12, 3, 2]] = ["manzana", "calabacines", "fresas", "piña"]
            borders = placement.iloc[[4, 12, 3, 2]].reset_index()
            all_parts_borders = pd.concat([all_parts_borders, borders])    
            
    return all_parts_placement, all_parts_borders


placement_fl, borders_fl = getting_placement(flipped, "flipped")
placement_og, borders_og = getting_placement(unflipped, "unflipped")

#### excluding boundaries...
exclude_stimulus = ["3_fresas","manzana-removebg-preview", "1_calabacines (1) (1)", "tomate (1) (1)","Captura_de_pantalla_2023-03-14_112329-removebg-preview", "1_calabacines (1)"]

all_placement = pd.concat([placement_og, placement_fl])
products_placement = all_placement.loc[~all_placement["stimulus"].isin(exclude_stimulus)]

######  ################################################################################################
################################ part 2 + 3 - original placement data at per subject level + how many points inside per participant !!!  
flipped_polygon = polygon_vertices[:, [1, 0]]  

def point_inside_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def check_point(row):
    polygon = flipped_polygon if row["participant"] in flipped else polygon_vertices
    return point_inside_polygon([row["confidence"], row["property"]], polygon)

products_placement["inside_polygon"] = products_placement.apply(check_point, axis=1)

inside_count = products_placement['inside_polygon'].sum()  # Número de valores True
total_count = len(products_placement)

dict_inside = {j: 0 for j in people}
for j in people:
    df_single = products_placement.loc[products_placement["participant"] == j]
    j_inside_count = df_single['inside_polygon'].sum()
    dict_inside[j] = j_inside_count

####################

'''
sns.set_style("whitegrid")

polygon_hull = ConvexHull(polygon_vertices)
fig, axes = plt.subplots(5, 6, figsize=(13, 13), constrained_layout=True)

for idx, j in enumerate(people):
    # stuff to put them in place 
    row = idx // 6
    col = idx % 6
    ax = axes[row, col]
    single_pl= products_placement.loc[products_placement["participant"] == j]
    
    ax=sns.scatterplot(data=single_pl, x  = "confidence", y = "property", ax=ax)
    ax.set_title(f"subj{j} \n {dict_inside[j]} points inside", fontweight="bold")
    
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    
    ax.set_xticks(np.arange(0, 1.1, 0.5))
    ax.set_yticks(np.arange(0, 1.1, 0.5))
    
    if j in unflipped: 
        ax.set_xlabel("Price")    
        ax.set_ylabel("Freshness")
        for simplex in polygon_hull.simplices:
            ax.plot(polygon_vertices[simplex, 0], polygon_vertices[simplex, 1], 'k-', alpha=0.5)
            ax.fill(polygon_vertices[polygon_hull.vertices, 0], 
                polygon_vertices[polygon_hull.vertices, 1], 
                'k', alpha=0.05) 
    elif j in flipped:
        ax.set_xlabel("Freshness")
        ax.set_ylabel("Price")
        for simplex in polygon_hull.simplices:
            ax.plot(polygon_vertices[simplex, 1], polygon_vertices[simplex, 0], 'k-',alpha=0.5)
            ax.fill(polygon_vertices[polygon_hull.vertices, 1], 
                polygon_vertices[polygon_hull.vertices, 0], 
                'k', alpha=0.05) 
    plt.subplots_adjust(wspace=0, hspace=0)


for empty_idx in range(len(people), 30):
    row = empty_idx // 6
    col = empty_idx % 6
    axes[row, col].axis("off")
    



products_placement["number"] = products_placement.apply(lambda row: 1 if row["inside_polygon"] == True else 0, axis=1)
'''
######################################################
print("###")
print(f"Mean number of points inside the polygon: {np.mean(products_placement['number']):.3f}")
print(f"Variance of points inside the polygon: {np.var(products_placement['number']):.3f}")



