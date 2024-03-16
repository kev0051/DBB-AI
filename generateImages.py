import bpy
from pascal_voc_writer import Writer
import random
from datetime import datetime
import math
import mathutils
import os


# Choose one of rendering option
#1. Render all
#2. Render batch
#3. Render individual parts
#4. Render backgrounds

# Run once with run_type = 1. This will generate around 400 images.
# Afterwards run script multiple times with run_type = 2. This will generate 500 extra images. 
#Suggested 4-5 runs on run_type 2 to generate ~2,5k images for dataset with 6 lego parts.
# First run will set batch 100 and next runs can be 500. 
# For me Blender crashes if more than 600 renderings generated in one script run.

run_type = 2
guarantee = "99773 (6009019)"

num = 255 
colors = []
materials = []
background = []
objects = []
object_count = []
start_range = 0 
step_size = 15 # Default 15. Choose step size for part rotation when rendering indivdual parts. 
batch_size = 100 # How many images to render. For me, more than 400 images @ 640x640 resolution crashes Blender.
i = 0

# Define colors, based on offical Lego RGB colors: 
# http://www.jennyscrayoncollection.com/2021/06/all-current-lego-colors.html
colors0=[(21, 21, 21), # Black [0] in materials arr
        (244, 244, 244), # White [1] in materials arr
        (221,196,142), # Tan [2] in materials arr
        (160,161,159), # Med Stone Grey [3] in materials arr
        (221, 26, 33), # Bright Red [4] in materials arr
        (255,205,3), # Bright Yellow [5] in materials arr
        (0, 108, 183), # Bright Blue [6] in materials arr
        (0, 146, 71), # Dark Green [7] in materials arr
        (100, 103, 101)] # Dark Stone Grey [8] in materials arr

# Dict maps from name to pos in materials array
colorsDict={"10928 (6012451)": 0,
            "13670 (6031821)": 2,
            "18575 (6346535)": 0,
            "2780 (4121715)": 0,
            "2815 (6028041)": 0,
            "29219 (6173127)": 3,
            "32009 (4495412)": 3,
            "32009 (6271161)": 1,
            "32013 (6284699)": 0,
            "32014 (6268905)": 0,
            "32034 (6271869)": 0,
            "32054 (4140806)": 4,
            "32062 (4142865)": 4,
            "32072 (6284188)": 0,
            "32073 (4211639)": 3,
            "32123 (4239601)": 5,
            "32140 (4141270)": 4,
            "32184 (4121667)": 0,
            "32270 (4177431)": 0,
            "32271 (6276836)": 3,
            "32278 (4542578)": 1,
            "32291 (6280394)": 0,
            "32316 (4211651)": 3,
            "32348 (4509912)": 3,
            "32348 (6278132)": 1,
            "32449 (4142236)": 0,
            "32498 (4177434)": 0,
            "32523 (4142822)": 0,
            "32523 (4153707)": 5,
            "32523 (4153718)": 4,
            "32523 (4509376)": 6,
            "32523 (6007973)": 7,
            "32524 (4495930)": 3,
            "32525 (4611705)": 3,
            "32526 (4211713)": 3,
            "32526 (4585040)": 1,
            "32556 (4514554)": 2,
            "32905 (6185471)": 3,
            "33299 (6313520)": 0,
            "3649 (6195314)": 3,
            "3673 (4211807)": 3,
            "3705 (370526)": 0,
            "3706 (370626)": 0,
            "3707 (370726)": 0,
            "3708 (370826)": 0,
            "3737 (373726)": 0,
            "40490 (4211866)": 3,
            "41239 (4522934)": 3,
            "41669 (6288218)": 1,
            "41678 (4162857)": 0,
            "4185 (4587275)": 3,
            "41897 (6035364)": 0,
            "42003 (6273715)": 8,
            "44294 (4211805)": 3,
            "44809 (6008527)": 4,
            "4519 (4211815)": 3,
            "45590 (4198367)": 0,
            "4716 (4211510)": 3,
            "48989 (4225033)": 3,
            "55013 (4499858)": 8,
            "56908 (4634091)": 3,
            "57518 (6325504)": 0,
            "57519 (4582792)": 0,
            "57585 (4502595)": 3,
            "59443 (4513174)": 4,
            "60483 (6265091)": 0,
            "60484 (4552347)": 0,
            "60485 (4535768)": 3,
            "63869 (6331441)": 3,
            "64178 (4540797)": 3,
            "64179 (4539880)": 3,
            "64392 (4541326)": 0,
            "64682 (4543490)": 0,
            "6536 (6261375)": 3,
            "6558 (4514553)": 6,
            "6562 (4666579)": 2,
            "6589 (4565452)": 2,
            "6590 (6275844)": 3,
            "6629 (6279881)": 0,
            "87080 (4566251)": 0,
            "87083 (6083620)": 8,
            "87086 (4566249)": 0,
            "94925 (4640536)": 3,
            "99009 (4652235)": 3,
            "99010 (6326620)": 0,
            "99773 (6009019)": 3}

# Set output resolution
r_settings = bpy.context.scene.render
r_settings.resolution_x = 300
r_settings.resolution_y = 300


print("######################")
print("Rendering lego parts. ", datetime.now())
print("Run_type: ", run_type)
print()


output_dir = '/Users/kyvang/Documents/CapstoneNEW/lego_renders'
output_dir_images = output_dir + '/images'
output_dir_annotations = output_dir + '/annotations'

print("Rendering output directory: ", output_dir)


if run_type == 2:
    batch_size = 5

        
# FUNCTIONS:

# Update camera location:
def update_camera(camera, focus_point=mathutils.Vector((0.0, 0.0, 0.0)), distance=10.0):
    """
    Focus the camera to a focus point and place the camera at a specific distance from that
    focus point. The camera stays in a direct line with the focus point.

    :param camera: the camera object
    :type camera: bpy.types.object
    :param focus_point: the point to focus on (default=``mathutils.Vector((0.0, 0.0, 0.0))``)
    :type focus_point: mathutils.Vector
    :param distance: the distance to keep to the focus point (default=``10.0``)
    :type distance: float
    """
    print("Moving camera")
    
    looking_direction = camera.location - focus_point
    rot_quat = looking_direction.to_track_quat('Z', 'Y')

    camera.rotation_euler = rot_quat.to_euler()
    # Use * instead of @ for Blender <2.8
    camera.location = rot_quat @ mathutils.Vector((0.0, 0.0, distance))


# Functions to get annotaion bounding boxes 
def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))

def camera_view_bounds_2d(scene, cam_ob, me_ob):
    """
    Returns camera space bounding box of mesh object.

    Negative 'z' value means the point is behind the camera.

    Takes shift-x/y, lens angle and sensor size into account
    as well as perspective/ortho projections.

    :arg scene: Scene to use for frame size.
    :type scene: :class:`bpy.types.Scene`
    :arg obj: Camera object.
    :type obj: :class:`bpy.types.Object`
    :arg me: Untransformed Mesh.
    :type me: :class:`bpy.types.Mesh´
    :return: a Box object (call its to_tuple() method to get x, y, width and height)
    :rtype: :class:`Box`
    """

    mat = cam_ob.matrix_world.normalized().inverted()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh_eval = me_ob.evaluated_get(depsgraph)
    me = mesh_eval.to_mesh()
    me.transform(me_ob.matrix_world)
    me.transform(mat)

    camera = cam_ob.data
    frame = [-v for v in camera.view_frame(scene=scene)[:3]]
    camera_persp = camera.type != 'ORTHO'

    lx = []
    ly = []

    for v in me.vertices:
        co_local = v.co
        z = -co_local.z

        if camera_persp:
            if z == 0.0:
                lx.append(0.5)
                ly.append(0.5)
            # Does it make any sense to drop these?
            # if z <= 0.0:
            #    continue
            else:
                frame = [(v / (v.z / z)) for v in frame]

        min_x, max_x = frame[1].x, frame[2].x
        min_y, max_y = frame[0].y, frame[1].y

        x = (co_local.x - min_x) / (max_x - min_x)
        y = (co_local.y - min_y) / (max_y - min_y)

        lx.append(x)
        ly.append(y)

    min_x = clamp(min(lx), 0.0, 1.0)
    max_x = clamp(max(lx), 0.0, 1.0)
    min_y = clamp(min(ly), 0.0, 1.0)
    max_y = clamp(max(ly), 0.0, 1.0)

    mesh_eval.to_mesh_clear()

    r = scene.render
    fac = r.resolution_percentage * 0.01
    dim_x = r.resolution_x * fac
    dim_y = r.resolution_y * fac

    # Sanity check
    if round((max_x - min_x) * dim_x) == 0 or round((max_y - min_y) * dim_y) == 0:
        return (0, 0, 0, 0)

    return (
        round(min_x * dim_x),            # X
        round(dim_y - max_y * dim_y),    # Y
        round((max_x - min_x) * dim_x),  # Width
        round((max_y - min_y) * dim_y)   # Height
    )


# Render scene in JPEG format
def render_scene(it):
    bpy.context.scene.render.image_settings.file_format='JPEG'
    bpy.context.scene.render.filepath = os.path.join(output_dir_images, f"{guarantee}_%0.5d.jpg" % it)
    bpy.ops.render.render(use_viewport = True, write_still=True)

# Export annotations of boundig boxes in VOC format
def save_annotations(object, it):
    # Include guarantee at the start of the file name for image and XML annotation
    image_file_name = f"{guarantee}_%0.5d.jpg" % it
    writer = Writer(os.path.join(output_dir_images, image_file_name), r_settings.resolution_x, r_settings.resolution_y)
    if object is not None:
        bound_x, bound_y, bound_w, bound_h = camera_view_bounds_2d(bpy.context.scene, bpy.context.scene.camera, object)
        part_name = str(object.name).split(".", 1)
        writer.addObject(part_name[0], bound_x, bound_y, bound_x+bound_w, bound_y+bound_h)
    writer.save(os.path.join(output_dir_annotations, f"{guarantee}_%0.5d.xml" % it))

# PROGRAM CODE:

# Normalise color values. Blender requires colors to be define between 0-1. 
for c in colors0:
    c = [x/num for x in c]
    c.append(1) # add 4th element alpha = 1, in case PNG format is required
    colors.append(c)
    
# Create list of materials.
for i in range(len(colors)):
    color_name = "color_"+str(i)
    mat1 = bpy.data.materials.new(color_name)    
    mat1.diffuse_color = colors[i]
    r1 = random.randint(0,1)
    if r1 > 0:
        mat1.shadow_method = ("NONE")
    else:
        mat1.shadow_method = ("OPAQUE")
    materials.append(mat1)


# Create list of backgrounds
# Backgrounds are located in 'Collection 2' object container
for obj_bg in bpy.data.collections['Collection 2'].all_objects:
    if (obj_bg.type == 'MESH'):
        background.append(obj_bg)


# Create list of lego part objects 
for obj in bpy.data.collections['Collection'].all_objects:
    if (obj.type == 'MESH'):
        objects.append(obj)
        

# Create a table to count every object appearance in rendering batches
# Used to check if random selection of parts has a uniform distribution
# Prints the table after script is finished
for x in objects:
    object_count.append([x.name]) # Part name
 
for c in range(0,len(objects)):
    object_count[c].append(0) # Number of times part has been used


# Reset camera location and orientation towards an object
objects[0].select_set(True)
objects[0].location = (0,0,0.5)
objects[0].rotation_euler = (0, 0, 0)
bpy.data.objects['Camera'].location = (0,0,5)
bpy.data.objects['Camera'].rotation_euler = (0,0,0)
bpy.ops.view3d.camera_to_view_selected()
# If resolution not 300x300, adjust y (-1.0) to suit, otherwise object part might be out of the picture.
update_camera(bpy.data.objects['Camera'],focus_point=mathutils.Vector((0.0, -1.0, 0.5)), distance=4)


# Hide all objects and backgrounds
for x in objects:
    x.hide_set(True)
    x.hide_render = True

for bg in background:
    bg.hide_set(True)
    bg.hide_render = True


# GENERATING INDIVIDUAL OBJECT RENDERS

# Render backgrounds
if (run_type == 1 or run_type == 4): 
    # Render few images without parts, only background 
    # This is not necessary (but recommended) since most object detection networks can handle empty scenes 
    for rr in range(0,3): 
        for bg in background:
            bg.hide_set(False) # Unhide one background
            bg.hide_render = False # Make it visible in renderings

            render_scene(start_range)
            save_annotations(None, start_range) # No object to annotate so passing None
            
            start_range += 1
            
            bg.hide_set(True)
            bg.hide_render = True
            

# Render individual parts rotated around all axes
if (run_type == 1 or run_type == 3):
    # Set camera to default location for rendering individual parts
    bpy.data.objects['Camera'].location = (0,0,5)
    bpy.data.objects['Camera'].rotation_euler = (0,0,0)
    update_camera(bpy.data.objects['Camera'],focus_point=mathutils.Vector((0.0, -1.0, 0)), distance=2.5)

    for x in objects: 
        # Check if the part is not a dublicate so not to render twise same parts. Duplicates have suffix ".001"
        if '.' not in x.name: 
            x.hide_set(False)
            x.hide_render = False
            x.select_set(True)    
            
            # Randomise part location a bit
            x.location = (0,0,0.5)
            x.rotation_euler = (0, 0, 0)
            
            # rotate around x
            for x_or in range(0,360,step_size):
                
                # Adjust part rotation
                x.rotation_euler = (math.radians(x_or), 0, 0)
                    
                # Choose material randomly/from dict
                if x.name in colorsDict:
                    x.active_material = materials[colorsDict[x.name]]
                else:
                    #r1 = random.randint(0, len(materials)-1) 
                    x.active_material = materials[1]
                
                # Hide all backgrounds    
                for bg in background:
                    bg.hide_set(True)
                    bg.hide_render = True
                    
                # unhide one background randomly
                r3 = random.randint(0,len(background)-1)
                background[r3].hide_set(False)
                background[r3].hide_render = False     
                   
                render_scene(start_range)
                save_annotations(x, start_range)
            
                # Reset orientation & randommise location of the object
                x.location = (round(random.uniform(-0.15, 0.15), 4), round(random.uniform(-0.15, 0.15), 4), 0.5)
                x.rotation_euler = (0, 0, 0)

                # increase counter
                start_range += 1   
            

            x.location = (0,0,0.5)
            x.rotation_euler = (0, 0, 0)
            
            
            # rotate around y
            for y_or in range(0,360,step_size):
                
                x.rotation_euler = (0, math.radians(y_or), 0)
                
                r1 = random.randint(0, len(materials)-1) 
                x.active_material = materials[r1]
                
                for bg in background:
                    bg.hide_set(True)
                    bg.hide_render = True
                    
                r3 = random.randint(0,len(background)-1)
                background[r3].hide_set(False)
                background[r3].hide_render = False     
                    
                render_scene(start_range)
                save_annotations(x, start_range)
                
                x.location = (round(random.uniform(-0.15, 0.15), 4), round(random.uniform(-0.15, 0.15), 4), 0.5)
                x.rotation_euler = (0, 0, 0)

                start_range += 1   

            
            x.location = (0,0,0.5)
            x.rotation_euler = (0, 0, 0)
            
            
            # rotate around z
            for z_or in range(0,360,step_size):
                
                x.rotation_euler = (0, 0, math.radians(z_or))
         
                r1 = random.randint(0, len(materials)-1) 
                x.active_material = materials[r1]
                
                for bg in background:
                    bg.hide_set(True)
                    bg.hide_render = True
                    
                r3 = random.randint(0,len(background)-1)
                background[r3].hide_set(False)
                background[r3].hide_render = False     

                render_scene(start_range)
                save_annotations(x, start_range)
                
                x.location = (round(random.uniform(-0.15, 0.15), 4), round(random.uniform(-0.15, 0.15), 4), 0.5)
                x.rotation_euler = (0, 0, 0)

                start_range += 1   
            

            # Hide the part after finished rendering it's set
            x.hide_set(True)
            x.hide_render = True
    # END OF GNEERATING INDIVIDUAL PARTS

if (run_type == 1 or run_type == 2):

    # CODE FOR BATCH RENDERING
    path, dirs, files = next(os.walk(output_dir+"/images"))
    file_count = len(files)
    start_range = file_count
    # start_range = 0 # Modify start_range to start from previous renders of individual parts
    image_set = start_range + batch_size

    print("Printing batch of images. Start_range from: ", start_range)
    
    # Deselect all objects
    for iii in range(start_range,image_set):
        
        bpy.ops.object.select_all(action='DESELECT')
        
        # Hide all objects
        for x in objects:
            x.hide_set(True)
            x.hide_render = True
            
        # Get a maximum number of 5 random lego part objects 
        specific_part_index = next((i for i, obj in enumerate(objects) if guarantee in obj.name), None)

        # Start with the specific part's index if it exists
        if specific_part_index is not None:
            list_of_objects_numbers = [specific_part_index]
            remaining_selection_count = random.randint(1, 4)  # Adjusted count since one part is already included

            # Sample additional, distinct indices, excluding the specific part's index
            list_of_objects_numbers += random.sample([i for i in range(len(objects)) if i != specific_part_index], remaining_selection_count)
        else:
            # Fallback to original random selection if the specific part was not found
            list_of_objects_numbers = random.sample(range(len(objects)), random.randint(2, 5))      
        print("Lego parts selected: ", list_of_objects_numbers)

        for obj_num in list_of_objects_numbers: 
            r1 = random.randint(0, len(materials)-1) 
            r2 = random.randint(0,1) # random number to decide in the part is in the scene
            
            x = objects[obj_num]

        # Unhide and select lego object that's in the list
            x.hide_set(False)
            x.hide_render = False
            x.select_set(True)
            
        # Set scale to 0.05 x 0.05 x 0.05
            x.scale = (0.05, 0.05, 0.05)

        # Move lego object to a random location within given constriants
            x.location = (round(random.uniform(-1.5, 1.5), 2), round(random.uniform(-1.5, 1.5), 2), 0.5)

            # Lego orientation randomisation
            x.rotation_euler = (math.radians(random.randint(0,180)), math.radians(random.randint(0,180)), math.radians(random.randint(0,180)))

            # Lego material randomisation
            # Check if any part of x.name matches a key in colorsDict
            matching_key = next((key for key in colorsDict.keys() if key in x.name), None)
            if matching_key is not None:
                # Use the matching key to set the active material
                x.active_material = materials[colorsDict[matching_key]]
            else:
                # If no matching key is found, use a default material
                x.active_material = materials[1]
            
        # Update lego part counting table
        cc = 0
        for x in objects:
            if x.hide_render == False:
                object_count[cc][1] += 1
            cc += 1
            
            
        # Background randomisation
        # Hide all backgrounds
        for bg in background:
            bg.hide_set(True)
            bg.hide_render = True
        
        # unhide one background randomly
        r3 = random.randint(0,len(background)-1)
        background[r3].hide_set(False)
        background[r3].hide_render = False      
            
        # Fit camera scene within the objects
        bpy.ops.view3d.camera_to_view_selected()

        render_scene(iii)
        
        #save_annotations(x, start_range) # WHY IS THIS HERE

        writer = Writer(output_dir_images + "/" + guarantee + "_%0.5d.jpg" % iii, r_settings.resolution_x, r_settings.resolution_y)

        # Save annotations
        for x in objects:
            if x.hide_render == False:
                bound_x, bound_y, bound_w, bound_h = (camera_view_bounds_2d(bpy.context.scene, bpy.context.scene.camera, x))
                part_name = str(x.name).split(".", 1)
                # Save annotations of rectangle around the object: x_min, y_min, x_max, y_max
                writer.addObject(part_name[0], bound_x, bound_y, bound_x+bound_w, bound_y+bound_h)
                
        writer.save(output_dir_annotations + "/" + guarantee + "_%0.5d.xml" % iii)


    # Print out times each Lego object was used
    print("Object count: ")
    for cc in object_count:
        print(cc)
    # END OF BATCH RENDERING CODE

print("All done.")

