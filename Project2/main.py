from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import os
import numpy as np

g_prev_mouse_x = 0
g_prev_mouse_y = 0
camera_pos = glm.vec3(2,1,2)
camera_tar = glm.vec3(0.0,0.0,0.0)
camera_up = glm.vec3(0,0.2,0)
azimuth = 45
elevation = 20.4
P = glm.perspective(45, 1, 1, 10)
is_ortho = False
w_up = glm.vec3(0,1,0)
vertices = []
normals = []
indices = []
triangle = 0
number = 0
quad = 0
over_quad = 0
h_vertices = []
h_normals = []
h_indices = []
h_triangle = 0
h_number = 0
h_normals = 0
h_quad = 0
h_over_quad = 0
n_index = []


single_mesh = True
is_wire = False

g_vertex_shader_src = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_normal; 

out vec3 vout_surface_pos;
out vec3 vout_normal;

uniform mat4 MVP;
uniform mat4 M;

void main()
{
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);
    gl_Position = MVP * p3D_in_hcoord;

    vout_surface_pos = vec3(M * vec4(vin_pos, 1));
    vout_normal = normalize( mat3(transpose(inverse(M))) * vin_normal);
}
'''

g_fragment_shader_src = '''
#version 330 core

in vec3 vout_surface_pos;
in vec3 vout_normal;

out vec4 FragColor;

uniform vec3 view_pos;
uniform vec3 vin_color;

void main()
{
    // light and material properties
    vec3 light_pos1 = vec3(3,2,4);
    vec3 light_color1 = vec3(1,1,1);
    vec3 light_color2 = vec3(1,1,1);
    vec3 light_pos2 = vec3(5,5,5);
    vec3 material_color = vin_color;
    float material_shininess = 32.0;

    // light components
    vec3 light_ambient1 = 0.1*light_color1;
    vec3 light_diffuse1 = light_color1;
    vec3 light_specular1 = light_color1;

    // material components
    vec3 material_ambient1 = material_color;
    vec3 material_diffuse1 = material_color;
    vec3 material_specular1 = light_color1;  // for non-metal material

        // light components
    vec3 light_ambient2 = 0.1*light_color2;
    vec3 light_diffuse2 = light_color2;
    vec3 light_specular2 = light_color2;

    // material components
    vec3 material_ambient2 = material_color;
    vec3 material_diffuse2 = material_color;
    vec3 material_specular2 = light_color2;  // for non-metal material
    // ambient

    vec3 ambient1 = light_ambient1 * material_ambient1;
    vec3 ambient2 = light_ambient2 * material_ambient2;

    // for diffiuse and specular
    vec3 normal = normalize(vout_normal);
    vec3 surface_pos = vout_surface_pos;
    vec3 light_dir1 = normalize(light_pos1 - surface_pos);
    vec3 light_dir2 = normalize(light_pos2 - surface_pos);

    // diffuse
    float diff1 = max(dot(normal, light_dir1), 0);
    vec3 diffuse1 = diff1 * light_diffuse1 * material_diffuse1;

    float diff2 = max(dot(normal, light_dir2), 0);
    vec3 diffuse2 = diff2 * light_diffuse2 * material_diffuse2;

    // specular
    vec3 view_dir = normalize(view_pos - surface_pos);
    vec3 reflect_dir1 = reflect(-light_dir1, normal);
    vec3 reflect_dir2 = reflect(-light_dir2, normal);
    float spec1 = pow( max(dot(view_dir, reflect_dir1), 0.0), material_shininess);
    float spec2 = pow( max(dot(view_dir, reflect_dir2), 0.0), material_shininess);
    
    vec3 specular1 = spec1 * light_specular1 * material_specular1;
    vec3 specular2 = spec2 * light_specular2 * material_specular2;


    vec3 color = ambient1 + diffuse1 + specular1 + ambient2 + diffuse2 + specular2;
    FragColor = vec4(color, 1.);
}
'''


class Node:
    def __init__(self, parent, shape_transform, color):
        # hierarchy
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)

        # transform
        self.transform = glm.mat4()
        self.global_transform = glm.mat4()

        # shape
        self.shape_transform = shape_transform
        self.color = color

    def set_transform(self, transform):
        self.transform = transform

    def update_tree_global_transform(self):
        if self.parent is not None:
            self.global_transform = self.parent.get_global_transform() * self.transform
        else:
            self.global_transform = self.transform

        for child in self.children:
            child.update_tree_global_transform()

    def get_global_transform(self):
        return self.global_transform
    def get_shape_transform(self):
        return self.shape_transform
    def get_color(self):
        return self.color

def key_callback(window, key, scancode, action, mods):
    global P
    global is_ortho, single_mesh, is_wire
    if(key == GLFW_KEY_V and action == GLFW_PRESS):
        if is_ortho:
            P = glm.perspective(45, 1, 1, 10)
            is_ortho = False
        else:
            P = glm.ortho(-1,1,-1,1,-100,100)
            is_ortho = True
    if(key == GLFW_KEY_H and action == GLFW_PRESS):
        if single_mesh:
            single_mesh = False
        else:
            single_mesh = True
    if(key == GLFW_KEY_Z and action == GLFW_PRESS):
        if is_wire:
            is_wire = False
        else:
            is_wire = True


def cursor_position_callback(window, xpos, ypos):
    global g_prev_mouse_x, g_prev_mouse_y, camera_pos, camera_tar, camera_up, azimuth, elevation, w_up

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS):
        if (g_prev_mouse_x == 0.0 and g_prev_mouse_y == 0.0):
            g_prev_mouse_x = xpos
            g_prev_mouse_y = ypos
        else:
            mouse_dx = (xpos - g_prev_mouse_x) * 0.005
            mouse_dy = -(ypos - g_prev_mouse_y) * 0.005
            azimuth = azimuth + np.degrees(mouse_dx)
            elevation = elevation + np.degrees(mouse_dy)

            if(np.cos(elevation) >= 0):
                w_up = glm.vec3(0,1,0)
            if(np.cos(np.radians(elevation)) <= 0):
                w_up = glm.vec3(0, -1, 0)
            front = glm.vec3(
            np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth)),
            np.sin(np.radians(elevation)),
            np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth))
            )
            distance = glm.distance(camera_pos, camera_tar)
            camera_pos = camera_tar + distance * front

            g_prev_mouse_x = xpos
            g_prev_mouse_y = ypos

    elif (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS):
            if (g_prev_mouse_x == 0.0 and g_prev_mouse_y == 0.0):
                g_prev_mouse_x = xpos
                g_prev_mouse_y = ypos
            else:
                mouse_dx = -(xpos - g_prev_mouse_x)
                mouse_dy = (ypos - g_prev_mouse_y)

                w = glm.normalize(camera_pos - camera_tar)
                u = glm.normalize(glm.cross(w_up, w))
                v = glm.cross(w, u)

                translate_distance = 0.005
                translate_vec = u * mouse_dx * translate_distance + v * mouse_dy * translate_distance
                camera_pos += translate_vec
                camera_tar += translate_vec

                g_prev_mouse_x = xpos
                g_prev_mouse_y = ypos
    else:
        g_prev_mouse_x = 0.0
        g_prev_mouse_y = 0.0

def scroll_callback(window, xoffset, yoffset):
    global camera_pos, camera_tar, scale
    distance = glm.distance(camera_pos, camera_tar)
    w = glm.normalize(camera_pos - camera_tar)
    # yoffset 값이 양수이면 스크롤을 위로 올리는 동작(줌인)
    if yoffset > 0:
        if(distance > 2):
            w = w * 0.9
    # yoffset 값이 음수이면 스크롤을 아래로 내리는 동작(줌아웃)
    elif yoffset < 0:
        w = w * 1.1
    new_cam_pos = camera_tar + w * distance
    camera_pos = new_cam_pos
    
def drop_callback(window, paths):
    global vertices, normals, indices, triangle, quad, number, over_quad, single_mesh

    single_mesh = True
    filepath = os.path.join(paths[0])
    filename = filepath.split('/')[-1]

    vertices = []
    normals = []
    indices = []
    number = 0
    triangle = 0
    quad = 0
    over_quad = 0

    faces = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                vertex = [float(x) for x in line[2:].split()]
                vertices.append(vertex)
            elif line.startswith('vn '):
                normal = [float(x) for x in line[3:].split()]
                normals.append(normal)
            elif line.startswith('f '):
                face = []
                for x in line[2:].split():
                    indices = x.split('/')
                    v_idx, n_idx = int(indices[0]), int(indices[2])
                    face.append((v_idx, n_idx))
                if(len(face) == 3):
                    number += 1
                    triangle += 1
                    faces.append(face)
                else:
                    if(len(face) == 4):
                        quad += 1
                    else:
                        over_quad += 1
                    for i in range(len(face)-2):
                        faces.append([face[0],face[1+i],face[2+i]])
                        number += 1

    vertices = np.array(vertices, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32)
    indices = np.array([i for f in faces for i in f], dtype=np.uint32) - 1

    print("Obj file name : " + filename)
    print("Total number of faces : " + str(triangle+quad+over_quad))
    print("Number of faces with 3 vertices : " + str(triangle))
    print("Number of faces with 4 vertices : " + str(quad))
    print("Number of faces with more than 4 vertices : " + str(over_quad))

def read_obj(filepath):
    global h_vertices, h_normals, h_indices, h_number, h_triangle, h_quad, h_over_quad

    h_vertices = []
    h_normals = []
    h_indices = []
    h_number = 0
    h_triangle = 0
    h_quad = 0
    h_over_quad = 0

    faces = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                vertex = [float(x) for x in line[2:].split()]
                h_vertices.append(vertex)
            elif line.startswith('vn '):
                normal = [float(x) for x in line[3:].split()]
                h_normals.append(normal)
            elif line.startswith('f '):
                face = []
                for x in line[2:].split():
                    h_indices = x.split('/')
                    v_idx, n_idx = int(h_indices[0]), int(h_indices[2])
                    face.append((v_idx, n_idx))
                if(len(face) == 3):
                    h_number += 1
                    h_triangle += 1
                    faces.append(face)
                else:
                    if(len(face) == 4):
                        h_quad += 1
                    else:
                        h_over_quad += 1
                    for i in range(len(face)-2):
                        faces.append([face[0],face[1+i],face[2+i]])
                        h_number += 1

    h_vertices = np.array(h_vertices, dtype=np.float32)
    h_normals = np.array(h_normals, dtype=np.float32)
    h_indices = np.array([i for f in faces for i in f], dtype=np.uint32) - 1
    n_index.append(h_number)

def load_shaders(vertex_shader_source, fragment_shader_source):
    # build and compile our shader program
    # ------------------------------------
    
    # vertex shader 
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)    # create an empty shader object
    glShaderSource(vertex_shader, vertex_shader_source) # provide shader source code
    glCompileShader(vertex_shader)                      # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(vertex_shader)
        print("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" + infoLog.decode())
        
    # fragment shader
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)    # create an empty shader object
    glShaderSource(fragment_shader, fragment_shader_source) # provide shader source code
    glCompileShader(fragment_shader)                        # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(fragment_shader)
        print("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" + infoLog.decode())

    # link shaders
    shader_program = glCreateProgram()               # create an empty program object
    glAttachShader(shader_program, vertex_shader)    # attach the shader objects to the program object
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)                    # link the program object

    # check for linking errors
    success = glGetProgramiv(shader_program, GL_LINK_STATUS)
    if (not success):
        infoLog = glGetProgramInfoLog(shader_program)
        print("ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + infoLog.decode())
        
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program    # return the shader program

def prepare_vao_single_mesh():

    combined_vertices = []
    for i in indices:
        combined_vertices.append(vertices[i[0]])
        combined_vertices.append(normals[i[1]])
    combined_vertices = np.array(combined_vertices, dtype=np.float32)
    combined_vertices = glm.array(combined_vertices)
    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, combined_vertices.nbytes, combined_vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_hierarchy(object_file):
    current_dir = os.path.dirname(__file__)

    # use os.path.join to create a platform-independent relative path to the obj file
    obj_path = os.path.join(current_dir, object_file)
    read_obj(obj_path)
    combined_vertices = []
    for i in h_indices:
        combined_vertices.append(h_vertices[i[0]])
        combined_vertices.append(h_normals[i[1]])
    combined_vertices = np.array(combined_vertices, dtype=np.float32)
    combined_vertices = glm.array(combined_vertices)

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, combined_vertices.nbytes, combined_vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_frame():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # normal
         -10.0, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis start
         10.0, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis end 
         0.0, 0.0, -10.0,  1.0, 0.0, 0.0, # z-axis start
         0.0, 0.0, 10.0,  1.0, 0.0, 0.0, # z-axis end 
    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_grid():
    # prepare vertex data (in main memory)
    vertices = []

    for i in range(0, 100):
        vertices += [-10, 0.0, i/10, 1.0, 1.0, 1.0]
        vertices += [10, 0.0, i/10, 1.0, 1.0, 1.0]
        vertices += [-10, 0.0, -i/10, 1.0, 1.0, 1.0]
        vertices += [10, 0.0, -i/10, 1.0, 1.0, 1.0]
        vertices += [i/10, 0.0, -10, 1.0, 1.0, 1.0]
        vertices += [i/10, 0.0, 10, 1.0, 1.0, 1.0]
        vertices += [-i/10, 0.0, -10, 1.0, 1.0, 1.0]
        vertices += [-i/10, 0.0, 10, 1.0, 1.0, 1.0]
    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, len(vertices)*4, np.array(vertices, dtype=np.float32), GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer
    
    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)
    return VAO

def draw_node(vao, node, VP, MVP_loc, color_loc):
    MVP = VP * node.get_global_transform() * node.get_shape_transform()
    color = node.get_color()
    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glUniform3f(color_loc, color.r, color.g, color.b)
    glDrawArrays(GL_TRIANGLES, 0, (h_number) * 3)

def main():
    global h_number
    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)   # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE) # for macOS

    # create a window and OpenGL context
    window = glfwCreateWindow(800, 800, '2019067429', None, None)
    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)

    # register event callbacks
    glfwSetCursorPosCallback(window, cursor_position_callback)
    glfwSetScrollCallback(window, scroll_callback)
    glfwSetKeyCallback(window, key_callback)
    glfwSetDropCallback(window, drop_callback)
    # load shaders
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)

    # get uniform locations
    MVP_loc = glGetUniformLocation(shader_program, 'MVP')
    color_loc = glGetUniformLocation(shader_program, 'vin_color')
    view_pos_loc = glGetUniformLocation(shader_program, 'view_pos')

    # prepare vaos
    vao_frame = prepare_vao_frame()
    vao_grid = prepare_vao_grid()
    vao_table = prepare_vao_hierarchy('15582_Table_Tennis_Table_v2_NEW.obj')
    vao_hand = prepare_vao_hierarchy('11538_Hand_v2.obj')
    vao_racket = prepare_vao_hierarchy('15583_TableTennisPaddle_ShakehandStyle_v1.obj')
    vao_ball = prepare_vao_hierarchy('15640_Tennis_Ball_v1.obj')
    vao_chair = prepare_vao_hierarchy('16948_fold_out_chair_v1_NEW.obj')
    vao_board = prepare_vao_hierarchy('19839_Drawing_Board_v1.obj')
    # # create a hirarchical model - Node(parent, shape_transform, color)
    table = Node(None, glm.translate((-1.2 , 0, 1.4)) * glm.scale((0.01, 0.01, 0.01)) * glm.rotate(np.radians(-90), glm.vec3(1,0,0)), glm.vec3(0.3,0.5,0.9))
    
    hand1 = Node(table, glm.translate(glm.vec3(-1.9,1.5,0.5)) * glm.scale((-0.03,0.03,0.03)) * glm.rotate(np.radians(220), glm.vec3(1,1,1)), glm.vec3(0.9,0.7,0.5))
    hand2 = Node(table, glm.translate(glm.vec3(1.8,1,0.5)) * glm.scale((0.03,0.03,0.03)) * glm.rotate(np.radians(-270), glm.vec3(1, 1, 1)), glm.vec3(0.9,0.7,0.5))
    chair = Node(table,  glm.translate(glm.vec3(-1,0,-1.2)) * glm.rotate(np.radians(-90), glm.vec3(1,0,0)) * glm.scale((0.01, 0.01, 0.01)), glm.vec3(.1,.1,.1))
    board = Node(table, glm.translate(glm.vec3(-0.2,0,-1.2)) * glm.scale((-0.05,0.05,0.05)) * glm.rotate(np.radians(-90), glm.vec3(1,0,0)), glm.vec3(1,1,1))
    
    racket1 = Node(hand1, glm.translate(glm.vec3(-1.45,1.5,0.3)) * glm.scale((-.03,.03,.03)) * glm.rotate(np.radians(10), glm.vec3(1,1,0)), glm.vec3(.7,0.2,.2))
    racket2 = Node(hand2, glm.translate(glm.vec3(1.65,0.7,0.3)) * glm.scale((.03,.03,.03)) * glm.rotate(np.radians(-270), glm.vec3(1,1,1)) * glm.rotate(np.radians(270), glm.vec3(1,0,1)), glm.vec3(.2,.2,.2))
    
    ball1 = Node(hand1, glm.translate(glm.vec3(-1.2, 1.2 ,0.3)) * glm.scale((-.03,.03,.03)) * glm.rotate(np.radians(10), glm.vec3(1,1,0)), glm.vec3(1,1,1))
    ball2 = Node(hand2, glm.translate(glm.vec3(1.4,0.9,0.3)) * glm.scale((.03,.03,.03)) * glm.rotate(np.radians(-270), glm.vec3(1,1,1)) * glm.rotate(np.radians(270), glm.vec3(1,0,1)), glm.vec3(1,1,1))

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # render

        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        if(is_wire):
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        V = glm.lookAt(camera_pos, camera_tar, w_up)
        M = glm.mat4()

        MVP = P*V*M
        glUseProgram(shader_program)
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
        glUseProgram(shader_program)
        glUniform3f(color_loc, 1, 1, 1)
        glUseProgram(shader_program)

        glBindVertexArray(vao_frame)
        glDrawArrays(GL_LINES, 0, 6)

        # draw the grid
        glBindVertexArray(vao_grid)
        glDrawArrays(GL_LINES, 0, 1000)
        
        # single_mesh rendering
        if single_mesh:
            vao_single_mesh = prepare_vao_single_mesh()
            glBindVertexArray(vao_single_mesh)
            glUniform3f(view_pos_loc, camera_pos.x, camera_pos.y, camera_pos.z)
            glDrawArrays(GL_TRIANGLES, 0, number*3)
        # hierarchy rendering
        else:
            t = glfwGetTime()
            table.set_transform(glm.rotate(.5*t, glm.vec3(0,1,0)))
            hand1.set_transform(glm.translate(glm.vec3(0, .2 * glm.sin(5*t) ,.1*glm.sin(5*t) )))
            hand2.set_transform(glm.translate(glm.vec3(0, .3 * glm.sin(5*t) ,0 )))
            ball1.set_transform(glm.translate(glm.vec3(-1.2, 1.2 ,0.3)) * glm.rotate(2*t, glm.vec3(0,1,0)) * glm.translate(glm.vec3(1.2, -1.2 ,-0.3)) * glm.translate(glm.vec3(0, .15 * glm.sin(5*t),0)))
            ball2.set_transform(glm.translate(glm.vec3(1.4,0.9,0.3)) * glm.rotate(2*t, glm.vec3(0,1,0)) * glm.translate(glm.vec3(-1.4,-0.9,-0.3)) * glm.translate(glm.vec3(0, .15 *glm.sin(5*t),0)))
            table.update_tree_global_transform()
            glUniform3f(view_pos_loc, camera_pos.x, camera_pos.y, camera_pos.z)
            h_number = n_index[0]
            draw_node(vao_table, table, P*V, MVP_loc, color_loc)
            h_number = n_index[1]
            draw_node(vao_hand, hand1, P*V, MVP_loc, color_loc)
            draw_node(vao_hand, hand2, P*V, MVP_loc, color_loc)
            h_number = n_index[2]
            draw_node(vao_racket, racket2, P*V, MVP_loc, color_loc)
            draw_node(vao_racket, racket1, P*V, MVP_loc, color_loc)
            h_number = n_index[3]
            draw_node(vao_ball, ball2, P*V, MVP_loc, color_loc)
            draw_node(vao_ball, ball1, P*V, MVP_loc, color_loc)
            h_number = n_index[4]
            draw_node(vao_chair, chair, P*V, MVP_loc, color_loc)
            h_number = n_index[5]
            draw_node(vao_board, board, P*V, MVP_loc, color_loc)

        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()





