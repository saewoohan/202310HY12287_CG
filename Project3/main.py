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
P = glm.perspective(45, 1, 1, 100000)
is_ortho = False
w_up = glm.vec3(0,1,0)
vao_line = 0
root_joint = 0
num_vertices = 0
nodes = []
is_line = True
is_box = False
is_render = False
is_motion = False
n_vertices = 2
vertices = []
vertices_for_box = []
frame_time = 0
is_outside = False
frame_count = 0
joint_names = []
multiplier = 1

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
    vec3 light_pos1 = vec3(100,100,100);
    vec3 light_color1 = vec3(1,1,1);
    vec3 light_color2 = vec3(1,1,1);
    vec3 light_pos2 = vec3(-10,-10,-10);
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
    def __init__(self, parent, link_transform_from_parent, shape_transform, color):
        # hierarchy
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)

        # transform
        self.link_transform_from_parent = link_transform_from_parent
        self.joint_transform = glm.mat4()
        self.global_transform = glm.mat4()
        self.channel = []
        self.offset = []
        self.index = 0
        self.name = ''

        # shape
        self.shape_transform = shape_transform
        self.color = color

    def set_joint_transform(self, joint_transform):
        self.joint_transform = joint_transform
    def get_channel(self):
        return self.channel
    def get_global_transform(self):
        return self.global_transform
    def get_shape_transform(self):
        return self.shape_transform
    def get_color(self):
        return self.color
    def update_tree_global_transform(self):
        if self.parent is not None:
            self.global_transform = self.parent.get_global_transform() * self.link_transform_from_parent * self.joint_transform
        else:
            self.global_transform = self.link_transform_from_parent * self.joint_transform

        for child in self.children:
            child.update_tree_global_transform()



def key_callback(window, key, scancode, action, mods):
    global P
    global is_ortho,is_line, is_box, is_motion
    if key == GLFW_KEY_V and action == GLFW_PRESS:
        if is_ortho:
            P = glm.perspective(45, 1, 1, 100)
            is_ortho = False
        else:
            P = glm.ortho(-1, 1, -1, 1, -100, 100)
            is_ortho = True

    if key == GLFW_KEY_1 and action == GLFW_PRESS:
        is_line = True
        is_box = False
    if key == GLFW_KEY_2 and action == GLFW_PRESS:
        is_line = False
        is_box = True
    if key == GLFW_KEY_SPACE and action == GLFW_PRESS:
        is_motion = True

def parse_bvh_file(content):
    global nodes, frame_time,joint_names, frame_count,multiplier
    joint_names = []
    joint_offsets = []

    motion_data = []
    channels = []  # 추가: 채널 정보를 저장할 리스트

    lines = content.split('\n')
    line_index = 0

    root = None
    current_node = None
    angles = []
    temp = 0
    
    if is_outside:
        multiplier = 50
    else:
        multiplier = 1
    while line_index < len(lines):
        line = lines[line_index].strip()
        line_index += 1
        if line.startswith('ROOT'):
            joint_name = line.split()[1]
            joint_names.append(joint_name)

            # Find the OFFSET line
            while not lines[line_index].strip().startswith('OFFSET'):
                line_index += 1

            offset_line = lines[line_index].strip()
            offset_values = [float(value)/multiplier for value in offset_line.split()[1:]]
            joint_offsets.append(offset_values)
            while not lines[line_index].strip().startswith('CHANNELS'):
                line_index += 1

            channels_line = lines[line_index].strip()
            channels = channels_line.split()[1:]
            # Create the node
            node = Node(None,glm.translate((offset_values[0],offset_values[1],offset_values[2])),glm.mat4(), glm.vec3(.5, .5, 1))
            nodes.append(node)
            node.name = joint_name
            node.channel = channels
            node.offset = [offset_values[0], offset_values[1], offset_values[2]]
            node.index = temp
            temp += 1
            root = node
            current_node = node

        elif line.startswith('JOINT'):
            joint_name = line.split()[1]
            joint_names.append(joint_name)


            # Find the OFFSET line
            while not lines[line_index].strip().startswith('OFFSET'):
                line_index += 1

            offset_line = lines[line_index].strip()
            offset_values = [float(value)/multiplier for value in offset_line.split()[1:]]
            # Create the node
            node = Node(current_node, glm.translate((offset_values[0],offset_values[1],offset_values[2])), glm.mat4(), glm.vec3(.5, .5, 1))
            nodes.append(node)
            joint_offsets.append(offset_values)
            node.offset = [offset_values[0], offset_values[1], offset_values[2]]
            node.index = temp
            node.name = joint_name
            temp += 1
            current_node = node
            # Find the CHANNELS line
            while not lines[line_index].strip().startswith('CHANNELS'):
                line_index += 1

            channels_line = lines[line_index].strip()
            channels = channels_line.split()[1:]
            current_node.channel = channels


        elif line.startswith('End Site'):
            while not lines[line_index].strip().startswith('OFFSET'):
                line_index += 1

            offset_line = lines[line_index].strip()
            offset_values = [float(value)/multiplier for value in offset_line.split()[1:]]
            joint_offsets.append(offset_values)

            # Create the endsite node
            endsite_node = Node(current_node, glm.translate((offset_values[0],offset_values[1],offset_values[2])), glm.mat4(), glm.vec3(.5, .5, 1))
            nodes.append(endsite_node)
            endsite_node.offset = [offset_values[0], offset_values[1], offset_values[2]]
            endsite_node.index = temp
            endsite_node.name = 'End'
            current_node = endsite_node
            temp += 1
        elif line.startswith('}'):
            if current_node.parent is not None:
                current_node = current_node.parent

        elif line.startswith('MOTION'):
            # MOTION 라인을 만나면 motion 데이터 파싱 시작
            # FRAME 개수 파싱
            frame_line = lines[line_index].strip()
            frame_count = int(frame_line.split()[-1])

            # FRAME TIME 파싱
            frame_time_line = lines[line_index + 1].strip()
            frame_time = float(frame_time_line.split()[-1])

            # MOTION 데이터 파싱
            motion_data_lines = lines[line_index + 3:line_index + 3 + frame_count]
            motion_data = [line.strip() for line in motion_data_lines]

            # 모션 데이터 추가
            for frame in motion_data:
                frame_angles = [float(angle) for angle in frame.strip().split()]
                angles.append(frame_angles)

    return root, angles



def drop_callback(window, paths):
    global root_joint, angles, is_render, vertices, vertices_for_box, nodes, is_outside

    is_render = True
    vertices = []
    vertices_for_box = []
    nodes = []
    filepath = os.path.join(paths[0])
    filename = filepath.split('/')[-1]
    if filename != 'sample-spin.bvh' and filename != 'sample-walk.bvh':
        is_outside = True
    else:
        is_outside = False
    with open(filepath, 'r') as file:
        content = file.read()

    root_joint, angles= parse_bvh_file(content)
    print("File name : " + filename)
    print("Number of frames : " + str(frame_count))
    print("FPS : " + str(frame_time))
    print("Number of joints : " + str(len(nodes)))
    print("List of all joint names : ", joint_names)


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

def prepare_vao_line(node):
    global vertices
    # prepare vertex data (in main memory)
    indices = []
    indices.append(node.parent.index)
    indices.append(node.index)
    vertices = np.array(vertices, dtype=np.float32)
    vertices = glm.array(vertices)
    indices = np.array(indices, dtype=np.uint32)
    indices = glm.array(indices)
    
    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW)

    # create and activate EBO (element buffer object)
    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)

    # copy index data to EBO
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices.ptr, GL_STATIC_DRAW)

    # configure vertex attributes
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3 * glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO


def prepare_vao_box(node):
    global vertices_for_box
    # prepare vertex data (in main memory)

    index_parent = node.parent.index * 4
    index_child = node.index * 4
    vertices_for_box = np.array(vertices_for_box, dtype=np.float32)
    vertices_for_box = glm.array(vertices_for_box)
    indices = glm.array(glm.uint32,
        index_parent, index_parent+2, index_parent+1,
        index_parent, index_parent+3, index_parent+2,
        index_child, index_child+1, index_child+2,
        index_child, index_child+2, index_child+3,
        index_parent, index_parent+1, index_child+1,
        index_parent+3, index_child+2, index_parent+2,
        index_parent+3, index_child+3, index_child+2,
        index_parent+1, index_parent+2, index_child+2,
        index_parent+1, index_child+2, index_child+1,
        index_parent, index_child+3, index_parent+3,
        index_parent, index_child, index_child+3,
    )
    
    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices_for_box.nbytes, vertices_for_box.ptr, GL_STATIC_DRAW)

    # create and activate EBO (element buffer object)
    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)

    # copy index data to EBO
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices.ptr, GL_STATIC_DRAW)

    # configure vertex attributes
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3 * glm.sizeof(glm.float32)))
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

def draw_node_line(vao, VP, MVP_loc, color_loc):
    MVP = VP * glm.mat4()
    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glUniform3f(color_loc, 0.1, 0.1, 0.7)
    glDrawElements(GL_LINES, 2, GL_UNSIGNED_INT, None)

def draw_node_box(vao, VP, MVP_loc, color_loc):
    MVP = VP * glm.mat4()
    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glUniform3f(color_loc, 0.1, 0.1, .7)
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)

def draw_bvh_nodes(VP, MVP_loc, color_loc, frame):
    global nodes, vertices, vertices_for_box
    if is_motion:
        channel_index = 0
        for node in nodes:
            # 현재 프레임에 해당하는 각도 가져오기
            angles_for_frame = angles[frame]
            # 노드의 채널 정보 가져오기
            channels = node.get_channel()

            # 노드의 조인트 변환 초기화
            joint_transform = glm.mat4()
            
            # 채널 정보를 사용하여 조인트 변환 계산
            for channel in channels[1:]:
                channel_axis = str(channel)
                channel_value = angles_for_frame[channel_index]
                channel_index += 1

                if channel_axis == 'XROTATION' or channel_axis == 'Xrotation':
                    joint_transform = joint_transform * glm.rotate(np.radians(channel_value), glm.vec3(1, 0, 0))
                elif channel_axis == 'YROTATION' or channel_axis == 'Yrotation':
                    joint_transform = joint_transform * glm.rotate(np.radians(channel_value), glm.vec3(0, 1, 0))
                elif channel_axis == 'ZROTATION' or channel_axis == 'Zrotation':
                    joint_transform = joint_transform * glm.rotate(np.radians(channel_value), glm.vec3(0, 0, 1))
                elif channel_axis == 'XPOSITION' or channel_axis == 'Xposition':
                    joint_transform = joint_transform * glm.translate(glm.vec3(channel_value/multiplier, 0, 0))
                elif channel_axis == 'YPOSITION'or channel_axis == 'Yposition':
                    joint_transform = joint_transform * glm.translate(glm.vec3(0, channel_value/multiplier, 0))
                elif channel_axis == 'ZPOSITION'or channel_axis == 'Zposition':
                    joint_transform = joint_transform * glm.translate(glm.vec3(0, 0, channel_value/multiplier))
            node.set_joint_transform(joint_transform)

    root_joint.update_tree_global_transform()
    joint_transform = glm.mat4()
    for i in nodes:
        joint_transform = i.get_global_transform()
        i.offset = joint_transform * glm.vec3(0, 0, 0)
    
    vertices = []
    vertices_for_box = []
    for i in nodes:
        vertices.append(i.offset)
        vertices.append([(i.offset[0] + i.offset[1] + i.offset[2]) /3 ,(i.offset[0] + i.offset[1] + i.offset[2]) /3, (i.offset[0] + i.offset[1] + i.offset[2]) /3])
    for i in nodes:
            vertices_for_box.append([i.offset[0] - 0.03, i.offset[1], i.offset[2] + 0.03])
            vertices_for_box.append([(i.offset[0]+0.03 + i.offset[1] + i.offset[2] + 0.03)/3, (i.offset[0]+0.03 + i.offset[1] + i.offset[2] + 0.03)/3, (i.offset[0]+0.03 + i.offset[1] + i.offset[2] + 0.03)/3])
            vertices_for_box.append([i.offset[0] + 0.03, i.offset[1], i.offset[2] + 0.03])
            vertices_for_box.append([(i.offset[0]+0.03 + i.offset[1] + i.offset[2] + 0.03)/3, (i.offset[0]+0.03 + i.offset[1] + i.offset[2] + 0.03)/3, (i.offset[0]+0.03 + i.offset[1] + i.offset[2] + 0.03)/3])
            vertices_for_box.append([i.offset[0] + 0.03, i.offset[1], i.offset[2] - 0.03 ])
            vertices_for_box.append([(i.offset[0]+0.03 + i.offset[1] + i.offset[2] + 0.03)/3, (i.offset[0]+0.03 + i.offset[1] + i.offset[2] + 0.03)/3, (i.offset[0]+0.03 + i.offset[1] + i.offset[2] + 0.03)/3])
            vertices_for_box.append([i.offset[0] - 0.03, i.offset[1], i.offset[2] - 0.03])
            vertices_for_box.append([(i.offset[0]+0.03 + i.offset[1] + i.offset[2] + 0.03)/3, (i.offset[0]+0.03 + i.offset[1] + i.offset[2] + 0.03)/3, (i.offset[0]+0.03 + i.offset[1] + i.offset[2] + 0.03)/3])
    for n in nodes:
        if(n.parent is None):
            continue
        if is_box:
            vao = prepare_vao_box(n)
            draw_node_box(vao, VP, MVP_loc, color_loc)
        if is_line:
            vao = prepare_vao_line(n)
            draw_node_line(vao, VP, MVP_loc, color_loc)

def main():
    global is_render
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
    vao_grid = prepare_vao_grid()

    t = 0
    glfwSetTime(0.0)  # 초기 시간을 0으로 설정
    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # render

        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        
        current_time = glfwGetTime()

        if current_time >= frame_time:
            t += 1
            glfwSetTime(0.0)  # 시간을 다시 0으로 설정
        if t >= (frame_count - 1):
            t = 0        

        glUseProgram(shader_program)
        glUniform3f(view_pos_loc, camera_pos.x, camera_pos.y, camera_pos.z)
        V = glm.lookAt(camera_pos, camera_tar, w_up)
        M = glm.mat4()
       
        MVP = P*V*M
        glUseProgram(shader_program)
        glUniform3fv(color_loc, 1,1,1)
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
        glUseProgram(shader_program)
        glUniform3f(color_loc, 1, 1, 1)
        glUseProgram(shader_program)

        # draw the grid
        glBindVertexArray(vao_grid)
        glDrawArrays(GL_LINES, 0, 1000)
        glUseProgram(shader_program)

        # draw triangle w.r.t. the current frame
        if is_render:
            if  is_box:
                draw_bvh_nodes(P * V, MVP_loc, color_loc, int(t))
            if  is_line:
                draw_bvh_nodes(P * V, MVP_loc, color_loc, int(t))
        
        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()





