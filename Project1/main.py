from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
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

g_vertex_shader_src = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_color; 

out vec4 vout_color;

uniform mat4 MVP;


void main()
{
    // 3D points in homogeneous coordinates
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);

    gl_Position = MVP * p3D_in_hcoord;

    vout_color = vec4(vin_color, 1.);
}
'''

g_fragment_shader_src = '''
#version 330 core

in vec4 vout_color;

out vec4 FragColor;

void main()
{
    FragColor = vout_color;
}
'''
def key_callback(window, key, scancode, action, mods):
    global P
    global is_ortho
    if(key == GLFW_KEY_V and action == GLFW_PRESS):
        if is_ortho:
            P = glm.perspective(45, 1, 1, 10)
            is_ortho = False
        else:
            P = glm.ortho(-1,1,-1,1,-100,100)
            is_ortho = True


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




def prepare_vao_triangle():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # color
         0.0, 0.0, 0.0,  1.0, 0.0, 0.0, # v0
         0.5, 0.0, 0.0,  0.0, 1.0, 0.0, # v1
         0.0, 0.5, 0.0,  0.0, 0.0, 1.0, # v2
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

def prepare_vao_frame():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # color
         -10.0, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis start
         10.0, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis end 
         0.0, 0.0, -10.0,  0.0, 0.0, 1.0, # z-axis start
         0.0, 0.0, 10.0,  0.0, 0.0, 1.0, # z-axis end 
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

def main():
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
    # load shaders
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)

    # get uniform locations
    MVP_loc = glGetUniformLocation(shader_program, 'MVP')
    
    # prepare vaos
    vao_triangle = prepare_vao_triangle()
    vao_frame = prepare_vao_frame()
    vao_grid = prepare_vao_grid()

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # render

        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        glUseProgram(shader_program)

        V = glm.lookAt(camera_pos, camera_tar, w_up)
        M = glm.mat4()

        MVP = P*V*M

        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))

        glBindVertexArray(vao_frame)
        glDrawArrays(GL_LINES, 0, 6)

        # draw the grid
        glBindVertexArray(vao_grid)
        glDrawArrays(GL_LINES, 0, 1000)

        # draw triangle w.r.t. the current frame
        glBindVertexArray(vao_triangle)
        glDrawArrays(GL_TRIANGLES, 0, 3)

        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()




