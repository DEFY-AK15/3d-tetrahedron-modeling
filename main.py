import matplotlib.pyplot as plt
import numpy as np
from math import *
from itertools import combinations

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_zlim(-5,5)
plt.xlim(-5,5)
plt.ylim(-5,5)

def get_x(p1,p2):
    return [p1[0],p2[0]]
def get_y(p1,p2):
    return [p1[1],p2[1]]
def get_z(p1,p2):
    return [p1[2],p2[2]]
def draw_line(A,B):
    ax.plot(get_x(A,B),get_y(A,B),get_z(A,B),color='#000000')
def draw(ver_list):
    temp = list(combinations(ver_list, 2))
    for temp2 in temp:
        draw_line(temp2[0],temp2[1])

def R(u,cos,sin):
    ux,uy,uz = list(u)
    return np.array([[cos+ux**2*(1-cos),ux*uy*(1-cos) - uz*sin,ux*uz*(1-cos)+uy*sin],
                     [ux*uy*(1-cos)+uz*sin, cos+uy**2 * (1-cos), uy*uz*(1-cos) - ux*sin],
                     [uz*ux*(1-cos)-uy*sin, uz*uy*(1-cos)+ux*sin, cos+uz**2*(1-cos)]])
def angle(ceta):
    return ceta * pi/180
def vec(A,O):
    return np.array([O[0]-A[0],O[1]-A[1],O[2]-A[2]])

def find_pos(A,O):
    AO = np.array([O[0]-A[0],O[1]-A[1],O[2]-A[2]])
    lenght = np.linalg.norm(AO)
    A_O = np.array([lenght,0,0])
    if AO[0]==A_O[0] and AO[1]==A_O[1] and AO[2]==A_O[2]:
        u=AO
    else:
        u = np.cross(AO,A_O)
        u /= np.linalg.norm(u)

    cos1 = AO@np.array([[A_O[i]] for i in range(3)]) / lenght**2
    sin1 = sqrt(1-cos1[0]**2)

    B= R(np.array([0,1,0]),1/3,sqrt(8)/3)@A_O + A_O
    b = np.array([B[0],0,0])
    bB = vec(b,B)
    Ob = vec(np.array([0,0,0]),b)
    C = R(np.array([1,0,0]),-1/2,sqrt(3)/2)@bB + Ob
    D = R(np.array([1,0,0]),-1/2,-sqrt(3)/2)@bB + Ob

    B = (R(u,cos1[0],-sin1)@B) +A
    C = (R(u,cos1[0],-sin1)@C) +A
    D = (R(u,cos1[0],-sin1)@D) +A
    return B,C,D

A = np.array([0,0,0])
O = np.array(list(map(int, input().split())))
B,C,D = find_pos(A,O)
draw([A,B,C,D])
print(f"{tuple(A)}\n{tuple(B)}\n{tuple(C)}\n{tuple(D)}\n{tuple(O)}")
ax.set_box_aspect([1,1,1])
plt.show()