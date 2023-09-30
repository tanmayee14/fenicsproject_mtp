# Code to study the effect of changing additives(Wb - Wa) on various process parameters 

import matplotlib.pyplot as plt
from dolfin import *
import numpy as np
from openpyxl import Workbook


#random initial condition generator 
def generate_random_initial(n):
    return  0.5*np.ones(n) + 0.02*np.random.random_sample(n) - 0.01

length_of_dom = 100
mesh_size = 5
dom_size = mesh_size*length_of_dom
interpolation_degree = 2

#allencahn Constants
delt = 1e-3
sim_time = 25.0
inter_loc = length_of_dom * 0.5
dt = delt
l_sigma = Constant(1.0)
l_eta = Constant(1.0)
alpha = Constant(0.5)
kappa = Constant(1)
nFbyRT = Constant(2*96485/(8.3144958*298))

del_eta = -0.1

#Additives Constants
#mobility = Constant(5.0e-4)
#RT = Constant(8.3144958*298)
#Wb = Constant( -48913.402)
#Wa = Constant(14886.57)

mobility = Constant(1)
RT = Constant(1)
Wb = Constant(-27.608)
Wa = Constant(8.391)
xb_bulk = 0.05

run_number = 5
save_step = 100

mesh = IntervalMesh(dom_size , 0, length_of_dom)
V = FunctionSpace(mesh, 'Lagrange', interpolation_degree)


def G(xi):
    return 4*xi**3 - 6*xi**2 + 2*xi

def H(xi):
    return 30*xi**4 -60*xi**3 +30*xi**2

def a(xb):
    return (1)


def b(xb):
    return (1-xb)*xb
    
boundary_marker = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)

class lbc(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-15
        return on_boundary and abs(x[0]) < (tol)
class rbc(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-15
        return on_boundary and abs(length_of_dom - x[0]) < tol

bx0 = lbc()
bx1 = rbc()

bx0.mark(boundary_marker,0)
bx1.mark(boundary_marker,1)


e = Function(V)
q = TestFunction(V)

xb = Function(V)
u = TestFunction(V)



#Initial Condition For Allen Cahn

e_old = Function(V)
e_old = interpolate(Expression('1/(1 + pow(const_e,(-(x[0] - inter_loc))))',degree = 2, const_e = np.e, inter_loc = inter_loc), V)
eta = Function(V)
eta = del_eta*e_old

#Initial Condition For Additives

xb_old = Function(V)
xb_old = interpolate(Constant(xb_bulk), V)
# coords = V.tabulate_dof_coordinates()
# initial = np.zeros(coords.shape[0])
# for i in range(coords.shape[0]):
#     if i < length_of_dom*0.89:
#         initial[i] = 1.0

# xb_old.vector().set_local(initial)
# print(xb_old.vector().get_local())

#defining Boundary Conditions for allen cahn equation

boundary_conditions = { 
                        0 : {'Neumann': 0.0},
                        1 : {'Neumann': 0.0}}


bcs1 = []
for i in boundary_conditions:
    if 'Dirichlet' in boundary_conditions[i]:
        expression, sub_domain = boundary_conditions[i]['Dirichlet']
        bc = DirichletBC(V, expression, sub_domain)
#        bc = DirichletBC(V, expression, boundary_marker, i)
        bcs1.append(bc) 

ds = Measure('ds', domain = mesh, subdomain_data = boundary_marker)

integral_N1 = []
integral_N3 = [] # Neumann Condition for 2nd term in Additives
for i in boundary_conditions:
    if 'Neumann' in boundary_conditions[i]:
        g = boundary_conditions[i]['Neumann']
        integral_N1.append(kappa*g*q*ds(i))
        integral_N3.append( b(xb_old)*G(e)*g*u*ds(i))

# Boundary Conditions For Additives


boundary_conditions = { 
                        0 : {'Dirichlet': (0.01, bx0)},
                        1 : {'Dirichlet': (0.01, bx1)}}


bcs2 = []
for i in boundary_conditions:
    if 'Dirichlet' in boundary_conditions[i]:
        expression, sub_domain = boundary_conditions[i]['Dirichlet']
        bc = DirichletBC(V, expression, sub_domain)
#        bc = DirichletBC(V, expression, boundary_marker, i)
        bcs2.append(bc) 

integral_N2 = []
for i in boundary_conditions:
    if 'Neumann' in boundary_conditions[i]:
        g = boundary_conditions[i]['Neumann']
        integral_N2.append(a(xb_old)*g*u*ds(i))
        

#Weak Form of Allen Cahn equation
A = (
    e * q * dx

    - e_old * q * dx

    + dt * l_sigma * (Wb*xb_old + Wa*(1-xb_old)) * G(e) * q * dx

    + dt * l_sigma *  inner(nabla_grad(q),kappa *nabla_grad(e)) * dx

    - dt * l_sigma * sum(integral_N1)

    + dt * l_eta * H(e_old) * (np.e**( (1-alpha)*nFbyRT*(eta)) - np.e**(-alpha*nFbyRT*(eta)) )* q * dx
)


F = (
    xb * u * dx

    - xb_old * u * dx

    - dt * mobility * RT  * sum(integral_N2)

    + dt * mobility * RT * inner(nabla_grad(u), a(xb_old)*nabla_grad(xb)) * dx

    - dt * mobility * (Wb - Wa) * sum(integral_N3)

    + dt * mobility * (Wb - Wa) *inner(nabla_grad(u), b(xb_old)*G(e)*nabla_grad(e)) * dx


)

e_File = File('Additives_peak_study/Wb_Wa{}/study_number{}/order_parameter.pvd'.format((Wb-Wa), run_number))
xb_File = File('Additives_peak_study/Wb_Wa{}/study_number{}/xb.pvd'.format((Wb-Wa), run_number))


#Prepping to write an XLSX file
coords = V.tabulate_dof_coordinates()

headers1 = ['Time', 'xb_max', 'xb_min']
headers2 = ['x', 'phi']
wb = Workbook()
file_name = 'Additives_peak_study/Wb_Wa{}/study_number{}/peak-conc_study_{}.xlsx'.format((Wb-Wa), run_number, run_number)

ws1 = wb.active
ws1.title = 'Wb - Wa study'
ws2 = wb.create_sheet('Order_parameter')
ws1.append(['Bulk Concentration', xb_bulk])
ws1.append(headers1)
ws2.append(headers2)

set_log_active(False)

# solving the Allen cahn and Additives Equations
tol = 1E-4
iter = 0.0
for t in np.arange(0, sim_time, delt):

    solve(A == 0, e, bcs1)
    print('Solved A-C equation')

    if iter % save_step == 0 :
        e_File << (e_old, t)
        
    e_old_dof = e_old.vector().get_local()
    e_dof = e.vector().get_local()
    norm1 = np.sqrt(np.sum((e_dof - e_old_dof)**2)) / len(e_dof)
    
    if iter % save_step == 0 :
        for coord, val in zip(coords, e_dof):
            ws2.append([coord[0], val])
        ws2.append([])    

    if norm1 < tol:
        e_File << (e_old, t)
        break

    e_old.assign(e)

    if t == sim_time:
        e_File << (e_old, t + delt)

    # iter += 1
    # plot(e_old, title = 'Order Parameter Evolution')
    # plot(xb_old, title = 'Additive Xb')
    # plt.draw()
    # plt.pause(0.1)
    # plt.clf()

tol = 1E-8
iter = 0.0
for t in np.arange(0, sim_time, delt):

    solve(F == 0, xb, bcs2)
    print('Solved diffusion equation')

    if iter % save_step == 0 :
        xb_File << (xb_old, t)

    xb_old_dof = xb_old.vector().get_local()
    xb_dof = xb.vector().get_local()
    norm2 = np.sqrt(np.sum((xb_dof - xb_old_dof)**2)) / len(xb_dof)

    if iter % save_step == 0 :
        print('time: ', t, np.max(xb_dof), np.min(xb_dof))
        ws1.append([t, np.max(xb_dof), np.min(xb_dof)])


    if norm2 < tol:
        xb_File << (xb_old, t)
        break

    xb_old.assign(xb)

    if t == sim_time:
        xb_File << (xb_old, t + delt)

wb.save(filename = file_name)
#plt.close()