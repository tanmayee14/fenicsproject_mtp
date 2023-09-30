#Code used to extract the variation of the order parmeter without the influence of additives and Cu diffusion.

import matplotlib.pyplot as plt
from dolfin import *
import numpy as np
from openpyxl import Workbook
from openpyxl import load_workbook

#random initial condition generator 
def generate_random_initial(n):
    return  0.5*np.ones(n) + 0.02*np.random.random_sample(n) - 0.01

length_of_dom = 100
mesh_size = 10
dom_size = mesh_size*length_of_dom
interpolation_degree = 2

#allencahn Constants
delt = 1e-3
sim_time = 10.0
inter_loc = length_of_dom * 0.9
dt = delt
l_sigma = Constant(1.0)
l_eta = Constant(1.0)
alpha = Constant(0.5)
kappa = Constant(1)
nFbyRT = Constant(2*96485/(8.3144958*298))

del_eta = -0.01

run_number = 9
save_step = 1000

mesh = IntervalMesh(dom_size , 0, length_of_dom)
V = FunctionSpace(mesh, 'Lagrange', interpolation_degree)


def G(xi):
    return 4*xi**3 - 6*xi**2 + 2*xi

def H(xi):
    return 30*xi**4 -60*xi**3 +30*xi**2
    
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


#Initial Condition For Allen Cahn

e_old = Function(V)
e_old = interpolate(Expression('1/(1 + pow(const_e,(-(x[0] - inter_loc))))',degree = 2, const_e = np.e, inter_loc = inter_loc), V)
eta = Function(V)
eta = del_eta*e_old

#Initial Condition For Additives

# xb_old = Function(V)
# xb_old = interpolate(Constant(0.01), V)
# # coords = V.tabulate_dof_coordinates()
# # initial = np.zeros(coords.shape[0])
# # for i in range(coords.shape[0]):
# #     if i < length_of_dom*0.89:
# #         initial[i] = 1.0

# # xb_old.vector().set_local(initial)
# # print(xb_old.vector().get_local())

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
for i in boundary_conditions:
    if 'Neumann' in boundary_conditions[i]:
        g = boundary_conditions[i]['Neumann']
        integral_N1.append(kappa*g*q*ds(i))
    

        

#Weak Form of Allen Cahn equation
A = (
    e * q * dx

    - e_old * q * dx

    + dt * l_sigma * G(e) * q * dx

    + dt * l_sigma *  inner(nabla_grad(q),kappa *nabla_grad(e)) * dx

    - dt * l_sigma * sum(integral_N1)

    + dt * l_eta * H(e_old) * (np.e**( (1-alpha)*nFbyRT*(eta)) - np.e**(-alpha*nFbyRT*(eta)) )* q * dx
)



e_File = File('Tafel_plot/study_number{}/e.pvd'.format(run_number))

#Prepping to write an XLSX file
coords = V.tabulate_dof_coordinates()
print(coords)

headers1 = ['time','grad_e_dof', 'Coordinate']
wb = Workbook()
file_name = 'Tafel_plot/study_number{}/Tafel_plot{}.xlsx'.format(run_number,del_eta)

ws1 = wb.active
ws1.title = 'Tafel Plot'
ws1.append(headers1)


set_log_active(False)

iter = 0.0
for t in np.arange(0, sim_time, delt):

    solve(A == 0, e, bcs1)
    print('Solved A-C equation')

    e_old_dof = e_old.vector().get_local()
    # Computing and storing maximum gradients 
    # W = VectorFunctionSpace(mesh, "Lagrange", 1)
    # grad_e =project(grad(e), W)
    # grad_e_dof = grad_e.vector().get_local()
    # coords_ge = W.tabulate_dof_coordinates()

    #     max_grad = np.max(grad_e_dof)
    #     for i in range(len(grad_e_dof)):
    #         if max_grad == grad_e_dof[i]:
    #             ws1.append([t, max_grad, coords_ge[i][0]])

    if iter % save_step == 0 :
        e_File << (e_old, t)

        for i in range(coords.shape[0]):
            ws1.append([t, e_old_dof[i], coords[i][0]])
         

            
    # if norm1 < tol :
    #     e_File << (e_old, t)
    #     print('norm reached')
    #     break
    

    if t == sim_time:
        e_File << (e_old, t + delt)
        
        for i in range(coords.shape[0]):
            ws1.append([t, e_old_dof[i], coords[i][0]])
          


 
    e_old.assign(e)
    iter += 1
    # plot(e_old, title = 'Order Parameter Evolution')
    # plt.draw()
    # plt.pause(0.1)
    # plt.clf()

wb.save(filename = file_name)
# plt.close()