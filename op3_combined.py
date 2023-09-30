#code to run the overall simulation of the system with the Phase Field Model, Diffusion of CU ions and Diffusion of Additives


import matplotlib.pyplot as plt
from dolfin import *
import numpy as np
from ufl import nabla_div
from openpyxl import Workbook


#random initial condition generator 
def generate_random_initial(n):
    return  0.5*np.ones(n) + 0.02*np.random.random_sample(n) - 0.01

length_of_dom = 100
mesh_size = 5
dom_size = mesh_size*length_of_dom
interpolation_degree = 2

# Allencahn Constants
delt = 1e-3
sim_time = 10.0
inter_loc = length_of_dom * 0.8
dt = delt
l_sigma = Constant(1.0)
l_eta = Constant(1.0)
alpha = Constant(0.5)
kappa = Constant(1)
nFbyRT = Constant(2*96485/(8.3144958*298))

del_eta = -0.1

# Additives Constants

#mobility = Constant(5.0e-4)
#RT = Constant(8.3144958*298)
#Wb = Constant( -48913.402)
#Wa = Constant(14886.57)

mobility = Constant(1)
RT = Constant(1)
Wb = Constant(-19.7428)
Wa = Constant(6.0085)

# Copper Diffusion Constants

Vm = 7.11 # molar volume of Cu [cc/mol]
Ds = 1.00
cs = 1/Vm
c0 = 1.0 # 1 M solution bulk concentration 
eta_e = -0.1
eta_s = 0.0


run_number = 1
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

# Phase-Field Variables
e = Function(V)
q = TestFunction(V)

# Additives Variables
xb = Function(V)
u = TestFunction(V)

# Copper Diffusion Variables
cp = Function(V)
delcp = TestFunction(V)


# Initial Condition For Allen Cahn

e_old = Function(V)
e_old = interpolate(Expression('1/(1 + pow(const_e,(-(x[0] - inter_loc))))',degree = 2, const_e = np.e, inter_loc = inter_loc), V)
eta = Function(V)
eta = del_eta*e_old


# Initial Condition For Additives

xb_old = Function(V)
xb_old = interpolate(Constant(0.01), V)
# coords = V.tabulate_dof_coordinates()
# initial = np.zeros(coords.shape[0])
# for i in range(coords.shape[0]):
#     if i < length_of_dom*0.89:
#         initial[i] = 1.0

# xb_old.vector().set_local(initial)
# print(xb_old.vector().get_local())


# Initial Conditions for copper diffusion

cp_old = Function(V)
cp_old = interpolate(Expression('cp_s*(1-1/(1 + pow(const_e,(-(x[0] - xm)))))',degree = 2, cp_s=1, const_e = np.e, xm=inter_loc), V)


# defining Boundary Conditions for allen cahn equation

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
        


# Boundary Conditions For Copper Diffusion
Dxi = Ds*(1-e_old)

boundary_conditions_cp = { 
                        0 : {'Dirichlet': (c0, bx1)},
                        1 : {'Neumann': 0.0}}


bcscp = []
for i in boundary_conditions_cp:
    if 'Dirichlet' in boundary_conditions_cp[i]:
        expression, sub_domain = boundary_conditions_cp[i]['Dirichlet']
        bc = DirichletBC(V, expression, sub_domain)
        bcscp.append(bc) 

bnd_integrals_cp = []
for i in boundary_conditions_cp:
    if 'Neumann' in boundary_conditions_cp[i]:
        grad_cp = boundary_conditions_cp[i]['Neumann']
        bnd_integrals_cp.append(delcp*Dxi*grad_cp*ds(i))




#Weak Form of Allen Cahn equation, Additives, Copper Diffusion
A = (
    e * q * dx

    - e_old * q * dx

    + dt * l_sigma * (Wb*xb_old + Wa*(1-xb_old)) * G(e) * q * dx

    + dt * l_sigma *  inner(nabla_grad(q),kappa *nabla_grad(e)) * dx

    - dt * l_sigma * sum(integral_N1)

    + dt * l_eta * H(e_old) * (np.e**( (1-alpha)*nFbyRT*(eta)) - (cp_old)*np.e**(-alpha*nFbyRT*(eta)) )* q * dx
)


F = (
    xb * u * dx

    - xb_old * u * dx

    - dt * mobility * RT  * sum(integral_N2)

    + dt * mobility * RT * inner(nabla_grad(u), a(xb_old)*nabla_grad(xb)) * dx

    - dt * mobility * (Wb - Wa) * sum(integral_N3)

    + dt * mobility * (Wb - Wa) *inner(nabla_grad(u), b(xb_old)*G(e)*nabla_grad(e)) * dx
)


Fcp = (
    cp * delcp * dx

    - cp_old * delcp * dx

    + dt * inner(nabla_grad(delcp), Dxi*nabla_grad(cp)) * dx

    - dt * sum(bnd_integrals_cp)

    - dt * nabla_div( Dxi*cp*(eta_e-eta_s)*nabla_grad(e_old) ) * delcp * dx

    + cp * cs/c0 * (e - e_old) * delcp * dx
)



e_File = File('op3_combined/study_number{}/order_parameter.pvd'.format(run_number))
xb_File = File('op3_combined/study_number{}/xb.pvd'.format(run_number))
cp_file = File('op3_combined/study_number{}/cp.pvd'.format(run_number))


#Prepping to write an XLSX file
# coords = V.tabulate_dof_coordinates()

# headers1 = ['Time', 'xb_max', 'xb_min']
# headers2 = ['x', 'phi']
# wb = Workbook()
# file_name = 'op3_combined/study_number{}/Wb-Wa_study_{}.xlsx'.format(run_number, run_number)

# ws1 = wb.active
# ws1.title = 'Wb - Wa study'
# ws2 = wb.create_sheet('Order_parameter')
# ws1.append(headers1)
# ws2.append(headers2)


# solving the Allen cahn and Additives Equations
tol = 1E-4
iter = 0.0
for t in np.arange(0, sim_time, delt):

    solve(A == 0, e, bcs1)
    print('Solved A-C equation')
    solve(F == 0, xb, bcs2)
    print('Solved diffusion equation')
    solve(Fcp == 0, cp, bcscp)
    print('solved Cu diffusion equation')
    if iter % save_step == 0 :
        e_File << (e_old, t)
        xb_File << (xb_old, t)
        cp_file << (cp_old, t)

    e_old.assign(e)
    xb_old.assign(xb)
    cp_old.assign(cp)


    if t == sim_time:
        e_File << (e_old, t + delt)
        xb_File << (xb_old, t + delt)
        cp_file << (cp_old, t + delt)


    iter += 1
    plot(e_old, title = 'Order Parameter Evolution')
    plot(cp_old, title = "Cu diffusion")
    plot(xb_old, title = 'Additive Xb')
    plt.legend(['Phi','Cu+2', 'Xb'])
    plt.draw()
    plt.pause(0.1)
    plt.clf()


plt.close()