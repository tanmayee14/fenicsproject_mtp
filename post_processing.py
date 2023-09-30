#Code to determine the average velocity of the interface from the excel file output for order-parameter vs x-coordinate "postprocessing"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

study_num = 9
domain_density = 2000
del_t = 1
over_potential = -0.01

file_loc = 'D:\IIT BBS\MTP\Tafel_plot\study_number{}\Tafel_plot{}.xlsx'
file_name = file_loc.format(study_num,over_potential)
df = pd.read_excel(file_name)


location = []
iter = 0
start = 2
for i in range(len(df.iloc[:,0].unique())):
    selected = df.iloc[(start+iter) + domain_density*i: (start+iter+domain_density)+ domain_density*i, 1]

    y = np.zeros(selected.shape)
    y_dash = np.ones(selected.shape)

    lower_limit = np.where(selected < 0.5, selected, y)
    lower_limit.sort()

    upper_limit = np.where(selected >= 0.5, selected, y_dash)
    upper_limit.sort()

    lo_idx = list(selected).index(lower_limit[-1])
    hi_idx = list(selected).index(upper_limit[0])

    lo = df.iloc[lo_idx+2, 2]
    hi = df.iloc[hi_idx+2, 2]
   # print(selected)
    print(iter)
    print('lower_limit :', lower_limit[-1], 'coordinate :', lo)
    print('upper_limit :', upper_limit[0], 'coordinate :', hi)
    
    loc = (hi - lo)/(upper_limit[0] - lower_limit[-1]) * (0.5 - lower_limit[-1]) + lo
    print('Location of 0.5 :', loc)
    location.append(loc)

    iter += 1

velocity = []
for i in range(len(location)-1):
    velocity.append((location[i + 1] - location[i]) / del_t)

fig, ax = plt.subplots()
ax.plot(range(len(velocity)), velocity)
ax.set_title('Velocity vs Iterations')
ax.set_xlabel('Iteration')
ax.set_ylabel('Velocity')
plt.show()
plt.close()

average_velocity = np.mean(velocity[-10:-1])
velocity.append(average_velocity)
print('average_velocity :', average_velocity)

index = list(range(len(velocity)-1))
index.append('Average Velocity')
vel = pd.DataFrame(velocity, columns=['Velocity'], index = index)
vel.to_excel('D:\IIT BBS\MTP\Tafel_plot\study_number{}\ velocity.xlsx'.format(study_num))



