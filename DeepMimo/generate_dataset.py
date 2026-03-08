#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import DeepMIMOv3
import numpy as np

# Load the default parameters
parameters = DeepMIMOv3.default_params()

# Set the main folder containing extracted scenarios
parameters['dataset_folder'] = '/home/lyx/rmc/DeepMimo/scenarios'

# Ray-tracing scenario
parameters['scenario'] = 'I3_2p4'  # The adopted ray tracing scenario

# Dynamic Scenario Scenes
parameters['dynamic_scenario_scenes'] = [1]  # 仅用于动态场景，[1]表示只使用第一个时间帧/场景快照

# Active base stations
parameters['active_BS'] = np.array([1,2])  # 同时激活基站1和基站2

# Active users
# parameters['user_row_first'] = 1    # The first row of the considered user section
# parameters['user_row_last'] = 1159  # The last row of the considered user section
parameters['user_rows'] = np.arange(0,1159)  #明确指定用户行的索引范围

# Subsampling of active users
parameters['row_subsampling'] = 1  # 不跳过任何用户行，每1行取1行
parameters['user_subsampling'] = 1 # 不跳过行内任何用户，每1个用户取1个

# Antenna array dimensions
parameters['bs_antenna'] = {
    'shape': np.array([8, 4]),  # Horizontal × Vertical BS antenna array elements
    'spacing': 0.5,  # Half-wavelength spacing
    'rotation': np.array([0, -45, 0]),  # Rotation angles [slant, downtilt, bearing]
    'FoV': np.array([180, 180]),  # Field of view (horizontal, vertical)
    'radiation_pattern': 'isotropic'  # 0 in MATLAB means isotropic
}

parameters['ue_antenna'] = {
    'shape': np.array([1, 1]),  # Horizontal × Vertical UE antenna elements
    'spacing': 0.5,  # Half-wavelength spacing
    'rotation': np.array([0, 0, 0]),  # UE antenna orientation
    'FoV': np.array([360, 180]),  # Field of view (horizontal, vertical)
    'radiation_pattern': 'isotropic'  # 0 in MATLAB means isotropic
}

# System parameters
parameters['bandwidth'] = 0.05  # The bandwidth in GHz
parameters['activate_RX_filter'] = 1  # Apply RX low-pass filter (ideal: Sinc in time domain)

# Channel parameters
parameters['num_paths'] = 25  # Maximum number of paths to be considered (1-25)

# OFDM parameters
parameters['OFDM'] = {
    'subcarriers': 512,  # Number of OFDM subcarriers
    'selected_subcarriers': np.arange(8, 513, 8),  # Sampled subcarriers (8, 16, ..., 512)
    'bandwidth': 0.05,  # 50 MHz
    'RX_filter': 1  # Activate RX filtering
}

# Additional settings
parameters['enable_doppler'] = 0  # Disable Doppler shift
parameters['enable_dual_polar'] = 0  # Disable cross dual-polar antenna
# 打印参数（可选）
for key, value in parameters.items():
    print(f"{key}: {value}")


# Generate data
dataset = DeepMIMOv3.generate_data(parameters)


# In[2]:


# 打印数据集的字段名
print(dataset[0].keys())

print(len(dataset[0]['user']['location']))


# In[3]:


# Number of basestations
len(dataset)


# In[4]:


# Keys of a basestation dictionary
dataset[0].keys()


# In[12]:


# Keys of a channel
dataset[0]['user'].keys()
dataset[0]['user']['LoS']


# In[13]:


# Number of UEs
len(dataset[0]['user']['channel'])


# In[14]:


# Shape of the channel matrix
dataset[0]['user']['channel'].shape


# In[29]:


# Shape of BS 0 - UE 0 channel
dataset[0]['user']['channel'][0].shape


# In[31]:


# Path properties of BS 0 - UE 0
print(dataset[0]['user']['paths'][0])
print(dataset[1]['user']['paths'][0])


# In[18]:


## Visualization of a channel matrix

from matplotlib import pyplot as plt


# Visualize channel magnitude response
# First, select indices of a user and bs
ue_idx = 1000
for bs_idx in range(len(parameters['active_BS'])):
    # Import channel
    plt.figure()
    channel = dataset[bs_idx]['user']['channel'][ue_idx]
    print(channel.shape)
    # Take only the first antenna pair
    plt.imshow(np.abs(channel.T))
    plt.title('Channel Magnitude Response'+' BS{}'.format(bs_idx+1))
    plt.xlabel('TX Antennas')
    plt.ylabel('Subcarriers')
    plt.show()


# # 保存数据到数据集

# In[ ]:


# 保存到文件 USER pos & CSI & LoS status
pos_houzhui = ['_BS1.txt','_BS2.txt']
csi_houzhui = ['_BS1.npy','_BS2.npy']
LoS_houzhui = ['_BS1.txt','_BS2.txt']

for bs_idx in range(len(parameters['active_BS'])):
    pos_save_name = parameters['dataset_folder'] + '/' + parameters['scenario'] + '/UE_pos'
    np.savetxt(pos_save_name + pos_houzhui[bs_idx], dataset[bs_idx]['user']['location'], fmt="%.6f", delimiter=" ")
    
    # 保存LoS状态数据
    LoS_status_save_name = parameters['dataset_folder'] + '/' + parameters['scenario'] + '/UE_LoS'
    np.savetxt(LoS_status_save_name + LoS_houzhui[bs_idx], 
              dataset[bs_idx]['user']['LoS'], 
              fmt="%d", 
              delimiter=" ")
    
    csi_save_name = parameters['dataset_folder'] + '/' + parameters['scenario'] + '/UE_CSI'
    dataset[bs_idx]['user']['channel'].shape
    np.save(csi_save_name + csi_houzhui[bs_idx],dataset[bs_idx]['user']['channel'])


# In[17]:


print(bs_idx)
print(f"BS{bs_idx+1} location:", dataset[bs_idx]['location'])


# In[19]:


## Visualization of the UE positions and path-losses

for bs_idx in range(len(parameters['active_BS'])):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    # 绘制当前BS覆盖的UE位置和路径损耗
    loc_x = dataset[bs_idx]['user']['location'][:, 0]
    loc_y = dataset[bs_idx]['user']['location'][:, 1]
    loc_z = dataset[bs_idx]['user']['location'][:, 2]
    pathloss = dataset[bs_idx]['user']['pathloss']
    im = ax.scatter(loc_x, loc_y, loc_z, c=pathloss)
    
    # 绘制当前BS的位置（红色）
    # 1. 绘制所有基站位置
    all_bs_locations = dataset[bs_idx]['basestation']['location']  # 所有基站位置
    for i, bs_loc in enumerate(all_bs_locations):
        if i == bs_idx:  # 当前激活基站
            ax.scatter(bs_loc[0], bs_loc[1], bs_loc[2], 
                      c='darkred', s=200, marker='.', label=f'Active BS{bs_idx+1}')
        else:  # 其他基站
            ax.scatter(bs_loc[0], bs_loc[1], bs_loc[2], 
                      c='lightcoral', s=100, marker='.', label=f'BS{i+1}')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    plt.title(f'UE and BS{bs_idx+1} Positions')
    plt.legend()
    plt.colorbar(im, label='Pathloss (dB)')
    plt.show()  # 显示当前图，避免叠加
    
    # 绘制二维图
    fig = plt.figure()
    ax = fig.add_subplot()
    im = ax.scatter(loc_x, loc_y, c=pathloss)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    fig.colorbar(im, ax=ax)
    ttl = plt.title('UE Grid Path-loss (dB)')
    # 绘制所有基站位置
    all_bs_locations = dataset[bs_idx]['basestation']['location']  # 所有基站位置
    for i, bs_loc in enumerate(all_bs_locations):
        if i == bs_idx:  # 当前激活基站
            ax.scatter(bs_loc[0], bs_loc[1], 
                      c='darkred',s=200, marker='.', label=f'Active BS{bs_idx+1}')
        else:  # 其他基站
            ax.scatter(bs_loc[0], bs_loc[1], 
                      c='lightcoral',s=100, marker='.', label=f'BS{i+1}')
    plt.show()


# In[22]:


from DeepMIMOv3.visualization import plot_LoS_status, plot_coverage
# Plotting LoS status for all users
bs_idx = 1  # 选择基站索引

# 提取数据（确保形状正确）
bs_location = dataset[bs_idx]['location'] # 当前基站位置 (3,)
user_location = dataset[bs_idx]['user']['location']  # 用户位置 (n_UE, 3)
LoS_status = dataset[bs_idx]['user']['LoS']  # 转换为 0/1

# 验证数据
print("BS location:", bs_location)
print("User locations shape:", user_location.shape)
print("LoS status samples:", LoS_status[:5])  # 查看前5个值

# 绘制LoS状态
plot_LoS_status(bs_location, user_location, LoS_status)
plt.title(f'LoS Status for BS{bs_idx + 1} and its UEs')
plt.show()


# In[18]:


# Automatic 2D/3D plot
var_names = ['LoS', 'pathloss', 'distance']
for var_name in var_names:
    plot_coverage(dataset[0]['user']['location'], dataset[0]['user'][var_name],
                  bs_pos=dataset[0]['location'], cbar_title=var_name, proj_3D=False)


# In[19]:


# Plot all path parameters
var_names = ['DoD_phi', 'DoD_theta', 'DoA_phi', 'DoA_theta', 'ToA', 'phase', 'power']
n_users = dataset[0]['user']['paths'].shape[0]
plt_map = np.zeros(n_users) * np.nan # set NaN to users with no paths
for var_name in var_names:
    plt_map = [dataset[0]['user']['paths'][i][var_name][0]
               if dataset[0]['user']['paths'][i]['num_paths'] else np.nan for i in range(n_users)]
    plot_coverage(dataset[0]['user']['location'], plt_map, bs_pos=dataset[0]['location'],
                  bs_ori=parameters['bs_antenna']['rotation']*np.pi/180,
                  proj_3D=False, title=var_name, cbar_title=var_name)


# In[20]:


from DeepMIMOv3.utils import steering_vec, uniform_sampling, trim_by_idx, \
                             LinearPath, get_idxs_in_xy_box

# Make a codebook matrix with 25 beam steering vectors from -60º to 60º azimuth
n_beams = 25
beam_angles = np.around(np.linspace(-60, 60, n_beams), 2)
F1 = np.array([steering_vec(parameters['bs_antenna']['shape'], phi=azi,
                            spacing=parameters['bs_antenna']['spacing']).squeeze()
               for azi in beam_angles])
# F1 is n_beams x n_ant

# Select one every four indices across x and y directions. Result 16x less indices
uniform_idxs = uniform_sampling([4,4], n_rows=len(parameters['user_rows']), users_per_row=411)
# The values <n_rows> and  <users_per_row> should be acquired from the scenario page.

# Trim dataset without regenerating.
# dataset_t = trim_by_idx(dataset[0], uniform_idxs) # uniform_idxs obtained from uniform_sampling(.)

# obtain the indices of the users in the zone
idxs_in_box = get_idxs_in_xy_box(dataset[0]['user']['location'],
                                 x_min=18, x_max=28, y_min=8, y_max=12)

# show map
plot_coverage(dataset[0]['user']['location'], dataset[0]['user']['LoS'],
              title='LoS status with selected users', cbar_title='LoS status')

# show selected users in the map
plt.scatter(dataset[0]['user']['location'][idxs_in_box,0],
            dataset[0]['user']['location'][idxs_in_box,1],
            label='box A', s=2, lw=.1, alpha=.1)


# In[21]:


# Create a linear path
linpath = LinearPath(dataset[0], first_pos=[100, 90], last_pos=[-50, 90], n_steps=75)

# Show it on the map
plot_coverage(dataset[0]['user']['location'], dataset[0]['user']['LoS'],
              title='LoS status with Linear Path', cbar_title='LoS status')
plt.scatter(linpath.pos[:,0], linpath.pos[:,1], c='blue', label='linear path')

# Plot features along the path
for var_name in linpath.get_feature_names():
    plt.plot(getattr(linpath, var_name), ls='-',  c='blue', marker='*', markersize=7)
    plt.xlabel('position index along the path')
    plt.ylabel(f'{var_name}')
    plt.grid()
    plt.show()

